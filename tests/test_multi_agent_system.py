"""
test_multi_agent_system.py

Tests para src/multi_agent_system.py.
Cubre: inicialización, pipeline completo process(), evaluación opcional,
manejo de errores (trace actualizado con ERROR), run_test_queries con y
sin expected_agent, routing correcto e incorrecto.

Todas las dependencias externas (Langfuse, agentes, vector stores) están
mockeadas para tests rápidos y deterministas.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures específicos del módulo
# ---------------------------------------------------------------------------


def _make_rag_result(domain="hr", agent_name="HRAgent", answer="Respuesta de prueba."):
    return {
        "agent": agent_name,
        "domain": domain,
        "query": "consulta de prueba",
        "answer": answer,
        "retrieved_docs": ["hr_docs/vacaciones.txt"],
        "context": "[Documento 1]\n15 días de vacaciones.",
    }


def _make_classification(domain="hr", confidence=0.95, reasoning="Consulta de RRHH."):
    return {"domain": domain, "confidence": confidence, "reasoning": reasoning}


@pytest.fixture
def mock_system():
    """
    MultiAgentSystem completamente mockeado.
    Reemplaza todas las dependencias externas y retorna el sistema listo para testear.
    """
    with (
        patch("src.multi_agent_system.Langfuse") as mock_lf,
        patch("src.multi_agent_system.CallbackHandler"),
        patch("src.multi_agent_system.build_all_vector_stores") as mock_build,
        patch("src.multi_agent_system.Orchestrator") as mock_orch_cls,
        patch("src.multi_agent_system.HRAgent") as mock_hr,
        patch("src.multi_agent_system.TechAgent") as mock_tech,
        patch("src.multi_agent_system.FinanceAgent") as mock_finance,
        patch("src.multi_agent_system.LegalAgent") as mock_legal,
        patch("src.multi_agent_system.EvaluatorAgent") as mock_eval_cls,
    ):
        mock_build.return_value = {d: MagicMock() for d in ("hr", "tech", "finance", "legal")}

        # Orchestrator mock
        mock_orchestrator = MagicMock()
        mock_orchestrator.classify.return_value = _make_classification()
        mock_orch_cls.return_value = mock_orchestrator

        # Agente HR mock (el que se usará por defecto)
        mock_hr_agent = MagicMock()
        mock_hr_agent.run.return_value = _make_rag_result()
        mock_hr.return_value = mock_hr_agent

        for mock_cls in (mock_tech, mock_finance, mock_legal):
            agent = MagicMock()
            agent.run.return_value = _make_rag_result()
            mock_cls.return_value = agent

        # Trace mock
        mock_trace = MagicMock()
        mock_lf.return_value.trace.return_value = mock_trace

        # Evaluator mock
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = {
            "relevance": 9,
            "completeness": 8,
            "accuracy": 9,
            "overall": 9,
            "feedback": "Excelente respuesta.",
        }
        mock_eval_cls.return_value = mock_evaluator

        from src.multi_agent_system import MultiAgentSystem

        system = MultiAgentSystem(enable_evaluation=True)

        # Exponer mocks para assertions en tests
        system._mock_orchestrator = mock_orchestrator
        system._mock_trace = mock_trace
        system._mock_evaluator = mock_evaluator
        system._mock_langfuse = mock_lf.return_value

        yield system


# ---------------------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------------------


class TestMultiAgentSystemInit:
    def test_init_with_evaluation_enabled(self):
        with (
            patch("src.multi_agent_system.Langfuse"),
            patch("src.multi_agent_system.CallbackHandler"),
            patch("src.multi_agent_system.build_all_vector_stores") as mb,
            patch("src.multi_agent_system.Orchestrator"),
            patch("src.multi_agent_system.HRAgent"),
            patch("src.multi_agent_system.TechAgent"),
            patch("src.multi_agent_system.FinanceAgent"),
            patch("src.multi_agent_system.LegalAgent"),
            patch("src.multi_agent_system.EvaluatorAgent") as mock_ev,
        ):
            mb.return_value = {d: MagicMock() for d in ("hr", "tech", "finance", "legal")}

            from src.multi_agent_system import MultiAgentSystem

            system = MultiAgentSystem(enable_evaluation=True)

        assert system.enable_evaluation is True
        mock_ev.assert_called_once()
        assert system.evaluator is not None

    def test_init_with_evaluation_disabled(self):
        with (
            patch("src.multi_agent_system.Langfuse"),
            patch("src.multi_agent_system.CallbackHandler"),
            patch("src.multi_agent_system.build_all_vector_stores") as mb,
            patch("src.multi_agent_system.Orchestrator"),
            patch("src.multi_agent_system.HRAgent"),
            patch("src.multi_agent_system.TechAgent"),
            patch("src.multi_agent_system.FinanceAgent"),
            patch("src.multi_agent_system.LegalAgent"),
            patch("src.multi_agent_system.EvaluatorAgent") as mock_ev,
        ):
            mb.return_value = {d: MagicMock() for d in ("hr", "tech", "finance", "legal")}

            from src.multi_agent_system import MultiAgentSystem

            system = MultiAgentSystem(enable_evaluation=False)

        assert system.enable_evaluation is False
        mock_ev.assert_not_called()
        assert system.evaluator is None

    def test_agents_dict_has_all_domains(self, mock_system):
        assert set(mock_system.agents.keys()) == {"hr", "tech", "finance", "legal"}


# ---------------------------------------------------------------------------
# process()
# ---------------------------------------------------------------------------


class TestMultiAgentSystemProcess:
    def test_returns_expected_keys(self, mock_system):
        result = mock_system.process("¿Cuántos días de vacaciones tengo?")
        expected_keys = {
            "trace_id",
            "query",
            "domain",
            "confidence",
            "reasoning",
            "agent",
            "answer",
            "retrieved_docs",
            "evaluation",
            "langfuse_url",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_calls_orchestrator_classify(self, mock_system):
        mock_system.process("query de prueba")
        mock_system._mock_orchestrator.classify.assert_called_once()
        call_args = mock_system._mock_orchestrator.classify.call_args
        assert call_args[0][0] == "query de prueba"

    def test_routes_to_correct_agent(self, mock_system):
        mock_system._mock_orchestrator.classify.return_value = _make_classification("tech")
        mock_system.agents["tech"].run.return_value = _make_rag_result("tech", "TechAgent")

        result = mock_system.process("Mi laptop no enciende.")

        mock_system.agents["tech"].run.assert_called_once()
        assert result["domain"] == "tech"

    def test_includes_evaluation_when_enabled(self, mock_system):
        result = mock_system.process("¿Cuántos días de vacaciones tengo?")
        assert result["evaluation"] is not None
        assert result["evaluation"]["overall"] == 9

    def test_calls_evaluator_with_context(self, mock_system):
        mock_system.process("query")
        call_kwargs = mock_system._mock_evaluator.evaluate.call_args[1]
        assert "context" in call_kwargs
        assert call_kwargs["context"] is not None

    def test_langfuse_trace_created(self, mock_system):
        mock_system.process("query")
        mock_system._mock_langfuse.trace.assert_called_once()

    def test_langfuse_flushed(self, mock_system):
        mock_system.process("query")
        mock_system._mock_langfuse.flush.assert_called()

    def test_langfuse_url_in_result(self, mock_system):
        result = mock_system.process("query")
        assert result["langfuse_url"].startswith("https://")
        assert result["trace_id"] in result["langfuse_url"]

    def test_user_id_passed_to_trace(self, mock_system):
        mock_system.process("query", user_id="user-123")
        trace_call = mock_system._mock_langfuse.trace.call_args[1]
        assert trace_call["user_id"] == "user-123"

    def test_anonymous_user_when_no_user_id(self, mock_system):
        mock_system.process("query")
        trace_call = mock_system._mock_langfuse.trace.call_args[1]
        assert trace_call["user_id"] == "anonymous"

    def test_evaluation_skipped_when_disabled(self):
        with (
            patch("src.multi_agent_system.Langfuse") as mock_lf,
            patch("src.multi_agent_system.CallbackHandler"),
            patch("src.multi_agent_system.build_all_vector_stores") as mb,
            patch("src.multi_agent_system.Orchestrator") as mock_orch_cls,
            patch("src.multi_agent_system.HRAgent") as mock_hr,
            patch("src.multi_agent_system.TechAgent"),
            patch("src.multi_agent_system.FinanceAgent"),
            patch("src.multi_agent_system.LegalAgent"),
            patch("src.multi_agent_system.EvaluatorAgent"),
        ):
            mb.return_value = {d: MagicMock() for d in ("hr", "tech", "finance", "legal")}
            mock_orch_cls.return_value.classify.return_value = _make_classification()
            mock_hr.return_value.run.return_value = _make_rag_result()
            mock_lf.return_value.trace.return_value = MagicMock()

            from src.multi_agent_system import MultiAgentSystem

            system = MultiAgentSystem(enable_evaluation=False)
            result = system.process("query")

        assert result["evaluation"] is None

    def test_exception_updates_trace_with_error(self, mock_system):
        mock_system._mock_orchestrator.classify.side_effect = RuntimeError("Fallo del LLM")

        with pytest.raises(RuntimeError, match="Fallo del LLM"):
            mock_system.process("query que falla")

        # El trace debe haberse actualizado con nivel ERROR
        update_calls = mock_system._mock_trace.update.call_args_list
        error_calls = [c for c in update_calls if c[1].get("level") == "ERROR"]
        assert len(error_calls) == 1
        assert "Fallo del LLM" in error_calls[0][1]["status_message"]

    def test_exception_flushes_langfuse_before_reraise(self, mock_system):
        mock_system._mock_orchestrator.classify.side_effect = ValueError("Error de clasificación")

        with pytest.raises(ValueError):
            mock_system.process("query")

        mock_system._mock_langfuse.flush.assert_called()

    def test_result_contains_query(self, mock_system):
        result = mock_system.process("Mi consulta específica.")
        assert result["query"] == "Mi consulta específica."

    def test_trace_id_is_uuid_format(self, mock_system):
        import re

        result = mock_system.process("query")
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        assert re.match(uuid_pattern, result["trace_id"])


# ---------------------------------------------------------------------------
# run_test_queries()
# ---------------------------------------------------------------------------


class TestRunTestQueries:
    def test_processes_all_queries(self, mock_system):
        queries = [
            {"query": "Consulta 1"},
            {"query": "Consulta 2"},
            {"query": "Consulta 3"},
        ]
        results = mock_system.run_test_queries(queries)
        assert len(results) == 3

    def test_routing_correct_when_domain_matches(self, mock_system):
        mock_system._mock_orchestrator.classify.return_value = _make_classification("hr")
        queries = [{"query": "Consulta HR", "expected_agent": "hr"}]

        results = mock_system.run_test_queries(queries)

        assert results[0]["routing_correct"] is True
        assert results[0]["expected_agent"] == "hr"

    def test_routing_incorrect_when_domain_mismatches(self, mock_system):
        mock_system._mock_orchestrator.classify.return_value = _make_classification("tech")
        queries = [{"query": "Consulta mal enrutada", "expected_agent": "hr"}]

        results = mock_system.run_test_queries(queries)

        assert results[0]["routing_correct"] is False

    def test_no_expected_agent_skips_validation(self, mock_system):
        queries = [{"query": "Consulta sin expected_agent"}]
        results = mock_system.run_test_queries(queries)
        assert "routing_correct" not in results[0]

    def test_accuracy_log_only_when_expected_agents_present(self, mock_system, caplog):
        import logging

        queries = [{"query": "Consulta sin expected_agent"}]

        with caplog.at_level(logging.INFO, logger="src.multi_agent_system"):
            mock_system.run_test_queries(queries)

        assert not any("ACCURACY" in r.message for r in caplog.records)

    def test_accuracy_log_when_expected_agents_present(self, mock_system, caplog):
        import logging

        mock_system._mock_orchestrator.classify.return_value = _make_classification("hr")
        queries = [{"query": "Consulta HR", "expected_agent": "hr"}]

        with caplog.at_level(logging.INFO, logger="src.multi_agent_system"):
            mock_system.run_test_queries(queries)

        assert any("ACCURACY" in r.message for r in caplog.records)
