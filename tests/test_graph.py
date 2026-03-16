"""
test_graph.py

Tests para src/graph.py.
Cubre: AgentState TypedDict, build_graph, nodo orchestrate, routing condicional
para los 4 dominios, nodos rag (hr/tech/finance/legal), evaluate_node con y sin
evaluador, y propagación de atributos de estado a través del grafo completo.
"""

from unittest.mock import MagicMock

import pytest

_TEST_CONFIG = {"configurable": {"thread_id": "test-thread"}}


def _invoke(graph, state):
    """Wrapper que añade el thread_id requerido por MemorySaver."""
    return graph.invoke(state, _TEST_CONFIG)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGENT_NAMES = {
    "hr": "HRAgent",
    "tech": "TechAgent",
    "finance": "FinanceAgent",
    "legal": "LegalAgent",
}


def _make_initial_state(query="consulta de prueba", trace_id="trace-abc-123"):
    return {
        "query": query,
        "user_id": "test-user",
        "trace_id": trace_id,
        "domain": "",
        "confidence": 0.0,
        "reasoning": "",
        "agent_name": "",
        "answer": "",
        "retrieved_docs": [],
        "context": "",
        "evaluation": None,
    }


def _make_orchestrator(domain="hr", confidence=0.9, reasoning="consulta de RRHH"):
    mock = MagicMock()
    mock.classify.return_value = {
        "domain": domain,
        "confidence": confidence,
        "reasoning": reasoning,
    }
    return mock


def _make_agents():
    agents = {}
    for domain in ("hr", "tech", "finance", "legal"):
        agent = MagicMock()
        agent.run.return_value = {
            "agent": AGENT_NAMES[domain],
            "answer": f"Respuesta del agente {domain}.",
            "retrieved_docs": [f"{domain}_docs/doc.txt"],
            "context": f"Contexto del dominio {domain}.",
        }
        agents[domain] = agent
    return agents


def _make_evaluator(scores=None):
    mock = MagicMock()
    mock.evaluate.return_value = scores or {
        "relevance": 9,
        "completeness": 8,
        "accuracy": 9,
        "overall": 9,
        "feedback": "Excelente respuesta.",
    }
    return mock


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_has_all_expected_keys(self):
        from src.graph import AgentState

        state: AgentState = _make_initial_state()
        expected_keys = {
            "query",
            "user_id",
            "trace_id",
            "domain",
            "confidence",
            "reasoning",
            "agent_name",
            "answer",
            "retrieved_docs",
            "context",
            "evaluation",
        }
        assert expected_keys == set(state.keys())

    def test_is_dict_subclass(self):
        from src.graph import AgentState

        assert issubclass(AgentState, dict)

    def test_evaluation_can_be_none(self):
        from src.graph import AgentState

        state: AgentState = _make_initial_state()
        assert state["evaluation"] is None

    def test_retrieved_docs_is_list(self):
        from src.graph import AgentState

        state: AgentState = _make_initial_state()
        assert isinstance(state["retrieved_docs"], list)


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_returns_non_none(self):
        from src.graph import build_graph

        graph = build_graph(_make_orchestrator(), _make_agents(), _make_evaluator())
        assert graph is not None

    def test_graph_is_invocable(self):
        from src.graph import build_graph

        graph = build_graph(_make_orchestrator(), _make_agents(), _make_evaluator())
        result = _invoke(graph, _make_initial_state())
        assert isinstance(result, dict)

    def test_graph_without_evaluator_is_invocable(self):
        from src.graph import build_graph

        graph = build_graph(_make_orchestrator(), _make_agents(), evaluator=None)
        result = _invoke(graph, _make_initial_state())
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# orchestrate_node (verificado vía graph.invoke)
# ---------------------------------------------------------------------------


class TestOrchestrateNode:
    def test_calls_orchestrator_classify_once(self):
        from src.graph import build_graph

        orchestrator = _make_orchestrator("hr")
        graph = build_graph(orchestrator, _make_agents(), None)

        _invoke(graph, _make_initial_state())

        orchestrator.classify.assert_called_once()

    def test_passes_query_to_classify(self):
        from src.graph import build_graph

        orchestrator = _make_orchestrator("hr")
        graph = build_graph(orchestrator, _make_agents(), None)

        _invoke(graph, _make_initial_state(query="¿Cuántos días de vacaciones?"))

        call_args = orchestrator.classify.call_args
        assert call_args[0][0] == "¿Cuántos días de vacaciones?"

    def test_passes_trace_id_to_classify(self):
        from src.graph import build_graph

        orchestrator = _make_orchestrator("hr")
        graph = build_graph(orchestrator, _make_agents(), None)

        _invoke(graph, _make_initial_state(trace_id="my-trace-id"))

        call_kwargs = orchestrator.classify.call_args[1]
        assert call_kwargs["trace_id"] == "my-trace-id"

    def test_domain_set_in_final_state(self):
        from src.graph import build_graph

        orchestrator = _make_orchestrator("tech")
        graph = build_graph(orchestrator, _make_agents(), None)

        result = _invoke(graph, _make_initial_state())

        assert result["domain"] == "tech"

    def test_confidence_set_in_final_state(self):
        from src.graph import build_graph

        orchestrator = _make_orchestrator("hr", confidence=0.85)
        graph = build_graph(orchestrator, _make_agents(), None)

        result = _invoke(graph, _make_initial_state())

        assert result["confidence"] == pytest.approx(0.85)

    def test_reasoning_set_in_final_state(self):
        from src.graph import build_graph

        orchestrator = MagicMock()
        orchestrator.classify.return_value = {
            "domain": "hr",
            "confidence": 0.9,
            "reasoning": "Es una consulta de RRHH.",
        }
        graph = build_graph(orchestrator, _make_agents(), None)

        result = _invoke(graph, _make_initial_state())

        assert result["reasoning"] == "Es una consulta de RRHH."

    def test_missing_confidence_defaults_to_zero(self):
        from src.graph import build_graph

        orchestrator = MagicMock()
        orchestrator.classify.return_value = {"domain": "hr"}
        agents = _make_agents()
        graph = build_graph(orchestrator, agents, None)

        result = _invoke(graph, _make_initial_state())

        assert result["confidence"] == pytest.approx(0.0)

    def test_missing_reasoning_defaults_to_empty(self):
        from src.graph import build_graph

        orchestrator = MagicMock()
        orchestrator.classify.return_value = {"domain": "hr"}
        agents = _make_agents()
        graph = build_graph(orchestrator, agents, None)

        result = _invoke(graph, _make_initial_state())

        assert result["reasoning"] == ""


# ---------------------------------------------------------------------------
# Routing condicional
# ---------------------------------------------------------------------------


class TestConditionalRouting:
    @pytest.mark.parametrize("domain", ["hr", "tech", "finance", "legal"])
    def test_routes_to_correct_domain_agent(self, domain):
        from src.graph import build_graph

        orchestrator = _make_orchestrator(domain)
        agents = _make_agents()
        graph = build_graph(orchestrator, agents, None)

        _invoke(graph, _make_initial_state())

        agents[domain].run.assert_called_once()

    @pytest.mark.parametrize("domain", ["hr", "tech", "finance", "legal"])
    def test_does_not_call_other_agents(self, domain):
        from src.graph import build_graph

        orchestrator = _make_orchestrator(domain)
        agents = _make_agents()
        graph = build_graph(orchestrator, agents, None)

        _invoke(graph, _make_initial_state())

        for other_domain, agent in agents.items():
            if other_domain != domain:
                agent.run.assert_not_called()


# ---------------------------------------------------------------------------
# rag_node (verificado vía graph.invoke)
# ---------------------------------------------------------------------------


class TestRagNode:
    def test_answer_in_final_state(self):
        from src.graph import build_graph

        orchestrator = _make_orchestrator("hr")
        agents = _make_agents()
        agents["hr"].run.return_value = {
            "agent": "HRAgent",
            "answer": "Tienes 15 días de vacaciones.",
            "retrieved_docs": ["hr_docs/vacaciones.txt"],
            "context": "Los empleados tienen 15 días.",
        }
        graph = build_graph(orchestrator, agents, None)

        result = _invoke(graph, _make_initial_state())

        assert result["answer"] == "Tienes 15 días de vacaciones."

    def test_agent_name_in_final_state(self):
        from src.graph import build_graph

        orchestrator = _make_orchestrator("hr")
        agents = _make_agents()
        graph = build_graph(orchestrator, agents, None)

        result = _invoke(graph, _make_initial_state())

        assert result["agent_name"] == "HRAgent"

    def test_retrieved_docs_in_final_state(self):
        from src.graph import build_graph

        orchestrator = _make_orchestrator("hr")
        agents = _make_agents()
        graph = build_graph(orchestrator, agents, None)

        result = _invoke(graph, _make_initial_state())

        assert "hr_docs/doc.txt" in result["retrieved_docs"]

    def test_context_in_final_state(self):
        from src.graph import build_graph

        orchestrator = _make_orchestrator("hr")
        agents = _make_agents()
        graph = build_graph(orchestrator, agents, None)

        result = _invoke(graph, _make_initial_state())

        assert result["context"] == "Contexto del dominio hr."

    def test_passes_query_to_agent_run(self):
        from src.graph import build_graph

        orchestrator = _make_orchestrator("hr")
        agents = _make_agents()
        graph = build_graph(orchestrator, agents, None)

        _invoke(graph, _make_initial_state(query="¿Cuántos días?"))

        call_args = agents["hr"].run.call_args
        assert call_args[0][0] == "¿Cuántos días?"

    def test_passes_trace_id_to_agent_run(self):
        from src.graph import build_graph

        orchestrator = _make_orchestrator("hr")
        agents = _make_agents()
        graph = build_graph(orchestrator, agents, None)

        _invoke(graph, _make_initial_state(trace_id="my-trace"))

        call_kwargs = agents["hr"].run.call_args[1]
        assert call_kwargs["trace_id"] == "my-trace"

    @pytest.mark.parametrize("domain", ["hr", "tech", "finance", "legal"])
    def test_correct_agent_name_per_domain(self, domain):
        from src.graph import build_graph

        orchestrator = _make_orchestrator(domain)
        agents = _make_agents()
        graph = build_graph(orchestrator, agents, None)

        result = _invoke(graph, _make_initial_state())

        assert result["agent_name"] == AGENT_NAMES[domain]


# ---------------------------------------------------------------------------
# evaluate_node (verificado vía graph.invoke)
# ---------------------------------------------------------------------------


class TestEvaluateNode:
    def test_evaluator_called_once(self):
        from src.graph import build_graph

        evaluator = _make_evaluator()
        graph = build_graph(_make_orchestrator(), _make_agents(), evaluator)

        _invoke(graph, _make_initial_state())

        evaluator.evaluate.assert_called_once()

    def test_evaluation_scores_in_final_state(self):
        from src.graph import build_graph

        evaluator = _make_evaluator()
        graph = build_graph(_make_orchestrator(), _make_agents(), evaluator)

        result = _invoke(graph, _make_initial_state())

        assert result["evaluation"] is not None
        assert result["evaluation"]["overall"] == 9

    def test_evaluation_none_when_evaluator_is_none(self):
        from src.graph import build_graph

        graph = build_graph(_make_orchestrator(), _make_agents(), evaluator=None)

        result = _invoke(graph, _make_initial_state())

        assert result["evaluation"] is None

    def test_evaluate_receives_correct_trace_id(self):
        from src.graph import build_graph

        evaluator = _make_evaluator()
        graph = build_graph(_make_orchestrator(), _make_agents(), evaluator)

        _invoke(graph, _make_initial_state(trace_id="specific-trace"))

        call_kwargs = evaluator.evaluate.call_args[1]
        assert call_kwargs["trace_id"] == "specific-trace"

    def test_evaluate_receives_answer_from_rag(self):
        from src.graph import build_graph

        agents = _make_agents()
        agents["hr"].run.return_value = {
            "agent": "HRAgent",
            "answer": "Mi respuesta específica.",
            "retrieved_docs": [],
            "context": "contexto",
        }
        evaluator = _make_evaluator()
        graph = build_graph(_make_orchestrator("hr"), agents, evaluator)

        _invoke(graph, _make_initial_state())

        call_kwargs = evaluator.evaluate.call_args[1]
        assert call_kwargs["answer"] == "Mi respuesta específica."

    def test_evaluate_receives_context_from_rag(self):
        from src.graph import build_graph

        agents = _make_agents()
        agents["hr"].run.return_value = {
            "agent": "HRAgent",
            "answer": "respuesta",
            "retrieved_docs": [],
            "context": "contexto específico del dominio HR",
        }
        evaluator = _make_evaluator()
        graph = build_graph(_make_orchestrator("hr"), agents, evaluator)

        _invoke(graph, _make_initial_state())

        call_kwargs = evaluator.evaluate.call_args[1]
        assert call_kwargs["context"] == "contexto específico del dominio HR"

    def test_evaluate_receives_agent_name_from_rag(self):
        from src.graph import build_graph

        agents = _make_agents()
        evaluator = _make_evaluator()
        graph = build_graph(_make_orchestrator("hr"), agents, evaluator)

        _invoke(graph, _make_initial_state())

        call_kwargs = evaluator.evaluate.call_args[1]
        assert call_kwargs["agent_name"] == "HRAgent"

    def test_evaluate_receives_query(self):
        from src.graph import build_graph

        evaluator = _make_evaluator()
        graph = build_graph(_make_orchestrator(), _make_agents(), evaluator)

        _invoke(graph, _make_initial_state(query="¿Qué es el onboarding?"))

        call_kwargs = evaluator.evaluate.call_args[1]
        assert call_kwargs["query"] == "¿Qué es el onboarding?"


# ---------------------------------------------------------------------------
# Flujo completo (estado final coherente)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_final_state_has_all_populated_keys(self):
        from src.graph import build_graph

        orchestrator = _make_orchestrator(
            "finance", confidence=0.88, reasoning="Consulta financiera."
        )
        agents = _make_agents()
        evaluator = _make_evaluator()
        graph = build_graph(orchestrator, agents, evaluator)

        result = _invoke(
            graph, _make_initial_state(query="¿Cómo presento un gasto?", trace_id="t-123")
        )

        assert result["query"] == "¿Cómo presento un gasto?"
        assert result["trace_id"] == "t-123"
        assert result["domain"] == "finance"
        assert result["confidence"] == pytest.approx(0.88)
        assert result["reasoning"] == "Consulta financiera."
        assert result["agent_name"] == "FinanceAgent"
        assert "finance" in result["answer"]
        assert result["retrieved_docs"] == ["finance_docs/doc.txt"]
        assert result["evaluation"]["overall"] == 9

    def test_exception_in_orchestrate_propagates(self):
        from src.graph import build_graph

        orchestrator = MagicMock()
        orchestrator.classify.side_effect = RuntimeError("LLM error")
        graph = build_graph(orchestrator, _make_agents(), None)

        with pytest.raises(RuntimeError, match="LLM error"):
            _invoke(graph, _make_initial_state())

    def test_exception_in_rag_propagates(self):
        from src.graph import build_graph

        agents = _make_agents()
        agents["hr"].run.side_effect = ValueError("FAISS error")
        graph = build_graph(_make_orchestrator("hr"), agents, None)

        with pytest.raises(ValueError, match="FAISS error"):
            _invoke(graph, _make_initial_state())
