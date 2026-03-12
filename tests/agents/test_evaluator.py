"""
test_evaluator.py

Tests para src/agents/evaluator.py.
Cubre: inicialización, evaluación con contexto, evaluación sin contexto
(placeholder + warning), registro de scores en Langfuse y retorno del dict.
"""

import json
import logging
from unittest.mock import MagicMock, patch


def _make_eval_response(
    relevance=8, completeness=7, accuracy=9, overall=8, feedback="Buena respuesta."
) -> MagicMock:
    """MagicMock que simula la respuesta JSON del LLM evaluador."""
    msg = MagicMock()
    msg.content = json.dumps(
        {
            "relevance": relevance,
            "completeness": completeness,
            "accuracy": accuracy,
            "overall": overall,
            "feedback": feedback,
        }
    )
    return msg


class TestEvaluatorAgentInit:
    def test_creates_chain(self):
        with patch("src.agents.evaluator.ChatOpenAI"), patch("src.agents.evaluator.Langfuse"):
            from src.agents.evaluator import EvaluatorAgent

            ev = EvaluatorAgent()
        assert ev.chain is not None

    def test_creates_langfuse_client(self):
        with (
            patch("src.agents.evaluator.ChatOpenAI"),
            patch("src.agents.evaluator.Langfuse") as mock_lf_cls,
        ):
            from src.agents.evaluator import EvaluatorAgent

            ev = EvaluatorAgent()
        mock_lf_cls.assert_called_once()
        assert ev.langfuse is mock_lf_cls.return_value


class TestEvaluatorAgentEvaluate:
    def _make_evaluator(self):
        with patch("src.agents.evaluator.ChatOpenAI"), patch("src.agents.evaluator.Langfuse"):
            from src.agents.evaluator import EvaluatorAgent

            ev = EvaluatorAgent()
        ev.chain = MagicMock()
        ev.langfuse = MagicMock()
        return ev

    def test_returns_dict_with_all_dimensions(self):
        ev = self._make_evaluator()
        ev.chain.invoke.return_value = _make_eval_response()

        result = ev.evaluate(
            trace_id="trace-001",
            query="¿Cuántos días de vacaciones tengo?",
            answer="Tienes 15 días según la política.",
            agent_name="HRAgent",
            context="[Documento 1]\n15 días de vacaciones al año.",
        )

        assert result["relevance"] == 8
        assert result["completeness"] == 7
        assert result["accuracy"] == 9
        assert result["overall"] == 8
        assert result["feedback"] == "Buena respuesta."

    def test_registers_all_four_scores_in_langfuse(self):
        ev = self._make_evaluator()
        ev.chain.invoke.return_value = _make_eval_response()

        ev.evaluate(
            trace_id="trace-001",
            query="query",
            answer="respuesta",
            agent_name="HRAgent",
            context="contexto",
        )

        score_names = {c.kwargs["name"] for c in ev.langfuse.score.call_args_list}
        assert score_names == {
            "eval_relevance",
            "eval_completeness",
            "eval_accuracy",
            "eval_overall",
        }

    def test_scores_associated_to_correct_trace(self):
        ev = self._make_evaluator()
        ev.chain.invoke.return_value = _make_eval_response()

        ev.evaluate(
            trace_id="my-trace-id",
            query="q",
            answer="a",
            agent_name="HRAgent",
            context="c",
        )

        for c in ev.langfuse.score.call_args_list:
            assert c.kwargs["trace_id"] == "my-trace-id"

    def test_without_context_uses_placeholder_and_warns(self, caplog):
        ev = self._make_evaluator()
        ev.chain.invoke.return_value = _make_eval_response()

        with caplog.at_level(logging.WARNING, logger="src.agents.evaluator"):
            ev.evaluate(
                trace_id="trace-002",
                query="query",
                answer="respuesta",
                agent_name="HRAgent",
                context=None,
            )

        # Debe haber un warning sobre el contexto faltante
        assert any("contexto" in r.message.lower() for r in caplog.records)

        # El chain debe haber recibido el placeholder
        call_input = ev.chain.invoke.call_args[0][0]
        assert "(contexto no disponible)" in call_input["context"]

    def test_with_context_passes_it_to_chain(self):
        ev = self._make_evaluator()
        ev.chain.invoke.return_value = _make_eval_response()

        ev.evaluate(
            trace_id="trace-003",
            query="query",
            answer="respuesta",
            agent_name="TechAgent",
            context="[Documento 1]\nContenido importante.",
        )

        call_input = ev.chain.invoke.call_args[0][0]
        assert "[Documento 1]" in call_input["context"]

    def test_passes_correct_agent_name_to_chain(self):
        ev = self._make_evaluator()
        ev.chain.invoke.return_value = _make_eval_response()

        ev.evaluate(
            trace_id="t",
            query="q",
            answer="a",
            agent_name="FinanceAgent",
            context="c",
        )

        call_input = ev.chain.invoke.call_args[0][0]
        assert call_input["agent_name"] == "FinanceAgent"

    def test_score_value_is_float(self):
        ev = self._make_evaluator()
        ev.chain.invoke.return_value = _make_eval_response(overall=9)

        ev.evaluate(trace_id="t", query="q", answer="a", agent_name="A", context="c")

        for c in ev.langfuse.score.call_args_list:
            assert isinstance(c.kwargs["value"], float)

    def test_feedback_passed_as_comment(self):
        ev = self._make_evaluator()
        ev.chain.invoke.return_value = _make_eval_response(feedback="Excelente.")

        ev.evaluate(trace_id="t", query="q", answer="a", agent_name="A", context="c")

        for c in ev.langfuse.score.call_args_list:
            assert c.kwargs["comment"] == "Excelente."
