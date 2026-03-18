"""
test_orchestrator.py

Tests para src/agents/orchestrator.py.
Cubre: constante DOMAINS_DESCRIPTION, inicialización, clasificación exitosa,
fallback ante dominio inválido, advertencia ante confianza baja y manejo de
callbacks opcionales.
"""

import json
from unittest.mock import MagicMock, patch

import pytest


def _make_llm_response(
    domain: str, confidence: float, reasoning: str = "Razón de prueba."
) -> MagicMock:
    """Construye un MagicMock que simula la respuesta del LLM en JSON mode."""
    msg = MagicMock()
    msg.content = json.dumps(
        {
            "domain": domain,
            "confidence": confidence,
            "reasoning": reasoning,
        }
    )
    return msg


class TestDomainsDescription:
    def test_contains_all_domains(self):
        from src.agents.orchestrator import DOMAINS_DESCRIPTION, VALID_DOMAINS

        for domain in VALID_DOMAINS:
            assert domain in DOMAINS_DESCRIPTION

    def test_is_string(self):
        from src.agents.orchestrator import DOMAINS_DESCRIPTION

        assert isinstance(DOMAINS_DESCRIPTION, str)

    def test_is_non_empty(self):
        from src.agents.orchestrator import DOMAINS_DESCRIPTION

        assert len(DOMAINS_DESCRIPTION) > 0


class TestOrchestratorInit:
    def test_creates_chain(self, mock_langfuse_handler):
        with patch("src.agents.orchestrator.ChatOpenAI"):
            from src.agents.orchestrator import Orchestrator

            orc = Orchestrator(langfuse_handler=mock_langfuse_handler)
        assert orc.chain is not None

    def test_stores_langfuse_handler(self, mock_langfuse_handler):
        with patch("src.agents.orchestrator.ChatOpenAI"):
            from src.agents.orchestrator import Orchestrator

            orc = Orchestrator(langfuse_handler=mock_langfuse_handler)
        assert orc.langfuse_handler is mock_langfuse_handler

    def test_none_handler_accepted(self):
        with patch("src.agents.orchestrator.ChatOpenAI"):
            from src.agents.orchestrator import Orchestrator

            orc = Orchestrator(langfuse_handler=None)
        assert orc.langfuse_handler is None


class TestOrchestratorClassify:
    def _make_orchestrator(self, handler=None):
        with patch("src.agents.orchestrator.ChatOpenAI"):
            from src.agents.orchestrator import Orchestrator

            orc = Orchestrator(langfuse_handler=handler)
        orc.chain = MagicMock()
        return orc

    def test_returns_valid_domain(self):
        orc = self._make_orchestrator()
        orc.chain.invoke.return_value = _make_llm_response("hr", 0.95)

        result = orc.classify("¿Cuántos días de vacaciones tengo?")

        assert result["domain"] == "hr"
        assert result["confidence"] == 0.95

    def test_returns_all_expected_keys(self):
        orc = self._make_orchestrator()
        orc.chain.invoke.return_value = _make_llm_response("tech", 0.9, "Es una consulta de IT.")

        result = orc.classify("Mi laptop no enciende.")

        assert "domain" in result
        assert "confidence" in result
        assert "reasoning" in result

    def test_fallback_on_invalid_domain(self):
        """Dominio inválido → 'unknown' con confidence 0.0 para activar clarification_node."""
        orc = self._make_orchestrator()
        orc.chain.invoke.return_value = _make_llm_response("invalid_domain", 0.8)

        result = orc.classify("Consulta ambigua.")

        assert result["domain"] == "unknown"
        assert result["confidence"] == 0.0

    def test_low_confidence_logs_warning(self, caplog):
        """Confianza < threshold genera un log WARNING."""
        import logging

        orc = self._make_orchestrator()
        orc.chain.invoke.return_value = _make_llm_response("finance", 0.1)

        with caplog.at_level(logging.WARNING, logger="src.agents.orchestrator"):
            orc.classify("Consulta poco clara.")

        assert any("Confianza baja" in r.message for r in caplog.records)

    def test_high_confidence_no_warning(self, caplog):
        """Confianza >= threshold NO genera WARNING."""
        import logging

        orc = self._make_orchestrator()
        orc.chain.invoke.return_value = _make_llm_response("legal", 0.95)

        with caplog.at_level(logging.WARNING, logger="src.agents.orchestrator"):
            orc.classify("Necesito revisar un contrato.")

        assert not any("Confianza baja" in r.message for r in caplog.records)

    def test_with_langfuse_handler_uses_callbacks(self, mock_langfuse_handler):
        orc = self._make_orchestrator(handler=mock_langfuse_handler)
        orc.chain.invoke.return_value = _make_llm_response("hr", 0.9)

        orc.classify("query", trace_id="trace-abc")

        call_config = orc.chain.invoke.call_args[1]["config"]
        assert mock_langfuse_handler in call_config["callbacks"]

    def test_without_langfuse_handler_empty_callbacks(self):
        orc = self._make_orchestrator(handler=None)
        orc.chain.invoke.return_value = _make_llm_response("hr", 0.9)

        orc.classify("query")

        call_config = orc.chain.invoke.call_args[1]["config"]
        assert call_config["callbacks"] == []

    def test_classify_without_trace_id(self):
        orc = self._make_orchestrator()
        orc.chain.invoke.return_value = _make_llm_response("tech", 0.85)

        result = orc.classify("query sin trace_id")
        assert result["domain"] == "tech"

    @pytest.mark.parametrize("domain", ["hr", "tech", "finance", "legal"])
    def test_all_valid_domains_pass_through(self, domain):
        orc = self._make_orchestrator()
        orc.chain.invoke.return_value = _make_llm_response(domain, 0.9)

        result = orc.classify(f"Consulta de {domain}.")
        assert result["domain"] == domain
