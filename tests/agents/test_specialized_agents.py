"""
test_specialized_agents.py

Tests para los cuatro agentes RAG especializados:
HRAgent, TechAgent, FinanceAgent, LegalAgent.

Verifica que cada agente:
  - Hereda de BaseRAGAgent
  - Define los atributos de clase correctos (domain, agent_name, system_prompt)
  - Se puede instanciar pasando un vector store (y opcionalmente un handler)
  - Delega correctamente a la lógica base en run()
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_agent(agent_cls, mock_vector_store, langfuse_handler=None):
    """Instancia un agente especializado con ChatOpenAI mockeado."""
    with patch("src.agents.base_rag_agent.ChatOpenAI"):
        return agent_cls(mock_vector_store, langfuse_handler)


# ---------------------------------------------------------------------------
# HRAgent
# ---------------------------------------------------------------------------


class TestHRAgent:
    def test_inherits_from_base(self, mock_vector_store):
        from src.agents.base_rag_agent import BaseRAGAgent
        from src.agents.hr_agent import HRAgent

        agent = make_agent(HRAgent, mock_vector_store)
        assert isinstance(agent, BaseRAGAgent)

    def test_domain(self, mock_vector_store):
        from src.agents.hr_agent import HRAgent

        agent = make_agent(HRAgent, mock_vector_store)
        assert agent.domain == "hr"

    def test_agent_name(self, mock_vector_store):
        from src.agents.hr_agent import HRAgent

        agent = make_agent(HRAgent, mock_vector_store)
        assert agent.agent_name == "HRAgent"

    def test_system_prompt_non_empty(self, mock_vector_store):
        from src.agents.hr_agent import HRAgent

        agent = make_agent(HRAgent, mock_vector_store)
        assert isinstance(agent.system_prompt, str)
        assert len(agent.system_prompt) > 0

    def test_system_prompt_mentions_rrhh(self, mock_vector_store):
        from src.agents.hr_agent import HRAgent

        agent = make_agent(HRAgent, mock_vector_store)
        assert "Recursos Humanos" in agent.system_prompt or "RRHH" in agent.system_prompt

    def test_run_returns_correct_domain(self, mock_vector_store):
        from src.agents.hr_agent import HRAgent

        agent = make_agent(HRAgent, mock_vector_store)
        agent.generation_chain = MagicMock(return_value=MagicMock())
        agent.generation_chain.invoke.return_value = "Respuesta HR."
        result = agent.run("¿Cuántos días de vacaciones tengo?")
        assert result["domain"] == "hr"

    def test_accepts_langfuse_handler(self, mock_vector_store, mock_langfuse_handler):
        from src.agents.hr_agent import HRAgent

        agent = make_agent(HRAgent, mock_vector_store, mock_langfuse_handler)
        assert agent.langfuse_handler is mock_langfuse_handler


# ---------------------------------------------------------------------------
# TechAgent
# ---------------------------------------------------------------------------


class TestTechAgent:
    def test_inherits_from_base(self, mock_vector_store):
        from src.agents.base_rag_agent import BaseRAGAgent
        from src.agents.tech_agent import TechAgent

        agent = make_agent(TechAgent, mock_vector_store)
        assert isinstance(agent, BaseRAGAgent)

    def test_domain(self, mock_vector_store):
        from src.agents.tech_agent import TechAgent

        agent = make_agent(TechAgent, mock_vector_store)
        assert agent.domain == "tech"

    def test_agent_name(self, mock_vector_store):
        from src.agents.tech_agent import TechAgent

        agent = make_agent(TechAgent, mock_vector_store)
        assert agent.agent_name == "TechAgent"

    def test_system_prompt_non_empty(self, mock_vector_store):
        from src.agents.tech_agent import TechAgent

        agent = make_agent(TechAgent, mock_vector_store)
        assert isinstance(agent.system_prompt, str)
        assert len(agent.system_prompt) > 0

    def test_system_prompt_mentions_it(self, mock_vector_store):
        from src.agents.tech_agent import TechAgent

        agent = make_agent(TechAgent, mock_vector_store)
        assert "IT" in agent.system_prompt or "Soporte" in agent.system_prompt

    def test_run_returns_correct_domain(self, mock_vector_store):
        from src.agents.tech_agent import TechAgent

        agent = make_agent(TechAgent, mock_vector_store)
        agent.generation_chain = MagicMock()
        agent.generation_chain.invoke.return_value = "Respuesta Tech."
        result = agent.run("Mi laptop no enciende.")
        assert result["domain"] == "tech"


# ---------------------------------------------------------------------------
# FinanceAgent
# ---------------------------------------------------------------------------


class TestFinanceAgent:
    def test_inherits_from_base(self, mock_vector_store):
        from src.agents.base_rag_agent import BaseRAGAgent
        from src.agents.finance_agent import FinanceAgent

        agent = make_agent(FinanceAgent, mock_vector_store)
        assert isinstance(agent, BaseRAGAgent)

    def test_domain(self, mock_vector_store):
        from src.agents.finance_agent import FinanceAgent

        agent = make_agent(FinanceAgent, mock_vector_store)
        assert agent.domain == "finance"

    def test_agent_name(self, mock_vector_store):
        from src.agents.finance_agent import FinanceAgent

        agent = make_agent(FinanceAgent, mock_vector_store)
        assert agent.agent_name == "FinanceAgent"

    def test_system_prompt_non_empty(self, mock_vector_store):
        from src.agents.finance_agent import FinanceAgent

        agent = make_agent(FinanceAgent, mock_vector_store)
        assert isinstance(agent.system_prompt, str)
        assert len(agent.system_prompt) > 0

    def test_system_prompt_mentions_finanzas(self, mock_vector_store):
        from src.agents.finance_agent import FinanceAgent

        agent = make_agent(FinanceAgent, mock_vector_store)
        assert "Finanzas" in agent.system_prompt or "financiero" in agent.system_prompt.lower()

    def test_run_returns_correct_domain(self, mock_vector_store):
        from src.agents.finance_agent import FinanceAgent

        agent = make_agent(FinanceAgent, mock_vector_store)
        agent.generation_chain = MagicMock()
        agent.generation_chain.invoke.return_value = "Respuesta Finance."
        result = agent.run("¿Cómo presento un gasto de viaje?")
        assert result["domain"] == "finance"


# ---------------------------------------------------------------------------
# LegalAgent
# ---------------------------------------------------------------------------


class TestLegalAgent:
    def test_inherits_from_base(self, mock_vector_store):
        from src.agents.base_rag_agent import BaseRAGAgent
        from src.agents.legal_agent import LegalAgent

        agent = make_agent(LegalAgent, mock_vector_store)
        assert isinstance(agent, BaseRAGAgent)

    def test_domain(self, mock_vector_store):
        from src.agents.legal_agent import LegalAgent

        agent = make_agent(LegalAgent, mock_vector_store)
        assert agent.domain == "legal"

    def test_agent_name(self, mock_vector_store):
        from src.agents.legal_agent import LegalAgent

        agent = make_agent(LegalAgent, mock_vector_store)
        assert agent.agent_name == "LegalAgent"

    def test_system_prompt_non_empty(self, mock_vector_store):
        from src.agents.legal_agent import LegalAgent

        agent = make_agent(LegalAgent, mock_vector_store)
        assert isinstance(agent.system_prompt, str)
        assert len(agent.system_prompt) > 0

    def test_system_prompt_mentions_legal(self, mock_vector_store):
        from src.agents.legal_agent import LegalAgent

        agent = make_agent(LegalAgent, mock_vector_store)
        assert "Legal" in agent.system_prompt or "legal" in agent.system_prompt

    def test_run_returns_correct_domain(self, mock_vector_store):
        from src.agents.legal_agent import LegalAgent

        agent = make_agent(LegalAgent, mock_vector_store)
        agent.generation_chain = MagicMock()
        agent.generation_chain.invoke.return_value = "Respuesta Legal."
        result = agent.run("¿Quién aprueba los contratos con proveedores?")
        assert result["domain"] == "legal"


# ---------------------------------------------------------------------------
# Parametrizado: todos los agentes deben tener las claves mínimas en run()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_cls,domain,agent_name",
    [
        ("src.agents.hr_agent.HRAgent", "hr", "HRAgent"),
        ("src.agents.tech_agent.TechAgent", "tech", "TechAgent"),
        ("src.agents.finance_agent.FinanceAgent", "finance", "FinanceAgent"),
        ("src.agents.legal_agent.LegalAgent", "legal", "LegalAgent"),
    ],
)
def test_all_agents_run_returns_required_keys(agent_cls, domain, agent_name, mock_vector_store):
    module_path, cls_name = agent_cls.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)

    with patch("src.agents.base_rag_agent.ChatOpenAI"):
        agent = cls(mock_vector_store)

    agent.generation_chain = MagicMock()
    agent.generation_chain.invoke.return_value = "Respuesta."
    result = agent.run("query de prueba")

    assert result["domain"] == domain
    assert result["agent"] == agent_name
    assert "answer" in result
    assert "retrieved_docs" in result
    assert "context" in result
