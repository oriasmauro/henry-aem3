"""
test_base_rag_agent.py

Tests para src/agents/base_rag_agent.py.
Cubre: _format_docs, inicialización de BaseRAGAgent y el pipeline run().

El pipeline run() hace UNA sola llamada al retriever (bug de doble retrieval
ya corregido). Los tests verifican explícitamente esto.
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# _format_docs
# ---------------------------------------------------------------------------


class TestFormatDocs:
    def test_formats_with_source_metadata(self):
        from src.agents.base_rag_agent import _format_docs

        doc = MagicMock()
        doc.metadata = {"source": "hr_docs/vacaciones.txt"}
        doc.page_content = "15 días de vacaciones."

        result = _format_docs([doc])

        assert "[Documento 1 - Fuente: hr_docs/vacaciones.txt]" in result
        assert "15 días de vacaciones." in result

    def test_fallback_source_when_metadata_missing(self):
        from src.agents.base_rag_agent import _format_docs

        doc = MagicMock()
        doc.metadata = {}  # sin clave "source"
        doc.page_content = "Contenido sin fuente."

        result = _format_docs([doc])

        assert "desconocido" in result

    def test_multiple_docs_separated_by_delimiter(self):
        from src.agents.base_rag_agent import _format_docs

        docs = []
        for i in range(3):
            d = MagicMock()
            d.metadata = {"source": f"doc{i}.txt"}
            d.page_content = f"Contenido {i}"
            docs.append(d)

        result = _format_docs(docs)

        assert "Documento 1" in result
        assert "Documento 2" in result
        assert "Documento 3" in result
        assert "---" in result  # separador entre docs

    def test_empty_list_returns_empty_string(self):
        from src.agents.base_rag_agent import _format_docs

        assert _format_docs([]) == ""


# ---------------------------------------------------------------------------
# BaseRAGAgent.__init__
# ---------------------------------------------------------------------------


class TestBaseRAGAgentInit:
    def _make_agent(self, vector_store, langfuse_handler=None):
        """Instancia un agente concreto (subclase mínima) para tests."""
        with patch("src.agents.base_rag_agent.ChatOpenAI"):
            from src.agents.base_rag_agent import BaseRAGAgent

            class ConcreteAgent(BaseRAGAgent):
                domain = "hr"
                agent_name = "TestAgent"
                system_prompt = "Eres un agente de prueba."

            return ConcreteAgent(vector_store, langfuse_handler)

    def test_stores_vector_store(self, mock_vector_store):
        agent = self._make_agent(mock_vector_store)
        assert agent.vector_store is mock_vector_store

    def test_creates_retriever_with_correct_params(self, mock_vector_store):
        from src.config import CONFIG

        self._make_agent(mock_vector_store)
        mock_vector_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": CONFIG["retriever_k"]},
        )

    def test_stores_langfuse_handler(self, mock_vector_store, mock_langfuse_handler):
        agent = self._make_agent(mock_vector_store, mock_langfuse_handler)
        assert agent.langfuse_handler is mock_langfuse_handler

    def test_none_langfuse_handler(self, mock_vector_store):
        agent = self._make_agent(mock_vector_store, None)
        assert agent.langfuse_handler is None

    def test_generation_chain_is_created(self, mock_vector_store):
        agent = self._make_agent(mock_vector_store)
        assert agent.generation_chain is not None


# ---------------------------------------------------------------------------
# BaseRAGAgent.run
# ---------------------------------------------------------------------------


class TestBaseRAGAgentRun:
    def _make_agent(self, vector_store, langfuse_handler=None):
        with patch("src.agents.base_rag_agent.ChatOpenAI"):
            from src.agents.base_rag_agent import BaseRAGAgent

            class ConcreteAgent(BaseRAGAgent):
                domain = "hr"
                agent_name = "TestAgent"
                system_prompt = "Eres un agente de prueba."

            return ConcreteAgent(vector_store, langfuse_handler)

    def test_returns_expected_keys(self, mock_vector_store):
        agent = self._make_agent(mock_vector_store)
        agent.generation_chain = MagicMock()
        agent.generation_chain.invoke.return_value = "Respuesta de prueba."

        result = agent.run("¿Cuántos días de vacaciones tengo?")

        assert set(result.keys()) == {
            "agent",
            "domain",
            "query",
            "answer",
            "retrieved_docs",
            "context",
        }

    def test_retriever_called_exactly_once(self, mock_vector_store):
        """Verifica que se eliminó el doble retrieval: el retriever se invoca una sola vez."""
        agent = self._make_agent(mock_vector_store)
        agent.generation_chain = MagicMock()
        agent.generation_chain.invoke.return_value = "Respuesta."

        agent.run("query de prueba")

        assert agent.retriever.invoke.call_count == 1

    def test_result_contains_correct_agent_and_domain(self, mock_vector_store):
        agent = self._make_agent(mock_vector_store)
        agent.generation_chain = MagicMock()
        agent.generation_chain.invoke.return_value = "Respuesta."

        result = agent.run("query")

        assert result["agent"] == "TestAgent"
        assert result["domain"] == "hr"
        assert result["query"] == "query"

    def test_result_contains_answer(self, mock_vector_store):
        agent = self._make_agent(mock_vector_store)
        agent.generation_chain = MagicMock()
        agent.generation_chain.invoke.return_value = "Respuesta esperada."

        result = agent.run("query")

        assert result["answer"] == "Respuesta esperada."

    def test_result_contains_retrieved_sources(self, mock_vector_store):
        agent = self._make_agent(mock_vector_store)
        agent.generation_chain = MagicMock()
        agent.generation_chain.invoke.return_value = "Respuesta."

        result = agent.run("query")

        # mock_vector_store retorna 2 docs con source en metadata
        assert isinstance(result["retrieved_docs"], list)
        assert len(result["retrieved_docs"]) == 2

    def test_result_contains_context_string(self, mock_vector_store):
        agent = self._make_agent(mock_vector_store)
        agent.generation_chain = MagicMock()
        agent.generation_chain.invoke.return_value = "Respuesta."

        result = agent.run("query")

        assert isinstance(result["context"], str)
        assert len(result["context"]) > 0

    def test_run_with_trace_id_passes_metadata(self, mock_vector_store):
        agent = self._make_agent(mock_vector_store)
        agent.generation_chain = MagicMock()
        agent.generation_chain.invoke.return_value = "Respuesta."

        agent.run("query", trace_id="trace-123")

        call_kwargs = agent.generation_chain.invoke.call_args[1]["config"]
        assert call_kwargs["metadata"]["trace_id"] == "trace-123"

    def test_run_without_trace_id(self, mock_vector_store):
        agent = self._make_agent(mock_vector_store)
        agent.generation_chain = MagicMock()
        agent.generation_chain.invoke.return_value = "Respuesta."

        result = agent.run("query")  # sin trace_id
        assert result is not None

    def test_run_with_langfuse_handler_uses_callbacks(
        self, mock_vector_store, mock_langfuse_handler
    ):
        agent = self._make_agent(mock_vector_store, mock_langfuse_handler)
        agent.generation_chain = MagicMock()
        agent.generation_chain.invoke.return_value = "Respuesta."

        agent.run("query")

        call_kwargs = agent.generation_chain.invoke.call_args[1]["config"]
        assert mock_langfuse_handler in call_kwargs["callbacks"]

    def test_run_without_langfuse_handler_empty_callbacks(self, mock_vector_store):
        agent = self._make_agent(mock_vector_store, None)
        agent.generation_chain = MagicMock()
        agent.generation_chain.invoke.return_value = "Respuesta."

        agent.run("query")

        call_kwargs = agent.generation_chain.invoke.call_args[1]["config"]
        assert call_kwargs["callbacks"] == []
