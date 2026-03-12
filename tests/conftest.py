"""
conftest.py

Fixtures y configuración global de pytest.

IMPORTANTE: las variables de entorno se setean a nivel de módulo (antes que
cualquier fixture) porque src.config ejecuta CONFIG = get_config() al momento
del import. Si los vars no están presentes cuando pytest importa los módulos
de test, get_config() lanzaría EnvironmentError antes de que cualquier test corra.
"""

import os

# Setear vars de entorno antes de cualquier import de src.*
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-test")
os.environ.setdefault("LANGFUSE_HOST", "https://cloud.langfuse.com")
os.environ.setdefault("FAISS_INDEX_DIR", "faiss_index_test")

import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_vector_store():
    """FAISS vector store mockeado con retriever que retorna docs de prueba."""
    doc1 = MagicMock()
    doc1.metadata = {"source": "hr_docs/vacaciones.txt"}
    doc1.page_content = "Los empleados tienen 15 días de vacaciones al año."

    doc2 = MagicMock()
    doc2.metadata = {"source": "hr_docs/politica.txt"}
    doc2.page_content = "Las vacaciones se solicitan con 5 días de anticipación."

    store = MagicMock()
    retriever = MagicMock()
    retriever.invoke.return_value = [doc1, doc2]
    store.as_retriever.return_value = retriever
    return store


@pytest.fixture
def mock_langfuse_handler():
    return MagicMock()


@pytest.fixture
def sample_docs():
    """Documentos LangChain de prueba."""
    doc1 = MagicMock()
    doc1.metadata = {"source": "archivo1.txt"}
    doc1.page_content = "Contenido del documento 1."

    doc2 = MagicMock()
    doc2.metadata = {}  # sin clave "source" → debe usar "desconocido"
    doc2.page_content = "Contenido del documento 2."

    return [doc1, doc2]


@pytest.fixture
def test_queries_file(tmp_path):
    """Archivo test_queries.json temporal."""
    data = {
        "test_queries": [
            {"query": "¿Cuántos días de vacaciones tengo?", "expected_agent": "hr"},
            {"query": "Mi laptop no enciende.", "expected_agent": "tech"},
        ]
    }
    p = tmp_path / "test_queries.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)
