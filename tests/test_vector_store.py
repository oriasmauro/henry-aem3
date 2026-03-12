"""
test_vector_store.py

Tests para src/vector_store.py.
Cubre: helpers internos, carga de documentos, construcción de índices FAISS
(desde disco y desde scratch), y orquestación de múltiples dominios.

Toda interacción con OpenAI, FAISS y el sistema de archivos está mockeada
para que los tests sean rápidos, deterministas y sin dependencias externas.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


class TestGetEmbeddings:
    def test_returns_openai_embeddings_instance(self):
        with patch("src.vector_store.OpenAIEmbeddings") as mock_cls:
            from src.vector_store import _get_embeddings

            result = _get_embeddings()
            mock_cls.assert_called_once()
            assert result is mock_cls.return_value

    def test_uses_correct_model(self):
        with patch("src.vector_store.OpenAIEmbeddings") as mock_cls:
            from src.vector_store import _get_embeddings

            _get_embeddings()
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model"] == "text-embedding-3-small"


class TestBuildTextSplitter:
    def test_returns_splitter_with_config_values(self):
        with patch("src.vector_store.RecursiveCharacterTextSplitter") as mock_cls:
            from src.config import CONFIG
            from src.vector_store import _build_text_splitter

            result = _build_text_splitter()
            mock_cls.assert_called_once_with(
                chunk_size=CONFIG["chunk_size"],
                chunk_overlap=CONFIG["chunk_overlap"],
                separators=["\n\n", "\n", " ", ""],
            )
            assert result is mock_cls.return_value


class TestIndexPath:
    def test_returns_correct_path(self):
        from src.config import CONFIG
        from src.vector_store import _index_path

        result = _index_path("hr")
        assert result == Path(CONFIG["faiss_index_dir"]) / "hr"

    def test_different_domains(self):
        from src.vector_store import _index_path

        for domain in ("hr", "tech", "finance", "legal"):
            path = _index_path(domain)
            assert domain in str(path)


# ---------------------------------------------------------------------------
# _load_domain_documents
# ---------------------------------------------------------------------------


class TestLoadDomainDocuments:
    def test_success(self, tmp_path):
        """Carga docs cuando el directorio existe y tiene archivos."""
        # Crear un archivo .txt de prueba en tmp_path/hr_docs/
        domain_path = tmp_path / "hr_docs"
        domain_path.mkdir()
        (domain_path / "vacaciones.txt").write_text("15 días de vacaciones.", encoding="utf-8")

        mock_doc = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]

        with (
            patch(
                "src.vector_store.CONFIG",
                {**__import__("src.config", fromlist=["CONFIG"]).CONFIG, "data_dir": str(tmp_path)},
            ),
            patch("src.vector_store.DirectoryLoader", return_value=mock_loader),
        ):
            from src.vector_store import _load_domain_documents

            docs = _load_domain_documents("hr_docs")

        assert docs == [mock_doc]

    def test_missing_directory_raises(self, tmp_path):
        """FileNotFoundError cuando el directorio no existe."""
        with patch(
            "src.vector_store.CONFIG",
            {**__import__("src.config", fromlist=["CONFIG"]).CONFIG, "data_dir": str(tmp_path)},
        ):
            from src.vector_store import _load_domain_documents

            with pytest.raises(FileNotFoundError, match="No se encontró el directorio"):
                _load_domain_documents("nonexistent_docs")

    def test_empty_directory_raises(self, tmp_path):
        """ValueError cuando el directorio existe pero no tiene .txt."""
        domain_path = tmp_path / "empty_docs"
        domain_path.mkdir()

        mock_loader = MagicMock()
        mock_loader.load.return_value = []

        with (
            patch(
                "src.vector_store.CONFIG",
                {**__import__("src.config", fromlist=["CONFIG"]).CONFIG, "data_dir": str(tmp_path)},
            ),
            patch("src.vector_store.DirectoryLoader", return_value=mock_loader),
        ):
            from src.vector_store import _load_domain_documents

            with pytest.raises(ValueError, match="No se encontraron documentos"):
                _load_domain_documents("empty_docs")


# ---------------------------------------------------------------------------
# build_vector_store
# ---------------------------------------------------------------------------


class TestBuildVectorStore:
    def test_loads_from_disk_when_index_exists(self, tmp_path):
        """Si el índice persiste en disco, lo carga sin re-embeddear."""
        index_dir = tmp_path / "faiss_index_test" / "hr"
        index_dir.mkdir(parents=True)

        mock_store = MagicMock()
        mock_embeddings = MagicMock()

        with (
            patch(
                "src.vector_store.CONFIG",
                {
                    **__import__("src.config", fromlist=["CONFIG"]).CONFIG,
                    "faiss_index_dir": str(tmp_path / "faiss_index_test"),
                },
            ),
            patch("src.vector_store._get_embeddings", return_value=mock_embeddings),
            patch("src.vector_store.FAISS") as mock_faiss_cls,
        ):
            mock_faiss_cls.load_local.return_value = mock_store

            from src.vector_store import build_vector_store

            result = build_vector_store("hr")

        mock_faiss_cls.load_local.assert_called_once()
        mock_faiss_cls.from_documents.assert_not_called()
        assert result is mock_store

    def test_builds_and_saves_when_no_index(self, tmp_path):
        """Si no existe índice en disco, lo construye y lo persiste."""
        mock_doc = MagicMock()
        mock_chunk = MagicMock()
        mock_store = MagicMock()
        mock_embeddings = MagicMock()
        mock_splitter = MagicMock()
        mock_splitter.split_documents.return_value = [mock_chunk]

        with (
            patch(
                "src.vector_store.CONFIG",
                {
                    **__import__("src.config", fromlist=["CONFIG"]).CONFIG,
                    "faiss_index_dir": str(tmp_path / "no_existing_index"),
                    "data_dir": "data",
                },
            ),
            patch("src.vector_store._get_embeddings", return_value=mock_embeddings),
            patch("src.vector_store._build_text_splitter", return_value=mock_splitter),
            patch("src.vector_store._load_domain_documents", return_value=[mock_doc]),
            patch("src.vector_store.FAISS") as mock_faiss_cls,
        ):
            mock_faiss_cls.from_documents.return_value = mock_store

            from src.vector_store import build_vector_store

            result = build_vector_store("hr")

        mock_faiss_cls.from_documents.assert_called_once_with([mock_chunk], mock_embeddings)
        mock_store.save_local.assert_called_once()
        assert result is mock_store


# ---------------------------------------------------------------------------
# build_all_vector_stores
# ---------------------------------------------------------------------------


class TestBuildAllVectorStores:
    def test_builds_all_four_domains(self):
        mock_stores = {d: MagicMock() for d in ("hr", "tech", "finance", "legal")}

        with patch("src.vector_store.build_vector_store", side_effect=lambda d: mock_stores[d]):
            from src.vector_store import build_all_vector_stores

            result = build_all_vector_stores()

        assert set(result.keys()) == {"hr", "tech", "finance", "legal"}
        for domain, store in mock_stores.items():
            assert result[domain] is store

    def test_calls_build_for_each_domain(self):
        with patch("src.vector_store.build_vector_store", return_value=MagicMock()) as mock_build:
            from src.vector_store import build_all_vector_stores

            build_all_vector_stores()

        called_domains = [c.args[0] for c in mock_build.call_args_list]
        assert set(called_domains) == {"hr", "tech", "finance", "legal"}
