"""
test_config.py

Tests para src/config.py.
Cubre: carga exitosa, valores por defecto, valores personalizados
y fallo ante variables de entorno requeridas faltantes.
"""

import os
from unittest.mock import patch

import pytest


def test_config_loads_required_keys():
    """get_config() carga las claves requeridas desde el entorno."""
    from src.config import get_config

    cfg = get_config()
    assert cfg["openai_api_key"] == os.environ["OPENAI_API_KEY"]
    assert cfg["langfuse_public_key"] == os.environ["LANGFUSE_PUBLIC_KEY"]
    assert cfg["langfuse_secret_key"] == os.environ["LANGFUSE_SECRET_KEY"]
    assert cfg["langfuse_host"] == os.environ["LANGFUSE_HOST"]


def test_config_default_values():
    """Los campos opcionales tienen los valores por defecto correctos cuando no se setean."""
    import os
    from unittest.mock import patch

    from src.config import get_config

    # Usar solo las vars requeridas y ninguna opcional → se aplican todos los defaults
    minimal_env = {
        "OPENAI_API_KEY": "test-key",
        "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
        "LANGFUSE_SECRET_KEY": "sk-lf-test",
        "LANGFUSE_HOST": "https://cloud.langfuse.com",
    }
    with patch.dict(os.environ, minimal_env, clear=True):
        cfg = get_config()

    assert cfg["orchestrator_model"] == "gpt-4o-mini"
    assert cfg["agent_model"] == "gpt-4o-mini"
    assert cfg["evaluator_model"] == "gpt-4o-mini"
    assert cfg["chunk_size"] == 800
    assert cfg["chunk_overlap"] == 100
    assert cfg["retriever_k"] == 4
    assert cfg["data_dir"] == "data"
    assert cfg["faiss_index_dir"] == "faiss_index"
    assert cfg["confidence_threshold"] == 0.4


def test_config_custom_values():
    """Los valores opcionales se sobreescriben desde el entorno."""
    from src.config import get_config

    custom_env = {
        "OPENAI_API_KEY": "test-openai-key",
        "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
        "LANGFUSE_SECRET_KEY": "sk-lf-test",
        "LANGFUSE_HOST": "https://cloud.langfuse.com",
        "ORCHESTRATOR_MODEL": "gpt-4o",
        "AGENT_MODEL": "gpt-4o",
        "EVALUATOR_MODEL": "gpt-4o",
        "CHUNK_SIZE": "400",
        "CHUNK_OVERLAP": "50",
        "RETRIEVER_K": "6",
        "DATA_DIR": "custom_data",
        "FAISS_INDEX_DIR": "custom_index",
        "CONFIDENCE_THRESHOLD": "0.6",
    }
    with patch.dict(os.environ, custom_env, clear=True):
        cfg = get_config()

    assert cfg["orchestrator_model"] == "gpt-4o"
    assert cfg["agent_model"] == "gpt-4o"
    assert cfg["evaluator_model"] == "gpt-4o"
    assert cfg["chunk_size"] == 400
    assert cfg["chunk_overlap"] == 50
    assert cfg["retriever_k"] == 6
    assert cfg["data_dir"] == "custom_data"
    assert cfg["faiss_index_dir"] == "custom_index"
    assert cfg["confidence_threshold"] == 0.6


def test_config_missing_required_vars_raises():
    """get_config() lanza EnvironmentError si faltan variables requeridas."""
    from src.config import get_config

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvironmentError, match="Faltan las siguientes variables de entorno"):
            get_config()


def test_config_missing_single_var():
    """El mensaje de error lista la variable específica que falta."""
    from src.config import get_config

    env = {
        "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
        "LANGFUSE_SECRET_KEY": "sk-lf-test",
        "LANGFUSE_HOST": "https://cloud.langfuse.com",
        # OPENAI_API_KEY ausente deliberadamente
    }
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            get_config()


def test_config_singleton_is_dict():
    """El CONFIG singleton exportado es un dict con las claves esperadas."""
    from src.config import CONFIG

    expected_keys = {
        "openai_api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
        "langfuse_host",
        "orchestrator_model",
        "agent_model",
        "evaluator_model",
        "chunk_size",
        "chunk_overlap",
        "retriever_k",
        "data_dir",
        "faiss_index_dir",
        "confidence_threshold",
    }
    assert expected_keys.issubset(set(CONFIG.keys()))
