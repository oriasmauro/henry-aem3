"""
config.py
Carga y valida las variables de entorno necesarias para el sistema.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def get_config() -> dict:
    """Retorna la configuración del sistema desde variables de entorno."""

    required_vars = [
        "OPENAI_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise OSError(
            f"Faltan las siguientes variables de entorno: {', '.join(missing)}\n"
            "Asegurate de copiar .env.example a .env y completar los valores."
        )

    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "langfuse_public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
        "langfuse_secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
        "langfuse_host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        # Modelos
        "orchestrator_model": os.getenv("ORCHESTRATOR_MODEL", "gpt-4o-mini"),
        "agent_model": os.getenv("AGENT_MODEL", "gpt-4o-mini"),
        "evaluator_model": os.getenv("EVALUATOR_MODEL", "gpt-4o-mini"),
        # RAG
        "chunk_size": int(os.getenv("CHUNK_SIZE", "800")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "100")),
        "retriever_k": int(os.getenv("RETRIEVER_K", "4")),
        # Paths
        "data_dir": os.getenv("DATA_DIR", "data"),
        # Persistencia FAISS: directorio donde se guardan/cargan los índices.
        # Si está vacío, los índices se construyen en memoria en cada startup.
        "faiss_index_dir": os.getenv("FAISS_INDEX_DIR", "faiss_index"),
        # Umbral mínimo de confianza del orquestador.
        # Clasificaciones por debajo de este valor generan una advertencia en logs.
        "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.4")),
    }


CONFIG = get_config()
