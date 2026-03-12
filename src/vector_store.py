"""
vector_store.py

Carga los documentos de cada dominio, los splitea en chunks
y construye índices FAISS independientes por agente.

Decisión técnica: un vector store por dominio (en lugar de uno único)
para que cada RAG agent recupere solo contexto relevante a su área,
evitando recuperaciones cruzadas que degradan la precisión.

Persistencia: si FAISS_INDEX_DIR está configurado (por defecto "faiss_index"),
los índices se guardan en disco tras la primera construcción y se cargan
en startups posteriores, evitando re-embeddings costosos.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CONFIG

logger = logging.getLogger(__name__)

# Mapeo dominio → carpeta de documentos
DOMAIN_DIRS: dict[str, str] = {
    "hr": "hr_docs",
    "tech": "tech_docs",
    "finance": "finance_docs",
    "legal": "legal_docs",
}


def _get_embeddings() -> OpenAIEmbeddings:
    """
    Instancia el modelo de embeddings.

    Decisión técnica: text-embedding-3-small ofrece excelente calidad
    a un costo ~5x menor que text-embedding-3-large, suficiente para
    documentos de políticas corporativas en español.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=CONFIG["openai_api_key"],
    )


def _build_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Decisión técnica: RecursiveCharacterTextSplitter con chunk_size y overlap
    configurables vía variables de entorno (default 800/100).
    - 800 chars da chunks con suficiente contexto semántico.
    - 100 chars de overlap evita cortar ideas en la frontera de chunks.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        separators=["\n\n", "\n", " ", ""],
    )


def _load_domain_documents(domain_dir: str):
    """Carga todos los .txt de una carpeta de dominio."""
    path = Path(CONFIG["data_dir"]) / domain_dir

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de documentos: {path}\n"
            "Asegurate de haber copiado los archivos de datos al proyecto."
        )

    loader = DirectoryLoader(
        str(path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()

    if not docs:
        raise ValueError(f"No se encontraron documentos .txt en: {path}")

    return docs


def _index_path(domain: str) -> Path:
    """Retorna la ruta del directorio donde se persiste el índice FAISS del dominio."""
    return Path(CONFIG["faiss_index_dir"]) / domain


def build_vector_store(domain: str) -> FAISS:
    """
    Retorna el índice FAISS para el dominio dado.

    Si existe un índice persistido en disco, lo carga directamente
    (evita re-embedding). Si no existe, lo construye desde los documentos
    y lo guarda en disco para el próximo startup.

    Decisión técnica: FAISS con persistencia local es el punto medio entre
    un setup puramente in-memory (sin estado) y un vector store administrado
    (Pinecone/pgvector). Ideal para proyectos medianos sin infraestructura adicional.
    """
    embeddings = _get_embeddings()
    index_path = _index_path(domain)

    if index_path.exists():
        logger.info("[VectorStore] Cargando índice persistido para dominio: %s", domain)
        return FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    logger.info("[VectorStore] Construyendo índice para dominio: %s", domain)
    domain_dir = DOMAIN_DIRS[domain]
    docs = _load_domain_documents(domain_dir)

    splitter = _build_text_splitter()
    chunks = splitter.split_documents(docs)

    logger.info("[VectorStore] %s: %d docs → %d chunks", domain, len(docs), len(chunks))

    vector_store = FAISS.from_documents(chunks, embeddings)

    # Persistir en disco para startups futuros
    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))
    logger.info("[VectorStore] Índice guardado en: %s", index_path)

    return vector_store


def build_all_vector_stores() -> dict[str, FAISS]:
    """
    Construye o carga los índices FAISS para todos los dominios.
    Se llama una sola vez al inicio del sistema.
    """
    logger.info("[VectorStore] Inicializando índices FAISS para todos los dominios...")
    stores: dict[str, FAISS] = {}
    for domain in DOMAIN_DIRS:
        stores[domain] = build_vector_store(domain)
    logger.info("[VectorStore] Índices listos.\n")
    return stores
