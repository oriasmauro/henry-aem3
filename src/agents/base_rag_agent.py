"""
agents/base_rag_agent.py

Clase base para todos los RAG agents especializados.

Decisión técnica: una clase base evita duplicar la lógica de RAG
en cada agente. Cada agente especializado solo define su system prompt
y hereda el mecanismo de retrieval + generation.

El pipeline separa recuperación de generación:
1. retriever.invoke(query)  → docs (una sola llamada)
2. generation_chain.invoke({context, query, system_prompt}) → respuesta

Esto es preferible a incluir el retriever dentro del chain porque:
- Elimina el doble retrieval que ocurría al llamar invoke() explícitamente
  para obtener fuentes y luego dejar que el chain lo vuelva a llamar internamente.
- Permite exponer los docs recuperados en el resultado sin overhead extra.
- El flujo es determinístico (retrieve → generate), no iterativo.
"""

import logging
from typing import Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler

from src.config import CONFIG

logger = logging.getLogger(__name__)


def _format_docs(docs: list) -> str:
    """Formatea los documentos recuperados como contexto numerado para el LLM."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "desconocido")
        formatted.append(f"[Documento {i} - Fuente: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


# Chain de generación pura: recibe contexto ya formateado, no hace retrieval.
# Separar esto del retriever evita el doble embedding por query.
GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """{system_prompt}

Usa ÚNICAMENTE la siguiente documentación interna de la empresa para responder.
Si la respuesta no está en la documentación, indícalo claramente en lugar de inventar información.
Cita la sección relevante cuando sea posible.

DOCUMENTACIÓN:
{context}""",
        ),
        ("human", "{query}"),
    ]
)


class BaseRAGAgent:
    """
    Agente RAG base. Recupera chunks relevantes del vector store
    del dominio y genera una respuesta fundamentada en la documentación.
    """

    # Cada subclase define estos atributos
    domain: str = ""
    agent_name: str = ""
    system_prompt: str = ""

    def __init__(self, vector_store: FAISS, langfuse_handler: Optional[CallbackHandler] = None):
        self.vector_store = vector_store
        self.langfuse_handler = langfuse_handler

        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": CONFIG["retriever_k"]},
        )

        self.llm = ChatOpenAI(
            model=CONFIG["agent_model"],
            temperature=0.1,
            openai_api_key=CONFIG["openai_api_key"],
        )

        # Chain de generación: recibe contexto pre-formateado (sin retriever interno).
        # El retrieval ocurre explícitamente en run() para capturar los docs en una
        # sola llamada y exponerlos en el resultado sin overhead adicional.
        self.generation_chain = GENERATION_PROMPT | self.llm | StrOutputParser()

    def run(self, query: str, trace_id: Optional[str] = None) -> dict[str, Any]:
        """
        Ejecuta el pipeline RAG y retorna la respuesta con metadata.

        Flujo:
          1. Retrieve docs (una sola llamada al vector store)
          2. Formatear docs como contexto
          3. Generar respuesta con el LLM

        Retorna:
        {
            "agent": str,
            "domain": str,
            "query": str,
            "answer": str,
            "retrieved_docs": list[str],   # fuentes usadas
            "context": str,                # texto formateado de los docs (para el evaluador)
        }
        """
        callbacks = [self.langfuse_handler] if self.langfuse_handler else []

        # Paso 1: recuperar docs (única llamada al retriever)
        retrieved_docs = self.retriever.invoke(query)
        sources = [doc.metadata.get("source", "desconocido") for doc in retrieved_docs]
        context = _format_docs(retrieved_docs)

        # Paso 2: generar respuesta pasando el contexto ya formateado
        answer = self.generation_chain.invoke(
            {
                "context": context,
                "query": query,
                "system_prompt": self.system_prompt,
            },
            config={
                "callbacks": callbacks,
                "run_name": f"{self.agent_name}_rag",
                "metadata": {
                    "trace_id": trace_id,
                    "domain": self.domain,
                    "query": query,
                },
            },
        )

        logger.info("[%s] Respuesta generada. Fuentes: %s", self.agent_name, sources)

        return {
            "agent": self.agent_name,
            "domain": self.domain,
            "query": query,
            "answer": answer,
            "retrieved_docs": sources,
            "context": context,
        }
