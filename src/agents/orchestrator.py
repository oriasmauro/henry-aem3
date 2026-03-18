"""
agents/orchestrator.py

Agente Orquestador: clasifica la intención de la consulta del usuario
y la enruta al agente RAG especializado correspondiente.

Decisión técnica: usamos un LLM con structured output (JSON mode) para
la clasificación. Es más robusto que regex o keyword matching porque
maneja lenguaje natural ambiguo y casos borde (ej: "licencia GPL" → legal,
no tech).

El orquestador NO responde la consulta. Solo clasifica y delega.
"""

import json
import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler

from src.config import CONFIG
from src.constants import VALID_DOMAINS

logger = logging.getLogger(__name__)

# Constante pre-computada al cargar el módulo: evita recalcular en cada classify()
DOMAINS_DESCRIPTION: str = "\n".join(
    f"- '{domain}': {description}" for domain, description in VALID_DOMAINS.items()
)

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres el orquestador de un sistema multiagente corporativo.
Tu única tarea es clasificar la consulta del usuario en uno de los siguientes dominios:

{domains_description}

Responde ÚNICAMENTE con un objeto JSON con el siguiente formato, sin texto adicional:
{{
  "domain": "<hr|tech|finance|legal>",
  "confidence": <0.0 a 1.0>,
  "reasoning": "<explicación breve en español de por qué elegiste ese dominio>"
}}

Si la consulta es ambigua, elige el dominio más probable y baja el confidence.
Si la consulta no corresponde a ningún dominio, usa el más cercano con confidence bajo.""",
        ),
        ("human", "Consulta del usuario: {query}"),
    ]
)


class Orchestrator:
    """
    Clasifica la intención del usuario y retorna el dominio destino.

    Decisión técnica: gpt-4o-mini como modelo de clasificación.
    Es suficientemente capaz para esta tarea y significativamente
    más barato que gpt-4o, lo que importa porque el orquestador
    se invoca en cada consulta.
    """

    def __init__(self, langfuse_handler: Optional[CallbackHandler] = None):
        self.llm = ChatOpenAI(
            model=CONFIG["orchestrator_model"],
            temperature=0,  # clasificación determinística
            model_kwargs={"response_format": {"type": "json_object"}},
            openai_api_key=CONFIG["openai_api_key"],
        )
        self.langfuse_handler = langfuse_handler
        self.chain = CLASSIFICATION_PROMPT | self.llm

    def classify(self, query: str, trace_id: Optional[str] = None) -> dict:
        """
        Clasifica la consulta y retorna:
        {
            "domain": "hr" | "tech" | "finance" | "legal",
            "confidence": float,
            "reasoning": str
        }
        """
        callbacks = [self.langfuse_handler] if self.langfuse_handler else []

        response = self.chain.invoke(
            {
                "domains_description": DOMAINS_DESCRIPTION,
                "query": query,
            },
            config={
                "callbacks": callbacks,
                "run_name": "orchestrator_classify",
                "metadata": {"trace_id": trace_id, "query": query},
            },
        )

        result = json.loads(response.content)

        # Validación defensiva: dominio inválido → "unknown" para activar clarification_node
        if result.get("domain") not in VALID_DOMAINS:
            logger.warning(
                "[Orchestrator] Dominio inválido retornado: '%s'. Usando 'unknown'.",
                result.get("domain"),
            )
            result["domain"] = "unknown"
            result["confidence"] = 0.0

        confidence = result.get("confidence", 0.0)

        # Advertencia cuando la clasificación es poco confiable
        if confidence < CONFIG["confidence_threshold"]:
            logger.warning(
                "[Orchestrator] Confianza baja (%.2f < %.2f) para dominio '%s'. "
                "Query: '%s'. Considera revisar este caso en Langfuse.",
                confidence,
                CONFIG["confidence_threshold"],
                result["domain"],
                query,
            )

        logger.info(
            "[Orchestrator] Clasificación: domain=%s confidence=%.2f | %s",
            result["domain"],
            confidence,
            result.get("reasoning", ""),
        )

        return result
