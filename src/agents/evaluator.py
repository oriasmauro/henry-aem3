"""
agents/evaluator.py

Agente Evaluador (BONUS): evalúa la calidad de cada respuesta RAG
y registra los scores en Langfuse usando la Score API.

Decisión técnica: evaluación LLM-as-a-judge con dimensiones múltiples.
Es más rico que un score único porque permite detectar exactamente
qué dimensión falla (ej: respuesta relevante pero incompleta).

Dimensiones evaluadas (1 a 10):
- relevance:    ¿La respuesta responde la pregunta original?
- completeness: ¿La respuesta cubre todos los aspectos de la pregunta?
- accuracy:     ¿La información de la respuesta está soportada por el contexto recuperado?
- overall:      Puntaje general (evaluación holística, no promedio simple)

Nota sobre accuracy: incluir el contexto recuperado en el prompt del evaluador
es clave para que esta dimensión sea real. Sin los docs, el LLM solo puede
juzgar si la respuesta "suena" correcta, no si está fundamentada en los datos.
"""

import json
import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse import Langfuse

from src.config import CONFIG

logger = logging.getLogger(__name__)


EVALUATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un evaluador experto de respuestas de sistemas de IA corporativos.
Tu tarea es evaluar la calidad de una respuesta generada por un agente RAG.

Evalúa en las siguientes dimensiones, asignando un puntaje del 1 al 10:

- relevance (relevancia): ¿La respuesta aborda directamente la pregunta formulada?
  1 = completamente irrelevante, 10 = perfectamente alineada con la pregunta

- completeness (completitud): ¿La respuesta cubre todos los aspectos importantes de la pregunta?
  1 = respuesta muy parcial, 10 = respuesta exhaustiva y completa

- accuracy (precisión): ¿La información en la respuesta está respaldada por el CONTEXTO RECUPERADO?
  1 = información inventada o contradice los documentos, 10 = información precisa y bien fundamentada en los docs

Responde ÚNICAMENTE con un JSON válido con este formato, sin texto adicional:
{{
  "relevance": <1-10>,
  "completeness": <1-10>,
  "accuracy": <1-10>,
  "overall": <1-10>,
  "feedback": "<comentario breve en español explicando los puntajes>"
}}

El 'overall' debe ser tu evaluación holística (no necesariamente el promedio exacto).""",
        ),
        (
            "human",
            """PREGUNTA ORIGINAL:
{query}

CONTEXTO RECUPERADO (documentos que el agente tenía disponibles):
{context}

RESPUESTA DEL AGENTE ({agent_name}):
{answer}

Por favor, evalúa la calidad de esta respuesta basándote en el contexto recuperado.""",
        ),
    ]
)


class EvaluatorAgent:
    """
    Evalúa respuestas RAG y registra los scores en Langfuse.

    Uso:
        evaluator = EvaluatorAgent()
        evaluator.evaluate(
            trace_id="...",
            query="¿Cuántos días de vacaciones tengo?",
            answer="Según la política...",
            agent_name="HRAgent",
            context="[Documento 1 - Fuente: ...]\n..."
        )
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=CONFIG["evaluator_model"],
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
            openai_api_key=CONFIG["openai_api_key"],
        )
        self.chain = EVALUATOR_PROMPT | self.llm

        # Cliente Langfuse directo para registrar scores via Score API
        self.langfuse = Langfuse(
            public_key=CONFIG["langfuse_public_key"],
            secret_key=CONFIG["langfuse_secret_key"],
            host=CONFIG["langfuse_host"],
        )

    def evaluate(
        self,
        trace_id: str,
        query: str,
        answer: str,
        agent_name: str,
        context: Optional[str] = None,
    ) -> dict:
        """
        Evalúa la respuesta y registra los scores en Langfuse.

        Args:
            trace_id:   ID del trace de Langfuse al que asociar los scores.
            query:      Consulta original del usuario.
            answer:     Respuesta generada por el agente RAG.
            agent_name: Nombre del agente que generó la respuesta.
            context:    Texto de los documentos recuperados (para evaluar accuracy).
                        Si no se provee, se usa un placeholder con advertencia.

        Returns:
            dict con los scores evaluados.
        """
        logger.info("[Evaluator] Evaluando respuesta de %s...", agent_name)

        if context is None:
            logger.warning(
                "[Evaluator] No se recibió contexto recuperado para trace %s. "
                "La dimensión 'accuracy' será menos precisa.",
                trace_id,
            )
            context = "(contexto no disponible)"

        response = self.chain.invoke(
            {
                "query": query,
                "answer": answer,
                "agent_name": agent_name,
                "context": context,
            }
        )

        scores = json.loads(response.content)

        # Registrar cada dimensión como un score separado en Langfuse
        score_dimensions = ["relevance", "completeness", "accuracy", "overall"]

        for dimension in score_dimensions:
            if dimension in scores:
                self.langfuse.score(
                    trace_id=trace_id,
                    name=f"eval_{dimension}",
                    value=float(scores[dimension]),
                    comment=scores.get("feedback", ""),
                )

        logger.info(
            "[Evaluator] Scores → relevance=%s | completeness=%s | accuracy=%s | overall=%s",
            scores.get("relevance"),
            scores.get("completeness"),
            scores.get("accuracy"),
            scores.get("overall"),
        )
        logger.info("[Evaluator] Feedback: %s", scores.get("feedback", ""))

        return scores
