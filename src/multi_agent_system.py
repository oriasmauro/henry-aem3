"""
multi_agent_system.py

Facade del sistema multiagente que usa un LangGraph StateGraph internamente.

Responsabilidades:
- Inicializar todos los componentes (vector stores, agentes, Langfuse)
- Construir el grafo de ejecución (src/graph.py)
- Gestionar el ciclo de vida de las trazas de Langfuse (crear → spans → flush)
- Exponer la API pública process() y run_test_queries()

El grafo (src/graph.py) encapsula el flujo condicional:
  orchestrate → route → {domain}_agent → evaluate

Cambio de arquitectura respecto a la versión anterior:
  Antes: MultiAgentSystem llamaba directamente a orchestrator, agents y evaluator
         mediante if/elif en process().
  Ahora: MultiAgentSystem invoca self.graph.invoke(initial_state) y el grafo
         LangGraph gestiona el routing declarativo y el estado tipado.
"""

import logging
import uuid
from typing import Optional

from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from src.agents.evaluator import EvaluatorAgent
from src.agents.finance_agent import FinanceAgent
from src.agents.hr_agent import HRAgent
from src.agents.legal_agent import LegalAgent
from src.agents.orchestrator import Orchestrator
from src.agents.tech_agent import TechAgent
from src.config import CONFIG
from src.graph import AgentState, build_graph
from src.vector_store import build_all_vector_stores

logger = logging.getLogger(__name__)


class MultiAgentSystem:
    """
    Sistema multiagente con orquestación LangGraph, RAG y observabilidad completa.

    Inicialización:
        system = MultiAgentSystem()

    Uso:
        result = system.process("¿Cuántos días de vacaciones tengo?")
        print(result["answer"])
    """

    def __init__(self, enable_evaluation: bool = True):
        logger.info("Inicializando MultiAgentSystem...")

        # Cliente Langfuse para crear traces
        self.langfuse = Langfuse(
            public_key=CONFIG["langfuse_public_key"],
            secret_key=CONFIG["langfuse_secret_key"],
            host=CONFIG["langfuse_host"],
        )

        # Construir o cargar índices FAISS para todos los dominios
        self.vector_stores = build_all_vector_stores()

        # Langfuse callback handler compartido para tracing LangChain
        self.langfuse_handler = CallbackHandler(
            public_key=CONFIG["langfuse_public_key"],
            secret_key=CONFIG["langfuse_secret_key"],
            host=CONFIG["langfuse_host"],
        )

        # Orquestador
        self.orchestrator = Orchestrator(langfuse_handler=self.langfuse_handler)

        # Agentes RAG especializados
        self.agents = {
            "hr": HRAgent(self.vector_stores["hr"], self.langfuse_handler),
            "tech": TechAgent(self.vector_stores["tech"], self.langfuse_handler),
            "finance": FinanceAgent(self.vector_stores["finance"], self.langfuse_handler),
            "legal": LegalAgent(self.vector_stores["legal"], self.langfuse_handler),
        }

        # Evaluador (bonus)
        self.enable_evaluation = enable_evaluation
        self.evaluator = EvaluatorAgent() if enable_evaluation else None

        # Construir el grafo LangGraph con las instancias ya inicializadas
        self.graph = build_graph(self.orchestrator, self.agents, self.evaluator)

        logger.info("Sistema inicializado correctamente.\n")

    def process(self, query: str, user_id: Optional[str] = None) -> dict:
        """
        Procesa una consulta a través del pipeline LangGraph completo.

        El grafo ejecuta: orchestrate → {domain}_agent → evaluate

        Args:
            query:   Consulta del usuario en lenguaje natural.
            user_id: Identificador opcional del usuario para trazabilidad.

        Returns:
            {
                "trace_id": str,
                "query": str,
                "domain": str,
                "confidence": float,
                "reasoning": str,
                "agent": str,
                "answer": str,
                "retrieved_docs": list[str],
                "evaluation": dict | None,
                "langfuse_url": str
            }
        """
        trace_id = str(uuid.uuid4())

        logger.info("=" * 60)
        logger.info("CONSULTA: %s", query)
        logger.info("TRACE ID: %s", trace_id)
        logger.info("=" * 60)

        # Crear trace principal en Langfuse
        trace = self.langfuse.trace(
            id=trace_id,
            name="multi_agent_query",
            input={"query": query},
            user_id=user_id or "anonymous",
            metadata={"system": "MultiAgentSystem", "version": "2.0"},
        )

        # Estado inicial que fluye por el grafo
        initial_state: AgentState = {
            "query": query,
            "user_id": user_id or "anonymous",
            "trace_id": trace_id,
            "domain": "",
            "confidence": 0.0,
            "reasoning": "",
            "agent_name": "",
            "answer": "",
            "retrieved_docs": [],
            "context": "",
            "evaluation": None,
        }

        try:
            logger.info("[1/3] Ejecutando grafo LangGraph...")
            final_state = self.graph.invoke(initial_state)

            # Registrar spans en Langfuse con los datos del estado final
            trace.span(
                name="orchestrator_classification",
                input={"query": query},
                output={
                    "domain": final_state["domain"],
                    "confidence": final_state["confidence"],
                    "reasoning": final_state["reasoning"],
                },
                metadata={
                    "domain": final_state["domain"],
                    "confidence": final_state["confidence"],
                },
            )

            trace.span(
                name=f"rag_{final_state['domain']}",
                input={"query": query, "domain": final_state["domain"]},
                output={
                    "answer": final_state["answer"],
                    "sources": final_state["retrieved_docs"],
                },
            )

            # Actualizar trace con el output final
            trace.update(
                output={
                    "domain": final_state["domain"],
                    "answer": final_state["answer"],
                    "evaluation": final_state["evaluation"],
                }
            )

        except Exception as exc:
            # Registrar el error en Langfuse antes de propagar la excepción
            logger.error("[MultiAgentSystem] Error procesando query '%s': %s", query, exc)
            trace.update(
                output={"error": str(exc)},
                level="ERROR",
                status_message=str(exc),
            )
            self.langfuse.flush()
            raise

        # Flush para asegurar que los datos se envíen a Langfuse
        self.langfuse.flush()

        langfuse_url = f"{CONFIG['langfuse_host']}/trace/{trace_id}"

        result = {
            "trace_id": trace_id,
            "query": query,
            "domain": final_state["domain"],
            "confidence": final_state["confidence"],
            "reasoning": final_state["reasoning"],
            "agent": final_state["agent_name"],
            "answer": final_state["answer"],
            "retrieved_docs": final_state["retrieved_docs"],
            "evaluation": final_state["evaluation"],
            "langfuse_url": langfuse_url,
        }

        logger.info("=" * 60)
        logger.info("RESPUESTA (%s):\n%s", final_state["domain"].upper(), final_state["answer"])
        if final_state["evaluation"]:
            logger.info(
                "EVALUACION: overall=%s | %s",
                final_state["evaluation"].get("overall"),
                final_state["evaluation"].get("feedback", ""),
            )
        logger.info("Trace: %s", langfuse_url)
        logger.info("=" * 60)

        return result

    def run_test_queries(self, queries: list[dict]) -> list[dict]:
        """
        Ejecuta una lista de queries de prueba y retorna los resultados.

        Args:
            queries: Lista de dicts con al menos la clave "query".
                     Opcionalmente "expected_agent" para validación de routing.
        """
        results = []
        correct = 0

        for i, q in enumerate(queries, 1):
            logger.info("[Test %d/%d]", i, len(queries))
            result = self.process(q["query"])

            expected = q.get("expected_agent")
            if expected:
                match = result["domain"] == expected
                if match:
                    correct += 1
                result["routing_correct"] = match
                result["expected_agent"] = expected

            results.append(result)

        if any("expected_agent" in q for q in queries):
            accuracy = correct / len(queries) * 100
            logger.info("=" * 60)
            logger.info("ACCURACY DE ROUTING: %d/%d (%.1f%%)", correct, len(queries), accuracy)
            logger.info("=" * 60)

        return results
