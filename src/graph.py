"""
graph.py

Define el StateGraph de LangGraph para el sistema multiagente.

El grafo modela el flujo como un DAG explícito:
  START → orchestrate → [routing condicional] → {domain}_agent → evaluate → END

Beneficios respecto a la orquestación imperativa anterior:
- Estado tipado (AgentState) con todas las claves del pipeline en un TypedDict.
- Routing declarativo mediante add_conditional_edges: la lógica de rutas es visible
  en la definición del grafo, no enterrada en bloques if/elif.
- Extensibilidad: agregar un nuevo dominio = un nodo + un edge condicional,
  sin modificar la lógica del orquestador ni de otros agentes.
- Compatible con checkpointing (InMemorySaver / PostgresSaver) para recuperación
  de estado ante fallos, sin cambios en los nodos.
"""

import logging
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Estado compartido que fluye a través de todos los nodos del grafo."""

    # Input
    query: str
    user_id: str
    trace_id: str

    # Populado por orchestrate_node
    domain: str
    confidence: float
    reasoning: str

    # Populado por rag_node
    agent_name: str
    answer: str
    retrieved_docs: list
    context: str

    # Populado por evaluate_node
    evaluation: Optional[dict]


def build_graph(orchestrator, agents: dict, evaluator):
    """
    Construye y compila el StateGraph del sistema multiagente.

    Args:
        orchestrator: Instancia de Orchestrator para clasificación de intención.
        agents:       Dict {domain: RAGAgent} con los 4 agentes especializados.
        evaluator:    Instancia de EvaluatorAgent, o None si la evaluación está deshabilitada.

    Returns:
        CompiledStateGraph listo para invocar con graph.invoke(initial_state).

    Estructura del grafo:
        START
          └─► orchestrate_node
                └─► [conditional: route_to_agent]
                      ├─► hr_agent_node
                      ├─► tech_agent_node
                      ├─► finance_agent_node
                      └─► legal_agent_node
                            └─► evaluate_node
                                  └─► END
    """

    # -----------------------------------------------------------------------
    # Nodos
    # -----------------------------------------------------------------------

    def orchestrate_node(state: AgentState) -> dict:
        """Clasifica la intención del usuario y determina el dominio destino."""
        result = orchestrator.classify(state["query"], trace_id=state["trace_id"])
        return {
            "domain": result["domain"],
            "confidence": result.get("confidence", 0.0),
            "reasoning": result.get("reasoning", ""),
        }

    def make_rag_node(domain: str):
        """Factory que crea el nodo RAG para el dominio dado."""

        def rag_node(state: AgentState) -> dict:
            result = agents[domain].run(state["query"], trace_id=state["trace_id"])
            return {
                "agent_name": result["agent"],
                "answer": result["answer"],
                "retrieved_docs": result["retrieved_docs"],
                "context": result["context"],
            }

        rag_node.__name__ = f"{domain}_agent"
        return rag_node

    def evaluate_node(state: AgentState) -> dict:
        """
        Evalúa la respuesta con LLM-as-a-judge.
        Si evaluator es None (evaluación deshabilitada), retorna evaluation=None.
        """
        if evaluator is None:
            return {"evaluation": None}

        scores = evaluator.evaluate(
            trace_id=state["trace_id"],
            query=state["query"],
            answer=state["answer"],
            agent_name=state["agent_name"],
            context=state.get("context"),
        )
        return {"evaluation": scores}

    # -----------------------------------------------------------------------
    # Routing function
    # -----------------------------------------------------------------------

    def route_to_agent(state: AgentState) -> str:
        """Retorna el nombre del nodo destino según el dominio clasificado."""
        return state["domain"]

    # -----------------------------------------------------------------------
    # Construcción del grafo
    # -----------------------------------------------------------------------

    builder = StateGraph(AgentState)

    # Nodo de orquestación
    builder.add_node("orchestrate", orchestrate_node)

    # Nodos RAG por dominio
    for domain in ("hr", "tech", "finance", "legal"):
        builder.add_node(f"{domain}_agent", make_rag_node(domain))

    # Nodo de evaluación
    builder.add_node("evaluate", evaluate_node)

    # Edges
    builder.add_edge(START, "orchestrate")

    # Routing condicional: orchestrate → agente del dominio correcto
    builder.add_conditional_edges(
        "orchestrate",
        route_to_agent,
        {
            "hr": "hr_agent",
            "tech": "tech_agent",
            "finance": "finance_agent",
            "legal": "legal_agent",
        },
    )

    # Todos los agentes convergen al nodo de evaluación
    for domain in ("hr", "tech", "finance", "legal"):
        builder.add_edge(f"{domain}_agent", "evaluate")

    builder.add_edge("evaluate", END)

    return builder.compile(checkpointer=MemorySaver())
