"""
main.py

Entry point del sistema multiagente de routing inteligente.

Modos de uso:
  1. Demo (4 queries de ejemplo): python main.py
  2. Consulta individual:         python main.py "¿Cuántos días de vacaciones tengo?"
  3. Test suite completa:         python main.py --test
  4. Modo interactivo:            python main.py --interactive

Flags opcionales:
  --no-eval   Deshabilitar el agente evaluador (más rápido, menos costo)
  --debug     Activar logging de nivel DEBUG (verbose)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.multi_agent_system import MultiAgentSystem


def setup_logging(debug: bool = False) -> None:
    """
    Configura el sistema de logging para toda la aplicación.

    Nivel INFO por defecto: muestra el flujo de ejecución sin ruido excesivo.
    Nivel DEBUG con --debug: incluye detalles internos de LangChain y FAISS.

    Formato: timestamp + nivel + módulo + mensaje.
    Esto facilita correlacionar logs con traces de Langfuse usando el timestamp.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Silenciar librerías externas verbosas en nivel INFO
    if not debug:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("langchain").setLevel(logging.WARNING)
        logging.getLogger("faiss").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


def load_test_queries(path: str = "test_queries.json") -> list[dict]:
    """Carga las queries de prueba desde el archivo JSON."""
    test_file = Path(path)
    if not test_file.exists():
        logger.error("No se encontró el archivo de test queries: %s", path)
        sys.exit(1)

    with open(test_file, encoding="utf-8") as f:
        data = json.load(f)

    return data.get("test_queries", [])


def run_single_query(system: MultiAgentSystem, query: str) -> None:
    """Ejecuta una consulta individual."""
    system.process(query)


def run_test_suite(system: MultiAgentSystem) -> None:
    """Ejecuta todas las queries de prueba y reporta resultados."""
    logger.info("\n" + "=" * 60)
    logger.info("EJECUTANDO TEST SUITE COMPLETA")
    logger.info("=" * 60)

    queries = load_test_queries()
    results = system.run_test_queries(queries)

    # Resumen tabular de resultados
    print("\nRESUMEN DE RESULTADOS:")
    print(f"{'#':<4} {'Esperado':<20} {'Obtenido':<20} {'OK':<6} {'Overall'}")
    print("-" * 60)

    for i, r in enumerate(results, 1):
        expected = r.get("expected_agent", "N/A")
        obtained = r.get("domain", "N/A")
        correct = "OK" if r.get("routing_correct", False) else "FAIL"
        overall = r.get("evaluation", {})
        overall_score = overall.get("overall", "N/A") if overall else "N/A"
        print(f"{i:<4} {expected:<20} {obtained:<20} {correct:<6} {overall_score}")


def run_interactive(system: MultiAgentSystem) -> None:
    """Modo interactivo: el usuario escribe consultas en la terminal."""
    print("\n" + "=" * 60)
    print("MODO INTERACTIVO - MultiAgent Support System")
    print("Escribi 'salir' para terminar")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Tu consulta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSaliendo...")
            break

        if query.lower() in ("salir", "exit", "quit"):
            print("Hasta luego.")
            break

        if not query:
            continue

        system.process(query)


def main():
    parser = argparse.ArgumentParser(
        description="Sistema Multiagente de Routing Inteligente - TechCorp SaaS"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Consulta a procesar (opcional)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Ejecutar la suite completa de test queries",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Modo interactivo: ingresar consultas manualmente",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Deshabilitar el agente evaluador (más rápido, menos costo)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activar logging de nivel DEBUG",
    )

    args = parser.parse_args()

    # Configurar logging antes de cualquier otra inicialización
    setup_logging(debug=args.debug)

    # Inicializar el sistema
    enable_evaluation = not args.no_eval
    system = MultiAgentSystem(enable_evaluation=enable_evaluation)

    # Modo de ejecución
    if args.test:
        run_test_suite(system)

    elif args.interactive:
        run_interactive(system)

    elif args.query:
        run_single_query(system, args.query)

    else:
        # Demo por defecto con algunas queries de ejemplo
        demo_queries = [
            "¿Cuántos días de vacaciones me corresponden después de 3 años en la empresa?",
            "Mi laptop se cayó y no enciende. ¿Qué hago?",
            "Necesito saber cómo presentar un gasto de viaje en Expensify.",
            "Queremos firmar un contrato con un proveedor nuevo. ¿Quién debe aprobarlo?",
        ]

        logger.info("\n" + "=" * 60)
        logger.info("DEMO - 4 consultas de ejemplo")
        logger.info("=" * 60)

        for query in demo_queries:
            system.process(query)


if __name__ == "__main__":
    main()
