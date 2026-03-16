"""
golden_evaluator.py

Evaluación sistemática del sistema multiagente contra el golden dataset.

Mide dos dimensiones objetivas por cada caso:
  1. Routing accuracy:   ¿El Orchestrator enrutó al agente correcto?
  2. Semantic similarity: ¿La respuesta generada es semánticamente similar
                           a la respuesta esperada del golden dataset?

La similitud semántica usa embeddings de OpenAI + similitud coseno.
Umbral de aprobación: similitud >= 0.80

Además registra los resultados como scores en Langfuse para tracking
histórico: si cambiás el modelo, el chunk_size o los prompts, podés
comparar runs y ver si mejoraste o empeorase.

Uso:
    python golden_evaluator.py
    python golden_evaluator.py --no-langfuse   # solo output local
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

from src.config import CONFIG
from src.multi_agent_system import MultiAgentSystem

# Umbral de similitud coseno para considerar una respuesta como correcta
SIMILARITY_THRESHOLD = 0.80


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calcula la similitud coseno entre dos vectores de embeddings."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_embedding(client: OpenAI, text: str) -> list[float]:
    """Obtiene el embedding de un texto usando text-embedding-3-small."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def load_golden_dataset(path: str = "golden_dataset.json") -> list[dict]:
    """Carga el golden dataset desde el archivo JSON."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"No se encontró el golden dataset en: {path}")
    with open(file, encoding="utf-8") as f:
        data = json.load(f)
    return data["golden_dataset"]


def evaluate_case(
    case: dict,
    system: MultiAgentSystem,
    openai_client: OpenAI,
) -> dict:
    """
    Evalúa un caso individual del golden dataset.

    Retorna un dict con:
    - Todos los campos del caso original
    - generated_answer: respuesta generada por el sistema
    - routed_agent: dominio al que enrutó el Orchestrator
    - routing_correct: bool
    - semantic_similarity: float (0.0 a 1.0)
    - answer_pass: bool (similitud >= SIMILARITY_THRESHOLD)
    - trace_id: ID del trace en Langfuse
    - latency_seconds: tiempo de respuesta
    """
    print(f"\n  Caso {case['id']}: {case['query'][:60]}...")

    start = time.time()
    result = system.process(case["query"])
    latency = round(time.time() - start, 2)

    # Routing accuracy
    routing_correct = result["domain"] == case["expected_agent"]

    # Similitud semántica entre respuesta generada y esperada
    emb_generated = get_embedding(openai_client, result["answer"])
    emb_expected  = get_embedding(openai_client, case["expected_answer"])
    similarity = round(cosine_similarity(emb_generated, emb_expected), 4)
    answer_pass = similarity >= SIMILARITY_THRESHOLD

    icon_routing = "✅" if routing_correct else "❌"
    icon_answer  = "✅" if answer_pass else "❌"

    print(f"    Routing: {icon_routing} ({result['domain']} | esperado: {case['expected_agent']})")
    print(f"    Similitud semántica: {icon_answer} {similarity:.3f} (umbral: {SIMILARITY_THRESHOLD})")
    print(f"    Latencia: {latency}s")

    return {
        **case,
        "generated_answer":   result["answer"],
        "routed_agent":        result["domain"],
        "routing_correct":     routing_correct,
        "semantic_similarity": similarity,
        "answer_pass":         answer_pass,
        "trace_id":            result["trace_id"],
        "latency_seconds":     latency,
        "llm_eval_overall":    result.get("evaluation", {}).get("overall") if result.get("evaluation") else None,
    }


def print_report(results: list[dict]) -> None:
    """Imprime el reporte final de la evaluación."""
    total = len(results)
    routing_ok    = sum(1 for r in results if r["routing_correct"])
    answer_ok     = sum(1 for r in results if r["answer_pass"])
    avg_similarity = sum(r["semantic_similarity"] for r in results) / total
    avg_latency    = sum(r["latency_seconds"] for r in results) / total

    print("\n" + "=" * 65)
    print("REPORTE GOLDEN DATASET EVALUATION")
    print("=" * 65)

    print(f"\n{'#':<4} {'Agente Esp.':<14} {'Agente Obtenido':<16} {'Routing':<10} {'Similitud':<12} {'Pass'}")
    print("-" * 65)

    for r in results:
        icon_r = "✅" if r["routing_correct"] else "❌"
        icon_a = "✅" if r["answer_pass"] else "❌"
        print(
            f"{r['id']:<4} "
            f"{r['expected_agent']:<14} "
            f"{r['routed_agent']:<16} "
            f"{icon_r:<10} "
            f"{r['semantic_similarity']:.3f}       "
            f"{icon_a}"
        )

    print("-" * 65)
    print("\n📊 MÉTRICAS GLOBALES")
    print(f"   Routing accuracy:      {routing_ok}/{total} ({routing_ok/total*100:.1f}%)")
    print(f"   Answer pass rate:      {answer_ok}/{total} ({answer_ok/total*100:.1f}%)")
    print(f"   Similitud promedio:    {avg_similarity:.3f}")
    print(f"   Latencia promedio:     {avg_latency:.2f}s")

    # Casos fallidos
    failed_routing = [r for r in results if not r["routing_correct"]]
    failed_answers = [r for r in results if not r["answer_pass"]]

    if failed_routing:
        print(f"\n⚠️  ROUTING INCORRECTO ({len(failed_routing)} casos):")
        for r in failed_routing:
            print(f"   Caso {r['id']}: esperado={r['expected_agent']} obtenido={r['routed_agent']}")
            print(f"   Query: {r['query'][:70]}...")

    if failed_answers:
        print(f"\n⚠️  RESPUESTAS BAJO UMBRAL ({len(failed_answers)} casos):")
        for r in failed_answers:
            print(f"   Caso {r['id']}: similitud={r['semantic_similarity']:.3f} | {r['query'][:60]}...")

    print("\n" + "=" * 65)


def save_results(results: list[dict], output_path: str = "golden_eval_results.json") -> None:
    """Guarda los resultados en un archivo JSON para análisis posterior."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Resultados guardados en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación del sistema multiagente contra el golden dataset"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Deshabilitar el EvaluatorAgent (más rápido)",
    )
    parser.add_argument(
        "--dataset",
        default="golden_dataset.json",
        help="Path al golden dataset (default: golden_dataset.json)",
    )
    parser.add_argument(
        "--output",
        default="golden_eval_results.json",
        help="Path para guardar los resultados (default: golden_eval_results.json)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("GOLDEN DATASET EVALUATION")
    print(f"Dataset:  {args.dataset}")
    print(f"Umbral similitud: {SIMILARITY_THRESHOLD}")
    print("=" * 65)

    # Cargar dataset
    cases = load_golden_dataset(args.dataset)
    print(f"\nCargados {len(cases)} casos del golden dataset.")

    # Inicializar sistema (sin EvaluatorAgent para no duplicar scoring)
    enable_eval = not args.no_eval
    system = MultiAgentSystem(enable_evaluation=enable_eval)
    openai_client = OpenAI(api_key=CONFIG["openai_api_key"])

    # Evaluar cada caso
    print("\nEjecutando evaluación...")
    results = []
    for case in cases:
        result = evaluate_case(case, system, openai_client)
        results.append(result)

    # Reporte final
    print_report(results)

    # Guardar resultados
    save_results(results, args.output)


if __name__ == "__main__":
    main()
