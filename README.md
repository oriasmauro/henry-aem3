# Sistema Multiagente de Routing Inteligente

Sistema de soporte corporativo basado en múltiples agentes de IA que clasifica automáticamente las consultas entrantes por departamento y las dirige a agentes RAG especializados, con observabilidad completa mediante Langfuse.

## Descripción

Un **Agente Orquestador** clasifica la intención de cada consulta (RRHH, IT, Finanzas o Legal) y la enruta condicionalmente al **Agente RAG especializado** correspondiente a través de un **StateGraph de LangGraph**. Cada agente recupera contexto relevante desde su base de conocimiento documental y genera una respuesta fundamentada en la documentación interna de la empresa. Todo el flujo queda trazado en **Langfuse** para observabilidad y depuración. Un **Agente Evaluador** (bonus) puntúa automáticamente cada respuesta en 4 dimensiones de calidad.

## Arquitectura: LangGraph StateGraph

El flujo de ejecución está modelado como un DAG explícito usando LangGraph:

```
Consulta del usuario
        │
        ▼
┌──────────────────────────────────────────────────────┐
│               LangGraph StateGraph                   │
│                                                      │
│  START → orchestrate_node                            │
│               │                                      │
│    [add_conditional_edges: route_to_agent]           │
│               │                                      │
│    ┌──────────┼──────────────────────┐               │
│    ▼          ▼          ▼           ▼               │
│  hr_agent  tech_agent  finance_agent  legal_agent    │
│  (FAISS)   (FAISS)     (FAISS)        (FAISS)        │
│    └──────────┴──────────┴────────────┘              │
│               │                                      │
│          evaluate_node                               │
│               │                                      │
│              END                                     │
└──────────────────────────────────────────────────────┘
        │
        ▼
 EvaluatorAgent → Score API → Langfuse
```

### AgentState (estado tipado)

Todos los nodos del grafo comparten un `TypedDict` que fluye a través del pipeline:

```python
class AgentState(TypedDict):
    query: str          # input del usuario
    user_id: str
    trace_id: str
    domain: str         # set por orchestrate_node
    confidence: float
    reasoning: str
    agent_name: str     # set por rag_node
    answer: str
    retrieved_docs: list
    context: str
    evaluation: Optional[dict]  # set por evaluate_node
```

### Checkpointing con MemorySaver

El grafo usa `MemorySaver` como checkpointer, lo que habilita persistencia de estado entre invocaciones por `thread_id`:

```python
# Conversación con usuario identificado (estado persiste entre mensajes)
result = system.process("¿Cuántos días de vacaciones tengo?", user_id="usuario-123")
result = system.process("¿Y si soy part-time?", user_id="usuario-123")  # mismo thread

# Sin user_id → thread_id = trace_id (sin persistencia entre requests)
result = system.process("¿Cómo configuro la VPN?")
```

Para persistencia real en producción, reemplazar `MemorySaver` por `PostgresSaver` en `src/graph.py`.

## Evaluación con golden dataset

El proyecto incluye un evaluador sistemático contra un dataset dorado de 12 casos curados:

```bash
# Evaluación completa (con scoring LLM)
python golden_evaluator.py

# Sin evaluador LLM (solo routing accuracy + similitud semántica)
python golden_evaluator.py --no-eval

# Dataset personalizado
python golden_evaluator.py --dataset mi_dataset.json
```

Los resultados se guardan en `golden_eval_results.json` y en Langfuse. Métricas reportadas:
- **Routing accuracy**: % de consultas correctamente derivadas al dominio esperado
- **Answer pass rate**: % de respuestas que superan el umbral de similitud semántica (≥ 0.80)
- **Similitud semántica**: coseno entre embeddings de la respuesta y la respuesta esperada

## Estructura del proyecto

```
aem3/
├── data/
│   ├── hr_docs/                    # Políticas de RRHH
│   │   ├── politica_vacaciones.txt
│   │   ├── desempeno_compensacion.txt
│   │   └── incorporacion_desvinculacion.txt
│   ├── tech_docs/                  # Documentación de IT
│   │   ├── politica_soporte_it.txt
│   │   └── infraestructura_incidentes.txt
│   ├── finance_docs/               # Políticas financieras
│   │   ├── politica_gastos_presupuesto.txt
│   │   └── facturacion_cuentas_pagar.txt
│   └── legal_docs/                 # Documentación legal
│       └── contratos_cumplimiento_legal.txt
├── src/
│   ├── __init__.py
│   ├── config.py                   # Carga y validación de variables de entorno
│   ├── vector_store.py             # Construcción y persistencia de índices FAISS
│   ├── graph.py                    # AgentState TypedDict + build_graph (LangGraph)
│   └── agents/
│       ├── __init__.py
│       ├── base_rag_agent.py       # Clase base con pipeline de retrieval + generación
│       ├── orchestrator.py         # Clasificación de intención con JSON mode
│       ├── hr_agent.py             # Agente de RRHH
│       ├── tech_agent.py           # Agente de IT
│       ├── finance_agent.py        # Agente de Finanzas
│       ├── legal_agent.py          # Agente de Legal
│       └── evaluator.py            # Agente evaluador con Score API (bonus)
├── tests/
│   ├── conftest.py                 # Fixtures y configuración global de pytest
│   ├── test_config.py
│   ├── test_vector_store.py
│   ├── test_graph.py               # Tests del StateGraph LangGraph
│   ├── test_multi_agent_system.py
│   ├── test_main.py
│   └── agents/
│       ├── test_base_rag_agent.py
│       ├── test_orchestrator.py
│       ├── test_evaluator.py
│       └── test_specialized_agents.py
├── faiss_index/                    # Índices FAISS persistidos (generado al ejecutar)
├── main.py                         # Entry point con CLI
├── golden_evaluator.py             # Evaluador sistemático contra dataset dorado
├── golden_dataset.json             # 12 casos curados con respuestas esperadas
├── test_queries.json               # 12 consultas de prueba con routing esperado
├── pytest.ini                      # Configuración de pytest y cobertura
├── requirements.txt
├── requirements-dev.txt            # Herramientas de desarrollo (ruff)
├── .python-version                 # Python 3.12 (requerido por faiss-cpu)
├── .github/workflows/ci.yaml       # CI: lint + format + tests con cobertura
├── .env.example
└── README.md
```

## Requisitos previos

- Python 3.12 o 3.13 (no compatible con 3.14+ por limitaciones de `faiss-cpu`)
- [uv](https://docs.astral.sh/uv/) instalado
- Cuenta de OpenAI con API key
- Cuenta de Langfuse (gratuita en https://cloud.langfuse.com)

## Instalación

### 1. Clonar o descargar el proyecto

```bash
cd aem3
```

### 2. Crear entorno virtual con Python 3.12

```bash
uv venv --python 3.12 venv
source venv/bin/activate       # Linux/Mac
```

### 3. Instalar dependencias

```bash
uv pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
```

Editar `.env` con tus credenciales:

```env
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

**Obtener keys de Langfuse:**
1. Crear cuenta en https://cloud.langfuse.com
2. Crear un nuevo proyecto (ej: `aem3-multiagente`)
3. Ir a Settings → API Keys → Create new API key

### Variables opcionales

Todas tienen valores por defecto funcionales. Editarlas solo si querés personalizar el comportamiento:

| Variable | Default | Descripción |
|---|---|---|
| `ORCHESTRATOR_MODEL` | `gpt-4o-mini` | Modelo para clasificación |
| `AGENT_MODEL` | `gpt-4o-mini` | Modelo para generación RAG |
| `EVALUATOR_MODEL` | `gpt-4o-mini` | Modelo para evaluación |
| `CHUNK_SIZE` | `800` | Tamaño de chunks en caracteres |
| `CHUNK_OVERLAP` | `100` | Solapamiento entre chunks |
| `RETRIEVER_K` | `4` | Documentos a recuperar por query |
| `FAISS_INDEX_DIR` | `faiss_index` | Directorio de índices persistidos |
| `CONFIDENCE_THRESHOLD` | `0.4` | Umbral de confianza del orquestador |

## Cómo ejecutar

### Demo rápida (4 consultas de ejemplo)

```bash
python main.py
```

### Consulta individual

```bash
python main.py "¿Cuántos días de vacaciones me corresponden después de 3 años?"
```

### Test suite completa (12 queries con validación de routing)

```bash
python main.py --test
```

### Modo interactivo

```bash
python main.py --interactive
```

### Sin evaluador (más rápido, menos costo de tokens)

```bash
python main.py --no-eval "¿Cómo configuro la VPN?"
```

### Logging verbose para debugging

```bash
python main.py --debug "¿Cuántos días de vacaciones tengo?"
```

## Ejemplos de uso

```bash
# RRHH
python main.py "¿Qué pasa si me enfermo más de 3 días seguidos?"

# IT
python main.py "Nuestro servicio de producción está caído, ¿qué hago?"

# Finanzas
python main.py "Un cliente lleva 45 días sin pagar su factura."

# Legal
python main.py "¿Podemos usar una librería con licencia GPL en nuestro producto?"

# Caso borde (parece IT pero es Legal)
python main.py "¿Qué licencias de open source están pre-aprobadas?"
```

## Tests

El proyecto tiene cobertura del 100% sobre todos los módulos de `src/` y `main.py`.

### Correr los tests

```bash
# Con reporte de cobertura completo
uv pip install -r requirements.txt -r requirements-dev.txt
pytest -q --cov=src --cov-report=term-missing

# Sin reporte de cobertura (más rápido)
pytest --no-cov

# Un archivo específico
pytest tests/test_graph.py -v

# Con output detallado de fallos
pytest -v --tb=short
```

### Lint y formato

```bash
# Corregir automáticamente
ruff check . --fix
ruff format .

# Solo verificar (como el CI)
ruff check .
ruff format --check .
```

### Estructura de tests

| Archivo | Qué testea |
|---|---|
| `test_config.py` | Carga de env vars, valores por defecto, error en vars faltantes |
| `test_vector_store.py` | Carga de docs, construcción de índice, persistencia en disco |
| `test_graph.py` | `AgentState`, `build_graph`, nodos, routing condicional (4 dominios), evaluate con/sin evaluador |
| `tests/agents/test_base_rag_agent.py` | `_format_docs`, pipeline RAG, verificación de un solo retrieval |
| `tests/agents/test_orchestrator.py` | Clasificación, fallback en dominio inválido, umbral de confianza |
| `tests/agents/test_evaluator.py` | Scoring 4 dimensiones, contexto en el prompt, registro en Langfuse |
| `tests/agents/test_specialized_agents.py` | Atributos y `run()` de HR, Tech, Finance y Legal agents |
| `test_multi_agent_system.py` | Pipeline completo, error handling con trace de Langfuse, routing |
| `test_main.py` | CLI (todos los flags), logging, modo interactivo, modo demo |

### Reporte de cobertura HTML

Después de correr `python -m pytest`, el reporte detallado está disponible en:

```
htmlcov/index.html
```

## Observabilidad con Langfuse

Cada consulta genera un **trace** en Langfuse con:
- Span del Orchestrator: clasificación, dominio seleccionado y `confidence`
- Span del RAG agent: chunks recuperados y respuesta generada
- Scores del Evaluador: `eval_relevance`, `eval_completeness`, `eval_accuracy`, `eval_overall`
- Nivel `ERROR` en el trace si ocurre una excepción (para incident response)

Al ejecutar, el sistema imprime la URL directa al trace:
```
Trace: https://cloud.langfuse.com/trace/<trace_id>
```

**Tip:** Si `confidence` es baja, el sistema emite un `WARNING` en los logs con el trace ID para facilitar la revisión en Langfuse.

## Decisiones técnicas

**¿Por qué LangGraph en lugar de orquestación imperativa?**
La versión anterior usaba `if/elif` en `process()` para rutear entre agentes. Con LangGraph, el routing es declarativo (`add_conditional_edges`), el estado es tipado (`AgentState`) y el grafo es inspeccionable. Agregar un nuevo dominio requiere solo un `add_node` + un edge condicional, sin modificar la lógica existente.

**¿Por qué no es A2A (Agent-to-Agent)?**
Este sistema usa orquestación hub-and-spoke: un orquestador central rutea a agentes especializados dentro del mismo proceso. El protocolo A2A de Google implica agentes independientes que exponen HTTP endpoints con Agent Cards y se comunican como peers entre distintos servicios. Esta arquitectura es correcta para un único servicio corporativo; A2A agregaría overhead innecesario sin beneficio real.

**¿Por qué un vector store por dominio?**
Índices separados por dominio evitan recuperaciones cruzadas. Si el HR agent consultara un índice unificado, podría traer chunks de IT o Legal como contexto, degradando la precisión de las respuestas.

**¿Por qué separar retrieval de generación en el pipeline RAG?**
La versión original tenía el retriever dentro del LCEL chain, lo que causaba que `run()` lo invocara dos veces (una explícita para obtener fuentes, otra implícita dentro del chain). El pipeline actual separa las dos etapas: una sola llamada al retriever, luego generación con el contexto ya formateado. Esto reduce llamadas a la API de embeddings a la mitad.

**¿Por qué LCEL en lugar de AgentExecutor?**
El flujo RAG es determinístico (retrieve → generate), no iterativo. LCEL es más predecible, más barato en tokens y más fácil de trazar en Langfuse que un AgentExecutor con herramientas.

**¿Por qué JSON mode para el Orchestrator?**
Garantiza que la clasificación siempre retorne un JSON parseable con `domain`, `confidence` y `reasoning`, sin necesidad de parsear texto libre propenso a errores.

**¿Por qué persistencia FAISS en disco?**
Los índices se construyen haciendo embedding de todos los documentos via la API de OpenAI. Sin persistencia, este proceso se repite en cada startup (costo + latencia). Con `FAISS.save_local()` / `load_local()`, el primer startup paga el costo y los siguientes son instantáneos.

**¿Por qué gpt-4o-mini para todos los agentes?**
Equilibrio costo/calidad adecuado para este caso. En producción, el Orchestrator podría usar un modelo más pequeño (clasificación es una tarea simple) y los agentes de Legal/Finance un modelo más potente para mayor precisión en respuestas complejas.

**¿Por qué el evaluador recibe el contexto recuperado?**
Sin los documentos recuperados en el prompt del evaluador, la dimensión `accuracy` solo puede juzgar si la respuesta "suena" correcta, no si está fundamentada en los datos reales. Pasar el contexto hace que la evaluación sea genuinamente verificable.

**¿Por qué logging estructurado en lugar de print()?**
Con `logging` se puede subir el nivel a `WARNING` en producción para silenciar el ruido operativo, o bajar a `DEBUG` con `--debug` para diagnóstico. Con `print()` es todo o nada.

## CI/CD

El pipeline de GitHub Actions (`.github/workflows/ci.yaml`) corre en cada push a `main` y en pull requests:

| Step | Herramienta | Descripción |
|---|---|---|
| Lint | `ruff check .` | Verifica errores de estilo e imports |
| Format | `ruff format --check .` | Verifica formato consistente |
| Tests | `pytest --cov=src --cov-fail-under=80` | Tests con cobertura mínima del 80% |

## Limitaciones conocidas

- El sistema está optimizado para consultas en español. Consultas en otros idiomas pueden degradar la calidad del routing.
- La evaluación agrega ~2-3 segundos y costo adicional de tokens por consulta. Usar `--no-eval` para pruebas rápidas o cuando el volumen de consultas es alto.
- Python 3.14+ no es compatible con `faiss-cpu`. Usar Python 3.12 (especificado en `.python-version`).
- Para volúmenes de producción altos, considerar migrar el vector store a Pinecone o pgvector en lugar de FAISS local.
- `MemorySaver` almacena estado en memoria del proceso. Al reiniciar el servidor, el historial de conversaciones se pierde.

## Próximos pasos

### Alta prioridad

- **Sesiones de usuario con memoria real**: implementar una capa de `session_id` estable que se pase como `user_id` para que `MemorySaver` persista el contexto entre múltiples mensajes del mismo usuario. Para producción, migrar a `PostgresSaver`.
- **Manejo de errores en golden_evaluator.py**: envolver cada caso en `try/except` para que errores puntuales (timeout de API, red) no interrumpan toda la evaluación y se guarden resultados parciales.
- **Timeout en `graph.invoke()`**: agregar límite de tiempo para evitar esperas indefinidas ante fallos de la API de OpenAI.

### Media prioridad

- **Validación de input**: rechazar queries vacíos o excesivamente largos antes de invocar el grafo.
- **Retry con backoff exponencial**: manejar errores 429 (rate limit) de OpenAI sin crashear.
- **Tests de integración**: agregar tests con `@pytest.mark.integration` que validen el flujo end-to-end con datos reales (marcados para no correr en CI).
- **Aislación de tests del grafo**: usar un `thread_id` único por test en lugar del `"test-thread"` compartido para evitar dependencias entre tests.

### Baja prioridad

- **Modelo diferenciado por dominio**: usar `gpt-4o` para agentes de Legal y Finance donde la precisión es más crítica.
- **Caché de queries frecuentes**: reducir costo y latencia para consultas duplicadas o muy similares.
- **Métricas de routing en Langfuse**: registrar accuracy de routing y latencia por dominio para detectar degradación sistemática.
- **Edge cases en golden dataset**: agregar casos ambiguos, fuera de dominio y queries que crucen múltiples dominios.
