# AI Delivery Portal — Architecture Overview

AI Delivery Portal is an Internal Developer Platform (IDP) for the MLOps and
LLMOps lifecycle, built on Backstage and running on top of OpenChoreo. This
page is the reference an AI assistant (or a person) should read to answer
"how is this platform put together" and "what runs where" questions.

## The three planes

The platform runs on a 3-node Kubernetes cluster, split by responsibility:

- **Control plane (master)** — managed by OpenChoreo itself: the OpenChoreo
  API (REST + RBAC), its controllers (which reconcile Component/Workload/
  Release CRDs into real Kubernetes resources), and Thunder (the OAuth2/OIDC
  identity provider).
- **Data plane / Portal plane (worker1)** — runs `orchestration-api` (the
  FastAPI backend that owns all MLOps/LLMOps business logic), the Adapter
  layer, and the 4 MCP servers, all deployed as OpenChoreo Components
  themselves.
- **Workflow plane + AI Platform zone (worker2)** — Argo Workflows (training
  and fine-tuning jobs), plus MLflow, Qdrant, MinIO, LiteLLM, and Feast —
  the AI Platform's own infrastructure, deployed as plain Kubernetes
  manifests outside OpenChoreo.

## Golden Paths

A Golden Path is a Backstage Scaffolder template that packages a whole
workflow (draft → validate → run → register) into one guided form, so a
Dev doesn't need to remember every manual step or which raw tool to call.

**MLOps track:**

1. **Train → Track → Register** (`train-track-register`) — trains or
   fine-tunes a model, logs the run to MLflow Tracking, and registers the
   resulting version in the Model Registry.
2. **Evaluate → Deploy** (`evaluate-deploy-model`) — before any deploy,
   `orchestration-api` runs the Evaluate Gate (`evaluations/evaluate_gate.py`)
   against the model's logged metrics; only a version that passes ever
   reaches the inference adapter. Supports deploy, rollback, promote (across
   environments), and promote-rollback.
3. **Set Up Model Monitoring** (`setup-model-monitoring`) — registers a
   periodic drift check against a reference dataset.

**LLMOps track** — the same "don't ship an untested change" discipline,
applied to system prompts and RAG indexes instead of trained models:

1. **Draft Prompt / Ingest RAG Data / Draft Eval Set** (`llm-draft-ingest`)
   — registers a new (inactive) version of a prompt, a RAG index, or a
   named reusable set of eval questions.
2. **Evaluate & Activate Prompt / RAG Version** (`llm-evaluate-activate`) —
   runs an LLM-as-judge Evaluate Gate against the drafted version and
   activates it only if it passes, or rolls back an already-registered
   version instantly with no re-evaluation. Activation is tracked
   independently per environment (development/staging/production).

**Serving LLM** (`llm-serve-deploy`) is a separate golden path for
self-hosted LLM serving (vLLM via KServe) — it does not currently run
through OpenChoreo, since no OpenChoreo ClusterComponentType exists yet for
GPU/vLLM workloads.

## The Adapter Pattern

Every integration with an external system (MLflow, Qdrant, KServe, Argo,
OpenChoreo itself) goes through an interface in `adapters/*/interfaces.py`.
Business logic (`orchestration-api`, the Scaffolder templates) only ever
calls the interface, never the concrete backend directly. Swapping a mock
implementation for a real one, or migrating a backend to OpenChoreo, means
adding one new class — never touching a caller. This is why the standard
model-serving path and the training/workflow path could migrate fully to
OpenChoreo (see `adapters/factory.py`'s `get_workflow_adapter()` and
`get_inference_adapter()`) without any Scaffolder template needing to change.

## The chat assistant

The Portal has a built-in chat assistant (`mcp-chat`), reachable through
`routers/chat.py`. It answers using:

- A **persona's active system prompt** (currently one persona, "mlops" —
  the MLOps Assistant), versioned through the same Prompt Registry the
  LLMOps Golden Path drafts and activates.
- Optionally, **RAG** over a named Qdrant collection (`use_rag=true`,
  `rag_collection=<name>`) — this is exactly how this document, once
  ingested, becomes something the assistant can answer questions from.
- Optionally, **MCP tool-calling** (`use_tools=true`), scoped per persona
  by `persona_tool_scope.py` (deny-by-default — a persona not explicitly
  listed there gets no tools at all). Any tool that mutates state
  (`activate_prompt`, `rag_activate`, `trigger_training`, ...) is only ever
  *proposed*, never auto-executed — the caller must resubmit it as
  `confirmed_tool_call` after a human explicitly approves it.

A Kubernetes-ops persona used to exist ("k8s") for read-only pod/log/event
lookups; it was removed once OpenChoreo shipped its own built-in MCP server
covering the same ground, so the Portal no longer duplicates that coverage.
