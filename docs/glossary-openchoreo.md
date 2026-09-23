# Glossary

Short, direct definitions — for when someone asks "what is a X" instead of
"how do I do X" (see `troubleshooting-faq.md` for the latter, and
`architecture-overview.md` for how these pieces fit together).

## OpenChoreo platform concepts

**Namespace** — the top-level scoping boundary in OpenChoreo; owns
Environments, DeploymentPipelines, and Projects.

**Project** — a grouping of related Components within a Namespace — the
OpenChoreo equivalent of "one application/system's worth of services."

**Component** — one deployable unit inside a Project (a service, a
scheduled task, a model-serving endpoint, ...). What a Component actually
*is* — how it builds, what CRDs it renders — is defined by its
**ComponentType** (or **ClusterComponentType** for one shared across every
Namespace).

**Workload** — the container/runtime spec for a Component (image, ports,
env vars, file mounts). Created alongside a Component when it has runtime
details to render (as opposed to, say, an external-CI-only Component).

**Trait** — a reusable, attachable cross-cutting capability layered onto a
Component (e.g. traffic routing, observability, prediction logging) without
that capability being baked into the ComponentType itself.

**Workflow** (or **ClusterWorkflow**) — a build/automation pipeline
definition a Component can reference (e.g. "build from source"). Triggering
one produces a **WorkflowRun**.

**ResourceType** / **Resource** — a template, and a concrete instance, of a
managed-infrastructure dependency a Component can consume (a database, a
queue, an object store, ...) without owning its lifecycle directly.

**Environment** — one deployment target (e.g. development/staging/
production) within a Namespace.

**DeploymentPipeline** — the ordered sequence of Environments a release is
allowed to promote through (e.g. development → staging → production), and
what "promote" and "rollback" actually operate over.

**Release** / **ReleaseBinding** — a Release is a concrete, versioned
rendering of a Component's resources; a ReleaseBinding is that Release
bound into one specific Environment. Promoting a change means creating a
new ReleaseBinding in the next Environment; rolling back means swapping a
ReleaseBinding back to what it pointed to before.

## MLOps/LLMOps platform concepts (this repo, not OpenChoreo's own vocabulary)

**Golden Path** — a Backstage Scaffolder template that packages a whole
multi-step workflow into one guided form, so following best practice is the
easiest option, not a separately-remembered checklist.

**Evaluate Gate** — the pass/fail check that decides whether a model
version or a generated response is fit to activate/deploy. For classical ML
it compares logged metrics against per-task-type thresholds
(`evaluations/evaluate_gate.py`'s `evaluate_metrics_gate`); for LLM output
it's an LLM-as-judge scoring safety/correctness/relevance (and
faithfulness, when the answer was generated from retrieved context) against
fixed thresholds (`evaluate_gate`).

**Model Registry** — MLflow's registry of trained model versions plus their
logged metrics and dataset lineage.

**Prompt Registry** — the versioned-and-aliased store of system prompt
text per persona (MLflow's native Prompt Registry). An "active" alias
(scoped per environment) marks which version the chat assistant currently
uses.

**RAG index** — a named, versioned Qdrant collection of embedded document
chunks, activated the same way a prompt is, so the chat assistant retrieves
from a known-evaluated version rather than whatever was ingested most
recently.

**Eval set** — a named, versioned, reusable list of eval questions, so two
versions of the same prompt/RAG index get graded against the same benchmark
instead of whatever questions happened to be typed into that run's form.

**DORA metrics** — the four standard delivery-performance metrics this
Portal tracks per change (deployment frequency, lead time, change failure
rate, MTTR), extended here with ML/LLM-specific dimensions (drift-triggered
recoveries, eval coverage, semantic vs. infra failure classification).
