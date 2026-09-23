"""Cost attribution for the AI Delivery Portal's golden paths.

Every step that spends money records one append-only ledger entry (see
adapters/ai_platform/cost_adapter.py), attributed to the artifact and lifecycle
stage it belongs to. The Cost Insights page then aggregates by stage/dimension
instead of by K8s topology.

`pricing` holds the rate tables used to turn a real quantity (GPU-hours, tokens,
chunks) into USD; `events` is the one helper every router calls.
"""
