"""Mock IHuggingFaceHubAdapter — no network calls, canned metadata for a
handful of well-known model ids plus deterministic synthetic data for
anything else, so the wizard's Model Source step and GPU sizing estimator
work in CI/local dev without hitting the real Hub.
"""

from adapters.interfaces import HuggingFaceModelInfo, IHuggingFaceHubAdapter

# A few real, commonly-demoed ids with their actual published architecture —
# lets a demo/test pick a recognizable name and get realistic numbers back.
_KNOWN_MODELS: dict[str, HuggingFaceModelInfo] = {
    "meta-llama/Llama-3.1-8B-Instruct": HuggingFaceModelInfo(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        exists=True,
        is_gated=True,
        param_count_billion=8.03,
        max_context_length=131072,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        license="llama3.1",
    ),
    "mistralai/Mistral-7B-Instruct-v0.3": HuggingFaceModelInfo(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        exists=True,
        is_gated=False,
        param_count_billion=7.25,
        max_context_length=32768,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        license="apache-2.0",
    ),
    "does-not-exist/not-a-real-model": HuggingFaceModelInfo(
        model_id="does-not-exist/not-a-real-model",
        exists=False,
        is_gated=False,
        param_count_billion=None,
        max_context_length=None,
        num_layers=None,
        hidden_size=None,
        num_attention_heads=None,
        num_key_value_heads=None,
        license=None,
    ),
}


class MockHuggingFaceHubAdapter(IHuggingFaceHubAdapter):
    def get_model_info(self, model_id: str) -> HuggingFaceModelInfo:
        known = _KNOWN_MODELS.get(model_id)
        if known is not None:
            return known
        # Anything else: treat as a real, ungated 7B-class model — good enough to
        # exercise the wizard/estimator end-to-end without a real Hub lookup.
        return HuggingFaceModelInfo(
            model_id=model_id,
            exists=True,
            is_gated=False,
            param_count_billion=7.0,
            max_context_length=8192,
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=32,
            license=None,
        )
