"""services/orchestration-api/training_presets.py"""

from training_presets import TRAINING_PRESETS, render_training_presets


def test_render_includes_every_use_case_and_its_algorithm() -> None:
    rendered = render_training_presets()
    for use_case, preset in TRAINING_PRESETS.items():
        assert use_case in rendered
        assert preset["algorithm"] in rendered


def test_render_omits_null_fields() -> None:
    # customer-segmentation has targetColumn=None and timeColumn=None —
    # neither should leak into the rendered line as the literal "None".
    rendered = render_training_presets()
    line = next(
        line for line in rendered.splitlines() if line.startswith("- customer-segmentation")
    )
    assert "None" not in line
    assert "targetColumn" not in line
    assert "timeColumn" not in line


def test_render_includes_id_columns_when_present() -> None:
    rendered = render_training_presets()
    line = next(
        line for line in rendered.splitlines() if line.startswith("- telco-fraud-detection")
    )
    assert "idColumns=['transaction_id']" in line
