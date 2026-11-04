"""Metadata templates never evaluate expressions or recursively render values."""

import pytest

from yanex.core.metadata_templates import (
    has_placeholders,
    render_metadata,
    render_template,
)
from yanex.utils.exceptions import ValidationError


def test_formats_and_nested_lookup():
    assert (
        render_template(
            "lr-{lr:.2f}-seed-{seed:03d}", {"model": {"lr": 0.125}, "seed": 4}
        )
        == "lr-0.12-seed-004"
    )
    assert render_template("{model.lr:.3g}", {"model": {"lr": 0.001234}}) == "0.00123"


def test_literal_braces_and_single_pass():
    assert (
        render_template("{{value}}-{value}", {"value": "{missing}"})
        == "{value}-{missing}"
    )
    assert not has_placeholders("{{value}}")
    assert has_placeholders("{value}")


@pytest.mark.parametrize(
    "template",
    [
        "{missing}",
        "{",
        "{}",
        "{x:.2f}",
        "{x!r}",
        "{x.__class__}",
        "{x[0]}",
        "{x:{width}}",
        "{len(x)}",
    ],
)
def test_invalid_templates(template):
    with pytest.raises(ValidationError):
        render_template(template, {"x": "text", "width": 2})


def test_ambiguous_local_suffix():
    with pytest.raises(ValidationError, match="matches multiple keys"):
        render_template("{lr}", {"model": {"lr": 1}, "optimizer": {"lr": 1}})


def test_metadata_validation_after_rendering():
    assert render_metadata("lr-{lr}", ["lr={lr}"], "[value]\n{lr}", {"lr": 0.01}) == (
        "lr-0.01",
        ["lr=0.01"],
        "[value]\n0.01",
    )
    for value in ["has space", "has\tspace", "glob*", "[x]"]:
        with pytest.raises(ValidationError):
            render_metadata(None, ["{value}"], None, {"value": value})
