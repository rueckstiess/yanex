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


def test_cache_shares_reads_but_not_combination_scope(tmp_path, monkeypatch):
    from collections import Counter

    from yanex.core.manager import ExperimentManager
    from yanex.core.metadata_templates import UpstreamParameterCache

    manager = ExperimentManager(tmp_path / "experiments")
    script = tmp_path / "run.py"
    script.write_text("pass")

    def create(config, deps=None):
        exp_id = manager.create_experiment(script, config=config, dependencies=deps)
        manager.complete_experiment(exp_id)
        return exp_id

    root = create({"shared": 7})
    first = create({"lr": 1}, {"data": root})
    second = create({"lr": 2}, {"data": root})
    config_reads, dependency_reads = Counter(), Counter()
    original_config = manager.storage.load_config
    original_dependencies = manager.storage.dependency_storage.load_dependencies

    def load_config(exp_id, **kwargs):
        config_reads[exp_id] += 1
        return original_config(exp_id, **kwargs)

    def load_dependencies(exp_id, **kwargs):
        assert kwargs["strict"] is True
        dependency_reads[exp_id] += 1
        return original_dependencies(exp_id, **kwargs)

    monkeypatch.setattr(manager.storage, "load_config", load_config)
    monkeypatch.setattr(
        manager.storage.dependency_storage, "load_dependencies", load_dependencies
    )
    cache = UpstreamParameterCache(manager)
    for _ in range(3):
        for exp_id, expected in [(first, 1), (second, 2)]:
            lookup = cache.lookup({"model": exp_id})
            assert lookup("lr") == expected
            assert lookup("shared") == 7
    assert config_reads == dependency_reads == Counter({root: 1, first: 1, second: 1})
    manager.storage.save_config(first, {"lr": 3})
    assert cache.lookup({"model": first})("lr") == 1
    assert UpstreamParameterCache(manager).lookup({"model": first})("lr") == 3


def test_cached_traversal_detects_cycles(tmp_path):
    from yanex.core.manager import ExperimentManager
    from yanex.core.metadata_templates import UpstreamParameterCache

    manager = ExperimentManager(tmp_path / "experiments")
    script = tmp_path / "run.py"
    script.write_text("pass")
    exp_id = manager.create_experiment(script, config={"lr": 1})
    manager.storage.dependency_storage.save_dependencies(exp_id, {"self": exp_id})
    with pytest.raises(ValidationError, match="Circular upstream dependencies"):
        UpstreamParameterCache(manager).lookup({"model": exp_id})("lr")
