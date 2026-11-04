"""Shared parameter lookup across experiment configurations."""

from typing import Any

from ..utils.exceptions import KeyNotFoundError
from .access_resolver import AccessResolver, flatten_dict, parse_canonical_key


def get_agreed_parameter(
    key: str, configs: dict[str, dict[str, Any]], resolver: AccessResolver | None = None
) -> Any:
    """Resolve one path, accepting repeated values only when they agree.

    Distinct matching paths remain ambiguous even when their values agree.
    Experiment IDs are retained for conflict diagnostics.
    """
    flattened = {exp_id: flatten_dict(config) for exp_id, config in configs.items()}
    if resolver is None:
        index = {}
        for params in flattened.values():
            index.update(params)
        resolver = AccessResolver(params=index)
    canonical = resolver.resolve(key, scope="param")
    _, path = parse_canonical_key(canonical)
    found = [
        (exp_id, params[path]) for exp_id, params in flattened.items() if path in params
    ]
    if not found:
        raise KeyNotFoundError(key, scope="param")
    values = [value for _, value in found]
    if all(value == values[0] for value in values):
        return values[0]
    details = ", ".join(f"{exp_id}={value!r}" for exp_id, value in found[:5])
    raise ValueError(
        f"Parameter '{key}' found in {len(found)} experiments with different values: {details}. "
        "Use a local parameter override or select dependencies with agreeing values."
    )
