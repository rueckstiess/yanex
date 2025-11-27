"""
Tests for dependency parsing and slot expansion functions.
"""

import pytest
from click import ClickException

from yanex.cli.commands.run import (
    _has_dependency_sweep,
    _parse_dependencies,
    expand_dependency_slots,
)


class TestParseDependencies:
    """Test _parse_dependencies() function."""

    def test_single_dependency_auto_slot(self):
        """Test single dependency with auto-generated slot name."""
        result = _parse_dependencies(("exp1",))
        assert result == [("dep1", ["exp1"])]

    def test_single_dependency_named_slot(self):
        """Test single dependency with explicit slot name."""
        result = _parse_dependencies(("train=exp1",))
        assert result == [("train", ["exp1"])]

    def test_comma_separated_sweep_auto_slot(self):
        """Test comma-separated IDs create sweep with auto slot."""
        result = _parse_dependencies(("exp1,exp2",))
        assert result == [("dep1", ["exp1", "exp2"])]

    def test_comma_separated_sweep_named_slot(self):
        """Test comma-separated IDs create sweep with named slot."""
        result = _parse_dependencies(("train=exp1,exp2",))
        assert result == [("train", ["exp1", "exp2"])]

    def test_multiple_flags_auto_slots(self):
        """Test multiple -D flags get sequential auto slot names."""
        result = _parse_dependencies(("exp1", "exp2"))
        assert result == [("dep1", ["exp1"]), ("dep2", ["exp2"])]

    def test_multiple_flags_named_slots(self):
        """Test multiple -D flags with explicit slot names."""
        result = _parse_dependencies(("data=exp1", "model=exp2"))
        assert result == [("data", ["exp1"]), ("model", ["exp2"])]

    def test_mixed_named_and_auto_slots(self):
        """Test mixing named and auto-generated slot names."""
        result = _parse_dependencies(("exp1", "model=exp2", "exp3"))
        assert result == [("dep1", ["exp1"]), ("model", ["exp2"]), ("dep3", ["exp3"])]

    def test_sweep_with_multiple_slots(self):
        """Test sweep in one slot, single in another."""
        result = _parse_dependencies(("data=exp1,exp2", "model=exp3"))
        assert result == [("data", ["exp1", "exp2"]), ("model", ["exp3"])]

    def test_empty_input(self):
        """Test empty tuple returns empty list."""
        result = _parse_dependencies(())
        assert result == []

    def test_whitespace_handling(self):
        """Test whitespace is stripped from IDs and slot names."""
        result = _parse_dependencies((" exp1 , exp2 ",))
        assert result == [("dep1", ["exp1", "exp2"])]

        result = _parse_dependencies((" train = exp1 ",))
        assert result == [("train", ["exp1"])]

    def test_empty_ids_filtered(self):
        """Test empty IDs from trailing commas are filtered."""
        result = _parse_dependencies(("exp1,,exp2,",))
        assert result == [("dep1", ["exp1", "exp2"])]

    def test_only_empty_ids_skipped(self):
        """Test -D flag with only empty IDs is skipped."""
        result = _parse_dependencies((",,,",))
        assert result == []

    def test_three_way_sweep(self):
        """Test sweep with three dependency IDs."""
        result = _parse_dependencies(("model=exp1,exp2,exp3",))
        assert result == [("model", ["exp1", "exp2", "exp3"])]

    def test_cross_product_setup(self):
        """Test setup for cross-product sweep."""
        result = _parse_dependencies(("data=d1,d2", "model=m1,m2"))
        assert result == [("data", ["d1", "d2"]), ("model", ["m1", "m2"])]

    def test_duplicate_named_slot_raises_error(self):
        """Test that duplicate named slot names raise an error."""
        with pytest.raises(ClickException) as exc_info:
            _parse_dependencies(("model=abc123", "model=def456"))

        error_msg = str(exc_info.value.message)
        assert "Duplicate dependency slot name 'model'" in error_msg
        assert "flag 1 and 2" in error_msg

    def test_duplicate_named_slot_three_flags(self):
        """Test duplicate detection with three flags."""
        with pytest.raises(ClickException) as exc_info:
            _parse_dependencies(("data=abc", "model=def", "model=ghi"))

        error_msg = str(exc_info.value.message)
        assert "Duplicate dependency slot name 'model'" in error_msg
        assert "flag 2 and 3" in error_msg

    def test_auto_slots_no_duplicate_error(self):
        """Test that auto-generated slot names don't clash."""
        # -D abc -D efg should work fine with dep1 and dep2
        result = _parse_dependencies(("abc", "efg"))
        assert result == [("dep1", ["abc"]), ("dep2", ["efg"])]

    def test_mixed_auto_and_named_no_clash(self):
        """Test auto and named slots don't clash when names differ."""
        result = _parse_dependencies(("abc", "model=def", "ghi"))
        assert result == [("dep1", ["abc"]), ("model", ["def"]), ("dep3", ["ghi"])]

    def test_duplicate_error_suggests_alternatives(self):
        """Test error message suggests valid alternatives."""
        with pytest.raises(ClickException) as exc_info:
            _parse_dependencies(("train=abc", "train=def"))

        error_msg = str(exc_info.value.message)
        assert "train1" in error_msg  # Suggests numbered slots
        assert "train=id1,id2" in error_msg  # Suggests sweep syntax


class TestExpandDependencySlots:
    """Test expand_dependency_slots() function."""

    def test_empty_slots_returns_empty_dict(self):
        """Test empty input returns single empty dict."""
        result = expand_dependency_slots([])
        assert result == [{}]

    def test_single_slot_single_id(self):
        """Test single slot with single ID."""
        result = expand_dependency_slots([("dep1", ["exp1"])])
        assert result == [{"dep1": "exp1"}]

    def test_single_slot_multiple_ids(self):
        """Test single slot with multiple IDs (sweep)."""
        result = expand_dependency_slots([("model", ["exp1", "exp2"])])
        assert result == [{"model": "exp1"}, {"model": "exp2"}]

    def test_multiple_slots_single_ids(self):
        """Test multiple slots each with single ID."""
        result = expand_dependency_slots([("data", ["d1"]), ("model", ["m1"])])
        assert result == [{"data": "d1", "model": "m1"}]

    def test_cross_product_two_by_two(self):
        """Test 2x2 cross-product expansion."""
        result = expand_dependency_slots(
            [("data", ["d1", "d2"]), ("model", ["m1", "m2"])]
        )
        assert len(result) == 4
        assert {"data": "d1", "model": "m1"} in result
        assert {"data": "d1", "model": "m2"} in result
        assert {"data": "d2", "model": "m1"} in result
        assert {"data": "d2", "model": "m2"} in result

    def test_cross_product_asymmetric(self):
        """Test asymmetric cross-product (2x1)."""
        result = expand_dependency_slots([("data", ["d1", "d2"]), ("model", ["m1"])])
        assert len(result) == 2
        assert {"data": "d1", "model": "m1"} in result
        assert {"data": "d2", "model": "m1"} in result

    def test_cross_product_three_slots(self):
        """Test three-way cross-product."""
        result = expand_dependency_slots(
            [
                ("data", ["d1"]),
                ("model", ["m1", "m2"]),
                ("config", ["c1", "c2"]),
            ]
        )
        assert len(result) == 4  # 1 * 2 * 2 = 4
        assert {"data": "d1", "model": "m1", "config": "c1"} in result
        assert {"data": "d1", "model": "m1", "config": "c2"} in result
        assert {"data": "d1", "model": "m2", "config": "c1"} in result
        assert {"data": "d1", "model": "m2", "config": "c2"} in result

    def test_preserves_slot_names(self):
        """Test that slot names are preserved in output."""
        result = expand_dependency_slots(
            [("my_data", ["exp1"]), ("my_model", ["exp2"])]
        )
        assert result == [{"my_data": "exp1", "my_model": "exp2"}]

    def test_order_preserved(self):
        """Test that order within each slot's sweep is preserved."""
        result = expand_dependency_slots([("model", ["a", "b", "c"])])
        assert result == [{"model": "a"}, {"model": "b"}, {"model": "c"}]


class TestHasDependencySweep:
    """Test _has_dependency_sweep() function."""

    def test_no_slots_no_sweep(self):
        """Test empty list is not a sweep."""
        assert _has_dependency_sweep([]) is False

    def test_single_id_no_sweep(self):
        """Test single ID per slot is not a sweep."""
        assert _has_dependency_sweep([("dep1", ["exp1"])]) is False
        assert _has_dependency_sweep([("data", ["d1"]), ("model", ["m1"])]) is False

    def test_multiple_ids_is_sweep(self):
        """Test multiple IDs in any slot triggers sweep."""
        assert _has_dependency_sweep([("dep1", ["exp1", "exp2"])]) is True

    def test_sweep_in_one_slot(self):
        """Test sweep detected when only one slot has multiple IDs."""
        assert (
            _has_dependency_sweep([("data", ["d1"]), ("model", ["m1", "m2"])]) is True
        )

    def test_sweep_in_all_slots(self):
        """Test sweep detected when all slots have multiple IDs."""
        assert (
            _has_dependency_sweep([("data", ["d1", "d2"]), ("model", ["m1", "m2"])])
            is True
        )


class TestIntegration:
    """Integration tests for parsing + expansion pipeline."""

    def test_single_dependency_flow(self):
        """Test single dependency from CLI to expansion."""
        parsed = _parse_dependencies(("exp1",))
        expanded = expand_dependency_slots(parsed)
        assert expanded == [{"dep1": "exp1"}]

    def test_named_dependency_flow(self):
        """Test named dependency from CLI to expansion."""
        parsed = _parse_dependencies(("data=abc12345",))
        expanded = expand_dependency_slots(parsed)
        assert expanded == [{"data": "abc12345"}]

    def test_sweep_flow(self):
        """Test sweep from CLI to expansion."""
        parsed = _parse_dependencies(("model=exp1,exp2,exp3",))
        expanded = expand_dependency_slots(parsed)
        assert len(expanded) == 3
        assert expanded[0] == {"model": "exp1"}
        assert expanded[1] == {"model": "exp2"}
        assert expanded[2] == {"model": "exp3"}

    def test_cross_product_flow(self):
        """Test cross-product from CLI to expansion."""
        parsed = _parse_dependencies(("data=d1,d2", "model=m1,m2"))
        expanded = expand_dependency_slots(parsed)
        assert len(expanded) == 4

        # Verify all combinations exist
        combinations = {(d["data"], d["model"]) for d in expanded}
        assert combinations == {("d1", "m1"), ("d1", "m2"), ("d2", "m1"), ("d2", "m2")}

    def test_has_sweep_detection(self):
        """Test sweep detection after parsing."""
        # No sweep
        parsed = _parse_dependencies(("data=d1", "model=m1"))
        assert _has_dependency_sweep(parsed) is False

        # Has sweep
        parsed = _parse_dependencies(("data=d1,d2", "model=m1"))
        assert _has_dependency_sweep(parsed) is True
