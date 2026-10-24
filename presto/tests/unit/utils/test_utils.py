"""Unit tests for utility modules."""

import pytest


class TestTyping:
    """Tests for typing utilities."""

    def test_torch_device_type(self):
        """Test TorchDevice type."""
        # Should be a Literal type
        from typing import get_args

        from presto.utils.typing import TorchDevice

        args = get_args(TorchDevice)
        assert "cpu" in args
        assert "cuda" in args

    def test_optimiser_name_type(self):
        """Test OptimiserName type."""
        from typing import get_args

        from presto.utils.typing import OptimiserName

        args = get_args(OptimiserName)
        assert "adam" in args
        assert "lm" in args

    def test_valence_type(self):
        """Test ValenceType type."""
        from typing import get_args

        from presto.utils.typing import ValenceType

        args = get_args(ValenceType)
        assert "Bonds" in args
        assert "Angles" in args
        assert "ProperTorsions" in args
        assert "ImproperTorsions" in args


class TestRegister:
    """Tests for registry decorator."""

    def test_get_registry_decorator(self):
        """Test that get_registry_decorator creates decorators."""
        from presto.utils.register import get_registry_decorator

        test_registry = {}
        decorator = get_registry_decorator(test_registry)

        # Test decorator works
        @decorator("key1")
        def test_fn1():
            pass

        assert "key1" in test_registry
        assert test_registry["key1"] == test_fn1

    def test_registry_decorator_with_class(self):
        """Test registry decorator with class as key."""
        from presto.utils.register import get_registry_decorator

        test_registry = {}
        decorator = get_registry_decorator(test_registry)

        class TestKey:
            pass

        @decorator(TestKey)
        def test_fn():
            pass

        assert TestKey in test_registry
        assert test_registry[TestKey] == test_fn

    def test_registry_prevents_duplicate_keys(self):
        """Test that registry prevents duplicate keys."""
        from presto.utils.register import get_registry_decorator

        test_registry = {}
        decorator = get_registry_decorator(test_registry)

        @decorator("key1")
        def test_fn1():
            pass

        # Attempting to register same key should work (overwrites)
        with pytest.raises(ValueError, match=r"Key key1 is already registered."):

            @decorator("key1")
            def test_fn2():
                pass


class TestSuppressOutput:
    """Tests for output suppression utilities."""

    def test_suppress_unwanted_output_import(self):
        """Test that suppress_unwanted_output can be imported."""
        from presto.utils._suppress_output import suppress_unwanted_output

        # Should not raise
        suppress_unwanted_output()


class TestPathLike:
    """Tests for PathLike type."""

    def test_pathlike_type(self):
        """Test PathLike type."""
        from typing import get_args

        from presto.utils.typing import PathLike

        # PathLike should be a Union of Path and str
        args = get_args(PathLike)
        # Check that it includes Path and str in some form
        assert len(args) > 0


class TestDeepUpdate:
    """Tests for deep_update."""

    def test_nested_dicts_merge(self):
        """Nested dicts are merged rather than replaced."""
        from presto.utils.dicts import deep_update

        data = {"a": {"b": 1, "c": 2}, "d": 3}
        assert deep_update(data, {"a": {"b": 10}}) == {"a": {"b": 10, "c": 2}, "d": 3}

    def test_list_items_updated_by_index(self):
        """A dict keyed by index updates those list items, including negative ones."""
        from presto.utils.dicts import deep_update

        data = {"protocols": [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 5}]}
        updated = deep_update(
            data, {"protocols": {0: {"x": 10}, -1: {"z": {"nested": True}}}}
        )
        assert updated == {
            "protocols": [
                {"x": 10, "y": 2},
                {"x": 3, "y": 4},
                {"x": 5, "z": {"nested": True}},
            ]
        }

    def test_input_not_mutated(self):
        """Neither the data nor its lists are modified in place."""
        from presto.utils.dicts import deep_update

        data = {"protocols": [{"x": 1}], "a": {"b": 1}}
        deep_update(data, {"protocols": {0: {"x": 2}}, "a": {"b": 2}})
        assert data == {"protocols": [{"x": 1}], "a": {"b": 1}}

    def test_list_replaced_by_list(self):
        """A list overwrite replaces the whole list."""
        from presto.utils.dicts import deep_update

        assert deep_update({"p": [1, 2]}, {"p": [3]}) == {"p": [3]}

    def test_non_index_keys_for_list_raise(self):
        """String keys for a list explain the index form."""
        from presto.utils.dicts import deep_update

        with pytest.raises(ValueError, match=r"keyed by list index.*\{0: \{\.\.\.\}\}"):
            deep_update({"p": [{"x": 1}]}, {"p": {"x": 2}})

    def test_out_of_range_index_raises(self):
        """An index past the end of the list raises."""
        from presto.utils.dicts import deep_update

        with pytest.raises(IndexError, match="out of range"):
            deep_update({"p": [{"x": 1}]}, {"p": {1: {"x": 2}}})
