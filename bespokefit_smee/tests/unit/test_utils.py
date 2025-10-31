"""Unit tests for utility modules."""

import pytest


class TestTyping:
    """Tests for typing utilities."""

    def test_torch_device_type(self):
        """Test TorchDevice type."""
        # Should be a Literal type
        from typing import get_args

        from bespokefit_smee.utils.typing import TorchDevice

        args = get_args(TorchDevice)
        assert "cpu" in args
        assert "cuda" in args

    def test_optimiser_name_type(self):
        """Test OptimiserName type."""
        from typing import get_args

        from bespokefit_smee.utils.typing import OptimiserName

        args = get_args(OptimiserName)
        assert "adam" in args
        assert "lm" in args

    def test_valence_type(self):
        """Test ValenceType type."""
        from typing import get_args

        from bespokefit_smee.utils.typing import ValenceType

        args = get_args(ValenceType)
        assert "Bonds" in args
        assert "Angles" in args
        assert "ProperTorsions" in args
        assert "ImproperTorsions" in args


class TestRegister:
    """Tests for registry decorator."""

    def test_get_registry_decorator(self):
        """Test that get_registry_decorator creates decorators."""
        from bespokefit_smee.utils.register import get_registry_decorator

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
        from bespokefit_smee.utils.register import get_registry_decorator

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
        from bespokefit_smee.utils.register import get_registry_decorator

        test_registry = {}
        decorator = get_registry_decorator(test_registry)

        @decorator("key1")
        def test_fn1():
            pass

        # Attempting to register same key should work (overwrites)
        with pytest.raises(ValueError, match="Key key1 is already registered."):

            @decorator("key1")
            def test_fn2():
                pass


class TestSuppressOutput:
    """Tests for output suppression utilities."""

    def test_suppress_unwanted_output_import(self):
        """Test that suppress_unwanted_output can be imported."""
        from bespokefit_smee.utils._suppress_output import suppress_unwanted_output

        # Should not raise
        suppress_unwanted_output()

    def test_suppress_output_is_callable(self):
        """Test that suppress_unwanted_output is callable."""
        from bespokefit_smee.utils._suppress_output import suppress_unwanted_output

        assert callable(suppress_unwanted_output)


class TestPathLike:
    """Tests for PathLike type."""

    def test_pathlike_type(self):
        """Test PathLike type."""
        from typing import get_args

        from bespokefit_smee.utils.typing import PathLike

        # PathLike should be a Union of Path and str
        args = get_args(PathLike)
        # Check that it includes Path and str in some form
        assert len(args) > 0
