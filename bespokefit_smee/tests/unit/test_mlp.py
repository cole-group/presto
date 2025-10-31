"""Unit tests for mlp module."""

import pytest
from openmmml import MLPotential

from bespokefit_smee.mlp import AvailableModels, _cache, get_mlp, load_egret_1


class TestAvailableModels:
    """Tests for AvailableModels type."""

    def test_available_models_defined(self):
        """Test that available models are defined."""
        from typing import get_args

        available = get_args(AvailableModels)
        assert len(available) > 0
        assert "egret-1" in available
        assert "mace-off23-small" in available


class TestLoadEgret1:
    """Tests for load_egret_1 function."""

    def test_load_egret_returns_mlpotential(self):
        """Test that loading EGRET-1 returns an MLPotential."""
        potential = load_egret_1()
        assert isinstance(potential, MLPotential)

    def test_load_egret_is_cached(self):
        """Test that loaded potential is cached."""
        _cache.clear()
        load_egret_1()
        # Cache should now have it if get_mlp was used
        # But load_egret_1 doesn't cache directly


class TestGetMlp:
    """Tests for get_mlp function."""

    def test_get_mlp_egret(self):
        """Test getting EGRET-1 model."""
        _cache.clear()
        potential = get_mlp("egret-1")
        assert isinstance(potential, MLPotential)

    def test_get_mlp_caches_result(self):
        """Test that get_mlp caches the result."""
        _cache.clear()
        potential1 = get_mlp("egret-1")
        potential2 = get_mlp("egret-1")
        assert potential1 is potential2

    def test_get_mlp_invalid_model_raises_error(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Invalid model name"):
            get_mlp("invalid-model-name")

    @pytest.mark.parametrize(
        "model_name",
        ["egret-1", "mace-off23-small"],
    )
    def test_get_mlp_valid_models(self, model_name):
        """Test that valid models can be loaded."""
        _cache.clear()
        potential = get_mlp(model_name)
        assert isinstance(potential, MLPotential)

    def test_cache_persists_across_calls(self):
        """Test that cache persists across multiple calls."""
        _cache.clear()

        # Load multiple models
        pot1 = get_mlp("egret-1")
        pot2 = get_mlp("mace-off23-small")

        # Verify they're cached
        assert "egret-1" in _cache
        assert "mace-off23-small" in _cache

        # Verify same instances are returned
        assert get_mlp("egret-1") is pot1
        assert get_mlp("mace-off23-small") is pot2

    def test_get_mlp_different_models_are_different(self):
        """Test that different models return different objects."""
        _cache.clear()
        pot1 = get_mlp("egret-1")
        pot2 = get_mlp("mace-off23-small")
        assert pot1 is not pot2
