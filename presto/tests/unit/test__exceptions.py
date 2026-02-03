"""Unit tests for exceptions module."""

import pytest

from presto._exceptions import InvalidSettingsError


class TestInvalidSettingsError:
    """Tests for InvalidSettingsError exception."""

    def test_is_value_error_subclass(self):
        """Test that InvalidSettingsError is a ValueError subclass."""
        assert issubclass(InvalidSettingsError, ValueError)

    def test_can_be_raised(self):
        """Test that the exception can be raised."""
        with pytest.raises(InvalidSettingsError):
            raise InvalidSettingsError("Test error")

    def test_error_message(self):
        """Test that error message is preserved."""
        message = "This is a test error message"
        with pytest.raises(InvalidSettingsError, match=message):
            raise InvalidSettingsError(message)

    def test_can_be_caught_as_value_error(self):
        """Test that exception can be caught as ValueError."""
        with pytest.raises(ValueError):
            raise InvalidSettingsError("Test error")

    def test_multiple_arguments(self):
        """Test that multiple arguments work."""
        with pytest.raises(InvalidSettingsError) as exc_info:
            raise InvalidSettingsError("Error", "Additional info")

        assert "Error" in str(exc_info.value)
