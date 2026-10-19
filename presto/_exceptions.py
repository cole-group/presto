"""Exceptions for presto."""


class InvalidSettingsError(ValueError):
    """Exception raised for invalid settings."""


class MissingOptionalDependencyError(ImportError):
    """Exception raised when an optional dependency required by the settings is missing."""


class MoleculeParameterisationError(RuntimeError):
    """Exception raised when one or more molecules cannot be parameterised."""
