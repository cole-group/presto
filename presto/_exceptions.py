"""Exceptions for presto."""

from dataclasses import dataclass


class InvalidSettingsError(ValueError):
    """Exception raised for invalid settings."""


class MoleculeParameterisationError(RuntimeError):
    """Exception raised when one or more molecules cannot be parameterised."""


@dataclass(frozen=True)
class MoleculeIssue:
    """A user-facing issue associated with one molecule and processing phase."""

    index: int
    description: str
    phase: str
    error: str


def format_molecule_issues(
    heading: str, issues: list[MoleculeIssue], n_molecules: int
) -> str:
    """Format molecule issues in deterministic input and phase order."""
    affected = len({issue.index for issue in issues})
    lines = [f"{heading} for {affected} of {n_molecules} molecules:"]
    current_index: int | None = None
    for issue in sorted(issues, key=lambda item: item.index):
        if issue.index != current_index:
            lines.append(f"  - {issue.description}")
            current_index = issue.index
        lines.extend(_indent_error(f"{issue.phase}: {issue.error}"))
    return "\n".join(lines)


def _indent_error(error: str) -> list[str]:
    """Indent an error under its molecule, keeping multi-line messages aligned."""
    return [f"      {line}" for line in error.splitlines()]
