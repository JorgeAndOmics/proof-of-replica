"""Exception hierarchy for proof_of_replica."""

from dataclasses import dataclass


class ProofOfReplicaError(Exception):
    """Base exception for proof_of_replica."""


class ProfileError(ProofOfReplicaError):
    """Raised when a profile is invalid or cannot be loaded."""


@dataclass(frozen=True, slots=True)
class ProfileIssue:
    """A single validation issue found in a profile."""

    location: str
    problem: str
    suggestion: str
    severity: str = "error"


class ProfileValidationError(ProfileError):
    """Raised when profile JSON fails schema validation.

    Carries structured issues with location, problem, and suggestion
    for each validation failure.
    """

    issues: list[ProfileIssue]

    def __init__(self, message: str, issues: list[ProfileIssue] | None = None) -> None:
        super().__init__(message)
        self.issues = issues or []


class GenerationError(ProofOfReplicaError):
    """Raised when replica generation fails."""


class ValidationError(ProofOfReplicaError):
    """Raised when replica validation fails."""


class FileIOError(ProofOfReplicaError):
    """Raised when file I/O fails."""


class ProfilingError(ProofOfReplicaError):
    """Raised when profiling fails."""
