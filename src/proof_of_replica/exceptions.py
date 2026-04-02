"""Exception hierarchy for proof_of_replica."""


class ProofOfReplicaError(Exception):
    """Base exception for proof_of_replica."""


class ProfileError(ProofOfReplicaError):
    """Raised when a profile is invalid or cannot be loaded."""


class ProfileValidationError(ProfileError):
    """Raised when profile JSON fails schema validation."""


class GenerationError(ProofOfReplicaError):
    """Raised when replica generation fails."""


class ValidationError(ProofOfReplicaError):
    """Raised when replica validation fails."""
