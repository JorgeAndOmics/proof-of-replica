"""Type aliases and protocols for proof_of_replica."""

from __future__ import annotations

from typing import Any

# Profile JSON is a nested dict structure
ProfileDict = dict[str, Any]

# Column statistics extracted during profiling
StatsDict = dict[str, Any]

# Distribution parameters (family-specific)
DistributionParams = dict[str, float | str]
