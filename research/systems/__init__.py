"""Multi-system support: scale VIS beyond the modified IEEE 39-bus benchmark.

The thesis's single biggest limitation (Ch. 6.2) is that results are shown on
one modified IEEE 39-bus system, so generalisation is asserted, not demonstrated.
This module provides the infrastructure to change that:

- `SystemSpec` + `SYSTEM_REGISTRY`: declarative description of a test system
  (case path, which static generators become grid-forming IBRs, M/D ranges),
  generalising the settings that were previously implicit in the IEEE 39-bus
  configs.
- `describe_system`: a light diagnostic (bus/line/machine/IBR counts + power-flow
  convergence) usable on any ANDES case.
- `augment_with_grid_forming_ibrs` + `REGCV1_TEMPLATE`: programmatically attach
  REGCV1 grid-forming converters to a case, generalising the hand-built
  `ieee39_full_ibrs.xlsx` so the same recipe applies to larger systems
  (e.g. the 140-bus NPCC case bundled with ANDES).

Candidate scale-up systems (per the research briefing): NPCC 140-bus (dynamic),
then IEEE 118 / 300 (need dynamic-model assignment) and Nordic-44.
"""

from research.systems.registry import (
    REGCV1_TEMPLATE,
    SYSTEM_REGISTRY,
    SystemSpec,
    augment_with_grid_forming_ibrs,
    describe_system,
    resolve_case_path,
)

__all__ = [
    "REGCV1_TEMPLATE",
    "SYSTEM_REGISTRY",
    "SystemSpec",
    "augment_with_grid_forming_ibrs",
    "describe_system",
    "resolve_case_path",
]
