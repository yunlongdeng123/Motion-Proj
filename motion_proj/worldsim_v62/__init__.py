"""WorldSim V6.2 constraint-aware physical state completion."""

from motion_proj.worldsim_v62.projection import (
    FREE_INDEX,
    OCCUPIED_INDEX,
    UNKNOWN_INDEX,
    ProjectionOutput,
    project_feasible_tristate,
)

__all__ = [
    "FREE_INDEX",
    "OCCUPIED_INDEX",
    "UNKNOWN_INDEX",
    "ProjectionOutput",
    "project_feasible_tristate",
]
