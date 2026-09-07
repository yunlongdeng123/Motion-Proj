"""EAS-VGGT 的多基座几何适配接口。"""

from motion_proj.worldsim_v72.eas_vggt.types import (
    BACKBONE_GEOMETRY_SCHEMA_VERSION,
    BackboneGeometry,
)
from motion_proj.worldsim_v72.eas_vggt.ordered_returns import (
    OrderedReturnDistribution,
    ordered_return_distribution,
)

__all__ = [
    "BACKBONE_GEOMETRY_SCHEMA_VERSION",
    "BackboneGeometry",
    "OrderedReturnDistribution",
    "ordered_return_distribution",
]
