"""把 M2 路由结果编译为不修改 base 的可撤销 repair delta。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

from .repair_candidates import GaussianAssetBinding, RepairCandidate
from .repair_router import RepairDecision, RiskRepairRouter


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class RepairRequest:
    request_id: str
    scene: str
    hole_id: str
    frames: tuple[int, ...]
    camera_ids: tuple[int, ...]
    erase_gaussian_ids: tuple[int, ...]
    base_checkpoint_sha256: str
    test_quality_read: bool = False

    def __post_init__(self) -> None:
        if not self.request_id or not self.scene or not self.hole_id:
            raise ValueError("request identity fields must be non-empty")
        if not self.frames or any(frame < 0 for frame in self.frames):
            raise ValueError("request frames must be non-empty and non-negative")
        if not self.camera_ids or any(camera < 0 for camera in self.camera_ids):
            raise ValueError("camera IDs must be non-empty and non-negative")
        if not self.erase_gaussian_ids or any(
            identifier < 0 for identifier in self.erase_gaussian_ids
        ):
            raise ValueError("erase Gaussian IDs must be non-empty and non-negative")
        if len(set(self.erase_gaussian_ids)) != len(self.erase_gaussian_ids):
            raise ValueError("erase Gaussian IDs must be unique")
        if len(self.base_checkpoint_sha256) != 64:
            raise ValueError("base checkpoint SHA must have 64 characters")
        if self.test_quality_read:
            raise ValueError("M2 compiler cannot consume test quality")


@dataclass(frozen=True)
class RepairDelta:
    request_id: str
    scene: str
    base_checkpoint_sha256: str
    composition_order: tuple[str, ...]
    requested_erase_gaussian_ids: tuple[int, ...]
    applied_erase_gaussian_ids: tuple[int, ...]
    insert_asset: GaussianAssetBinding | None
    decision: RepairDecision
    base_mutated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "worldsim_v4_m2_repair_delta_v1",
            "request_id": self.request_id,
            "scene": self.scene,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "composition_order": list(self.composition_order),
            "requested_erase_gaussian_ids": list(
                self.requested_erase_gaussian_ids
            ),
            "applied_erase_gaussian_ids": list(self.applied_erase_gaussian_ids),
            "insert_asset": self.insert_asset.to_dict() if self.insert_asset else None,
            "decision": self.decision.to_dict(),
            "base_mutated": self.base_mutated,
            "rollback_checkpoint_sha256": self.base_checkpoint_sha256,
            "test_quality_read": False,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()


def compile_repair_delta(
    *,
    request: RepairRequest,
    candidates: Iterable[RepairCandidate],
    router: RiskRepairRouter,
) -> RepairDelta:
    rows = list(candidates)
    decision = router.route(rows)
    by_id = {candidate.candidate_id: candidate for candidate in rows}
    if decision.accepted:
        selected = by_id[decision.candidate_id]
        applied_erase = request.erase_gaussian_ids
        insert_asset = selected.gaussians
    else:
        # 高风险时整次 edit 原子拒绝，避免留下只删除未修复的半成品。
        applied_erase = ()
        insert_asset = None
    return RepairDelta(
        request_id=request.request_id,
        scene=request.scene,
        base_checkpoint_sha256=request.base_checkpoint_sha256,
        composition_order=("BASE", "ERASE", "INSERT_REPAIR"),
        requested_erase_gaussian_ids=request.erase_gaussian_ids,
        applied_erase_gaussian_ids=applied_erase,
        insert_asset=insert_asset,
        decision=decision,
    )
