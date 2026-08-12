"""风险最小化与 hard-priority 两种 matched repair router。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

from .repair_candidates import REPAIR_METHODS, RepairCandidate
from .repair_risk import RepairRiskScore, RepairRiskWeights, score_repair_candidate


@dataclass(frozen=True)
class RepairDecision:
    policy: str
    action: str
    candidate_id: str | None
    selected_score: float | None
    risk_threshold: float | None
    abstain_reason: str | None
    scores: tuple[RepairRiskScore, ...]

    @property
    def accepted(self) -> bool:
        return self.action != "ABSTAIN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "action": self.action,
            "candidate_id": self.candidate_id,
            "selected_score": self.selected_score,
            "risk_threshold": self.risk_threshold,
            "accepted": self.accepted,
            "abstain_reason": self.abstain_reason,
            "scores": [score.to_dict() for score in self.scores],
        }


def _unique_candidates(candidates: Iterable[RepairCandidate]) -> list[RepairCandidate]:
    rows = list(candidates)
    identifiers = [candidate.candidate_id for candidate in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("repair candidate IDs must be unique")
    return rows


class RiskRepairRouter:
    def __init__(
        self,
        *,
        weights: RepairRiskWeights,
        threshold: float,
        tie_priority: Sequence[str] = REPAIR_METHODS,
    ) -> None:
        threshold = float(threshold)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("risk threshold must be finite and non-negative")
        if tuple(sorted(tie_priority)) != tuple(sorted(REPAIR_METHODS)):
            raise ValueError("tie priority must contain every repair method exactly once")
        self.weights = weights
        self.threshold = threshold
        self.tie_priority = tuple(tie_priority)

    def route(self, candidates: Iterable[RepairCandidate]) -> RepairDecision:
        rows = _unique_candidates(candidates)
        if not rows:
            return RepairDecision(
                policy="risk_minimization",
                action="ABSTAIN",
                candidate_id=None,
                selected_score=None,
                risk_threshold=self.threshold,
                abstain_reason="ABSTAIN_NO_CANDIDATE",
                scores=(),
            )
        priority = {method: index for index, method in enumerate(self.tie_priority)}
        scored = [score_repair_candidate(candidate, self.weights) for candidate in rows]
        scored.sort(
            key=lambda score: (score.total, priority[score.method], score.candidate_id)
        )
        selected = scored[0]
        if selected.total > self.threshold:
            return RepairDecision(
                policy="risk_minimization",
                action="ABSTAIN",
                candidate_id=None,
                selected_score=selected.total,
                risk_threshold=self.threshold,
                abstain_reason="ABSTAIN_RISK_THRESHOLD",
                scores=tuple(scored),
            )
        return RepairDecision(
            policy="risk_minimization",
            action=selected.method,
            candidate_id=selected.candidate_id,
            selected_score=selected.total,
            risk_threshold=self.threshold,
            abstain_reason=None,
            scores=tuple(scored),
        )


def route_hard_priority(
    candidates: Iterable[RepairCandidate],
    *,
    priority: Sequence[str] = REPAIR_METHODS,
) -> RepairDecision:
    """主消融用 hard priority；它只在完全无候选时拒绝。"""

    rows = _unique_candidates(candidates)
    if tuple(sorted(priority)) != tuple(sorted(REPAIR_METHODS)):
        raise ValueError("priority must contain every repair method exactly once")
    by_method = {method: [] for method in priority}
    for candidate in rows:
        by_method[candidate.method].append(candidate)
    for method in priority:
        if by_method[method]:
            selected = min(by_method[method], key=lambda candidate: candidate.candidate_id)
            return RepairDecision(
                policy="hard_priority",
                action=selected.method,
                candidate_id=selected.candidate_id,
                selected_score=None,
                risk_threshold=None,
                abstain_reason=None,
                scores=(),
            )
    return RepairDecision(
        policy="hard_priority",
        action="ABSTAIN",
        candidate_id=None,
        selected_score=None,
        risk_threshold=None,
        abstain_reason="ABSTAIN_NO_CANDIDATE",
        scores=(),
    )
