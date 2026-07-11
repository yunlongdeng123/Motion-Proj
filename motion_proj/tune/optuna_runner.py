"""严格执行 16×100、4×300、2×800 与九小时截止的 Optuna 执行器。"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from omegaconf import OmegaConf

from ..runtime.atomic import atomic_write_json
from .policy import STAGES, SearchBudget, derived_run_id, objective_score, prune_reason, suggest_params


@dataclass
class TrialRecord:
    stage_steps: int
    run_id: str
    parent_run_id: str | None
    params: dict
    metrics: dict
    prune_reason: str | None
    score: float | None


def promotion_candidates(records: list[TrialRecord], count: int) -> list[TrialRecord]:
    valid = [record for record in records if record.prune_reason is None and record.score is not None]
    return sorted(valid, key=lambda record: (-float(record.score), float(record.metrics["lpips"])))[:count]


class OptunaExecutor:
    def __init__(self, storage: str, work_root: str, base_lpips: float,
                 runner: Callable[[dict, int, str, str | None], dict],
                 hard_limit_hours: float = 9.0, study_name: str = "motionproj-p2"):
        import optuna

        Path(work_root).mkdir(parents=True, exist_ok=True)
        if "://" not in storage:
            storage = f"sqlite:///{os.path.abspath(storage)}"
        self.optuna = optuna
        self.study = optuna.create_study(
            study_name=study_name, storage=storage, direction="maximize", load_if_exists=True
        )
        self.work_root = work_root
        self.base_lpips = float(base_lpips)
        self.runner = runner
        self.budget = SearchBudget.start(hard_limit_hours)
        self.records: list[TrialRecord] = []
        self.state_path = os.path.join(work_root, "executor_state.json")
        self._load_records()

    def _load_records(self) -> None:
        try:
            rows = json.loads(Path(self.state_path).read_text(encoding="utf-8"))["records"]
            self.records = [TrialRecord(**row) for row in rows]
        except (OSError, ValueError, KeyError, TypeError):
            self.records = []

    def _save(self, status: str) -> None:
        reasons = [record.prune_reason for record in self.records if record.prune_reason]
        summary = {
            "status": status, "stages": STAGES,
            "records": [asdict(record) for record in self.records],
            "all_pruned_same_reason": len(set(reasons)) == 1 and len(reasons) >= STAGES[0][0],
            "thresholds_relaxed": False,
        }
        atomic_write_json(self.state_path, summary)

    def _run_record(self, params: dict, steps: int, parent: TrialRecord | None,
                    run_prefix: str) -> TrialRecord:
        parent_id = parent.run_id if parent else None
        run_id = derived_run_id(run_prefix, params, steps)
        metrics = self.runner(params, steps, run_id, parent_id)
        reason = prune_reason(metrics, self.base_lpips)
        score = None if reason else objective_score(metrics)
        record = TrialRecord(steps, run_id, parent_id, params, metrics, reason, score)
        self.records.append(record)
        self._save("running")
        return record

    def execute(self) -> dict:
        initial = [record for record in self.records if record.stage_steps == 100]
        while len(initial) < STAGES[0][0] and self.budget.may_start_trial():
            trial = self.study.ask()
            params = suggest_params(trial)
            record = self._run_record(params, 100, None, f"tune-t{trial.number}")
            if record.prune_reason:
                self.study.tell(trial, state=self.optuna.trial.TrialState.PRUNED)
            else:
                self.study.tell(trial, float(record.score))
            initial.append(record)

        previous = initial
        for promote_count, target_steps in STAGES[1:]:
            existing = [record for record in self.records if record.stage_steps == target_steps]
            selected = promotion_candidates(previous, promote_count)
            completed_parents = {record.parent_run_id for record in existing}
            for parent in selected:
                if parent.run_id in completed_parents:
                    continue
                if not self.budget.may_start_trial():
                    self._save("deadline")
                    return json.loads(Path(self.state_path).read_text(encoding="utf-8"))
                existing.append(self._run_record(parent.params, target_steps, parent, parent.run_id))
            previous = existing

        status = "completed" if len(initial) == STAGES[0][0] else "deadline"
        self._save(status)
        return json.loads(Path(self.state_path).read_text(encoding="utf-8"))


def command_runner(command_template: str, work_root: str):
    def run(params: dict, steps: int, run_id: str, parent_run_id: str | None) -> dict:
        run_dir = os.path.join(work_root, run_id)
        summary = os.path.join(run_dir, "summary.json")
        if os.path.isfile(summary):
            return json.loads(Path(summary).read_text(encoding="utf-8"))
        values = {**params, "target_steps": steps, "run_id": run_id,
                  "parent_run_id": parent_run_id or "none", "run_dir": run_dir}
        command = [part.format(**values) for part in shlex.split(command_template)]
        env = dict(os.environ, MOTIONPROJ_TRIAL_JSON=json.dumps(values, sort_keys=True))
        subprocess.run(command, check=True, env=env)
        return json.loads(Path(summary).read_text(encoding="utf-8"))
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-lpips", type=float, required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--command-template", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    executor = OptunaExecutor(
        str(cfg.storage), args.work_root, args.base_lpips,
        command_runner(args.command_template, args.work_root),
        hard_limit_hours=float(cfg.hard_limit_hours),
    )
    print(json.dumps(executor.execute(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
