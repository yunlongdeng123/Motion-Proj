"""真实约十分钟 smoke：300s fresh +300s exact resume +下一阶段恢复入口。

正式实验从零开始，smoke 和 entry 更新单独记账。不会启动正式训练或关机。
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from inspect_bundle import inspect


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def execute(command, output, name):
    started = time.monotonic()
    with (output / (name + ".log")).open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    write(output / (name + "_execution.json"), dict(command=command, returncode=result.returncode,
          wall_seconds=time.monotonic() - started))
    if result.returncode:
        raise RuntimeError("真实 smoke 阶段失败，请保留现场并查看 " + str(output / (name + ".log")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--steps", type=int, default=30000, help="正式预算合同；smoke 提前停但不缩短 LR 计划")
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--leg-seconds", type=float, default=300)
    parser.add_argument("--entry-steps", type=int, default=2)
    parser.add_argument("--reference-root", type=Path, help="可选独立参考，仅用于下一阶段渲染评价")
    args = parser.parse_args()
    args.scene, args.output = args.scene.resolve(), args.output.resolve()
    if args.output.exists():
        raise FileExistsError("不覆盖已有 smoke 现场")
    args.output.mkdir(parents=True)
    worker = Path(__file__).resolve().with_name("native_worker.py")
    common = [sys.executable, "-u", str(worker), "--scene", str(args.scene), "--steps", str(args.steps),
              "--checkpoint-every", str(args.checkpoint_every)]
    for key, path in (("--source", args.source), ("--base-config", args.base_config)):
        if path:
            common += [key, str(path.resolve())]
    fresh, resumed, entry = [args.output / name for name in ("fresh", "resumed", "next_stage_entry")]
    execute(common + ["--output", str(fresh), "--smoke-seconds", str(args.leg_seconds)], args.output, "fresh")
    fresh_acceptance = inspect(fresh)
    write(args.output / "fresh_acceptance.json", fresh_acceptance)
    first_event = read(fresh_acceptance["event"])
    execute(common + ["--output", str(resumed), "--smoke-seconds", str(args.leg_seconds), "--resume", first_event["checkpoint_paths"]["static"]], args.output, "resumed")
    resumed_acceptance = inspect(resumed)
    write(args.output / "resumed_acceptance.json", resumed_acceptance)
    restored = read(resumed / "restore_verification.json")
    if restored.get("all_passed") is not True:
        raise RuntimeError("真实 CUDA 完整恢复未通过")
    second_event = read(resumed_acceptance["event"])
    execute(common + ["--output", str(entry), "--entry-steps", str(args.entry_steps), "--resume", second_event["checkpoint_paths"]["static"]], args.output, "next_stage_entry")
    entry_acceptance = inspect(entry)
    write(args.output / "entry_acceptance.json", entry_acceptance)
    if read(entry / "restore_verification.json").get("all_passed") is not True:
        raise RuntimeError("下一阶段真实恢复入口未通过")
    render_output = args.output / "render_entry"
    render_command = [sys.executable, "-u", str(worker.with_name("evaluate_edits.py")), "--event", entry_acceptance["event"],
                      "--output", str(render_output), "--edits", "none,delete", "--keyframes",
                      read(args.scene / "preparation_manifest.json")["frames"][0]["id"]]
    if args.source:
        render_command += ["--source", str(args.source.resolve())]
    if args.reference_root:
        render_command += ["--reference-root", str(args.reference_root.resolve())]
    execute(render_command, args.output, "render_entry")
    render_entry = read(render_output / "render_manifest.json")
    if (render_entry["status"] != "NATIVE_BRANCH_ROUTING_EDIT_RENDERED"
            or not render_entry["native_none_identity_checked_frames"]
            or render_entry["native_none_identity_max_abs"] > 2e-6
            or not render_entry["original_checkpoint_unchanged"]):
        raise RuntimeError("下一阶段真实 none/delete 光栅入口未通过")
    returns = {name: read(path / "returned.json") for name, path in (("fresh", fresh), ("resumed", resumed), ("entry", entry))}
    triggers = {name: returns[name]["timing"]["stop_trigger"] for name in ("fresh", "resumed")}
    if any(trigger is None or trigger["reason"] != "smoke" for trigger in triggers.values()):
        raise RuntimeError("两段必须各自达到登记的 smoke 计时阈值后停止，不能用预算结束或评价补足")
    timed = sum(trigger["training_loop_wall_seconds_excluding_measured_eval_checkpoint"] for trigger in triggers.values())
    each_leg_reached = all(trigger["training_loop_wall_seconds_excluding_measured_eval_checkpoint"] >= args.leg_seconds for trigger in triggers.values())
    phase_totals = {}
    for value in returns.values():
        for name, seconds in value["timing"]["phase_wall_seconds"].items():
            phase_totals[name] = phase_totals.get(name, 0.) + seconds
    smoke = dict(status="PASS_REAL_NATIVE_SMOKE", approximately_ten_minutes_contract=each_leg_reached and timed >= 600,
        timed_leg_training_loop_wall_seconds_excluding_measured_eval_checkpoint=timed,
        each_leg_reached_requested_loop_wall_seconds=each_leg_reached, stop_trigger_timing=triggers,
        phase_wall_seconds=phase_totals, phases=returns,
        timing_scope="host training-loop wall proxy; measured evaluation/checkpoint overhead excluded; not pure GPU kernel seconds; restore is a subset of initialization, not added to it",
        extra_smoke_updates=returns["fresh"]["observed_finite_step_calls"] + returns["resumed"]["observed_finite_step_calls"],
        extra_entry_updates=returns["entry"]["observed_finite_step_calls"], formal_updates=0,
        formal_must_start_fresh=True, complete_restore=restored, next_stage_entry=entry_acceptance,
        actual_next_stage_render=dict(manifest=str(render_output / "render_manifest.json"),
            native_none_identity_max_abs=render_entry["native_none_identity_max_abs"], training_updates=0),
        scientific_contamination_claim=False, real_scene_generalization_claim=False)
    write(args.output / "smoke_summary.json", smoke)
    formal = common + ["--output", str(args.output.parent / "formal")]
    write(args.output / "formal_entry.json", dict(command=formal, executed=False, starts_fresh=True,
        steps=args.steps, position_lr_max_steps=30000, smoke_updates_charged_separately=True,
        requires_approximately_ten_minute_smoke_for_hour_scale_run=True))
    print(json.dumps(dict(status=smoke["status"], training_loop_wall_seconds_excluding_measured_eval_checkpoint=timed,
        ten_minute_contract=smoke["approximately_ten_minutes_contract"], formal_executed=False)))


if __name__ == "__main__":
    main()
