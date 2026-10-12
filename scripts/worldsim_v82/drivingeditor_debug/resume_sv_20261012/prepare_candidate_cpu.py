"""只复制 S 的小型源码并加入历史 V；不导入训练模块，不加载 checkpoint。"""

import argparse
import ast
import difflib
import json
import os
from pathlib import Path
import shutil


os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
BASE = Path("/root/autodl-tmp/drivingeditor_repro")
SOURCE = BASE / "campaign/vgpu_bugfix_20261012/loss_audit/next_diagnostics_cpu/observed_static_rgb_S/frozen_repository"
ROOT = BASE / "campaign/resume_sv_20261012"
HISTORICAL_V = BASE / "campaign/vgpu_bugfix_20261012/loss_audit/long_pair_30k_32000/variant_frozen_repository/utils/paper_losses.py"
IMPORT_ANCHOR = "from utils.loss_utils import psnr, ssim\n"
IMPORT_ADDITION = ("from utils.resume_sv_losses import (diagnostic_dynamic_background_opacity_loss,\n"
                   "                                    raw_alpha_domain_stats, validate_sv_config,\n"
                   "                                    reject_unbound_gpu_training)\n")
S_BLOCK = '''            if args.get("diagnostic_observed_static_rgb_enabled", False):
                log_dict["loss_original_objective_before_diagnostic"] = loss.item()
                observed_static_mask = 1.0 - dynamic_mask
                observed_static_prediction = static_image * observed_static_mask
                observed_static_truth = gt_image * observed_static_mask
                observed_static_rgb = ((1.0 - args.lambda_dssim) * F.l1_loss(observed_static_prediction, observed_static_truth)
                                       + args.lambda_dssim * (1.0 - ssim(observed_static_prediction, observed_static_truth)))
                loss = loss + args.diagnostic_lambda_observed_static_rgb * observed_static_rgb
                log_dict["loss_diagnostic_observed_static_rgb"] = observed_static_rgb.item()
'''
V_BLOCK = '''            # S+V：原 Eq13 前景监督 + 历史 V 掩码外监督，均以全 HW 归一化。
            # 必须从 render_pkg 读取 raw alpha，位于后续显示用 premask 之前。
            raw_dynamic_alpha = render_pkg["dynamic_alpha"]
            diagnostic_background_opacity = diagnostic_dynamic_background_opacity_loss(raw_dynamic_alpha, dynamic_mask)
            log_dict["loss_resume_sv_original"] = log_dict["loss_original_objective_before_diagnostic"]
            log_dict["loss_resume_sv_S"] = observed_static_rgb.item()
            log_dict["loss_resume_sv_V"] = diagnostic_background_opacity.item()
            log_dict["loss_resume_sv_V_weighted"] = 0.05 * diagnostic_background_opacity.item()
            loss = loss + 0.05 * diagnostic_background_opacity
            log_dict["loss_resume_sv_total"] = loss.item()
            log_dict.update(raw_alpha_domain_stats(raw_dynamic_alpha, dynamic_mask))
'''
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".sh",
                   ".bat", ".ps1", ".cpp", ".cu", ".c", ".cc", ".h", ".hpp", ".cuh",
                   ".md", ".rst", ".txt", ".cmake"}
CACHE_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
              ".cache", "build", "dist", "data", "datasets", "runs", "outputs",
              "checkpoints", "point_cloud", "visualization", "eval"}


def patch_source(text):
    text = text.replace("\r\n", "\n")
    replacements = ((IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_ADDITION),
                    ("def training(args):\n\n", "def training(args):\n    reject_unbound_gpu_training()\n    validate_sv_config(args)\n\n"),
                    (S_BLOCK, S_BLOCK + V_BLOCK),
                    ('if __name__ == "__main__":\n', 'if __name__ == "__main__":\n    reject_unbound_gpu_training()\n'))
    if "resume_sv_losses" in text or "loss_resume_sv_V" in text:
        raise ValueError("拒绝重复插入 S+V")
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError("S 源码插入点必须且只能出现一次: " + before.splitlines()[0])
        text = text.replace(before, after, 1)
    ast.parse(text)
    return text


def ignore_assets(directory, names):
    ignored = []
    for name in names:
        item = Path(directory) / name
        if item.is_symlink() or name in CACHE_DIRS or "approval" in name.lower() or "binding" in name.lower():
            ignored.append(name)
        elif item.is_file() and (item.suffix.lower() not in SOURCE_SUFFIXES
                                and name not in {"LICENSE", "Makefile", "CMakeLists.txt"}):
            ignored.append(name)
        elif item.is_file() and item.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("拒绝复制超出源码大小范围的文件: " + str(item))
    return ignored


def materialize_source_copy(source, destination, helper):
    # 先检查目标，拒绝覆盖或复用半成品；只创建全新目录。
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("候选 frozen_repository 已存在，拒绝覆盖: " + str(destination))
    if not source.is_dir() or source == destination or source in destination.parents:
        raise ValueError("来源目录无效，或目标落在来源内")
    original = (source / "train.py").read_text(encoding="utf-8")
    patched = patch_source(original)
    helper_text = helper.read_text(encoding="utf-8")
    ast.parse(helper_text)
    if not (source / "utils/paper_losses.py").is_file():
        raise FileNotFoundError("S 原 foreground loss 来源缺失")
    if (source / "utils/resume_sv_losses.py").exists():
        raise ValueError("来源已含 S+V helper，拒绝重用")
    shutil.copytree(source, destination, ignore=ignore_assets)
    (destination / "train.py").write_text(patched, encoding="utf-8", newline="\n")
    (destination / "utils/resume_sv_losses.py").write_text(helper_text, encoding="utf-8", newline="\n")
    return original.replace("\r\n", "\n"), patched, helper_text


def source_delta(original, patched, helper_text):
    return ("".join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                        fromfile="S/train.py", tofile="SV/train.py"))
            + "".join(difflib.unified_diff([], helper_text.splitlines(True),
                                           fromfile="/dev/null", tofile="SV/utils/resume_sv_losses.py")))


def candidate_plan(root=ROOT):
    parent = BASE / "runs/scene0017_000_048_paper_50000_seed0_resume_vgpu1011_baseline"
    return {"schema_version": 1, "task_id": "WS-V82-DRIVINGEDITOR-SV-20261012", "run_id": "r1",
            "status": "source_preparation_only", "candidate_root": str(root),
            "candidate_repository": str(root / "frozen_repository"), "source_S": str(SOURCE),
            "historical_V_losses": str(HISTORICAL_V), "algorithm_changes_from_S": ["historical V outside-opacity term, coefficient 0.05"],
            "supervision": {"foreground": "original Eq13: -mean[Md*(1-clamp(alpha,1e-4,1-1e-4))*log(clamp(alpha,1e-4,1-1e-4))]",
                            "foreground_coefficient": 0.05,
                            "outside": "historical V: -mean[(1-Md)*a*log1p(-a/(1+1e-4))], a=clamp(raw_alpha,0,1)",
                            "outside_coefficient": 0.05, "normalization": "full H*W for both domains",
                            "static_rgb": "unchanged S observed static RGB L1+SSIM on (1-Md), full image reduction",
                            "static_rgb_coefficient": 1.0, "lambda_dssim": 0.2,
                            "alpha_input": "render_pkg dynamic_alpha, before premask",
                            "mask": "existing body-only binary dynamic appearance mask; shadow remains absent",
                            "author_formula_verified": False},
            "continuation": {"parent_iteration": 30000, "first_iteration": 30001, "stop_iteration": 32000,
                             "steps": 2000, "args_iterations": 50000, "position_lr_max_steps": 50000, "seed": 0,
                             "train_frames": 49, "frame_range": "FRONT raw0-48", "terminal_eval_frames": 49,
                             "checkpoint_files": [str(parent / filename) for filename in
                                                  ("chkpnt30000.pth", "dynamic_chkpnt30000.pth", "env_light_chkpnt30000.pth", "training_state30000.pth")],
                             "sampling_reference": str(BASE / "campaign/vgpu_bugfix_20261012/continuation_diagnostic/original30k_to32000"),
                             "sampling_requirement": "reuse B1 all2000 camera/time_shift/LR trajectory; restore saved RNG/sampler without reseed",
                             "checkpoint_loading": "none during CPU source preparation"},
            "execution": {"allow_training": False, "gpu_harness_connected": False, "new_runtime_binding": None,
                          "reuse_historical_binding_or_approval": False, "gpu_launcher_written": False,
                          "training_started": False, "cpu_only": True, "CPU_threads": 1},
            "failure_ledger_refs": ["V82-F01"], "failure_ledger_delta": "none"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.source.resolve() != SOURCE.resolve() or args.candidate_root.resolve() != ROOT.resolve():
        raise ValueError("只允许指定 S 来源与新的 campaign/resume_sv_20261012 目标")
    args.candidate_root.mkdir(parents=True, exist_ok=True)
    destination = args.candidate_root / "frozen_repository"
    original, patched, helper = materialize_source_copy(args.source, destination, HERE / "utils/resume_sv_losses.py")
    (args.candidate_root / "source_delta.patch").write_text(source_delta(original, patched, helper), encoding="utf-8", newline="\n")
    plan = candidate_plan(args.candidate_root)
    files = [p for p in destination.rglob("*") if p.is_file()]
    plan["source_copy"] = {"materialized": True, "source_train_bytes": (args.source / "train.py").stat().st_size,
                           "candidate_file_count": len(files), "candidate_total_bytes": sum(p.stat().st_size for p in files),
                           "modified_existing_files": ["train.py"], "added_files": ["utils/resume_sv_losses.py"],
                           "CPU_math_checks_passed": None}
    (args.candidate_root / "candidate_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": plan["status"], "candidate_repository": str(destination),
                      "files": len(files), "bytes": plan["source_copy"]["candidate_total_bytes"],
                      "allow_training": False, "GPU_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
