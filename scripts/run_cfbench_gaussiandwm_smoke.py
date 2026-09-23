"""官方合成 QA 合同样例的 GPU smoke；没有编辑或真实下游计分声明。"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-unreleased-trajectory-head", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = Path("/root/autodl-tmp/external/worldsim_v75_downstream_bench/GaussianDWM")
    package = repo / "src/gaussiandwm_cvpr"
    sys.path.insert(0, str(repo / "src"))
    os.chdir(repo)
    cfgdir = out / "configs"
    shutil.copytree(package / "configs/qa", cfgdir)
    infer = yaml.safe_load((cfgdir / "infer.yaml").read_text())
    infer.update(infer_id="gpu-smoke-20260923", batch_size=1)
    infer["qa"]["max_new_tokens"] = 64
    (cfgdir / "infer.yaml").write_text(yaml.safe_dump(infer))
    result = {"task_id": "WS-V75-DOWNSTREAM-GPU-SMOKE-01", "run_id": "20260923-r1",
              "model_id": "gaussiandwm", "role": "consumer_only", "status": "started",
              "input_role": "upstream_synthetic_contract_example", "benchmark_aligned": False,
              "seed": 42, "max_new_tokens": 64, "human_verdict": None,
              "claim_boundary": "仅验证官方 QA 完整调用链，不评价真实驾驶质量；不是 paired downstream outcome。",
              "failure_ledger_refs": []}
    started = time.monotonic()
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        from gaussiandwm_cvpr.infer.qa import run_qa_inference_from_config_dir
        if args.exclude_unreleased_trajectory_head:
            # README 明确 CVPR 不含 trajectory code。仅排除已观测的六个闲置参数，
            # 其余参数继续由官方 strict=True 加载；不修改官方源码或权重文件。
            from gaussiandwm_cvpr.models.unified_model import UnifiedGaussianDWM
            original_loader = UnifiedGaussianDWM.load_pretrained_state_dict
            excluded = {f"traj_head.mlp.{layer}.{kind}" for layer in [0, 1, 3] for kind in ["bias", "weight"]}
            def load_qa_state(*loader_args, **loader_kwargs):
                state = original_loader(*loader_args, **loader_kwargs)
                found = {key for key in state if key.startswith("traj_head.")}
                if found != excluded:
                    raise RuntimeError(f"未预期的 trajectory 参数集合: {sorted(found)}")
                result["excluded_unused_parameters"] = {key: list(state[key].shape) for key in sorted(excluded)}
                return {key: value for key, value in state.items() if key not in excluded}
            UnifiedGaussianDWM.load_pretrained_state_dict = staticmethod(load_qa_state)
            result["implementation_variant"] = "qa_compatibility_adapter_not_raw_upstream"
        prediction = run_qa_inference_from_config_dir(
            config_dir=cfgdir, run_dir=out,
            model_id="/root/autodl-tmp/models/worldsim_v75_downstream_bench/gaussiandwm/model",
            data_root=package / "examples/dummy_data",
            annotation_path=package / "examples/dummy_data/qa_val.jsonl",
            gauss_cache_root=out / "gauss-cache",
        )
        result.update(status="complete", predictions_path=str(prediction))
    except BaseException as exc:
        result.update(status="oom_stopped" if isinstance(exc, torch.OutOfMemoryError) else "failed_stopped",
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result.update(wall_s=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      failure_ledger_delta="none")
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
