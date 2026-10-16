"""GPU 后先测各阶段吞吐/显存；不凭旧 18k 微调耗时估计本次从零重建。"""
import argparse
import json
import time
from pathlib import Path
import torch
from .config import Config
from .data import Dataset
from .model import SceneModel
from .train import objective, write
from .gpu_probe import probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status="GPU_BENCHMARK_NOT_STARTED", planned_stages=["geometry", "material", "joint"])))
        return
    if args.steps < 3:
        raise ValueError("吞吐探测至少3步，首2步作为预热")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = Config.load(args.config)
    dataset = Dataset(args.data, "cpu")
    from .train import preflight
    inputs = preflight(dataset, config)
    abi = probe(config)
    dataset.device = "cuda"
    reports = []
    for stage, budget in (("geometry", config.geometry_steps), ("material", config.material_steps), ("joint", config.joint_steps)):
        model = SceneModel(dataset.seed, config).cuda()
        model.field.anchor()
        optimizer = model.optimizer(stage)
        generator = torch.Generator().manual_seed(config.seed)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(args.steps):
            frame = dataset.frame(dataset.train_indices[i % len(dataset.train_indices)])
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = objective(model, frame, len(dataset.actor_ids), "cuda", stage, generator)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if i >= 2:
                times.append(time.perf_counter()-started)
        ordered = sorted(times)
        median = ordered[len(ordered)//2]
        reports.append(dict(stage=stage, seconds_per_update_median=median, seconds_per_update_max=max(times),
                            estimated_stage_hours=median*budget/3600, max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                            caveat="initial seed only; densification, teacher, I/O and full49 editing evaluation add cost"))
        write(output/"benchmark.json", dict(inputs=inputs, ABI=abi, stages=reports, quality_validated=False))
        del loss, optimizer, model
        torch.cuda.empty_cache()
    print(json.dumps(dict(stages=reports, projected_training_hours=sum(r["estimated_stage_hours"] for r in reports)), ensure_ascii=False))


if __name__ == "__main__":
    main()
