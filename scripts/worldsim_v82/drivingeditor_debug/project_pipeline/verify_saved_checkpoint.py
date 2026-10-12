#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--iteration", type=int, required=True)
parser.add_argument("--including-dynamic", action="store_true")
args = parser.parse_args()
static, iteration = torch.load(args.run / f"chkpnt{args.iteration}.pth", map_location="cpu")
assert iteration == args.iteration
steps = sorted({float(value["step"]) for value in static[14]["state"].values() if "step" in value})
assert steps == [float(args.iteration)]
assert torch.isfinite(static[1]).all()
light, light_iteration = torch.load(args.run / f"env_light_chkpnt{args.iteration}.pth", map_location="cpu")
light_steps = sorted({float(value["step"]) for value in light[1]["state"].values() if "step" in value})
assert light_iteration == iteration and light_steps == steps
assert torch.isfinite(light[0]).all()
result = {"status": "PASS", "checkpoint_iteration": iteration, "static_optimizer_steps": steps,
          "environment_optimizer_steps": light_steps, "xyz_points": len(static[1]),
          "static_xyz_and_environment_finite": True}
if args.including_dynamic:
    dynamic, dynamic_iteration = torch.load(args.run / f"dynamic_chkpnt{args.iteration}.pth", map_location="cpu")
    dynamic_steps = sorted({float(value["step"]) for value in dynamic[14]["state"].values() if "step" in value})
    assert dynamic_iteration == iteration and dynamic_steps == steps
    assert torch.isfinite(dynamic[1]).all()
    result.update(dynamic_optimizer_steps=dynamic_steps, dynamic_xyz_points=len(dynamic[1]),
                  dynamic_xyz_finite=True)
(args.run / "checkpoint_verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
