"""将 CPU 数学/集成检查和小场景梯度 smoke 的实际结果写入指定新目录。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
import argparse
from dataclasses import replace
import contextlib
import io
from pathlib import Path
import sys
import time
import unittest
import ast
import json
import torch
import numpy as np
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import run, write, preflight
from shadow_rebuild.evaluate import evaluate


def cases(suite):
    for value in suite:
        if isinstance(value, unittest.TestSuite):
            yield from cases(value)
        else:
            yield value.id()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).resolve().parent
    sys.path.insert(0, str(code/"tests"))
    torch.set_num_threads(2)
    started = time.monotonic()
    syntax = []
    for path in sorted(code.rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8-sig"), str(path))
        syntax.append(path.relative_to(code).as_posix())
    suite = unittest.defaultTestLoader.discover(str(code/"tests"), pattern="test_*.py")
    names = list(cases(suite))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (output/"unittest.log").write_text(stream.getvalue(), encoding="utf-8")
    if not result.wasSuccessful():
        write(output/"cpu_validation.json", dict(status="FAILED", tests=result.testsRun, log="unittest.log"))
        raise SystemExit(stream.getvalue())
    from fixture import create
    config = replace(create(output/"synthetic_input"), geometry_steps=12, material_steps=12, joint_steps=6)
    dataset = Dataset(output/"synthetic_input", synthetic=True)
    model0 = SceneModel(dataset.seed, config)
    frame = dataset.frame(0)

    def image_error(model):
        with torch.no_grad():
            g, p = model.buffers(frame, 1, "reference")
            image, _ = model.full_image(p, g)
            return float((image-frame.rgb).abs().mean())

    before = image_error(model0)
    with contextlib.redirect_stdout(io.StringIO()):
        model, completed = run(dataset, config, output/"synthetic_run")
    after = image_error(model)
    rows = evaluate(model, dataset, output/"synthetic_evaluation", "reference", videos=False)
    receipt = dict(status="PASSED_CPU_ONLY", test_count=result.testsRun, test_cases=names,
                   syntax_files=syntax, torch_version=torch.__version__, interpreter=sys.executable,
                   cuda_initialized=torch.cuda.is_initialized(), GPU_started=False, SSH_attempted=False, subagents_used=False,
                   CPU_fixture=dict(completed, frames=len(dataset.rows), initial_image_mae=before, final_image_mae=after,
                                    image_error_improved=after < before, denominator=24*24*3,
                                    continuous_edit_PNGs=40, delete_all_shadow_off_max=max(r["delete_all_shadow_off_max"] for r in rows),
                                    fixed_static_max=max(r["fixed_static_max"] for r in rows)),
                   real_scene_GPU_updates=0, actual_remote_inputs_checked=False, CUDA_ABI_checked=False,
                   reference_contract=preflight(dataset, config), human_verdict=None,
                   elapsed_seconds=time.monotonic()-started)
    if receipt["cuda_initialized"] or not after < before:
        raise RuntimeError("CPU smoke 未达到预期约束或损失改善；不应交付为通过")
    write(output/"cpu_validation.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
