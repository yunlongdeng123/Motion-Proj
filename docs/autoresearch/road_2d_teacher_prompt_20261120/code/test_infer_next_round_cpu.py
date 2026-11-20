"""针对真实本地快照的计划、12门、截断与显式GPU边界检查；不调用远端。"""
from pathlib import Path
import copy
import json
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import infer_next_round as entry

ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "work/teacher_audit/pilot_manifest_r2.json"
HISTORY = ROOT / "work/gpu_20261119/predictions.json"


class NextRoundCPUContract(unittest.TestCase):
    def make_plan(self, root):
        config = entry.read(entry.DEFAULT_CONFIG)
        path = root / "config.json"
        entry.write(path, config)
        return entry.prepare(path, BASELINE, HISTORY, root / "plan")

    def test_real_matrix_twelve_full1152_replays_and_frozen_model(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            manifest = self.make_plan(root)
            m = entry.plan_check(manifest)
            self.assertEqual(len(m["jobs"]), 60)
            combos = {(j["case_id"], j["seed"], j["prompt_id"]) for j in m["jobs"]}
            self.assertEqual(len(combos), 60)
            self.assertTrue(all(j["historical_replay_expected"] for j in m["jobs"][:12]))
            self.assertTrue(all(not j["historical_replay_expected"] for j in m["jobs"][12:]))
            self.assertEqual({j["seed"] for j in m["jobs"][:12]}, {101, 202})
            self.assertTrue(all("full1152_T0" in j["historical_job_id"] for j in m["jobs"][:12]))
            self.assertTrue(all("legacy_B_expected" not in case for case in m["cases"]))
            m["checkpoint"]["sha256"] = "wrong"
            entry.write(manifest, m)
            with self.assertRaisesRegex(entry.ContractError, "冻结模型/源码"):
                entry.plan_check(manifest)

    def test_changed_seed_or_method_is_not_silently_accepted(self):
        baseline = entry.read(BASELINE)
        for field, value in (("seeds", [101, 202, 303, 404, 999]), ("maximum_diffusion_calls", 61), ("automatic_candidate_selection", True)):
            config = copy.deepcopy(entry.read(entry.DEFAULT_CONFIG))
            config[field] = value
            with self.assertRaises(entry.ContractError):
                entry.validate_config(config, baseline)

    def test_non_gpu_run_only_checks_plan_and_does_not_create_GPU_directory(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            manifest = self.make_plan(root)
            output = root / "MUST_NOT_EXIST_GPU"
            process = subprocess.run([sys.executable, str(Path(entry.__file__)), "run", "--manifest", str(manifest), "--out", str(output)], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            self.assertTrue(result["local_plan_passed"])
            self.assertFalse(result["runtime_preflight_passed"])
            self.assertFalse(result["GPU_model_loaded"])
            self.assertFalse(output.exists())

    def test_tokenizer_over77_and_replay_mismatch_stop(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            fake_tokenizer = types.SimpleNamespace(model_max_length=77, encode=lambda *a, **kw: list(range(78)))
            fake_module = types.SimpleNamespace(CLIPTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: fake_tokenizer))
            with patch.dict(sys.modules, {"transformers": fake_module}), patch.object(entry.importlib.util, "find_spec", return_value=object()):
                with self.assertRaisesRegex(entry.ContractError, "超过77"):
                    entry.token_check([{"id": "candidate", "text": "fixture only"}], root, required=True)
            expected, folder = {}, root / "generated"
            folder.mkdir()
            for name in ("raw_model", "raw_native", "final_native"):
                image = Image.fromarray(np.full((2, 2, 3), 50, np.uint8))
                old = root / (name + "_old.png")
                image.save(old); image.save(folder / (name + ".png"))
                expected[name] = entry.pin(old)
            job = dict(id="first_full1152_replay", historical_replay_expected=True, historical_expected=expected)
            self.assertTrue(all(entry.replay_exact_or_stop(job, folder).values()))
            Image.fromarray(np.full((2, 2, 3), 51, np.uint8)).save(folder / "final_native.png")
            with self.assertRaisesRegex(entry.ContractError, "立即停止"):
                entry.replay_exact_or_stop(job, folder)


if __name__ == "__main__":
    unittest.main()
