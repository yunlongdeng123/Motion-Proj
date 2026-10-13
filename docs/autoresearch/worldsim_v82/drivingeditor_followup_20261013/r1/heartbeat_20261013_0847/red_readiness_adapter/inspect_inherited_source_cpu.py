#!/usr/bin/env python3
"""只读核对原GMC CPU模板与独立prepare所用真实源；不生成prepared receipt。"""
import ast
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
root = Path("/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013")
matrix = Path("/root/autodl-tmp/drivingeditor_repro/campaign/hybrid30k_20261012")
sys.path.insert(0, str(root / "tools/followup"))
from contracts import read_json, require, validate_gn11_config
from prepare_followup import source_contract, worker_source

source_contract(root / "frozen_repository")
validate_gn11_config(read_json(matrix / "configs/GN11.json"))
worker = worker_source((matrix / "tools/run_matrix.py").read_text(encoding="utf-8"))
require((matrix / "tools/memory_monitor.py").is_file(), "真实memory monitor缺失")
tree = ast.parse((root / "tools/prepare_scene0008_cpu.py").read_text(encoding="utf-8"), feature_version=(3, 9))
functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
require({"import_cpu_reader", "cpu_views"}.issubset(functions), "成功GMC CPU模板的真实入口缺失")
for path in ("red_readiness_cpu.py", "inspect_inherited_source_cpu.py"):
    ast.parse((Path(__file__).parent / path).read_text(encoding="utf-8"), feature_version=(3, 9))
require("torch" not in sys.modules and not any(n.startswith("tensorflow") for n in sys.modules), "只读源检查不应导入框架")
print(json.dumps({"status": "PASS_READONLY_INHERITED_SOURCE_CONTRACTS", "python_version": sys.version.split()[0],
                  "actual_frozen_source_contract": True, "actual_GN11_config_contract": True,
                  "actual_worker_extraction_python39_parse": True, "actual_GMC_CPU_template_entries": sorted({"import_cpu_reader", "cpu_views"}),
                  "matrix_tools": str(matrix / "tools"), "training_source_prepared": False,
                  "Torch_imported": False, "TF_imported": False, "GPU_used": False,
                  "training_started": False, "files_written": False}, ensure_ascii=False, indent=2))
