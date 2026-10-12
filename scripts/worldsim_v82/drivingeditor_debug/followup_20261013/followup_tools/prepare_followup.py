"""复制小型冻结源码并提取原worker，保留训练/内存/RNG实现；CPU准备不需要gate passed。"""
import ast
import difflib
from pathlib import Path
import shutil

from contracts import below, require, write_new

REQUIRED_SOURCE = ("train.py", "utils/paper_checkpoint.py", "utils/hybrid30k_runtime.py",
                   "utils/hybrid30k_losses.py", "utils/hybrid30k_metrics.py",
                   "utils/hybrid30k_static_time.py", "scene/paper_sgcm.py",
                   "scene/gaussian_model.py", "scene/hybrid_sgcm_helpers.py")


def worker_source(text):
    tree = ast.parse(text)
    names = ("install_memory_hooks", "write", "run_arm")
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    require(all(name in functions for name in names), "当前matrix worker结构不符，拒绝猜patch")
    body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    body += [functions[name] for name in names]
    source = ast.unparse(ast.Module(body=body, type_ignores=[])) + "\n"
    seed = "training.seed_everything(0)"
    require(source.count(seed) == 1, "真实seed初始化位置非唯一")
    lines = []
    for line in source.splitlines():
        lines.append(line)
        if line.strip() == seed:
            lines.append(line[:len(line) - len(line.lstrip())] + "bind_training_contract(training, cfg)")
    source = "\n".join(lines) + "\n"
    anchor = "time_control['anchor_pose_frame'] != '0017010'"
    require(source.count(anchor) == 1, "旧anchor检查非唯一，需review当前源码")
    source = source.replace(anchor, "time_control['anchor_pose_frame'] != cfg['followup_static_anchor_frame']")
    done = "status.update(status='complete', iteration=30000,"
    require(source.count(done) == 1, "原worker完成位置非唯一")
    lines = []
    for line in source.splitlines():
        if line.lstrip().startswith(done):
            lines.append(line[:len(line) - len(line.lstrip())] + "validate_completion(output, cfg)")
        lines.append(line)
    source = "\n".join(lines) + "\n"
    require("0017010" not in source and "def main(" not in source and "PARENT" not in source,
            "独立worker仍含scene17/four-arm入口")
    ast.parse(source, feature_version=(3, 9))
    return '"""由原matrix worker提取三个函数；无原main、旧ROOT/scene17常量。"""\n' + source


def copy_code(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    require(target != source and source not in target.parents and target not in source.parents,
            "新training_source不能覆盖或递归复制原source")
    require(not target.exists(), "training_source已存在，拒绝覆盖准备证据")
    ignored = {".git", "__pycache__", ".pytest_cache", "runs", "outputs", "build", "dist"}
    suffixes = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".cu", ".cpp", ".h", ".cuh", ".sh"}
    selected = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink() or any(part in ignored for part in path.relative_to(source).parts):
            continue
        if path.suffix not in suffixes and path.name not in ("LICENSE", "Makefile"):
            continue
        require(path.stat().st_size <= 16 * 1024**2, "拒绝复制大型非代码资产: " + str(path))
        selected.append(path.relative_to(source))
    require(selected, "冻结source没有可复制代码")
    target.mkdir(parents=True, exist_ok=False)
    for relative in selected:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    return [p.as_posix() for p in selected]


def source_contract(source):
    root = Path(source)
    require(all((root / name).is_file() for name in REQUIRED_SOURCE), "必须使用已整合GN11的完整小型frozen repository")
    train = (root / "train.py").read_text(encoding="utf-8")
    for marker in ("compose_hybrid_loss", "hybrid_sampler", "training_views=scene.getTrainCameras()",
                   "if iteration < args.densify_until_iter", "publish_checkpoint_event"):
        require(marker in train, "冻结训练源缺合同: " + marker)
    require("hybrid_sampler" in (root / "utils/paper_checkpoint.py").read_text(encoding="utf-8"),
            "完整checkpoint缺独立sampler")
    require("guard_additions" in (root / "scene/paper_sgcm.py").read_text(encoding="utf-8")
            and "capture_sgcm_optimizer" in (root / "scene/gaussian_model.py").read_text(encoding="utf-8"),
            "corrected SGCM/lineage未整合")
    for name in REQUIRED_SOURCE:
        ast.parse((root / name).read_text(encoding="utf-8"), feature_version=(3, 9))
    return True


def prepare(paths, config, spec, readiness_status):
    campaign = Path(paths["campaign_root"]).resolve()
    source = Path(paths["source_repository"]).resolve()
    tools = Path(paths["matrix_tools_dir"]).resolve()
    source_contract(source)
    worker = worker_source((tools / "run_matrix.py").read_text(encoding="utf-8"))
    require((tools / "memory_monitor.py").is_file(), "当前memory monitor缺失")
    target = below(campaign / "training_source" / spec["scene_id"], campaign)
    support = below(campaign / "training_support" / spec["scene_id"], campaign)
    config_path = below(campaign / "training_configs" / (spec["scene_id"] + "_GN11.json"), campaign)
    receipt_path = below(campaign / "training_metadata" / (spec["scene_id"] + "_prepared.json"), campaign)
    for path in (target, support, config_path, receipt_path):
        require(not path.exists(), "独立prepare目标已存在，拒绝覆写: " + str(path))
    here = Path(__file__).resolve().parent
    files = copy_code(source, target)
    old = (target / "utils/hybrid30k_static_time.py").read_text(encoding="utf-8")
    new = (here / "static_time_scene.py").read_text(encoding="utf-8")
    (target / "utils/hybrid30k_static_time.py").write_text(new, encoding="utf-8")
    support.mkdir(parents=True, exist_ok=False)
    (support / "matrix_worker.py").write_text(worker, encoding="utf-8")
    shutil.copy2(tools / "memory_monitor.py", support / "memory_monitor.py")
    patch = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                       fromfile="frozen/utils/hybrid30k_static_time.py",
                                       tofile="training_source/utils/hybrid30k_static_time.py"))
    (support / "source_delta.patch").write_text(patch, encoding="utf-8")
    write_new(config_path, config)
    receipt = {"status": "FOLLOWUP_CPU_SOURCE_PREPARED_GPU_BLOCKED", "scene_id": spec["scene_id"],
               "source_repository": str(source), "training_source": str(target),
               "matrix_tools_dir": str(tools), "training_support": str(support),
               "training_config": str(config_path), "copied_code_files": files,
               "changed_source_files": ["utils/hybrid30k_static_time.py"],
               "anchor_pose_frame": spec["anchor_frame"], "scene_cpu_status": readiness_status,
               "gpu_started": False, "gate_required_at_run": True,
               "actual_model_sgcm_runtime_binding": "pending until real --run",
               "restoration_verified": False, "paths": {k: str(Path(v).resolve()) for k, v in paths.items()}}
    write_new(receipt_path, receipt)
    return receipt


def verify_prepared(paths, config, spec):
    campaign = Path(paths["campaign_root"]).resolve()
    receipt_path = campaign / "training_metadata" / (spec["scene_id"] + "_prepared.json")
    from contracts import read_json, strict_dict_equal, DATA_IGNORED
    record = read_json(receipt_path)
    require(record.get("status") == "FOLLOWUP_CPU_SOURCE_PREPARED_GPU_BLOCKED"
            and record.get("gpu_started") is False and record.get("scene_id") == spec["scene_id"],
            "独立CPU prepare receipt不完整")
    for key in ("source_repository", "matrix_tools_dir", "campaign_root", "scene_spec", "readiness_file", "reference_config"):
        require(record["paths"].get(key) == str(Path(paths[key]).resolve()), "prepare路径绑定已改变: " + key)
    require(read_json(record["training_config"]) == config, "prepared config与实际候选不一致")
    source, target = Path(record["source_repository"]), Path(record["training_source"])
    for relative in record["copied_code_files"]:
        current = target / relative
        expected = (Path(__file__).parent / "static_time_scene.py"
                    if relative == "utils/hybrid30k_static_time.py" else source / relative)
        equal = (current.read_text(encoding="utf-8") == expected.read_text(encoding="utf-8")
                 if relative == "utils/hybrid30k_static_time.py" else current.read_bytes() == expected.read_bytes())
        require(current.is_file() and equal,
                "冻结源码在prepare后发生变化: " + relative)
    expected_worker = worker_source((Path(paths["matrix_tools_dir"]) / "run_matrix.py").read_text(encoding="utf-8"))
    support = Path(record["training_support"])
    require((support / "matrix_worker.py").read_text(encoding="utf-8") == expected_worker,
            "worker来源在prepare后发生变化")
    require((support / "memory_monitor.py").read_bytes() ==
            (Path(paths["matrix_tools_dir"]) / "memory_monitor.py").read_bytes(), "memory monitor改变")
    return record
