"""将相似路面高斯迁移追加到原道路报告，保留55条旧实验记录。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
BATCH_ID = "gaussian_transfer"
BATCH_LABEL = "相似路面高斯迁移"
RAWS = ("86", "110", "134", "179", "198")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
FILE_EXTENSIONS = IMAGE_EXTENSIONS | {".json", ".csv", ".npz", ".md"}
BEGIN = "<!-- GAUSSIAN_TRANSFER_REPORT_BEGIN -->"
END = "<!-- GAUSSIAN_TRANSFER_REPORT_END -->"



def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def extract_data(html: str):
    found = re.search(r"\bconst\s+D\s*=\s*", html)
    if not found:
        raise ValueError("既有报告没有内嵌D数据")
    data, length = json.JSONDecoder().raw_decode(html[found.end():])
    return data, found.end(), found.end() + length


def image_size(path: Path):
    if path.suffix.lower() == ".svg":
        return None
    from PIL import Image
    with Image.open(path) as image:
        return list(image.size)


class Assets:
    def __init__(self, report_root: Path, dataset_root: Path, prefix: str):
        self.report_root = report_root.resolve()
        self.dataset_root = dataset_root.resolve()
        self.prefix = Path(prefix)
        if self.prefix.is_absolute() or ".." in self.prefix.parts or not self.prefix.parts:
            raise ValueError("asset-prefix须为报告内相对目录")
        self.copied = {}
        self.files = {}

    def find(self, value: str) -> Path:
        if value.startswith(("http://", "https://", "data:", "file:")):
            raise ValueError("报告资源须为本地文件: " + value[:160])
        raw = Path(value)
        candidates = [raw] if raw.is_absolute() else [self.dataset_root / raw, self.report_root / raw]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError("缺少报告资源: " + value)

    def add(self, value: str) -> str:
        source = self.find(value)
        if source.is_relative_to(self.report_root):
            target = source
        else:
            try:
                relative = source.relative_to(self.dataset_root)
            except ValueError:
                relative = Path("external") / (sha(source)[:12] + "_" + source.name)
            target = (self.report_root / self.prefix / relative).resolve()
            if not target.is_relative_to(self.report_root):
                raise ValueError("资源目标越出报告目录")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target != source and (not target.exists() or sha(target) != sha(source)):
                shutil.copy2(source, target)
                self.copied[target.relative_to(self.report_root).as_posix()] = str(source)
        relative = target.relative_to(self.report_root).as_posix()
        self.files[relative] = {"sha256": sha(target), "bytes": target.stat().st_size}
        return relative

    def tree(self, value):
        if isinstance(value, dict):
            return {key: self.tree(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.tree(item) for item in value]
        if isinstance(value, str) and Path(value).suffix.lower() in FILE_EXTENSIONS:
            return self.add(value)
        return value


def adapt(base_index, payload_path, output_html, *, dataset_root=None, report_root=None,
          asset_prefix="gaussian_transfer", audit_json=None, extra_evidence=None):
    base_index, payload_path, output_html = [Path(path).resolve() for path in (base_index, payload_path, output_html)]
    root = Path(report_root or base_index.parent).resolve()
    if output_html.parent != root:
        raise ValueError("输出HTML须与report-root同目录，保持原资源路径")
    original_html = base_index.read_text(encoding="utf-8-sig")
    data, start, end = extract_data(original_html)
    catalog = copy.deepcopy(data["catalog"])
    catalog["batches"] = [item for item in catalog["batches"] if item["id"] != BATCH_ID]
    for key in ("models", "rows", "gaps", "newly_covered_rows"):
        catalog[key] = [item for item in catalog.get(key, []) if item.get("batch") != BATCH_ID
                        and not item.get("id", "").startswith(BATCH_ID + "::")]
    if len(catalog["models"]) != 55:
        raise ValueError("基准报告须为原55条记录；本模块不接入旧多源蒸馏组")
    frozen = {key: canonical(catalog.get(key, [])) for key in ("batches", "models", "rows", "gaps", "newly_covered_rows")}
    old_counts = {key: len(catalog.get(key, [])) for key in frozen}
    payload = json.loads(payload_path.read_text(encoding="utf-8-sig"))
    assets = Assets(root, Path(dataset_root or payload_path.parent), asset_prefix)
    new_models = []
    for original in payload["models"]:
        model = copy.deepcopy(original)
        original_id = str(model["id"]).removeprefix(BATCH_ID + "::")
        model.update(id=BATCH_ID + "::" + original_id, original_id=original_id, batch=BATCH_ID)
        model.setdefault("name", original_id)
        model.setdefault("renderer", "actual1152_native3DGS")
        model["frames"] = {str(raw): views for raw, views in model["frames"].items()}
        if set(model["frames"]) != set(RAWS):
            raise ValueError("每阶段须提供五帧真实渲染: " + original_id)
        model["frame_sizes"] = {}
        for raw, frame in model["frames"].items():
            if not frame.get("road") or not frame.get("joint"):
                raise ValueError("缺少道路或联合渲染")
            model["frame_sizes"][raw] = {}
            for key, path in list(frame.items()):
                if isinstance(path, str) and Path(path).suffix.lower() in IMAGE_EXTENSIONS:
                    frame[key] = assets.add(path)
                    model["frame_sizes"][raw][key] = image_size(root / frame[key])
        new_models.append(model)
    identifiers = {model["original_id"] for model in new_models}
    if not {"parent", "transfer"}.issubset(identifiers) or len(identifiers) != len(new_models):
        raise ValueError("须提供唯一parent和transfer阶段")
    patches = copy.deepcopy(payload.get("source_patches", {}))
    for patch in patches.values():
        patch["frames"] = assets.tree(patch.get("frames", {}))
    scope = payload.get("scope", "相似路块推断；保距配准源高斯纹理，先零步验收，再局部接缝拟合。")
    batch = {"id": BATCH_ID, "label": BATCH_LABEL, "kind": "gaussian_patch_transfer", "count": len(new_models),
             "scope": scope, "metric_scope": scope, "teacher": {"frames": {}}, "geometry_support": {},
             "defaults": [BATCH_ID + "::parent", BATCH_ID + "::transfer"],
             "final_model_id": BATCH_ID + "::" + ("appearance" if "appearance" in identifiers else "transfer"),
             "source_patches": patches, "source_layers": assets.tree(payload.get("source_layers", {"frames": {}})),
             "diagnostics": copy.deepcopy(payload.get("diagnostics", {})), "summary": copy.deepcopy(payload.get("summary", {})),
             "resources": copy.deepcopy(payload.get("resources", {})),
             "architecture": assets.add(str(HERE / "architecture_components.svg")),
             "new_diffusion_calls": 0}
    if batch["diagnostics"].get("visual_contact_sheet"):
        batch["diagnostics"]["visual_contact_sheet"] = assets.add(batch["diagnostics"]["visual_contact_sheet"])
    catalog["batches"].append(batch)
    catalog["models"].extend(new_models)
    for original in payload.get("gaps", []):
        row = copy.deepcopy(original)
        original_id = str(row["id"]).removeprefix(BATCH_ID + "::")
        row.update(id=BATCH_ID + "::" + original_id, original_id=original_id, batch=BATCH_ID)
        catalog["gaps"].append(row)
    catalog["default_pair"] = batch["defaults"]
    catalog["metric_recomputed"] = False
    for key, expected in frozen.items():
        restored = [item for item in catalog[key] if item.get("batch") != BATCH_ID
                    and (key != "batches" or item.get("id") != BATCH_ID)]
        if canonical(restored) != expected:
            raise AssertionError("原实验记录被改动: " + key)
    data["catalog"] = catalog
    data["gaussian_transfer_evidence"] = assets.tree(payload.get("evidence", []))
    data["gaussian_transfer_evidence"].insert(0, {"name": "高斯迁移完整数据", "path": assets.add(str(payload_path))})
    for path in extra_evidence or []:
        path = Path(path).resolve()
        data["gaussian_transfer_evidence"].append({"name": path.stem, "path": assets.add(str(path))})
    data["gaussian_transfer_disclaimer"] = payload.get("disclaimer")
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html = original_html[:start] + encoded + original_html[end:]
    html = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), "", html, flags=re.S)
    script = (HERE / "gaussian_transfer_viewer.js").read_text(encoding="utf-8")
    block = BEGIN + '\n<script id="gaussian-transfer-report-script">\n' + script + "\n</script>\n" + END
    close = html.rfind("</html>")
    html = html[:close] + block + "\n" + html[close:] if close >= 0 else html + block
    output_html.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_html.with_name(output_html.name + ".gaussian-transfer.tmp")
    temporary.write_text(html, encoding="utf-8")
    os.replace(temporary, output_html)
    audit = {"schema": "gaussian_transfer_report_integration.v1", "passed": True,
             "base_index": str(base_index), "output_html": str(output_html), "output_sha256": sha(output_html),
             "payload": str(payload_path), "payload_sha256": sha(payload_path),
             "old_record_counts": old_counts, "old_record_SHA256": frozen, "old_records_unchanged": True,
             "records": len(catalog["models"]), "batches": len(catalog["batches"]),
             "new_stages": [model["original_id"] for model in new_models], "new_diffusion_calls": 0,
             "default_pair": catalog["default_pair"], "assets": assets.files, "copied_assets": assets.copied,
             "old_metrics_recomputed": False}
    if audit_json:
        Path(audit_json).parent.mkdir(parents=True, exist_ok=True)
        Path(audit_json).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--asset-prefix", default="gaussian_transfer")
    parser.add_argument("--audit-json", type=Path)
    args = parser.parse_args()
    result = adapt(args.base_index, args.payload, args.output_html, dataset_root=args.dataset_root,
                   report_root=args.report_root, asset_prefix=args.asset_prefix, audit_json=args.audit_json)
    print(json.dumps({key: result[key] for key in ("passed", "records", "batches", "new_stages", "old_records_unchanged", "output_html")}, ensure_ascii=False))


