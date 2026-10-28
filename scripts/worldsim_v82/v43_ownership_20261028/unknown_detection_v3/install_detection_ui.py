"""把目标检测证据接到原第一场景；不修改共用模板或历史实验。

用法：python work/v43_unknown_detection_20261028/install_detection_ui.py \
    --delivery /path/to/delivery.json

delivery.json 的最小合同：
{
  "schema": "target-detection-ui-v1", "context": "147053...",
  "target_track": "...", "static_total": 918850,
  "source_checkpoint": "冻结检查点路径", "checkpoint_frozen": true,
  "geometry_base_group": "v43_actor_10000",
  "actor": {"cloud": "v43:10000", "role_default": 1,
            "row_namespace": "frozen_v43_actor10k"},
  "globalsource_frames": [1, 2, 3],
  "frames": {"109": {
    "evidence": "perraw/109.json",  // 也支持 packed 对象或 npz
    "images": {"baseline": "baseline/109.png", ...},
    "render_receipts": {"baseline": {
      "status": "complete", "renderer": "native_gaussian", "source_raw": 109
    }, ...}
  }}
}
每帧 evidence：ids / static_role / reason_bits / body_mass / shadow_mass /
bg_mass / total_mass / target_fraction / shadow_fraction；后六项可按实际导出提供。
static_role: 0本目标外或未纳入，1car，2目标U，3shadow_U，4mixed_U。
bg_protected / shadow_effect 是独立布尔数组；mixed_U仅在显式shadow_effect=true时增加阴影effect，角色仍为4。
pixel_role 使用另一字典：0outside,1body,2boundary,3shadow,4mixed。
可选pixel_shadow_effect为480×320独立布尔/u8；紫色effect仅覆盖显式标记，允许与pixel_role=1并存，不改车身类别。
三剥离默认删除角色 car=[1], u=[2,3,4], car_u=[1,2,3,4]。
可通过 removal_roles 覆盖；car 组只删除显式 actor.cloud，不删除其他动态资产。
所有原生 PNG 必须有同帧 native_gaussian receipt；缺项保存为 pending。
"""

from __future__ import annotations

import argparse
import array
import base64
import copy
import json
import re
import shutil
import struct
import sys
import zlib
from pathlib import Path


BASE = Path(__file__).resolve().parents[2]
OUTPUT = BASE / "outputs/GS_Scene_Diagnostics_20261027"
CONTEXT = "14705303724557273004_3105_000_3125_000"
CASES = ("baseline", "remove_car", "remove_u", "remove_car_u")
TITLES = {
    "baseline": "检测诊断 · 冻结Actor10K基线",
    "remove_car": "检测诊断 · 只去car",
    "remove_u": "检测诊断 · 只去本目标U",
    "remove_car_u": "检测诊断 · 去car＋本目标U",
}
DEFAULT_REMOVAL = {"baseline": [], "remove_car": [1], "remove_u": [2, 3, 4], "remove_car_u": [1, 2, 3, 4]}
GROUP_PREFIX = "v43_target_detection_"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_assignment(path: Path, name: str):
    source = path.read_text(encoding="utf-8-sig")
    match = re.search(r"(?:window\.)?" + re.escape(name) + r"\s*=\s*", source)
    if not match:
        raise ValueError(f"{path}: 找不到 {name}")
    return json.JSONDecoder().raw_decode(source[match.end():])[0]


def unpack(payload):
    """只用于验证/筛行；输出仍沿用原 packed 数据。"""
    if not isinstance(payload, dict):
        raise ValueError("evidence 必须是数组字典或 packed 对象")
    if "arrays" not in payload:
        return payload
    blob = zlib.decompress(base64.b64decode(payload["data"]))
    types = {"<i4": "i", "<u4": "I", "<u2": "H", "|u1": "B", "|b1": "B", "<f4": "f", "<f8": "d"}
    result = {}
    for key, meta in payload["arrays"].items():
        dtype = meta["dtype"]
        raw = blob[meta["offset"]:meta["offset"] + meta["bytes"]]
        if len(raw) != meta["bytes"]:
            raise ValueError(f"{key}: packed 长度不足")
        if dtype == "<f2":
            result[key] = [v[0] for v in struct.iter_unpack("<e", raw)]
        else:
            if dtype not in types:
                raise ValueError(f"{key}: 不支持 {dtype}")
            values = array.array(types[dtype])
            values.frombytes(raw)
            if sys.byteorder != "little" and values.itemsize > 1:
                values.byteswap()
            result[key] = values
    return result


def load_evidence(value, delivery_root: Path):
    if isinstance(value, str):
        path = resolve_source(value, delivery_root)
        if path.suffix.lower() == ".npz":
            import numpy as np

            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
            blob, metadata = bytearray(), {}
            for key, values in arrays.items():
                if values.dtype.kind == "b":
                    values = values.astype("uint8")
                if values.dtype.kind in "iu" and values.dtype.itemsize > 4:
                    minimum, maximum = (int(values.min()), int(values.max())) if values.size else (0, 0)
                    if -(2**31) <= minimum and maximum < 2**31:
                        values = values.astype("int32")
                    elif minimum >= 0 and maximum < 2**32:
                        values = values.astype("uint32")
                    else:
                        raise ValueError(f"{key}: 整数超出UI可精确表达的32位范围")
                values = values.astype(values.dtype.newbyteorder("<"), copy=False)
                raw = values.tobytes()
                metadata[key] = dict(offset=len(blob), bytes=len(raw), dtype=values.dtype.str, shape=list(values.shape))
                blob.extend(raw)
            return dict(arrays=metadata, data=base64.b64encode(zlib.compress(blob, 6)).decode("ascii"))
        return read_json(path)
    return value


def resolve_source(value, root: Path):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def validate_evidence(payload, static_total: int, raw: int):
    values = unpack(payload)
    ids, roles = values.get("ids"), values.get("static_role")
    if ids is None or roles is None or len(ids) != len(roles):
        raise ValueError(f"raw{raw}: ids/static_role 必须同长")
    previous = -1
    for row, role in zip(ids, roles):
        if int(row) != row or row <= previous or not 0 <= row < static_total:
            raise ValueError(f"raw{raw}: ids 必须是 0—{static_total - 1} 的升序唯一原static行号")
        if int(role) != role or role not in range(5):
            raise ValueError(f"raw{raw}: static_role 必须为0—4")
        previous = row
    row_fields = ("reason_bits", "body_mass", "shadow_mass", "bg_mass", "total_mass", "target_fraction", "shadow_fraction", "bg_protected", "shadow_effect", "body_frames", "shadow_frames", "bg_frames")
    for key in row_fields:
        if key in values and len(values[key]) != len(ids):
            raise ValueError(f"raw{raw}: {key} 与 ids 长度不符")
    for key in ("bg_protected", "shadow_effect"):
        if key in values and any(v not in (0, 1) for v in values[key]):
            raise ValueError(f"raw{raw}: {key} 必须是独立布尔量")
    if "pixel_role" in values:
        if len(values["pixel_role"]) != 480 * 320 or any(v not in range(5) for v in values["pixel_role"]):
            raise ValueError(f"raw{raw}: pixel_role 必须为480×320，值0—4")
    if "pixel_shadow_effect" in values:
        if len(values["pixel_shadow_effect"]) != 480 * 320 or any(v not in (0, 1) for v in values["pixel_shadow_effect"]):
            raise ValueError(f"raw{raw}: pixel_shadow_effect 必须为480×320独立布尔/u8，值0/1")
    offsets, sources = values.get("source_raw_offsets"), values.get("source_raw_ids")
    if offsets is not None or sources is not None:
        if offsets is None or sources is None or len(offsets) != len(ids) + 1 or offsets[0] != 0 or offsets[-1] != len(sources):
            raise ValueError(f"raw{raw}: source raw CSR 长度不符")
        if any(a > b for a, b in zip(offsets, offsets[1:])):
            raise ValueError(f"raw{raw}: source raw CSR offsets 非递增")
    return values


def png_size(path: Path):
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"不是 PNG: {path}")
    return struct.unpack(">II", header[16:24])


def receipt_for(frame, case):
    value = frame.get("images", {}).get(case)
    inline = value.get("receipt") if isinstance(value, dict) else None
    return inline or frame.get("render_receipts", {}).get(case) or frame.get("render_receipt")


def native_image(frame, case, raw: int, delivery_root: Path):
    value = frame.get("images", {}).get(case)
    path_value = value.get("path") if isinstance(value, dict) else value
    if not path_value:
        return None, "pending · 未提供原生RGB"
    path = resolve_source(path_value, delivery_root)
    if not path.is_file():
        return None, "pending · 原生RGB文件未到达"
    receipt = receipt_for(frame, case)
    if not isinstance(receipt, dict) or receipt.get("status") not in ("complete", "completed", "done") or receipt.get("renderer") != "native_gaussian":
        return None, "pending · 缺原生执行凭据"
    if int(receipt.get("source_raw", -1)) != raw:
        raise ValueError(f"raw{raw}/{case}: receipt 源raw不匹配")
    if path.stem.isdigit() and int(path.stem) != raw:
        raise ValueError(f"raw{raw}/{case}: 文件帧号不匹配 {path.name}")
    if png_size(path) != (480, 320):
        raise ValueError(f"raw{raw}/{case}: 原生RGB须为480×320，不能静默重采样")
    return path, "complete · 同帧原生RGB"


def write_image(source: Path, scene: Path, relative: Path):
    target = scene / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    uri = "data:image/png;base64," + base64.b64encode(source.read_bytes()).decode("ascii")
    target.with_name(target.name + ".js").write_text("window.IMAGE_DATA[" + json.dumps(relative.as_posix()) + "]=" + json.dumps(uri) + ";\n", encoding="utf-8")
    return relative.as_posix()


def original_target(state, geometry, raw):
    reference = geometry.get("motion_no_edit", {}).get(raw) or geometry.get("reconstruction_30000", {}).get(raw) or state
    for obj in reference.get("objects", []):
        if obj.get("key") == "target" and obj.get("corners_local"):
            return dict(target=True, label="TARGET 原位参考", corners_local=obj["corners_local"])
    return None


def replace_block(source: str, name: str, body: str):
    start, end = f"<!-- {name}-start -->", f"<!-- {name}-end -->"
    block = start + body + end
    if start in source:
        return re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _match: block, source, count=1, flags=re.S)
    return None


def build(delivery_path: Path, scene: Path, shared_entry: Path | None = None):
    delivery = read_json(delivery_path)
    delivery_root = delivery_path.parent
    if not str(delivery.get("schema", "")).startswith("target-detection"):
        raise ValueError("delivery 缺 target-detection schema")
    scene_data = read_assignment(scene / "scene.js", "SCENE_DATA")
    context = scene_data["context"]
    if delivery.get("context") != context or str(delivery.get("target_track")) != str(scene_data.get("target_track")):
        raise ValueError("delivery 场景或目标track与当前页面不一致")
    if delivery.get("static_total") != 918850:
        raise ValueError("本安装器只接受第一场景的918850行原static")
    if not delivery.get("source_checkpoint") or delivery.get("checkpoint_frozen") is not True:
        raise ValueError("必须记录 source_checkpoint 且 checkpoint_frozen=true")
    actor = delivery.get("actor", {})
    if not actor.get("cloud") or actor.get("role_default") != 1 or not actor.get("row_namespace"):
        raise ValueError("必须显式指定冻结Actor的独立cloud/row_namespace及role_default=1")
    original_geometry = read_assignment(scene / "edit_geometry.js", "EDIT_GEOMETRY")["groups"]
    v43 = read_json(scene / "v43_groups_data.json")
    geometry = dict(original_geometry, **v43.get("geometry", {}))
    base_group = delivery.get("geometry_base_group", "v43_actor_10000")
    if base_group not in geometry or actor["cloud"] not in v43.get("clouds", {}):
        raise ValueError("冻结Actor或几何基线尚未安装；不能用原30K资产替代")
    removal = copy.deepcopy(DEFAULT_REMOVAL)
    removal.update(delivery.get("removal_roles", {}))
    for case in CASES:
        if any(role not in range(1, 5) for role in removal[case]):
            raise ValueError("removal_roles 只可包含目标角色1—4，不能删除本目标外role0")
    requested = sorted({int(v) for v in delivery.get("raws", scene_data["frames"].keys())})
    if not requested or any(str(raw) not in scene_data["frames"] for raw in requested):
        raise ValueError("raws 必须是该共享场景中实际存在的源帧")
    manifest = {key: value for key, value in delivery.items() if key not in ("frames", "groups", "geometry", "clouds")}
    manifest.update(schema="target-detection-ui-v1", frames={}, groups=[], geometry={}, removal_roles=removal)
    evidence_files, image_files = {}, []
    receipts = {case: {"complete": 0, "pending": 0} for case in CASES}
    for raw in requested:
        key = str(raw)
        frame = delivery.get("frames", {}).get(key, {})
        if int(frame.get("source_raw", raw)) != raw:
            raise ValueError(f"raw{raw}: 帧证据source_raw不匹配")
        entry = dict(source_raw=raw, images={}, render_status={}, render_receipts={})
        values = None
        if frame.get("evidence") is not None:
            packed = load_evidence(frame["evidence"], delivery_root)
            values = validate_evidence(packed, delivery["static_total"], raw)
            evidence_path = f"target_detection/evidence/{raw}.js"
            evidence_files[evidence_path] = "window.TARGET_DETECTION_FRAMES=window.TARGET_DETECTION_FRAMES||{};window.TARGET_DETECTION_FRAMES[" + json.dumps(key) + "]=" + json.dumps(packed, separators=(",", ":"), ensure_ascii=False) + ";\n"
            entry.update(evidence_path=evidence_path, evidence_rows=len(values["ids"]), status="evidence_ready")
        else:
            entry.update(status="pending", evidence_rows=0)
        for field in ("globalsource_frames", "source_note", "thresholds", "metrics"):
            if field in frame:
                entry[field] = frame[field]
        for case in CASES:
            image, status = native_image(frame, case, raw, delivery_root)
            entry["render_status"][case] = status
            receipts[case]["complete" if image else "pending"] += 1
            if image:
                relative = Path("assets/v43_target_detection") / case / f"{raw:03d}.png"
                image_files.append((image, relative))
                entry["images"][case] = relative.as_posix()
                entry["render_receipts"][case] = receipt_for(frame, case)
        manifest["frames"][key] = entry
        source = geometry[frame.get("geometry_base_group", base_group)].get(key)
        for case in CASES:
            gid = GROUP_PREFIX + case
            manifest["geometry"].setdefault(gid, {})
            if source is None or values is None:
                state = dict(supported=False, objects=[], note="pending：本源帧尚无对应检测证据/冻结Actor几何；不替换为邻帧或旧模型。")
            else:
                state = copy.deepcopy(source)
                state.pop("semantic_render_filter", None)
                state.pop("static_adapter", None)
                suppressed = set(state.get("static_suppressed_rows", []))
                suppressed.update(int(row) for row, role in zip(values["ids"], values["static_role"]) if role in removal[case])
                state["static_suppressed_rows"] = sorted(suppressed)
                state["objects"] = [obj for obj in state.get("objects", []) if not (1 in removal[case] and obj.get("cloud") == actor["cloud"])]
                for obj in state["objects"]:
                    if obj.get("cloud") == actor["cloud"]:
                        obj["render_only"] = True
                target = original_target(source, original_geometry, key)
                state["reference_boxes"] = [target] if target else []
                state["target_detection"] = dict(case=case, source_raw=raw, removed_roles=removal[case], actor_cloud=actor["cloud"])
                state["note"] = "冻结检查点的目标专属检测剥离；role0仅本目标外/未纳入，bg_protected独立。阴影为vehicle_shadow_candidate effect，不重编码car/road/19类语义。三维只显示真实高斯中心，RGB执行状态见同帧卡片。"
            manifest["geometry"][gid][key] = state
    for case in CASES:
        gid = GROUP_PREFIX + case
        frames = {raw: dict(channels={"rgb": frame["images"][case]} if case in frame["images"] else {}, target_detection_case=case, render_status=frame["render_status"][case]) for raw, frame in manifest["frames"].items()}
        manifest["groups"].append(dict(id=gid, title=TITLES[case], category="v43", kind="v43_target_detection", frames=frames, summary="target_detection/install_receipt.json"))
    data_script = "window.TARGET_DETECTION_DELIVERY=" + json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + ";\n"
    data_script += "window.SCENE_DATA.groups=window.SCENE_DATA.groups.filter(g=>!window.TARGET_DETECTION_DELIVERY.groups.some(n=>n.id===g.id));window.SCENE_DATA.groups.push(...window.TARGET_DETECTION_DELIVERY.groups);Object.assign(window.EDIT_GEOMETRY.groups,window.TARGET_DETECTION_DELIVERY.geometry);\n"
    index_path = scene / "index.html"
    index = index_path.read_text(encoding="utf-8")
    data_block = '<script src="target_detection_data.js?v=target-detection-2"></script>'
    runtime_block = '<script src="../scene_target_detection.js?v=target-detection-2"></script>'
    updated = replace_block(index, "target-detection-data", data_block)
    if updated is None:
        updated, count = re.subn(r'(?=<script src="\.\./viewer\.js(?:\?[^\"]*)?"></script>)', '<!-- target-detection-data-start -->' + data_block + '<!-- target-detection-data-end -->', index, count=1)
        if count != 1:
            raise ValueError("找不到当前共享viewer脚本；未修改模板")
    final = replace_block(updated, "target-detection-runtime", runtime_block)
    if final is None:
        final = updated.replace("</body>", '<!-- target-detection-runtime-start -->' + runtime_block + '<!-- target-detection-runtime-end --></body>', 1)
    shared_updated = None
    if shared_entry is not None:
        shared = shared_entry.read_text(encoding="utf-8")
        link = '<div class="notice"><a href="' + context + '/index.html#targetDetection">第一场景：目标专属检测、阴影候选 effect 与同帧三组原生剥离 →</a>（未通过：整行误伤背景；冻结检查点，缺项为pending）</div>'
        shared_updated = replace_block(shared, "target-detection-link", link)
        if shared_updated is None:
            shared_updated = shared.replace('<main id="workspace">', '<main id="workspace"><!-- target-detection-link-start -->' + link + '<!-- target-detection-link-end -->', 1)
            if shared_updated == shared:
                raise ValueError("共享入口缺 workspace 注入位置")
    # 先完成全部合同/帧号/注入位置验证，再写入新的专属资产和入口。
    for relative, source in evidence_files.items():
        target = scene / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    for image, relative in image_files:
        write_image(image, scene, relative)
    (scene / "target_detection").mkdir(parents=True, exist_ok=True)
    (scene / "target_detection_data.js").write_text(data_script, encoding="utf-8")
    shutil.copy2(Path(__file__).with_name("scene_target_detection.js"), scene / "scene_target_detection.js")
    shutil.copy2(Path(__file__).with_name("scene_target_detection.js"), scene.parent / "scene_target_detection.js")
    index_path.write_text(final, encoding="utf-8")
    if shared_updated is not None:
        shared_entry.write_text(shared_updated, encoding="utf-8")
    receipt = dict(schema="target-detection-install-v1", context=context, requested_raws=requested, static_total=delivery["static_total"], actor=actor, source_checkpoint=delivery["source_checkpoint"], checkpoint_frozen=True, rendered=receipts, evidence_ready_frames=sum(frame["status"] == "evidence_ready" for frame in manifest["frames"].values()), semantic_map_modified=False, history_groups_preserved=True, shared_templates_modified=False, delivery_path=str(delivery_path.resolve()))
    (scene / "target_detection/install_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=OUTPUT / CONTEXT)
    parser.add_argument("--no-shared-entry", action="store_true", help="独立测试场景时不注入总入口")
    args = parser.parse_args()
    result = build(args.delivery.resolve(), args.scene.resolve(), None if args.no_shared_entry else OUTPUT / "index.html")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
