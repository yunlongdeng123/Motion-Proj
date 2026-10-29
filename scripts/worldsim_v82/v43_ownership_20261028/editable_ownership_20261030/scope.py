"""CPU通用局部范围。原body mask -> bbox -> 固定归一化蓝框 -> 本帧查询。

只定义屏幕范围，不生成全局semantic unknown。没有mask则skip，不能回退大域。
支持原R2与独立source-only两个显式arm，不隐式复用旧seed或删除RGB。
"""
import argparse
import base64
import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REFERENCE_BBOX = np.array([292., 176., 352., 223.])  # half-open，原raw113观测mask。
REFERENCE_BLUE = np.array([[265, 166], [265, 157], [291, 137], [319, 134], [341, 139],
                           [417, 139], [422, 147], [421, 212], [382, 244], [265, 244]], float)
REFERENCE_CONTACT = np.array([[276, 203], [357, 203], [357, 236], [276, 236]], float)
REFERENCE_CONTEXT = "14705303724557273004_3105_000_3125_000"
QUERY_MODES = ("source_body_contact", "r2_existing_saturation", "source_local_band")


def default_template():
    size = REFERENCE_BBOX[2:] - REFERENCE_BBOX[:2]
    return {"name": "user_blue113_bbox_affine_v1", "reference_context": REFERENCE_CONTEXT,
            "reference_raw": 113, "reference_bbox_xyxy": REFERENCE_BBOX.tolist(),
            "blue_polygon_normalized": ((REFERENCE_BLUE - REFERENCE_BBOX[:2]) / size).tolist(),
            "contact_polygon_normalized": ((REFERENCE_CONTACT - REFERENCE_BBOX[:2]) / size).tolist(),
            "display_top_offset_in_bbox_height": -8. / size[1],
            "rasterization": "numpy.rint polygon vertices; PIL filled polygon includes boundary pixels",
            "missing_mask_policy": "skip; no bbox/neighbor/predicted-mask/big-domain fallback",
            "source": "user screenshot blue polygon traced at raw113; original body bbox affine transport",
            "scope_semantics": "frame-local projected-center eligibility and local composition; not global semantic unknown"}


def _binary_body(mask):
    if mask is None:
        return None, "missing_body_mask"
    body = np.asarray(mask)
    if body.ndim != 2 or not body.size:
        return None, "invalid_body_mask_shape"
    if not np.isfinite(body).all():
        return None, "nonfinite_body_mask"
    unique = np.unique(body)
    if not np.all(np.isin(unique, [0, 1, 255])):
        return None, "body_mask_is_not_binary"
    body = body > 0
    if not body.any():
        return None, "empty_body_mask"
    return body, None


def mask_bbox(mask):
    body, reason = _binary_body(mask)
    if reason:
        raise ValueError(reason)
    y, x = np.where(body)
    return np.array([x.min(), y.min(), x.max() + 1, y.max() + 1], dtype=float)


def transport_polygon(normalized, bbox):
    bbox = np.asarray(bbox, dtype=float)
    if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
        raise ValueError("invalid_half_open_bbox")
    return np.asarray(normalized, dtype=float) * (bbox[2:] - bbox[:2]) + bbox[:2]


def raster_polygon(points, image_shape):
    h, w = image_shape
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("invalid_polygon")
    image = Image.new("1", (w, h))
    ImageDraw.Draw(image).polygon([tuple(p) for p in np.rint(points).astype(int)], fill=1)
    return np.asarray(image, dtype=bool)


def build_scope(body_mask, *, context, raw, frame_id=None, template=None,
                image_shape=None, body_source=None):
    """唯一有效bbox来源是给定原始二值body；缺失时返回显式skip。"""
    template = default_template() if template is None else dict(template)
    body, reason = _binary_body(body_mask)
    result = {"schema": "observed-bbox-local-scope-v1", "context": str(context), "raw": int(raw),
              "frame_id": frame_id, "template": template, "body_source": body_source,
              "status": "skipped" if reason else "ok", "skip_reason": reason,
              "automatic_semantic_unknown": False, "labels_for_training": False,
              "allowed": None, "user_contact_query": None, "display_scope": None, "body": None}
    if reason:
        return result
    if image_shape is not None and tuple(image_shape) != body.shape:
        result.update(status="skipped", skip_reason="mask_resolution_mismatch_no_resizing")
        return result
    h, w = body.shape
    box = mask_bbox(body)
    polygon = transport_polygon(template["blue_polygon_normalized"], box)
    contact_polygon = transport_polygon(template["contact_polygon_normalized"], box)
    allowed = raster_polygon(polygon, body.shape)
    contact = raster_polygon(contact_polygon, body.shape) & allowed
    # 与原R2的int()截断规则一致，不擅自换round/floor。
    top = int(box[1] + template["display_top_offset_in_bbox_height"] * (box[3] - box[1]))
    display = allowed & (np.arange(h)[:, None] >= top)
    if not allowed.any() or not display.any() or not contact.any():
        result.update(status="skipped", skip_reason="transported_scope_empty_after_image_clip")
        return result
    result.update(body=body, allowed=allowed, user_contact_query=contact, display_scope=display,
                  image_shape=[h, w], body_bbox_xyxy=box.tolist(), allowed_polygon_px=polygon.tolist(),
                  contact_polygon_px=contact_polygon.tolist(), display_top_y=top,
                  body_pixels=int(body.sum()), allowed_pixels=int(allowed.sum()),
                  contact_pixels=int(contact.sum()), display_pixels=int(display.sum()),
                  body_pixels_outside_allowed=int((body & ~allowed).sum()),
                  polygon_vertices_outside_image=int(((polygon[:, 0] < 0) | (polygon[:, 0] >= w) |
                                                      (polygon[:, 1] < 0) | (polygon[:, 1] >= h)).sum()))
    return result


def _chw(rgb, image_shape):
    rgb = np.asarray(rgb)
    h, w = image_shape
    if rgb.shape == (h, w, 3):
        rgb = rgb.transpose(2, 0, 1)
    if rgb.shape != (3, h, w) or not np.isfinite(rgb).all():
        raise ValueError("RGB shape/nonfinite mismatch; no resizing or nan filling")
    return rgb


def build_query(scope, query_mode="source_body_contact", native_rgb=None):
    if scope["status"] != "ok":
        return {"status": "skipped", "skip_reason": scope["skip_reason"], "query": None,
                "query_mode": query_mode}
    if query_mode not in QUERY_MODES:
        raise ValueError("unknown query_mode: " + query_mode)
    allowed = scope["allowed"]
    uses_native = query_mode == "r2_existing_saturation"
    if uses_native and native_rgb is None:
        return {"status": "skipped", "skip_reason": "r2_existing_unclipped_RGB_missing",
                "query": None, "query_mode": query_mode}
    if uses_native:
        rgb = _chw(native_rgb, allowed.shape)
        clipped = ((rgb < 0).any(0) | (rgb > 1).any(0)) & allowed
        query = (scope["body"] | scope["user_contact_query"] | clipped) & allowed
        source = "same-frame existing unclipped RGB out-of-range query plus observed body/manual contact; diagnostic, not detector GT"
    elif query_mode == "source_body_contact":
        query = (scope["body"] | scope["user_contact_query"]) & allowed
        source = "original observed body and fixed normalized manual contact only"
    else:
        # 仅显式调用才可使用这个额外控制臂，绝不作为缺mask/RGB回退。
        query = scope["display_scope"].copy()
        source = "explicit independent arm: fixed source-local display band; includes background, not semantic GT"
    return {"status": "ok", "skip_reason": None, "query": query, "query_mode": query_mode,
            "query_pixels": int(query.sum()), "native_RGB_content_used": uses_native,
            "query_provenance": source, "labels_for_training": False}


def project_native_centers(model_xyz, full_proj_transform, world_view_transform, image_shape):
    """输入native模型坐标与同帧导出的转置相机矩阵；不是原global世界坐标。"""
    xyz = np.asarray(model_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("model_xyz must be N x 3")
    homogeneous = np.column_stack([xyz, np.ones(len(xyz))])
    clip = homogeneous @ np.asarray(full_proj_transform, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = clip[:, :3] / clip[:, 3:4]
    h, w = image_shape
    screen = np.column_stack([(ndc[:, 0] + 1) * w / 2 - .5, (ndc[:, 1] + 1) * h / 2 - .5])
    z = (homogeneous @ np.asarray(world_view_transform, dtype=np.float64))[:, 2]
    return screen, z


def eligible_from_screen(screen_centers, camera_z, allowed):
    """精确复用原R2 rint像素中心规则；不因协方差越界扩大eligible。"""
    allowed = np.asarray(allowed, bool)
    screen, z = np.asarray(screen_centers, float), np.asarray(camera_z).reshape(-1)
    if screen.shape != (len(z), 2):
        raise ValueError("screen/z shape mismatch")
    h, w = allowed.shape
    candidate = np.isfinite(screen).all(1) & np.isfinite(z) & (z > 0)
    # 仅近图像范围的有限值转整数，避免后方/Inf/极大NDC溢出。
    candidate &= (screen[:, 0] >= -1) & (screen[:, 0] <= w) & (screen[:, 1] >= -1) & (screen[:, 1] <= h)
    index = np.flatnonzero(candidate)
    uv = np.rint(screen[index]).astype(np.int64)
    on = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    result = np.zeros(len(z), bool)
    result[index[on]] = allowed[uv[on, 1], uv[on, 0]]
    return result


def selection_start(init_mode="empty", existing_removed_ids=None):
    """每帧重新调用。函数无历史状态，不接受previous-frame候选。"""
    if init_mode == "empty":
        return np.empty(0, np.int32)
    if init_mode != "same_frame_existing" or existing_removed_ids is None:
        raise ValueError("same-frame existing IDs required; no previous-frame or empty fallback")
    ids = np.asarray(existing_removed_ids, np.int64).reshape(-1)
    if (ids < 0).any():
        raise ValueError("negative static row id")
    return np.unique(ids).astype(np.int32)


def compose_bounded_white(scope, gaussian_after, alpha_after, same_frame_baseline):
    if scope["status"] != "ok":
        raise ValueError("cannot compose skipped scope")
    shape = tuple(scope["image_shape"])
    g, baseline = _chw(gaussian_after, shape), _chw(same_frame_baseline, shape)
    alpha = np.asarray(alpha_after)
    if alpha.shape != shape or not np.isfinite(alpha).all():
        raise ValueError("after alpha shape/nonfinite mismatch")
    white = g + (1 - alpha)[None]
    output = baseline.copy()
    output[:, scope["display_scope"]] = white[:, scope["display_scope"]]
    return output


def read_identity(path):
    """只解析JSON赋值，不执行JS。缺完整body bits不能从outline/box造mask。"""
    text = Path(path).read_text(encoding="utf-8-sig")
    start = text.find("{")
    if start < 0:
        raise ValueError("identity JSON not found")
    return json.JSONDecoder().raw_decode(text[start:])[0]


def identity_body(record, image_shape=(320, 480)):
    if not record or not record.get("body_mask_bits"):
        return None
    packed = zlib.decompress(base64.b64decode(record["body_mask_bits"]))
    h, w = image_shape
    if len(packed) != (h * w + 7) // 8:
        raise ValueError("packed body size mismatch; no resize fallback")
    body = np.unpackbits(np.frombuffer(packed, np.uint8))[:h * w].reshape(h, w).astype(bool)
    expected = record.get("body_bbox")
    if expected is not None and body.any() and not np.array_equal(mask_bbox(body), expected):
        raise ValueError("identity bbox disagrees with original body bits")
    return body


def plan_manifest(dataset_manifest, *, body_loader, template=None):
    """保留所有manifest帧（包括skip），frame camera_index使用manifest位置而非raw。"""
    frames = []
    context = dataset_manifest["context"]
    for camera_index, frame in enumerate(dataset_manifest["frames"]):
        try:
            body = body_loader(frame)
            result = build_scope(body, context=context, raw=frame["raw"], frame_id=frame.get("frame_id"), template=template)
        except (OSError, ValueError, KeyError) as error:
            result = build_scope(None, context=context, raw=frame["raw"], frame_id=frame.get("frame_id"), template=template)
            result["skip_reason"] = "body_input_error:" + str(error)
        result["camera_index"] = camera_index
        result["model_time"] = frame.get("model_time")
        result["track"] = dataset_manifest.get("track")
        frames.append(result)
    return frames


def metadata(scope):
    return {key: value for key, value in scope.items() if key not in ["body", "allowed", "display_scope", "user_contact_query"]}


def local_inventory(workspace):
    root = Path(workspace)
    base = root / "outputs/V43_Ownership_20261027/gpu_v43/actor_delivery_early"
    diagnostics = root / "outputs/GS_Scene_Diagnostics_20261027"
    reports = []
    for manifest_path in sorted(base.glob("*/dataset_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        context = manifest["context"]
        identity_path = diagnostics / context / "identity.js"
        identity = read_identity(identity_path) if identity_path.exists() else {"frames": {}}
        identities = identity.get("frames", {})
        plan = plan_manifest(manifest, body_loader=lambda f: identity_body(identities.get(str(f["raw"]))))
        group_path = diagnostics / context / "v43_groups_data.json"
        groups = json.loads(group_path.read_text(encoding="utf-8-sig")).get("groups", []) if group_path.exists() else []
        reports.append({"context": context, "dataset_manifest": str(manifest_path), "identity_source": str(identity_path),
                        "frames": len(plan), "raw_range": [plan[0]["raw"], plan[-1]["raw"]] if plan else None,
                        "scopes_ok": sum(p["status"] == "ok" for p in plan),
                        "skipped": [{"raw": p["raw"], "reason": p["skip_reason"]} for p in plan if p["status"] != "ok"],
                        "local_v43_group_ids": [g["id"] for g in groups],
                        "local_actor_10000_present": any(g["id"] == "v43_actor_10000" for g in groups)})
    return {"schema": "local-scope-input-inventory-v1", "scenes": reports,
            "remote_data_checked": False, "missing_masks_do_not_fallback": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-workspace", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(local_inventory(args.inventory_workspace), ensure_ascii=False, indent=2))
