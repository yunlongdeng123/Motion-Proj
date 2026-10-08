"""R001 SAM3 video instance -> real-contour DELETE hole. No pseudo-Y is made here.

Only `--self-test` is CPU-only. The normal command requires an already-installed
official SAM3 package, an existing local SAM3 checkpoint, and an authorized GPU.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


def nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """PyTorch interpolate(mode='nearest') source indices for downsampling."""
    ys = np.floor(np.arange(height) * mask.shape[0] / height).astype(int)
    xs = np.floor(np.arange(width) * mask.shape[1] / width).astype(int)
    return mask[np.ix_(ys, xs)]


def shape_record(mask: np.ndarray, label: str) -> dict:
    if mask.dtype != np.bool_ or mask.ndim != 2 or not mask.any():
        raise ValueError(f"{label}: empty/nonbinary instance mask")
    yy, xx = np.where(mask)
    box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    area = int(mask.sum())
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    if area / box_area >= 0.98:
        raise ValueError(f"{label}: near-filled bounding rectangle; refuse mask fallback")
    return {"size_hw": list(mask.shape), "pixels": area, "mask_bbox_xyxy": box,
            "bbox_fill": area / box_area}


def validate_contour(mask: np.ndarray, index: int) -> dict:
    if mask.shape != (576, 1024):
        raise ValueError(f"frame {index}: expected original 576x1024 mask")
    result = {"frame": index, "full": shape_record(mask, f"frame {index} full")}
    # Native DriveEditor mask_concat is 72x128; r19 training uses 320x576,
    # then 40x72 latent. Reject a contour lost at either model input scale.
    for name, hw in (("native_latent", (72, 128)),
                     ("train_rgb", (320, 576)), ("train_latent", (40, 72))):
        result[name] = shape_record(nearest(mask, *hw), f"frame {index} {name}")
    return result


def one_instance(output: dict, expected_id=None) -> tuple[int, np.ndarray]:
    ids = np.asarray(output["out_obj_ids"]).reshape(-1)
    masks = np.asarray(output["out_binary_masks"])
    if masks.ndim != 3 or len(ids) != len(masks) or len(set(ids.tolist())) != len(ids):
        raise ValueError("SAM3 object IDs and HxW masks disagree")
    if expected_id is None:
        if len(ids) != 1:
            raise ValueError(f"box prompt yielded {len(ids)} instances; review, do not guess")
        expected_id = int(ids[0])
    matches = np.where(ids == expected_id)[0]
    if len(matches) != 1:
        raise ValueError(f"SAM3 tracked ID {expected_id} missing/duplicated; no fallback")
    mask = masks[int(matches[0])]
    if mask.dtype != np.bool_:
        raise ValueError("SAM3 out_binary_masks must be boolean")
    return expected_id, mask


def load_sam3_handoff(output: Path, priors: dict, frame_indices=tuple(range(10))):
    """Build one DELETE hole for the native request and the fixed r47 branch.

    ``frame_indices`` selects source SAM3 frames in request order, including
    repeated frames for a static diagnostic. The caller supplies priors already
    sliced to that request length. Nothing from the old alpha or hole is reused.
    """
    output = Path(output)
    manifest = json.loads((output / "mask_contract.json").read_text(encoding="utf-8"))
    if (manifest.get("case_id") != "R001" or
            manifest.get("sam3_instance_dir") != "sam3_instance" or
            manifest.get("edit_hole_dir") != "model_mask" or
            manifest.get("tracked_sam3_obj_id") is None or
            len(manifest.get("frames", [])) != 10):
        raise ValueError("not a complete R001 SAM3 instance handoff")
    if not frame_indices or any(not isinstance(i, int) or i not in range(10)
                                for i in frame_indices):
        raise ValueError("frame_indices must select R001 frames 0..9")
    if len(frame_indices) != len(priors["geometry"]):
        raise ValueError("request frames and r47 geometry are misaligned")
    masks = []
    for index in frame_indices:
        paths = [output / role / f"{index:05d}.png"
                 for role in ("sam3_instance", "model_mask")]
        values = [np.asarray(Image.open(path).convert("L")) for path in paths]
        if any(value.shape != (576, 1024) or
               not np.isin(value, (0, 255)).all() for value in values):
            raise ValueError(f"frame {index}: SAM3 masks must be 576x1024 binary PNGs")
        if not np.array_equal(*values):
            raise ValueError(f"frame {index}: instance and edit hole differ")
        mask = values[0] > 0
        validate_contour(mask, index)
        masks.append(mask)
    edit_mask = np.stack(masks)
    # r47 prepare_inputs.py used INTER_AREA > 0 at 1/4 resolution. For
    # 576x1024 -> 144x256 this is exactly any positive pixel in each 4x4 cell.
    branch_hole = edit_mask.reshape(len(edit_mask), 144, 4, 256, 4).any(axis=(2, 4))
    if priors["hole"].shape != branch_hole.shape:
        raise ValueError("r47 hole prior must be T x 144 x 256")
    aligned_priors = dict(priors)
    aligned_priors["hole"] = branch_hole
    return edit_mask, edit_mask.astype(np.float32), aligned_priors


def run(args):
    paths = [args.rgb_dir / f"{i:05d}.jpg" for i in range(10)]
    if any(not p.is_file() for p in paths) or len(list(args.rgb_dir.glob("*.jpg"))) != 10:
        raise ValueError("R001 requires exactly ten ordered 00000.jpg..00009.jpg frames")
    if not args.checkpoint.is_file() or args.checkpoint.name != "sam3.pt":
        raise ValueError("provide an existing official SAM3 sam3.pt; no download")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output must be empty: {args.output}")
    sizes = [Image.open(p).size for p in paths]
    if sizes != [(1024, 576)] * 10:
        raise ValueError("R001 original RGB must be ten 1024x576 frames")
    x0, y0, x1, y1 = args.gt_box
    if not (0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 576):
        raise ValueError("GT prompt box must be pixel XYXY inside original RGB")
    box_xywh = [[x0 / 1024, y0 / 576, (x1 - x0) / 1024, (y1 - y0) / 576]]

    # Explicit checkpoint prevents build_sam3_video_model's HF auto-download.
    import torch
    from sam3.model_builder import build_sam3_video_predictor
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 video inference requires CUDA; use --self-test on CPU")
    predictor = build_sam3_video_predictor(checkpoint_path=str(args.checkpoint),
                                           gpus_to_use=[torch.cuda.current_device()], compile=False)
    session = None
    try:
        response = predictor.handle_request({"type": "start_session",
                                             "resource_path": str(args.rgb_dir)})
        session = response["session_id"]
        seed = predictor.handle_request({"type": "add_prompt", "session_id": session,
                                         "frame_index": args.prompt_frame,
                                         "bounding_boxes": box_xywh,
                                         "bounding_box_labels": [1]})
        obj_id, first = one_instance(seed["outputs"])
        masks = {args.prompt_frame: first}
        for response in predictor.handle_stream_request({"type": "propagate_in_video",
                                                          "session_id": session,
                                                          "propagation_direction": "both",
                                                          "start_frame_index": args.prompt_frame}):
            index = int(response["frame_index"])
            _, mask = one_instance(response["outputs"], obj_id)
            if index != args.prompt_frame:
                if index in masks:
                    raise ValueError(f"duplicate propagation frame {index}")
                masks[index] = mask
        if set(masks) != set(range(10)):
            raise ValueError(f"incomplete propagation: {sorted(masks)}")
        checks = [validate_contour(masks[i], i) for i in range(10)]
    finally:
        if session is not None:
            predictor.handle_request({"type": "close_session", "session_id": session})
        predictor.shutdown()

    # Separate semantic roles. Both files currently carry the SAM3 contour;
    # neither is a rasterized prompt box or a composed target RGB.
    for role in ("sam3_instance", "model_mask"):
        (args.output / role).mkdir(parents=True, exist_ok=True)
        for i in range(10):
            Image.fromarray(masks[i].astype(np.uint8) * 255).save(
                args.output / role / f"{i:05d}.png")
    for i in range(10):
        instance = np.asarray(Image.open(args.output / "sam3_instance" / f"{i:05d}.png"))
        hole = np.asarray(Image.open(args.output / "model_mask" / f"{i:05d}.png"))
        if not np.array_equal(instance, hole) or not np.array_equal(hole > 0, masks[i]):
            raise RuntimeError(f"frame {i}: saved mask/hole changed the SAM3 contour")
    manifest = {"case_id": "R001", "source": "official SAM3 video box prompt + propagation",
                "checkpoint": str(args.checkpoint), "prompt_frame": args.prompt_frame,
                "prompt_box_xyxy": list(args.gt_box), "tracked_sam3_obj_id": obj_id,
                "sam3_instance_dir": "sam3_instance", "edit_hole_dir": "model_mask",
                "hole_policy": "exact SAM3 contour; no rectangle/dilation/GT rasterization",
                "gt_box_used_only_as_prompt": True, "identity_review": "pending",
                "frames": checks}
    (args.output / "mask_contract.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def self_test():
    rect = np.zeros((576, 1024), bool)
    rect[200:300, 300:500] = True
    try:
        validate_contour(rect, 0)
    except ValueError as e:
        assert "rectangle" in str(e)
    else:
        raise AssertionError("rectangle was accepted")
    yy, xx = np.ogrid[:576, :1024]
    contour = ((xx - 400) / 100) ** 2 + ((yy - 250) / 55) ** 2 < 1
    assert validate_contour(contour, 0)["train_latent"]["bbox_fill"] < 1
    try:
        one_instance({"out_obj_ids": np.array([], dtype=int),
                      "out_binary_masks": np.zeros((0, 576, 1024), bool)})
    except ValueError:
        pass
    else:
        raise AssertionError("empty identity was accepted")
    with TemporaryDirectory() as temp:
        root = Path(temp)
        for role in ("sam3_instance", "model_mask"):
            (root / role).mkdir()
            for i in range(10):
                shifted = np.roll(contour, i * 3, axis=1)
                Image.fromarray(shifted.astype(np.uint8) * 255).save(
                    root / role / f"{i:05d}.png")
        (root / "mask_contract.json").write_text(json.dumps({
            "case_id": "R001", "tracked_sam3_obj_id": 1,
            "sam3_instance_dir": "sam3_instance", "edit_hole_dir": "model_mask",
            "frames": [{} for _ in range(10)]}), encoding="utf-8")
        stale = {"geometry": np.zeros((2, 12, 144, 256), np.float32),
                 "hole": np.ones((2, 144, 256), bool)}
        edit, alpha, updated = load_sam3_handoff(root, stale, (0, 5))
        assert np.array_equal(edit[0], contour)
        assert np.array_equal(alpha, edit.astype(np.float32))
        assert np.array_equal(updated["hole"], edit.reshape(2, 144, 4, 256, 4).any((2, 4)))
        assert not np.array_equal(updated["hole"], stale["hole"])
        assert np.all(alpha[~edit] == 0) and np.all(alpha[edit] == 1)
        Image.fromarray(rect.astype(np.uint8) * 255).save(root / "model_mask" / "00000.png")
        try:
            load_sam3_handoff(root, stale, (0, 5))
        except ValueError as e:
            assert "differ" in str(e)
        else:
            raise AssertionError("diverged instance/edit masks were accepted")
    print("CPU contract: contour/alpha/r47 hole aligned; stale prior, rectangle and mismatch rejected")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--rgb-dir", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--prompt-frame", type=int, default=0)
    p.add_argument("--gt-box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    a = p.parse_args()
    if a.self_test:
        self_test()
    else:
        if any(v is None for v in (a.rgb_dir, a.checkpoint, a.output, a.gt_box)):
            p.error("--rgb-dir, --checkpoint, --output, and --gt-box are required")
        if not 0 <= a.prompt_frame < 10:
            p.error("--prompt-frame must be 0..9")
        run(a)
