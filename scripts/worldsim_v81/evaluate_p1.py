"""Seen-to-Scene P1 evaluation using fixed Follow-Your-Canvas metrics.

Input JSON: {"cases": [{"dataset": "davis2017"|"youtube_vos",
"sequence_id": str, "source_id": str, "mask_total_ratio": 0.25|0.66,
"gt": "/abs/gt.mp4", "pred": "/abs/pred.mp4", "comp": "/abs/comp.mp4",
"source_dir": "/abs/original/JPEGImages/sequence"}]}.
Formal full-video evaluation requires source_dir; short-window evaluation is diagnostic.
The train source IDs file is a JSON string list or one ID per line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FYC_REVISION = "0e6af915b93266b3a0297768c32edc579b3b1f6e"
DEFAULT_FYC = Path("/root/autodl-tmp/external/follow_your_canvas_metrics_v81")
DAVIS_IMAGESETS = Path("/root/autodl-tmp/data/worldsim_v81/davis_2017_480p/DAVIS/ImageSets/2017")
DEFAULT_I3D = Path("/root/autodl-tmp/external/weights_v81/i3d_torchscript.pt")
I3D_SOURCE = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
I3D_SHA256 = "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2"
DATASETS = {"davis2017": 90, "youtube_vos": 60}
RATIOS = (0.25, 0.66)  # Full canvas masked width; 0.125 and 0.33 on each side.
FRAMES = 16  # Follow-Your-Canvas official low-resolution metrics code.
YOUTUBE_IDS = frozenset("""0c7a4680db 0d349f8286 2e21c7e59b
2e129b0b09 3b72dc1941 3f2012d518 4b31a18d91 4f5b3310e3 5c3d2d3155
06a5dfb511 6a75316e99 6cced81d30 7daa6343e6 8dea7458de 9c4419eb12
13c3cea202 24e2b52a4d 37b4ec2e1a 37dc952545 45fd60997a 54ad024bb3
83a5056a16 95ef69d827 97b38cabcc 97fa40286c 397dccb3a0 459e70cd8e
03664dc880 4035d3275c 9787f452bf 40718bb478 90949b2059 547416bda1
607001c98f 1320830fd2 4348676053 6031809500 a9839ec6f2 ac4653b61d
b8bd20a472 b175cd8138 b492f67a89 b715879d5a ba5dde67e9 c9ef04fe59
c16d9a4ade cbea8f6bea d1dd586cfd d7ff44ea97 d59c093632 da9713ef3e
dab44991de dc197289ef e10236eb37 e1925724ab eb49ce8027 f9eedb8691
f58212429d fab725059c fb104c286f""".split())  # Seen-to-Scene Appendix E.
UNVERIFIED = [
    "Seen-to-Scene未公开自身四指标评测脚本；使用固定Follow-Your-Canvas实现。",
    "论文未明确FVD两倍率汇总、pred/comp选择、视频编码与抽帧细节。",
    "DAVIS论文只给90条数量，未公布精确序列清单；采用官方2017 train+val ImageSets。",
    "论文称YouTube-VOS test，但附录E的60条ID均位于2019 valid split；需声明年份与全帧版本假设。",
    "论文称纯高斯前向推理与Adam；固定官方源码另跑DDIM inversion并使用AdamW。",
]


def official_metrics(fyc_root: Path):
    """Import official metric functions from an external pinned source checkout."""
    fyc_root = fyc_root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(fyc_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != FYC_REVISION:
        raise ValueError(f"Follow-Your-Canvas revision {revision}, expected {FYC_REVISION}")
    metric_dir = fyc_root / "video_metrics"
    if not metric_dir.is_dir():
        raise FileNotFoundError(f"missing official metric source: {metric_dir}")
    sys.path.insert(0, str(metric_dir))
    from calculate_lpips import calculate_lpips
    from calculate_psnr import calculate_psnr
    from calculate_ssim import calculate_ssim
    from our_fvd import compute_fvd

    return calculate_psnr, calculate_ssim, calculate_lpips, compute_fvd


def train_source_ids(path: Path) -> set[str]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError("训练来源ID文件为空；无法排除来源重叠")
    if raw.startswith("["):
        items = json.loads(raw)
    else:
        items = [line.strip() for line in raw.splitlines() if line.strip()]
    if not isinstance(items, list) or any(not isinstance(x, str) or not x for x in items):
        raise ValueError("训练来源ID应为非空字符串列表")
    return set(items)


def davis_ids() -> set[str] | None:
    train_file, val_file = (DAVIS_IMAGESETS / "train.txt", DAVIS_IMAGESETS / "val.txt")
    if not train_file.is_file() or not val_file.is_file():
        return None
    train = [line.strip() for line in train_file.read_text().splitlines() if line.strip()]
    val = [line.strip() for line in val_file.read_text().splitlines() if line.strip()]
    if len(train) != 60 or len(val) != 30 or len(set(train) | set(val)) != 90:
        raise ValueError("DAVIS官方ImageSets/2017须为不重叠的60 train + 30 val")
    return set(train) | set(val)


def load_manifest(path: Path, known_train: set[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest必须有非空cases列表")
    seen: set[tuple[str, str, float]] = set()
    by_dataset: dict[str, dict[float, set[str]]] = defaultdict(lambda: defaultdict(set))
    expected_davis = davis_ids()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case必须是对象")
        dataset, sequence, source = (case.get(k) for k in ("dataset", "sequence_id", "source_id"))
        if dataset not in DATASETS or not isinstance(sequence, str) or not sequence or not isinstance(source, str) or not source:
            raise ValueError("每项须有合法dataset、sequence_id、source_id")
        if dataset == "youtube_vos" and sequence not in YOUTUBE_IDS:
            raise ValueError(f"YouTube-VOS序列不在论文附录E的60条内: {sequence}")
        if dataset == "davis2017" and expected_davis is not None and sequence not in expected_davis:
            raise ValueError(f"DAVIS序列不在官方2017 train+val ImageSets内: {sequence}")
        if source in known_train:
            raise ValueError(f"评测来源与训练来源重叠: {source}")
        ratio = float(case.get("mask_total_ratio", -1))
        if ratio not in RATIOS:
            raise ValueError(f"仅接受总宽遮挡比例{RATIOS}: {ratio}")
        key = (dataset, sequence, ratio)
        if key in seen:
            raise ValueError(f"重复评测项: {key}")
        seen.add(key)
        by_dataset[dataset][ratio].add(sequence)
        for name in ("gt", "pred", "comp"):
            video_path = Path(case.get(name, ""))
            if not video_path.is_absolute() or not video_path.is_file():
                raise FileNotFoundError(f"{key} {name}需要已存在的绝对视频文件: {video_path}")
    for dataset, ratios in by_dataset.items():
        if ratios[0.25] != ratios[0.66]:
            raise ValueError(f"{dataset}两个倍率的序列集合不同")
    return cases


def coverage(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    expected_davis = davis_ids()
    for dataset, expected in DATASETS.items():
        ids = {case["sequence_id"] for case in cases if case["dataset"] == dataset and float(case["mask_total_ratio"]) == RATIOS[0]}
        if ids:
            inventory_verified = (ids == YOUTUBE_IDS if dataset == "youtube_vos"
                                  else expected_davis is not None and ids == expected_davis)
            result[dataset] = {"n_sequences": len(ids), "expected": expected,
                               "inventory_verified": inventory_verified,
                               "complete": len(ids) == expected and inventory_verified}
    return result


def require_complete_benchmark(cases: list[dict[str, Any]]) -> None:
    """正式清单必须同时包含两套完整基准，不能漏掉整个数据集。"""
    inventory = coverage(cases)
    if set(inventory) != set(DATASETS) or any(not row["complete"] for row in inventory.values()):
        raise ValueError("基准序列不完整：DAVIS须官方train+val全部90条，YouTube-VOS须附录E全部60条；调试可显式--allow-partial")


def require_25_frame_videos(cases: list[dict[str, Any]]) -> None:
    """仅供明示的本地25帧诊断协议。"""
    from decord import VideoReader, cpu

    checked: set[Path] = set()
    for case in cases:
        if Path(case["pred"]).resolve() == Path(case["comp"]).resolve():
            raise ValueError(f"原生pred与硬合成comp必须分别保存：{case['dataset']}/{case['sequence_id']}")
        for name in ("gt", "pred", "comp"):
            path = Path(case[name])
            if path in checked:
                continue
            checked.add(path)
            video = VideoReader(str(path), ctx=cpu(0))
            count = len(video)
            if count != 25:
                raise ValueError(f"本地25帧协议要求GT/pred/comp恰好25帧：{case['dataset']}/{case['sequence_id']} {name}={count} {path}")
            if tuple(video[0].shape[:2]) != (256, 256):
                raise ValueError(f"正式评测要求256x256帧：{case['dataset']}/{case['sequence_id']} {name} {path}")


def require_full_video_lengths(cases: list[dict[str, Any]]) -> dict[str, int]:
    """正式全视频协议：原始JPEG数量必须逐项等于GT/pred/comp帧数。"""
    from decord import VideoReader, cpu

    verified = {}
    source_by_sequence = {}
    for case in cases:
        key = (case["dataset"], case["sequence_id"])
        raw_source = case.get("source_dir")
        if not isinstance(raw_source, str) or not raw_source:
            raise ValueError(f"全视频评测需要原始source_dir：{key}")
        source_dir = Path(raw_source)
        if not source_dir.is_absolute() or not source_dir.is_dir():
            raise ValueError(f"source_dir需要已存在的绝对目录：{key} {source_dir}")
        resolved = source_dir.resolve()
        if key in source_by_sequence and source_by_sequence[key] != resolved:
            raise ValueError(f"同一序列两个倍率必须使用同一来源目录：{key}")
        source_by_sequence[key] = resolved
        source_count = sum(path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                           for path in source_dir.iterdir())
        if source_count < FRAMES:
            raise ValueError(f"原始视频不足{FRAMES}帧：{key} {source_count}")
        if Path(case["pred"]).resolve() == Path(case["comp"]).resolve():
            raise ValueError(f"原生pred与硬合成comp必须分别保存：{key}")
        counts = {}
        for name in ("gt", "pred", "comp"):
            video = VideoReader(str(case[name]), ctx=cpu(0))
            counts[name] = len(video)
            if counts[name] < FRAMES or tuple(video[0].shape[:2]) != (256, 256):
                raise ValueError(f"全视频评测要求至少{FRAMES}帧且256x256：{key} {name}")
        if any(count != source_count for count in counts.values()):
            raise ValueError(f"原始来源、GT、pred、comp帧数不一致：{key} source={source_count} {counts}")
        verified[f"{key[0]}/{key[1]}/{float(case['mask_total_ratio']):g}"] = source_count
    return verified


def read_first_16(path: Path, *, fvd: bool = False):
    """Match FYC demo.py/fvd2.py frame selection and 224-pixel transforms."""
    import torch
    import torchvision.transforms as transforms
    from decord import VideoReader, cpu

    video = VideoReader(str(path), ctx=cpu(0))
    if len(video) < FRAMES:
        raise ValueError(f"视频少于{FRAMES}帧: {path}")
    frames = video.get_batch(list(range(FRAMES))).asnumpy().astype(np.uint8)
    tensor = torch.from_numpy(frames.transpose(0, 3, 1, 2))
    if fvd:
        tensor = transforms.Resize((224, 224))(tensor)
    else:
        tensor = transforms.CenterCrop(224)(transforms.Resize(224)(tensor))
    return tensor.unsqueeze(0).float() / 255.0


def frame_metric_means(real, fake, functions, device: str) -> dict[str, float]:
    psnr_fn, ssim_fn, lpips_fn, _ = functions
    result = {
        "psnr": psnr_fn(real, fake),
        "ssim": ssim_fn(real, fake),
        "lpips": lpips_fn(real, fake, device),
    }
    return {name: float(np.mean(list(obj["value"].values()))) for name, obj in result.items()}


def i3d_features(detector, video, device: str):
    import torch

    x = video.permute(0, 2, 1, 3, 4).to(device)
    with torch.no_grad():
        features = detector(x, rescale=False, resize=False, return_features=True)
    return features.detach().cpu().numpy()


def average_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {name: float(np.mean([row[name] for row in rows])) for name in ("psnr", "ssim", "lpips")}


def evaluate(cases: list[dict[str, Any]], fyc_root: Path, i3d_path: Path | None,
             device: str, *, generation_protocol: str = "local-first25"):
    import torch

    functions = official_metrics(fyc_root)
    detector = None
    fvd_status = "missing_i3d_torchscript_pt"
    if i3d_path is not None and i3d_path.is_file():
        actual_hash = hashlib.sha256(i3d_path.read_bytes()).hexdigest()
        if actual_hash != I3D_SHA256:
            raise ValueError(f"I3D SHA256错误: {actual_hash}, expected {I3D_SHA256}")
        detector = torch.jit.load(str(i3d_path), map_location=device).eval().to(device)
        fvd_status = "computed_with_fyc_i3d_interface"

    rows: dict[tuple[str, float, str], list[dict[str, float]]] = defaultdict(list)
    features: dict[tuple[str, float, str], list[np.ndarray]] = defaultdict(list)
    case_rows = []
    for case in cases:
        dataset, ratio = case["dataset"], float(case["mask_total_ratio"])
        real = read_first_16(Path(case["gt"]))
        real_fvd = read_first_16(Path(case["gt"]), fvd=True) if detector else None
        if detector:
            features[(dataset, ratio, "gt")].append(i3d_features(detector, real_fvd, device))
        for output in ("pred", "comp"):
            fake = read_first_16(Path(case[output]))
            row = frame_metric_means(real, fake, functions, device)
            rows[(dataset, ratio, output)].append(row)
            case_rows.append({"dataset": dataset, "sequence_id": case["sequence_id"],
                              "source_id": case["source_id"], "mask_total_ratio": ratio,
                              "output": output, **row})
            if detector:
                fake_fvd = read_first_16(Path(case[output]), fvd=True)
                features[(dataset, ratio, output)].append(i3d_features(detector, fake_fvd, device))

    grouped = {}
    for dataset in DATASETS:
        if not any(case["dataset"] == dataset for case in cases):
            continue
        grouped[dataset] = {}
        for output in ("pred", "comp"):
            ratio_rows = {}
            for ratio in RATIOS:
                key = (dataset, ratio, output)
                if not rows[key]:
                    continue
                vals = average_rows(rows[key])
                vals["n_sequences"] = len(rows[key])
                if detector and len(rows[key]) >= 2:
                    real_feats = np.concatenate(features[(dataset, ratio, "gt")], axis=0)
                    fake_feats = np.concatenate(features[key], axis=0)
                    vals["fvd"] = float(functions[3](fake_feats, real_feats))
                else:
                    vals["fvd"] = None
                ratio_rows[str(ratio)] = vals
            mean = {name: float(np.mean([ratio_rows[str(r)][name] for r in RATIOS]))
                    for name in ("psnr", "ssim", "lpips")}
            enough = detector and all(ratio_rows[str(r)]["fvd"] is not None for r in RATIOS)
            mean["fvd"] = (float(np.mean([ratio_rows[str(r)]["fvd"] for r in RATIOS]))
                           if enough else None)
            grouped[dataset][output] = {"by_total_mask_ratio": ratio_rows, "mean_of_ratios": mean}
    if detector and any(block[output]["mean_of_ratios"]["fvd"] is None
                        for block in grouped.values() for output in ("pred", "comp")):
        fvd_status = "insufficient_sequences_for_covariance"
    reason = ("本地仅生成每序列前25帧，再评分前16帧；此协议仅供诊断，不能称论文表1完整复现。"
              if generation_protocol == "local-first25" else
              "全视频生成采用25帧重叠窗口/stride16，指标取前16帧；论文定量视频长度与抽帧仍未公开。")
    return {"protocol_verified": False, "unverified_reasons": [*UNVERIFIED, reason],
            "fvd_status": fvd_status, "fyc_revision": FYC_REVISION,
            "i3d_source": I3D_SOURCE if detector else None,
            "i3d_sha256": I3D_SHA256 if detector else None,
            "generation_protocol": ("local_first25_sorted_start0_diagnostic"
                                    if generation_protocol == "local-first25" else
                                    "full_video_sliding_window_25_stride16"),
            "first_frames": FRAMES, "coverage": coverage(cases), "groups": grouped, "cases": case_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-source-ids", type=Path, required=True)
    parser.add_argument("--fyc-root", type=Path, default=DEFAULT_FYC)
    parser.add_argument("--i3d", type=Path, default=DEFAULT_I3D)
    parser.add_argument("--allow-partial", action="store_true", help="仅调试；允许不足90/60条，结果不能充当完整基准")
    parser.add_argument("--generation-protocol", choices=("local-first25", "full-video"),
                        default="local-first25",
                        help="正式评测须显式full-video；local-first25仅供诊断")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(repo):
        parser.error("评测产物必须保存到仓库外")
    cases = load_manifest(args.manifest, train_source_ids(args.train_source_ids))
    if not args.allow_partial and args.generation_protocol != "full-video":
        parser.error("前25帧短窗仅供诊断；正式评测必须显式--generation-protocol full-video")
    if not args.allow_partial:
        try:
            require_complete_benchmark(cases)
        except ValueError as exc:
            parser.error(str(exc))
    if not args.allow_partial and not args.i3d.is_file():
        parser.error(f"正式四指标评测必须有已冻结的I3D权重：{args.i3d}")
    frame_counts = (require_full_video_lengths(cases)
                    if args.generation_protocol == "full-video" else None)
    if args.generation_protocol == "local-first25":
        require_25_frame_videos(cases)
    result = evaluate(cases, args.fyc_root, args.i3d, args.device,
                      generation_protocol=args.generation_protocol)
    if frame_counts is not None:
        result["validated_frame_counts"] = frame_counts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_verified": False, "fvd_status": result["fvd_status"],
                      "cases": len(cases), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
