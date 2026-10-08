from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REGISTRATION_THRESHOLDS = {
    "minimum_ratio_test_matches": 30,
    "minimum_ransac_inliers": 24,
    "minimum_inlier_fraction": 0.35,
    "maximum_median_reprojection_error_px": 1.5,
    "maximum_p90_reprojection_error_px": 3.0,
    "maximum_corner_displacement_px": 12.0,
    "minimum_linear_scale": 0.96,
    "maximum_linear_scale": 1.04,
    "maximum_abs_rotation_deg": 2.0,
    "maximum_abs_perspective_term": 0.0005,
}


class CompositionError(RuntimeError):
    pass


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise CompositionError(f"cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise CompositionError(f"cannot write image: {path}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def clean_label(path: Path) -> str:
    source = path.parent.name or path.stem
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", source).strip("._")
    if not label:
        raise CompositionError(f"cannot derive candidate label from: {path}")
    return label


def mask_bbox_xyxy_exclusive(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise CompositionError("edit mask is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def hard_compose(original: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if original.shape != candidate.shape:
        raise CompositionError(
            f"compose shape mismatch: original={original.shape}, candidate={candidate.shape}"
        )
    if mask.shape != original.shape[:2]:
        raise CompositionError(f"mask shape mismatch: mask={mask.shape}, image={original.shape}")
    composed = original.copy()
    composed[mask] = candidate[mask]
    return composed


def outside_difference(original: np.ndarray, composed: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    difference = np.abs(
        composed[~mask].astype(np.int16) - original[~mask].astype(np.int16)
    )
    return {
        "maximum_absolute_difference": int(difference.max(initial=0)),
        "changed_channel_values": int(np.count_nonzero(difference)),
    }


def registration_support_mask(edit_mask: np.ndarray) -> np.ndarray:
    exclusion = cv2.dilate(
        edit_mask.astype(np.uint8) * 255,
        np.ones((17, 17), dtype=np.uint8),
        iterations=1,
    )
    return np.where(exclusion > 0, 0, 255).astype(np.uint8)


def estimate_small_homography(
    original_roi: np.ndarray,
    candidate_roi: np.ndarray,
    edit_mask: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any], np.ndarray]:
    support_mask = registration_support_mask(edit_mask)
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02, edgeThreshold=10)
    original_gray = cv2.cvtColor(original_roi, cv2.COLOR_BGR2GRAY)
    candidate_gray = cv2.cvtColor(candidate_roi, cv2.COLOR_BGR2GRAY)
    original_keypoints, original_descriptors = sift.detectAndCompute(original_gray, support_mask)
    candidate_keypoints, candidate_descriptors = sift.detectAndCompute(candidate_gray, support_mask)

    stats: dict[str, Any] = {
        "method": "SIFT candidate-to-original homography",
        "feature_support": "only pixels outside the independent edit mask, with an 8px exclusion margin",
        "support_mask_pixels": int(np.count_nonzero(support_mask)),
        "original_keypoints": len(original_keypoints),
        "candidate_keypoints": len(candidate_keypoints),
        "ratio_test_matches": 0,
        "ransac_inliers": 0,
        "thresholds": REGISTRATION_THRESHOLDS,
        "pass": False,
        "failure_reasons": [],
    }
    blank_matches = np.hstack([candidate_roi, original_roi])

    if original_descriptors is None or candidate_descriptors is None:
        stats["failure_reasons"] = ["missing_sift_descriptors"]
        return None, stats, blank_matches

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(candidate_descriptors, original_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
    stats["ratio_test_matches"] = len(good)
    if len(good) < 4:
        stats["failure_reasons"] = ["fewer_than_four_ratio_test_matches"]
        return None, stats, blank_matches

    source = np.float32([candidate_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    target = np.float32([original_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        source,
        target,
        cv2.RANSAC,
        2.0,
        maxIters=5000,
        confidence=0.999,
    )
    if homography is None or inlier_mask is None:
        stats["failure_reasons"] = ["homography_estimation_failed"]
        return None, stats, blank_matches

    homography = homography / homography[2, 2]
    inliers = inlier_mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(source, homography)
    errors = np.linalg.norm(projected - target, axis=2).ravel()
    inlier_errors = errors[inliers]

    height, width = original_roi.shape[:2]
    corners = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    ).reshape(-1, 1, 2)
    projected_corners = cv2.perspectiveTransform(corners, homography)
    corner_displacements = np.linalg.norm(projected_corners - corners, axis=2).ravel()
    center = np.float32([[[0.5 * (width - 1), 0.5 * (height - 1)]]])
    projected_center = cv2.perspectiveTransform(center, homography)
    center_shift = (projected_center - center).reshape(2)

    singular_values = np.linalg.svd(homography[:2, :2], compute_uv=False)
    rotation_deg = math.degrees(math.atan2(homography[1, 0], homography[0, 0]))
    perspective_max = float(np.max(np.abs(homography[2, :2])))
    inlier_count = int(np.count_nonzero(inliers))
    inlier_fraction = float(inliers.mean())
    median_error = float(np.median(inlier_errors)) if inlier_errors.size else float("inf")
    p90_error = (
        float(np.percentile(inlier_errors, 90)) if inlier_errors.size else float("inf")
    )

    stats.update(
        {
            "ransac_inliers": inlier_count,
            "inlier_fraction": inlier_fraction,
            "inlier_reprojection_median_px": median_error,
            "inlier_reprojection_p90_px": p90_error,
            "homography_candidate_to_original": homography.tolist(),
            "center_shift_xy_px": [float(center_shift[0]), float(center_shift[1])],
            "corner_displacement_px": [float(value) for value in corner_displacements],
            "maximum_corner_displacement_px": float(corner_displacements.max()),
            "linear_singular_values": [float(value) for value in singular_values],
            "rotation_deg": float(rotation_deg),
            "maximum_abs_perspective_term": perspective_max,
        }
    )

    failures: list[str] = []
    thresholds = REGISTRATION_THRESHOLDS
    if len(good) < thresholds["minimum_ratio_test_matches"]:
        failures.append("insufficient_ratio_test_matches")
    if inlier_count < thresholds["minimum_ransac_inliers"]:
        failures.append("insufficient_ransac_inliers")
    if inlier_fraction < thresholds["minimum_inlier_fraction"]:
        failures.append("low_inlier_fraction")
    if median_error > thresholds["maximum_median_reprojection_error_px"]:
        failures.append("high_median_reprojection_error")
    if p90_error > thresholds["maximum_p90_reprojection_error_px"]:
        failures.append("high_p90_reprojection_error")
    if corner_displacements.max() > thresholds["maximum_corner_displacement_px"]:
        failures.append("homography_not_small_at_corners")
    if singular_values.min() < thresholds["minimum_linear_scale"]:
        failures.append("linear_scale_too_small")
    if singular_values.max() > thresholds["maximum_linear_scale"]:
        failures.append("linear_scale_too_large")
    if abs(rotation_deg) > thresholds["maximum_abs_rotation_deg"]:
        failures.append("rotation_too_large")
    if perspective_max > thresholds["maximum_abs_perspective_term"]:
        failures.append("perspective_term_too_large")

    stats["failure_reasons"] = failures
    stats["pass"] = not failures

    inlier_matches = [match for match, keep in zip(good, inliers) if keep]
    inlier_matches.sort(key=lambda match: match.distance)
    match_visualization = cv2.drawMatches(
        candidate_roi,
        candidate_keypoints,
        original_roi,
        original_keypoints,
        inlier_matches[:80],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return homography, stats, match_visualization


def titled_panel(image: np.ndarray, title: str) -> np.ndarray:
    panel = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(
        panel,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return panel


def make_review(
    original_roi: np.ndarray,
    edit_mask: np.ndarray,
    candidate_unregistered: np.ndarray,
    composed_unregistered: np.ndarray,
    candidate_registered: np.ndarray | None,
    composed_registered: np.ndarray | None,
) -> np.ndarray:
    overlay = original_roi.copy()
    red = np.zeros_like(overlay)
    red[..., 2] = 255
    overlay[edit_mask] = np.rint(
        0.55 * overlay[edit_mask].astype(np.float32)
        + 0.45 * red[edit_mask].astype(np.float32)
    ).astype(np.uint8)
    if candidate_registered is None or composed_registered is None:
        candidate_registered = np.zeros_like(original_roi)
        composed_registered = np.zeros_like(original_roi)
        cv2.putText(
            candidate_registered,
            "registration unavailable",
            (20, original_roi.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        composed_registered[:] = candidate_registered
    top = np.hstack(
        [
            titled_panel(original_roi, "original ROI"),
            titled_panel(candidate_unregistered, "candidate resized / unregistered"),
            titled_panel(candidate_registered, "candidate registered"),
        ]
    )
    bottom = np.hstack(
        [
            titled_panel(overlay, "independent write mask"),
            titled_panel(composed_unregistered, "hard compose / unregistered"),
            titled_panel(composed_registered, "hard compose / registered"),
        ]
    )
    return np.vstack([top, bottom])


def process_candidate(
    candidate_path: Path,
    output_dir: Path,
    original_full: np.ndarray,
    original_roi: np.ndarray,
    edit_mask_roi: np.ndarray,
    edit_mask_full: np.ndarray,
    roi_bbox: tuple[int, int, int, int],
    expected_generation_wh: tuple[int, int] | None,
) -> dict[str, Any]:
    label = clean_label(candidate_path)
    candidate_dir = output_dir / label
    if candidate_dir.exists():
        raise CompositionError(f"candidate output already exists: {candidate_dir}")
    candidate_dir.mkdir(parents=True)

    candidate_full_resolution = read_image(candidate_path)
    actual_generation_wh = (
        candidate_full_resolution.shape[1],
        candidate_full_resolution.shape[0],
    )
    if expected_generation_wh and actual_generation_wh != expected_generation_wh:
        raise CompositionError(
            f"unexpected candidate size {actual_generation_wh}; expected {expected_generation_wh}"
        )

    roi_height, roi_width = original_roi.shape[:2]
    if abs((actual_generation_wh[0]/actual_generation_wh[1])/(roi_width/roi_height)-1) > .02:
        raise CompositionError('generated ROI aspect ratio differs by more than 2 percent')
    candidate_unregistered = cv2.resize(
        candidate_full_resolution,
        (roi_width, roi_height),
        interpolation=cv2.INTER_AREA,
    )
    composed_roi_unregistered = hard_compose(
        original_roi, candidate_unregistered, edit_mask_roi
    )
    composed_full_unregistered = original_full.copy()
    x0, y0, x1, y1 = roi_bbox
    composed_full_unregistered[y0:y1, x0:x1] = composed_roi_unregistered

    raw_roi_outside = outside_difference(
        original_roi, composed_roi_unregistered, edit_mask_roi
    )
    raw_full_outside = outside_difference(
        original_full, composed_full_unregistered, edit_mask_full
    )
    if raw_roi_outside["changed_channel_values"] != 0 or raw_full_outside[
        "changed_channel_values"
    ] != 0:
        raise CompositionError("unregistered hard composition changed pixels outside mask")

    write_image(candidate_dir / "candidate_roi_resized_unregistered.png", candidate_unregistered)
    write_image(candidate_dir / "composed_roi_unregistered.png", composed_roi_unregistered)
    write_image(candidate_dir / "composed_fullframe_unregistered.png", composed_full_unregistered)

    homography, registration, match_visualization = estimate_small_homography(
        original_roi, candidate_unregistered, edit_mask_roi
    )
    write_image(candidate_dir / "registration_feature_support_mask.png", registration_support_mask(edit_mask_roi))
    write_image(candidate_dir / "registration_inlier_matches.png", match_visualization)

    candidate_registered: np.ndarray | None = None
    composed_roi_registered: np.ndarray | None = None
    composed_full_registered: np.ndarray | None = None
    registered_roi_outside: dict[str, int] | None = None
    registered_full_outside: dict[str, int] | None = None
    if homography is not None:
        candidate_registered = cv2.warpPerspective(
            candidate_unregistered,
            homography,
            (roi_width, roi_height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT101,
        )
        composed_roi_registered = hard_compose(
            original_roi, candidate_registered, edit_mask_roi
        )
        composed_full_registered = original_full.copy()
        composed_full_registered[y0:y1, x0:x1] = composed_roi_registered
        registered_roi_outside = outside_difference(
            original_roi, composed_roi_registered, edit_mask_roi
        )
        registered_full_outside = outside_difference(
            original_full, composed_full_registered, edit_mask_full
        )
        if registered_roi_outside["changed_channel_values"] != 0 or registered_full_outside[
            "changed_channel_values"
        ] != 0:
            raise CompositionError("registered hard composition changed pixels outside mask")
        write_image(candidate_dir / "candidate_roi_registered.png", candidate_registered)
        write_image(candidate_dir / "composed_roi_registered.png", composed_roi_registered)
        write_image(candidate_dir / "composed_fullframe_registered.png", composed_full_registered)

    review = make_review(
        original_roi,
        edit_mask_roi,
        candidate_unregistered,
        composed_roi_unregistered,
        candidate_registered,
        composed_roi_registered,
    )
    write_image(candidate_dir / "review.png", review)

    outputs = {
        "candidate_roi_resized_unregistered": "candidate_roi_resized_unregistered.png",
        "composed_roi_unregistered": "composed_roi_unregistered.png",
        "composed_fullframe_unregistered": "composed_fullframe_unregistered.png",
        "registration_feature_support_mask": "registration_feature_support_mask.png",
        "registration_inlier_matches": "registration_inlier_matches.png",
        "review": "review.png",
    }
    if homography is not None:
        outputs.update(
            {
                "candidate_roi_registered": "candidate_roi_registered.png",
                "composed_roi_registered": "composed_roi_registered.png",
                "composed_fullframe_registered": "composed_fullframe_registered.png",
            }
        )

    audit = {
        "artifact_type": "pseudo_candidate",
        "qa_status": "pending_independent_review",
        "training_admitted": False,
        "candidate_label": label,
        "candidate_source": str(candidate_path),
        "candidate_generation_wh": list(actual_generation_wh),
        "candidate_resized_to_roi_wh": [roi_width, roi_height],
        "candidate_pixel_edits": "none; only resize and optional audited global homography",
        "hard_composition": {
            "mask_expansion_px": 0,
            "write_rule": "candidate pixels only where the independent edit mask is nonzero",
            "outside_mask_rule": "pixel-identical source query",
            "unregistered_roi_outside_difference": raw_roi_outside,
            "unregistered_fullframe_outside_difference": raw_full_outside,
            "registered_roi_outside_difference": registered_roi_outside,
            "registered_fullframe_outside_difference": registered_full_outside,
        },
        "registration": registration,
        "registered_variant_status": (
            "pseudo_candidate_qa_pending"
            if registration["pass"]
            else "saved_for_review_not_eligible_due_to_registration_gate"
        ),
        "outputs": outputs,
    }
    write_json(candidate_dir / "composition_audit.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize ROI image-edit candidates, audit small SIFT registration, and hard-compose only inside an independent mask."
    )
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cv2.setNumThreads(4)
    cv2.ocl.setUseOpenCL(False)

    evidence_dir = args.evidence_dir.resolve()
    output_dir = args.outdir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CompositionError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = evidence_dir / "roi_v2.json"
    if not metadata_path.exists():
        metadata_path = evidence_dir / "roi.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    full_width, full_height = map(int, metadata["full_image_wh"])
    x0, y0, x1, y1 = map(int, metadata["roi_full_xyxy_exclusive"])
    roi_width, roi_height = map(int, metadata["roi_wh"])
    generation_wh = tuple(map(int, metadata["generation_output_wh"])) if "generation_output_wh" in metadata else None
    if (x1 - x0, y1 - y0) != (roi_width, roi_height):
        raise CompositionError("ROI bbox and roi_wh disagree")
    if int(metadata["mask_expansion_px"]) != 0:
        raise CompositionError("mask expansion is forbidden for this composition")

    query_path = evidence_dir / metadata["source"]["query_rgb"]
    original_full = read_image(query_path)
    if original_full.shape[:2] != (full_height, full_width):
        raise CompositionError("source query dimensions disagree with roi metadata")
    original_roi = original_full[y0:y1, x0:x1].copy()
    write_image(output_dir / 'original_full.png', original_full)

    edit_mask_gray = read_image(evidence_dir / "editmask_binary.png", cv2.IMREAD_GRAYSCALE)
    if edit_mask_gray.shape != (roi_height, roi_width):
        raise CompositionError("independent edit mask dimensions disagree with ROI")
    edit_mask_roi = edit_mask_gray > 127
    if int(np.count_nonzero(edit_mask_roi)) != int(metadata["mask_pixels_roi"]):
        raise CompositionError("independent edit mask pixel count disagrees with metadata")

    mask_bbox_roi = mask_bbox_xyxy_exclusive(edit_mask_roi)
    mask_bbox_full = [
        mask_bbox_roi[0] + x0,
        mask_bbox_roi[1] + y0,
        mask_bbox_roi[2] + x0,
        mask_bbox_roi[3] + y0,
    ]
    if mask_bbox_full != [int(value) for value in metadata["mask_bbox_full_xyxy_exclusive"]]:
        raise CompositionError("independent edit mask bbox disagrees with metadata")

    packaged_original_roi_path = evidence_dir / "A_original_RGB_ROI.png"
    packaged_roi_check: dict[str, Any] | None = None
    if packaged_original_roi_path.exists():
        packaged_original_roi = read_image(packaged_original_roi_path)
        if packaged_original_roi.shape != original_roi.shape:
            raise CompositionError("packaged original ROI dimensions disagree with source crop")
        packaged_difference = np.abs(
            packaged_original_roi.astype(np.int16) - original_roi.astype(np.int16)
        )
        packaged_roi_check = {
            "reference_only": True,
            "source_of_truth_for_hard_composition": str(query_path),
            "maximum_absolute_difference": int(packaged_difference.max(initial=0)),
            "mean_absolute_difference": float(packaged_difference.mean()),
            "changed_channel_values": int(np.count_nonzero(packaged_difference)),
        }

    edit_mask_full = np.zeros((full_height, full_width), dtype=bool)
    edit_mask_full[y0:y1, x0:x1] = edit_mask_roi
    if int(np.count_nonzero(edit_mask_full)) != int(metadata["mask_pixels_full"]):
        raise CompositionError("full-frame mask pixel count disagrees with metadata")

    labels = [clean_label(path) for path in args.candidate]
    if len(labels) != len(set(labels)):
        raise CompositionError("candidate labels collide")

    audits = []
    for candidate_path in args.candidate:
        audits.append(
            process_candidate(
                candidate_path.resolve(),
                output_dir,
                original_full,
                original_roi,
                edit_mask_roi,
                edit_mask_full,
                (x0, y0, x1, y1),
                generation_wh,
            )
        )

    manifest = {
        "artifact_type": "pseudo_candidate_collection",
        "qa_status": "pending_independent_review",
        "training_admitted": False,
        "case_id": metadata["case_id"],
        "frame_index": int(metadata["frame_index"]),
        "source_query": str(query_path),
        "roi_full_xyxy_exclusive": [x0, y0, x1, y1],
        "roi_wh": [roi_width, roi_height],
        "mask_bbox_full_xyxy_exclusive": mask_bbox_full,
        "mask_pixels": int(np.count_nonzero(edit_mask_full)),
        "mask_expansion_px": 0,
        "packaged_roi_vs_source_crop": packaged_roi_check,
        "candidates": [
            {
                "label": audit["candidate_label"],
                "registration_pass": bool(audit["registration"]["pass"]),
                "audit": f"{audit['candidate_label']}/composition_audit.json",
                "review": f"{audit['candidate_label']}/review.png",
            }
            for audit in audits
        ],
    }
    write_json(output_dir / "composition_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
