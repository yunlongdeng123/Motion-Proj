"""Prepare isolated, reviewable patches against the immutable run snapshot."""
import difflib
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_frozen"
CANDIDATES = ROOT / "candidates"

DEPTH_HELPER = '''"""Permutation-invariant nearest visible sparse depth; NumPy only.

This is an explicit geometric projection choice. The paper does not specify
how multiple LiDAR points in a raster pixel are reduced.
"""
import numpy as np


def nearest_positive_depth(uv, depth, height, width):
    """Return1xHxW nearest finite positive depths; invalid/empty pixels are0.

    uv may contain finite nonnegative float coordinates or integer pixels.
    Coordinate quantization matches camera_utils' positive-coordinate truncation.
    np.minimum.at resolves repeated indices without dependence on point order.
    """
    uv = np.asarray(uv)
    depth = np.asarray(depth)
    if uv.ndim != 2 or uv.shape[1] != 2 or depth.shape != (len(uv),):
        raise ValueError("Expected Nx2 uv and N depth")
    if height <= 0 or width <= 0:
        raise ValueError("Raster dimensions must be positive")
    valid = (np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0)
             & (uv[:, 0] >= 0) & (uv[:, 0] < width)
             & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    pixels = uv[valid].astype(np.int64)
    indices = pixels[:, 1] * width + pixels[:, 0]
    nearest = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(nearest, indices, depth[valid])
    nearest[~np.isfinite(nearest)] = 0.0
    return nearest.reshape(1, height, width)
'''

OPACITY_HELPER = '''


def diagnostic_dynamic_background_opacity_loss(dynamic_alpha, dynamic_mask, eps=1e-4):
    """Experimental background-domain complement, NOT a verified author term.

    Retains Eq13 on the foreground. Outside Md, use the Eq7-shaped soft
    zero-opacity penalty -a*log((1-a+eps)/(1+eps)), normalized by full image area.
    This stabilizes the logarithm without a high-alpha zero-gradient plateau.
    Mask uncertainty and the extra coefficient require a controlled diagnosis.
    No old100*MSE or old composite/static RGB objective is restored.
    """
    foreground = _mask_like(dynamic_mask, dynamic_alpha)
    opacity = dynamic_alpha.clamp(0.0, 1.0)
    return -((1.0 - foreground) * opacity * torch.log1p(-opacity / (1.0 + eps))).mean()
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def preserve_newlines(text, original_path):
    return text.replace("\n", "\r\n") if b"\r\n" in original_path.read_bytes() else text


def replace_one(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def patch(original, candidate, relative):
    return "".join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
                                        fromfile=("a/" + relative if original else "/dev/null"), tofile="b/" + relative))


def main():
    original_camera = (SOURCE / "utils/camera_utils.py").read_text(encoding="utf-8")
    depth_camera = replace_one(original_camera, "import numpy as np\n",
                               "import numpy as np\nfrom utils.sparse_depth import nearest_positive_depth\n")
    depth_camera = replace_one(depth_camera,
                              "        pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]\n",
                              "        pts_depth = nearest_positive_depth(uv, uvz[:, 2], h, w)\n")
    depth = CANDIDATES / "nearest_depth"
    depth_camera = preserve_newlines(depth_camera, SOURCE / "utils/camera_utils.py")
    write(depth / "utils/camera_utils.py", depth_camera)
    write(depth / "utils/sparse_depth.py", DEPTH_HELPER)
    depth_patch = patch((SOURCE / "utils/camera_utils.py").read_bytes().decode("utf-8"), depth_camera, "utils/camera_utils.py")
    depth_patch += patch("", DEPTH_HELPER, "utils/sparse_depth.py")
    write(CANDIDATES / "nearest_depth.patch", depth_patch)

    original_losses = (SOURCE / "utils/paper_losses.py").read_text(encoding="utf-8")
    candidate_losses = original_losses.rstrip() + OPACITY_HELPER
    original_train = (SOURCE / "train.py").read_text(encoding="utf-8")
    candidate_train = replace_one(original_train,
        "                                       nps_loss, sparse_inverse_depth_loss, compose_paper_loss)\n",
        "                                       nps_loss, sparse_inverse_depth_loss, compose_paper_loss,\n"
        "                                       diagnostic_dynamic_background_opacity_loss)\n")
    candidate_train = replace_one(candidate_train,
        "            paper_opacity = zero\n",
        "            paper_opacity = zero\n"
        "            diagnostic_background_opacity = zero\n"
        "            diagnostic_lambda_background = float(args.get(\"diagnostic_lambda_dynamic_background_opacity\", 0.0))\n")
    candidate_train = replace_one(candidate_train,
        "                paper_opacity = dynamic_opacity_loss(dynamic_alpha, dynamic_mask)\n",
        "                paper_opacity = dynamic_opacity_loss(dynamic_alpha, dynamic_mask)\n"
        "                if diagnostic_lambda_background > 0.0:\n"
        "                    diagnostic_background_opacity = diagnostic_dynamic_background_opacity_loss(dynamic_alpha, dynamic_mask)\n")
    candidate_train = replace_one(candidate_train,
        "                                      lambda_dynamic_rgb=args.get(\"paper_lambda_dynamic_rgb\", 1.0))\n",
        "                                      lambda_dynamic_rgb=args.get(\"paper_lambda_dynamic_rgb\", 1.0))\n"
        "            # Isolated diagnostic only: explicitly add the unverified background-domain term.\n"
        "            log_dict[\"loss_original_objective_before_diagnostic\"] = loss.item()\n"
        "            loss = loss + diagnostic_lambda_background * diagnostic_background_opacity\n"
        "            log_dict[\"loss_diagnostic_dynamic_background_opacity\"] = diagnostic_background_opacity.item()\n")
    opacity = CANDIDATES / "opacity_outside"
    candidate_train = preserve_newlines(candidate_train, SOURCE / "train.py")
    write(opacity / "utils/paper_losses.py", candidate_losses)
    write(opacity / "train.py", candidate_train)
    opacity_patch = patch(original_losses, candidate_losses, "utils/paper_losses.py")
    opacity_patch += patch((SOURCE / "train.py").read_bytes().decode("utf-8"), candidate_train, "train.py")
    write(CANDIDATES / "opacity_outside.patch", opacity_patch)
    write(opacity / "diagnostic_config_snippet.yaml",
          "# Experimental complement to foreground Eq13; not certified author formula.\n"
          "# Apply only in an isolated controlled diagnosis. Baseline default is0.\n"
          "diagnostic_background_domain_complement: true\n"
          "diagnostic_lambda_dynamic_background_opacity: 0.05\n")
    manifest = {
        "scope": "Reviewable isolated copies/patches; formal source/config/data/checkpoints unchanged",
        "baseline_root": str(SOURCE),
        "source_pins": {name: sha(SOURCE / name) for name in ("utils/camera_utils.py", "utils/paper_losses.py", "train.py")},
        "candidates": [
            {"name": "nearest_depth", "patch": str(CANDIDATES / "nearest_depth.patch"),
             "classification": "Inherited order-dependent sparse-depth projection bug; nearest visible policy explicit, not specified by paper",
             "variables_changed": "Only duplicate sparse-depth target reduction; photometric losses/weights and opacity resets unchanged",
             "paper_reference": "Eq16 PDF7-8/printed1702-1703", "source_reference": "camera_utils.py133"},
            {"name": "opacity_outside", "patch": str(CANDIDATES / "opacity_outside.patch"),
             "classification": "Single new complementary background-domain diagnostic; unverified final-paper author objective",
             "variables_changed": "Only lambda_background term, preserving foreground Eq13 and all original paper losses",
             "default_lambda": 0.0, "proposed_diagnostic_lambda": .05,
             "formula": "L_background=-sum[(1-M_d)*alpha_d*log((1-alpha_d+eps)/(1+eps))]/(h*w),eps=1e-4",
             "paper_reference": "Eq13 domain omitted; Eq7 shape used as an explicit experimental complement"},
        ],
    }
    write(CANDIDATES / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
