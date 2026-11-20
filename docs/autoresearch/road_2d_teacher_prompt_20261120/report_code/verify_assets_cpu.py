"""本地取回验收：60原PNG字节、原生ROI逆变换与实际W/K像素；无GPU。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from PIL import Image


def need(value, message):
    if not value:
        raise RuntimeError(message)


def rgb(path):
    with Image.open(path) as im:
        need(im.mode == "RGB", "not RGB: " + str(path))
        return np.asarray(im).copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    m = json.loads((a.assets/"asset_manifest.json").read_text(encoding="utf-8"))
    need(m["complete"] and m["candidate_count"] == 60, "asset export incomplete")
    digests = {}
    for f in m["files"]:
        path = a.assets/f["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        need(digest == f["sha256"] and path.stat().st_size == f["bytes"], "takeback mismatch: " + f["path"])
        with Image.open(path) as image:
            image.load()
            need(list(image.size) == f["wh"], "decoded size mismatch: " + f["path"])
        digests[f["path"]] = digest
    count = 0
    for c in m["cases"]:
        need({(j["prompt_id"],j["seed"]) for j in c["jobs"]} == {(p,s) for p in ("legacy","material_match_v1") for s in (101,202,303,404,505)}, "candidate matrix mismatch")
        observation = rgb(a.assets/c["reference"]["roi"])
        with Image.open(a.assets/c["masks"]["W"]) as im:
            W = np.asarray(im) == 255
        with Image.open(a.assets/c["masks"]["K"]) as im:
            K = np.asarray(im) == 255
        for j in c["jobs"]:
            need(digests[j["raw_model"]] == j["remote_originals"]["raw_model"]["sha256"], "raw_model source bytes changed")
            with Image.open(a.assets/j["raw_model"]) as model:
                need(model.size == (1152,768), "wrong source resolution")
                reconstructed = np.asarray(model.resize((1920,1280),Image.Resampling.BILINEAR).crop(c["roi_xyxy"]))
            raw, final = rgb(a.assets/j["raw_roi"]), rgb(a.assets/j["final_roi"])
            need(np.array_equal(raw,reconstructed), "raw ROI not exact inverse: " + j["id"])
            need(np.array_equal(final[~W],observation[~W]) and np.array_equal(final[K],observation[K]), "local actual W/K failed")
            need(np.array_equal(final[W],raw[W]), "W is not raw hard-writeback")
            count += 1
    need(count == m["raw_model_original_count"] == 60, "not all60 originals")
    result = dict(schema="road_2d_teacher_prompt_assets_local_validation.v1",complete=True,CPU_only=True,GPU_executed=False,
        files_verified=len(digests),images_decoded=len(digests),original_raw_PNG_byte_exact_count=60,raw_ROI_inverse_exact_count=60,
        final_ROI_actual_W_K_exact_count=60,cases=6,seeds=5,prompts=2,hidden_truth_opened=False)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":
    main()
