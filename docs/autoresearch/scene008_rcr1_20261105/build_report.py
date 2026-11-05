"""Build a local review snapshot from existing, verified final-render evidence.

No model imports, remote writes, new rendering, or training are performed.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import shutil

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
SOURCE = ROOT / "work/native_final_acceptance"
ANALYSIS = ROOT / "work/native_global_road_replay/final_analysis"
RAWS = (105, 113, 130, 137)
EXPECTED = {
    "reference": {
        105: "995c29c9c33e6db924c851c7584d90104d8185b98ab979c72eaa55574c8fa3d2",
        113: "91049c0d03b25aca3e76e0f2c6da26fa256e705d13188c580734060b6842c0f2",
        130: "027da08df22e1d65ad07ab092954f9b730f88f18014acf8e6a15bc55ab21ea7f",
        137: "e43f1cf1eca7793a268cdb13d69dda27f31cb96551735c4e5fc756037c8dc228",
    },
    "candidate": {
        105: "b61a482952e1ab2b81c2b1ae763bca3b1d4e6c76a017a59e4b24cb2a1b944e29",
        113: "28ecdcdff00fded584579a36b2588c55cbf33d80218eddbf70747fced2d2049c",
        130: "1980fe61b8608f7f388dbf22091c4bfc38d7c3b4135e51a0b74a0890ccb11b80",
        137: "27e71db947a58ecc9e0d92231f905f3cbba9cc7963880de65453eea8a2292efa",
    },
}


def load(path):
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"))


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def font(size):
    path = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


assets = OUT / "assets"
evidence = OUT / "evidence"
assets.mkdir(exist_ok=True)
evidence.mkdir(exist_ok=True)
images = {}
verified = []
for state, sub in (("reference", "reference"), ("candidate", "render")):
    for raw in RAWS:
        source = SOURCE / sub / f"raw{raw}_delete.png"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == EXPECTED[state][raw], (state, raw, "remote/local SHA256 mismatch")
        destination = assets / f"{state}_{raw}.png"
        shutil.copyfile(source, destination)
        image = Image.open(destination).convert("RGB")
        assert image.size == (480, 320)
        images[state, raw] = image
        verified.append(dict(state=state, raw=raw, source=str(source),
                             file=str(destination.relative_to(OUT)).replace("\\", "/"), sha256=digest))

receipt = load(SOURCE / "receipt.json")
candidate_manifest = load(SOURCE / "render/render_manifest.json")
reference_manifest = load(SOURCE / "reference/render_manifest.json")
assert receipt["binding"]["native_iteration"] == 10000
assert receipt["binding"]["road_iteration"] == 1000
assert receipt["M2_all_27_tensors_bitwise_unchanged_CPU"] is True
assert candidate_manifest["M2_sha256"] == reference_manifest["M2_sha256"]
assert candidate_manifest["only_a37_changes_with_delete"] is True
assert candidate_manifest["original_own_camera_per_raw"] is True
assert candidate_manifest["preflight_only"] is False
for source, name in ((SOURCE / "receipt.json", "candidate_receipt.json"),
                     (SOURCE / "render/render_manifest.json", "candidate_render_manifest.json"),
                     (SOURCE / "reference/render_manifest.json", "reference_render_manifest.json"),
                     (ANALYSIS / "analysis_summary.txt", "analysis_summary.txt"),
                     (ANALYSIS / "tables/delete4_fixed_domain.csv", "delete4_fixed_domain.csv"),
                     (ANALYSIS / "tables/ROI_common_mask_focus8.csv", "ROI_common_mask_focus8.csv")):
    shutil.copyfile(source, evidence / name)

with (ANALYSIS / "tables/ROI_common_mask_focus8.csv").open(encoding="utf-8-sig", newline="") as stream:
    blue_rows = [r for r in csv.DictReader(stream) if r["state"] == "delete"]
with (ANALYSIS / "tables/delete4_fixed_domain.csv").open(encoding="utf-8-sig", newline="") as stream:
    fixed_rows = list(csv.DictReader(stream))

data = dict(
    method_code="RCR-1",
    method_name="RGB Constraint Restoration / RGB 约束恢复",
    created_at=datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds"),
    scope="Existing complete final candidate versus the accepted-current M2/delete reference; four saved delete frames only.",
    quality_accepted=False,
    acceptance="NOT_ACCEPTED_VISUAL_REGRESSION",
    regression="raw130/raw137: silver residual. Net brightening is native-static dominated; environment*T still carries car-shaped structure. Exact static rows and training cause remain unisolated.",
    reference="Original ownership native10K + original road1K/gates/fine + fixed complete M2",
    candidate="Native10K with only the missing global RGB coefficient1 restored + fresh road1K/gates + fresh coarse6x5/fine11x9 + the same fixed complete M2",
    M2_sha256=candidate_manifest["M2_sha256"],
    M2_all_27_tensors_bitwise_unchanged_CPU=True,
    source_candidate="scene008_blue_global_M2_final_20261105_r1",
    source_reference="scene008_blue_current_M2_reference_20261105_r1",
    image_verification="All eight PNG SHA256 hashes match a fresh read-only remote check on 2026-11-05.",
    images=verified, blue_rows=blue_rows, fixed_domain_rows=fixed_rows,
    frames={
        "105": dict(note="检查删车后的道路接管与车尾衔接。", default_roi="tail", rois={
            "tail": dict(label="车尾与接管区域", xyxy=[205, 178, 305, 238]),
            "road": dict(label="道路衔接", xyxy=[190, 215, 330, 285])}),
        "113": dict(note="原框内偏蓝明显减轻；同时检查补全边界与车尾。", default_roi="blue", rois={
            "blue": dict(label="用户原框：白线旁偏蓝", xyxy=[276, 175, 309, 220]),
            "tail": dict(label="车尾与接管区域", xyxy=[202, 185, 300, 242])}),
        "130": dict(note="回退：车尾银灰亮片增显；净亮化主要来自新静态层，环境×T仍有车形结构。", default_roi="tail", rois={
            "tail": dict(label="车尾银灰亮残影", xyxy=[202, 192, 298, 244]),
            "road": dict(label="道路衔接", xyxy=[190, 215, 330, 285])}),
        "137": dict(note="回退：银灰亮片增显；该块不在道路接管范围，具体训练成因仍未隔离。", default_roi="tail", rois={
            "tail": dict(label="车尾银灰亮残影", xyxy=[202, 192, 298, 244]),
            "blue": dict(label="右侧原蓝斑区域", xyxy=[276, 175, 309, 220]),
            "band": dict(label="原横条区域", xyxy=[210, 213, 290, 233])}),
    },
)
for frame in data["frames"].values():
    frame["rois"]["a5"] = dict(label="a5 左侧蓝车（待追溯）", xyxy=[0, 160, 175, 320])
silver = load(evidence / "rcr1_silver315_contributions.json")
data["silver315"] = {key: silver[key] for key in (
    "pixels", "mask_definition", "component_premultiplied_RGB", "output_delta_mean_RGB",
    "confirmed_source", "unresolved", "spatial_structure_observation", "validation")}
data["closeout"] = dict(status="ARCHIVED_NOT_ADOPTED", native_updates=10000,
    road_updates=1000, task_total_updates=26609, formal_linear_solves=54,
    preflight_linear_solves=54, original_M2_retained=True, human_verdict=None,
    next_RCR_training_controller=False, automatic_shutdown_executed=False,
    prepared_not_executed="backgroundJoint1: CPU-only gradient-domain check, no GPU run",
    pending_followup="a5 blue-car damage may already exist in reference; trace earliest affected stage separately")
data["holdout_scope"] = "HELD9 only withholds road-RGB calibration targets; native/road-gate stages used the same49-frame clip. This is internal diagnostic evaluation, not unseen-native generalization."
for source, name in ((ANALYSIS / "final_cpu_comparison.json", "final_cpu_comparison.json"),
                     (ROOT / "work/native_lineage_counterfactual/code_audit/inference_code_audit.json", "inference_code_audit.json")):
    shutil.copyfile(source, evidence / name)
dump(OUT / "report_data.json", data)

# Lossless full-frame overview: no colour or exposure changes.
overview = Image.new("RGB", (1020, 1576), "white")
draw = ImageDraw.Draw(overview)
draw.text((22, 15), "RCR-1 完整 delete 对照：原版 / 去蓝候选（未采用）", font=font(24), fill="#192936")
draw.text((22, 50), "两端均已接道路层，使用同一完整 M2。PNG 原图；未调色。", font=font(16), fill="#52616c")
for i, raw in enumerate(RAWS):
    y = 90 + i * 365
    draw.text((24, y), f"raw{raw} · 原版 delete", font=font(18), fill="#192936")
    draw.text((516, y), f"raw{raw} · RCR-1 delete · 未采用", font=font(18), fill="#9a421b")
    overview.paste(images["reference", raw], (24, y + 30))
    overview.paste(images["candidate", raw], (516, y + 30))
overview.save(OUT / "delete_overview.png")

# Matched nearest-neighbour crops expose the regression without smoothing.
tails = Image.new("RGB", (1016, 656), "white")
draw = ImageDraw.Draw(tails)
draw.text((22, 14), "车尾亮残影：原版 delete / 去蓝版 delete", font=font(24), fill="#192936")
draw.text((22, 50), "同坐标局部 4× 最近邻放大；去蓝候选尚未通过质量验收。", font=font(16), fill="#52616c")
for i, raw in enumerate((130, 137)):
    y = 88 + i * 278
    draw.text((24, y), f"raw{raw} · 原版", font=font(18), fill="#192936")
    draw.text((516, y), f"raw{raw} · 去蓝候选", font=font(18), fill="#9a421b")
    for state, x in (("reference", 24), ("candidate", 516)):
        crop = images[state, raw].crop((202, 192, 298, 244)).resize((384, 208), Image.Resampling.NEAREST)
        tails.paste(crop, (x + 45, y + 32))
tails.save(OUT / "delete_tail_130_137.png")

template = (OUT / "report_template.html").read_text(encoding="utf-8")
html = template.replace("__REPORT_DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
(OUT / "index.html").write_text(html, encoding="utf-8")
print(json.dumps(dict(report=str(OUT / "index.html"), frames=list(RAWS),
                      verified_images=len(verified), quality_accepted=False,
                      source_native_iteration=10000, source_road_iteration=1000), ensure_ascii=False))
