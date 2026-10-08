"""R001 mask restart: record CPU evidence without generating masks or training."""

import html
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


TASK = "WS-V77-TARGET-PROTECTED-20260929"
BASE = Path("/root/autodl-tmp/runs/worldsim_v77") / TASK
OLD = BASE / "r50/inputs/R001"
OUT = BASE / "r53"
MODEL = Path("/root/autodl-tmp/models/worldsim_v77_sam3")
REPO = Path("/root/autodl-tmp/motion_proj_v77")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    web = OUT / "review"
    assets = web / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    c = json.loads((OLD / "case.json").read_text())
    provenance = json.loads((MODEL / "provenance.json").read_text())
    assert (MODEL / "sam3.pt").stat().st_size == provenance["bytes"]
    box = c["frames"][0]["target"]["box_xyxy"]
    command = (
        "/root/autodl-tmp/envs/worldsim-v77-sam3/bin/python "
        f"{REPO}/scripts/worldsim_v77/target_protected/iteration20/sam3_real_mask.py "
        f"--rgb-dir {OLD}/rgb --checkpoint {MODEL}/sam3.pt "
        f"--prompt-frame 0 --gt-box {' '.join(str(x) for x in box)} "
        f"--output {OUT}/sam3_R001"
    )
    (OUT / "gpu_command.txt").write_text(command + "\n", encoding="utf-8")
    rows, pictures = [], []
    for i in (0, 4, 9):
        rgb = np.asarray(Image.open(OLD / "rgb" / f"{i:05d}.jpg").convert("RGB"))
        sam = np.asarray(Image.open(OLD / "sam" / f"{i:05d}.png")) > 0
        rect = np.asarray(Image.open(OLD / "model_mask" / f"{i:05d}.png")) > 0
        yy, xx = np.where(rect)
        box_area = int((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1))
        row = {"frame": i, "old_sam2_pixels": int(sam.sum()),
               "old_model_mask_pixels": int(rect.sum()),
               "old_model_bbox_area": box_area,
               "old_model_is_filled_rectangle": int(rect.sum()) == box_area}
        rows.append(row)
        for role, mask, color in (("rgb", None, None), ("sam2", sam, (0, 230, 170)),
                                  ("old_model_rectangle", rect, (255, 140, 30))):
            arr = rgb.copy()
            if mask is not None:
                arr[mask] = (arr[mask] * .55 + np.asarray(color) * .45).astype(np.uint8)
            im = Image.fromarray(arr)
            draw = ImageDraw.Draw(im)
            draw.rectangle([8, 8, 430, 36], fill=(15, 20, 25))
            draw.text((14, 14), f"R001 f{i:02d} / historical {role} / NOT SAM3 output", fill="white")
            name = f"f{i:02d}_{role}.jpg"
            im.save(assets / name, quality=94)
            pictures.append((i, role, name))
    save(OUT / "old_mask_evidence.json", rows)
    save(OUT / "registration.json", {
        "task_id": TASK, "run_id": "r53", "phase": "SAM3 CPU preparation",
        "case_id": "R001", "source": str(OLD), "source_role": "real train DEV; already exposed",
        "checkpoint": provenance, "gt_box_role": "prompt only; not deletion region or Y",
        "mask_policy": "exact instance silhouette; synchronize native mask, r47 hole and new alpha",
        "gpu_inference_windows": 0, "training_steps": 0, "training_admission": False,
        "identity_review": "pending actual SAM3 GPU result", "human_verdict": None,
        "seed": None, "resources": "CPU only; GPU memory/driver compatibility untested",
        "failure_ledger_refs": ["V77-F02"],
        "failure_ledger_delta": "record contour-to-rectangle interface; no new model result",
        "next": "Run one R001 SAM3 sequence, inspect identity/coverage/contour, then determine supervision; no auto training"
    })
    cleanup = json.loads(Path("/root/autodl-tmp/cleanup_manifests/v77_20261009_200GiB/deletion_complete.json").read_text())
    disk = shutil.disk_usage(OUT)
    save(OUT / "disk_after_sam3.json", {"free_bytes": disk.free, "free_GiB": disk.free / 2**30,
                                        "cleanup": cleanup, "sam3_weight_bytes": provenance["bytes"]})
    content = '''<!doctype html><meta charset="utf-8"><title>r53 · SAM3 CPU 准备</title>
<style>body{background:#101820;color:#e3edf4;font:16px/1.65 system-ui;margin:32px;max-width:1500px}a{color:#89cfff}h1,h2{line-height:1.3}.banner{background:#473922;padding:18px;border-radius:8px}.flow{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.flow b{background:#25405b;padding:16px;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}img{width:100%}figure{margin:0}figcaption{font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1a2937;padding:16px}table{border-collapse:collapse}td,th{padding:8px 16px;border:1px solid #41586a}</style>
<h1>r53 · SAM3 真轮廓 CPU 准备</h1>
<p class="banner">尚未运行 SAM3 分割、DriveEditor 推理或训练。以下图片全部来自历史 SAM2，仅用于定位旧接口把轮廓改成矩形的问题。权重已下载，不代表实例质量已通过。</p>
<div class="flow"><b>原始 RGB + 目标框 prompt</b>→<b>SAM3 视频分割（待 GPU）</b>→<b>同一实例轮廓</b>→<b>DELETE mask / r47 hole / 新 alpha</b>→<b>逐帧身份与覆盖检查</b></div>
<h2>本轮关闭的数据</h2><p>r52 矩形道路遮挡重复原模型已有的背景补全任务，不提供银车背后的正确监督；矩形硬拼接伪标签出现接缝、标线和结构错误，单例微调回退。两类数据均退出训练；原候选、分数与权重保留作为失败证据。分数大于 1 不再使这些旧样本自动准入。</p>
<p><a href="../v77-single-case-r52/index.html">r52 完整历史对照与失败评分（本地报告）</a> · <a href="https://arxiv.org/html/2412.19458v2#S3">DriveEditor 数据构造</a></p>
<h2>旧 mask 接口证据</h2><p>绿：历史 SAM2 实例；橙：旧模型真正接收的矩形洞。新入口跳过外接矩形转换。真实轮廓仍需检查目标身份、边缘漏分和邻车侵入；它本身不能提供被遮挡区域的真实 RGB。</p>
'''
    for i in (0, 4, 9):
        content += f"<h3>f{i:02d}</h3><div class='grid'>"
        for frame, role, name in pictures:
            if frame == i:
                label = {"rgb": "原始 RGB", "sam2": "历史 SAM2 轮廓", "old_model_rectangle": "旧 model_mask：矩形"}[role]
                content += f"<figure><img src='assets/{name}'><figcaption>{label}</figcaption></figure>"
        content += "</div>"
    content += "<pre>" + html.escape(json.dumps(rows, ensure_ascii=False, indent=2)) + "</pre>"
    content += f"<h2>权重与磁盘</h2><p>SAM3 来源：<a href='{provenance['mirror_page']}'>ModelScope facebook/sam3</a>；仅原生 sam3.pt，{provenance['bytes']:,} 字节。数据盘当前可用 {disk.free / 2**30:.1f} GiB；清理实际释放 203.7 GiB（约 218.7 GB）。nuScenes、关键权重与实验结果保留；AV2 原始 sensors 缓存需要时重下。</p>"
    content += "<h2>开 GPU 后只做第一步</h2><p>先运行 R001 十帧分割并检查真实目标轮廓；不把模型换代当成质量结论，不恢复 r52 训练。</p><pre>" + html.escape(command) + "</pre>"
    (web / "index.html").write_text(content, encoding="utf-8")
    print(web / "index.html")


if __name__ == "__main__":
    main()
