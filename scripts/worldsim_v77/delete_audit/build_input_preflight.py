"""Make a CPU-only visual audit of frozen targets and verified public RGB."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read(path: Path):
    return json.loads(path.read_text())


def target_crop(image: Image.Image, box: list[float]) -> Image.Image:
    left, top, right, bottom = box
    cx, cy = (left + right) / 2, (top + bottom) / 2
    side = max(180, (right - left) * 2.4, (bottom - top) * 2.4)
    side = min(side, image.width, image.height)
    x0 = max(0, min(image.width - side, cx - side / 2))
    y0 = max(0, min(image.height - side, cy - side / 2))
    return image.crop((round(x0), round(y0), round(x0 + side), round(y0 + side)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    selection = read(args.run / "selection.json")
    sources = read(args.run / "expected_sources.json")
    geometry = read(args.run / "clip_geometry.json")
    check = read(args.run / "pub_rgb_check.json")
    assert check["all_selected_rgb_available"], "Do not publish an incomplete input gallery"
    assert len(selection["clips"]) == len(sources["clips"]) == len(geometry["clips"]) == 70
    src_by_id = {c["clip_id"]: c for c in sources["clips"]}
    geom_by_id = {c["clip_id"]: c for c in geometry["clips"]}
    args.out.mkdir(parents=True, exist_ok=True)
    images = args.out / "images"
    images.mkdir(exist_ok=True)
    cards = []
    for clip in selection["clips"]:
        cid = clip["clip_id"]
        geo = geom_by_id[cid]
        src = src_by_id[cid]
        anchor = geo["prompt_frame"]
        grow = geo["frames"][anchor]
        srow = src["frames"][anchor]
        assert grow["filename"] == srow["filename"]
        assert grow["target"] and grow["target"]["box_xyxy"]
        image_path = args.run / "rgb" / srow["filename"]
        with Image.open(image_path) as original:
            frame = original.convert("RGB")
        # GT projection is in the 1024×576 model-input grid; source JPEG is 1600×900.
        box = [grow["target"]["box_xyxy"][i] * (frame.width / 1024 if i % 2 == 0 else frame.height / 576)
               for i in range(4)]
        full = frame.resize((800, 450), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(full)
        scaled = [box[i] * (800 / frame.width if i % 2 == 0 else 450 / frame.height)
                  for i in range(4)]
        for neighbor in grow["neighbors"]:
            hull = [(x * 800 / 1024, y * 450 / 576) for x, y in neighbor["hull"]]
            draw.polygon(hull, outline="#35bdf3", width=2)
        draw.rectangle(scaled, outline="#ffd600", width=4)
        target_hull = [(x * 800 / 1024, y * 450 / 576) for x, y in grow["target"]["hull"]]
        draw.polygon(target_hull, outline="#ffd600", width=3)
        label = f"{cid}  TARGET"
        lx = max(0, min(680, int(scaled[0])))
        ly = max(0, int(scaled[1]) - 26)
        draw.rectangle((lx, ly, lx + 118, ly + 24), fill="#242000")
        draw.text((lx + 5, ly + 3), label, fill="#fff566")
        detail = target_crop(frame, box).resize((400, 400), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (1200, 450), "#12151a")
        canvas.paste(full, (0, 0))
        canvas.paste(detail, (800, 25))
        cd = ImageDraw.Draw(canvas)
        cd.text((810, 5), "YELLOW GT TARGET / CYAN OTHER GT", fill="#e9eef3")
        path = images / f"{cid}.jpg"
        canvas.save(path, "JPEG", quality=83, optimize=True)
        cards.append({
            "clip_id": cid, "scene": clip["scene"], "category": clip["category"],
            "camera": clip["camera"], "instance_token": clip["instance_token"],
            "anchor_frame": anchor, "source_file": srow["filename"],
            "max_area_frame": geo["prompt_policy"]["max_area_frame"],
            "other_gt_fraction_in_prompt_box": geo["prompt_policy"]["chosen_other_GT_fraction_in_box"],
            "unclipped_prompt_available": geo["prompt_policy"]["unclipped_candidate_available"],
            "size": clip["size_bucket"], "visibility": clip["occlusion_proxy"],
            "behind_vehicle_proxy": clip["behind_vehicle_proxy"],
            "night_proxy": clip["night_proxy"],
            "input_difficulty_proxy": clip["input_difficulty_proxy"],
            "difficulty_factors": clip["difficulty_factors"],
            "image": f"images/{cid}.jpg", "human_verdict": None,
        })
    (args.out / "manifest.json").write_text(json.dumps({
        "task_id": selection["task_id"], "run_id": selection["run_id"],
        "stage": "CPU input preflight only; no SAM2 or DriveEditor inference",
        "audit_scenes": selection["audit_scenes"],
        "final_quarantined_scenes": selection["final_quarantined_scenes"],
        "required_unique_rgb": check["required_unique_rgb"],
        "extracted_and_decoded": check["extracted_and_decoded"],
        "prompt_policy_revision": "p2_uniform_instance_separability_before_GPU",
        "cards": cards, "human_verdict": None,
    }, ensure_ascii=False, indent=2) + "\n")
    rows = []
    for card in cards:
        cid = html.escape(card["clip_id"])
        scene = html.escape(card["scene"])
        factors = ", ".join(card["difficulty_factors"]) or "无代理难点"
        a025 = ('<p><a href="diagnostics/A025_old_maxarea.jpg">旧第2帧提示框</a> · '
                '<a href="diagnostics/A025_overlap_montage.jpg">第0–19帧抽样目标/邻车投影对照</a></p>'
                if cid == "A025" else "")
        rows.append(f'''<article id="{cid}"><div class="heading"><h2>{cid} · {scene}</h2>
          <span>{html.escape(card["camera"])} · {html.escape(card["category"])} · {html.escape(card["input_difficulty_proxy"])}</span></div>
          <img loading="lazy" src="{card["image"]}" alt="{cid} 目标黄框与原始RGB局部裁剪">
          <p>目标实例：<code>{card["instance_token"]}</code>；SAM2候选提示帧：{card["anchor_frame"]}/25
          （最大面积帧为{card["max_area_frame"]}）；提示框内其他GT车辆投影占比：{card["other_gt_fraction_in_prompt_box"]:.1%}；
          大小：{card["size"]}；GT可见度：{card["visibility"]}；后方邻车投影：{card["behind_vehicle_proxy"]}；
          夜间代理：{card["night_proxy"]}。输入难度依据：{html.escape(factors)}。</p>
          {a025}<details><summary>原始文件</summary><code>{html.escape(card["source_file"])}</code></details></article>''')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>v77 DELETE · RGB输入核验</title>
<style>body{font:16px/1.55 system-ui,sans-serif;background:#101318;color:#e9eef3;max-width:1280px;margin:auto;padding:28px}
h1{font-size:30px}h2{margin:0;font-size:19px}p{margin:12px 0}code{overflow-wrap:anywhere;color:#d8dcff}
article{background:#1b2028;border:1px solid #394354;border-radius:12px;margin:24px 0;padding:18px}
article img{width:100%;height:auto;border-radius:5px}.heading{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}
.note{background:#203348;border-left:5px solid #73bbf5;padding:14px 18px}svg{width:100%;height:auto}
summary{cursor:pointer;color:#9dd6ff}a{color:#9dd6ff}</style>
<h1>v77 DELETE · CPU输入核验</h1><p class="note">本页仅展示预先冻结的目标和公共盘原始RGB。
黄色框与多边形是同一个目标车辆的3D GT投影；蓝色是其他GT车辆的投影。黄色框可能包进邻车像素，
它只用于SAM2提示，绝不直接作为删除区域；真正单车实例mask还须GPU阶段验证。
右侧是同一原图的放大裁剪。所有图像均来自原始nuScenes RGB；尚未运行SAM2或DriveEditor，
不代表DELETE效果，也不填写人工质量分。A025原最大面积帧曾同时框入旁车；统一候选帧规则已把提示改到第12帧，
其框内其他GT车辆投影由14.6%降至3.3%。</p>
<svg viewBox="0 0 1180 100" role="img" aria-label="输入核验组件图"><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#b8d4e9"/></marker></defs>
<g fill="#213752" stroke="#75b9e2" stroke-width="2"><rect x="5" y="18" width="235" height="65" rx="9"/><rect x="320" y="18" width="235" height="65" rx="9"/><rect x="635" y="18" width="235" height="65" rx="9"/><rect x="950" y="18" width="225" height="65" rx="9"/></g>
<g fill="white" text-anchor="middle" font-size="17"><text x="122" y="56">val 元数据 + 公共盘RGB</text><text x="437" y="56">固定scene / 单车clip</text><text x="752" y="56">解码 + 目标黄框</text><text x="1062" y="56">GPU推理待启动</text></g>
<g stroke="#b8d4e9" stroke-width="3" marker-end="url(#a)"><path d="M242 50H315"/><path d="M557 50H630"/><path d="M872 50H945"/></g></svg>
<p>官方nuScenes val：150 scene；本次审计45 scene / 70个单车clip；25个scene隔离给后续final评测。
70段各约2.6秒，合计1820个帧位置；实际涉及1802张不同JPEG，均已从公共盘提取并逐张解码。</p>
<p>昼夜、大小、相机、可见度和邻车覆盖来自固定GT/元数据代理，不是人工看结果后挑样。
4个帧位置的最近原始相机曝光差70–85ms，已单列为输入时间误差。完整清单见同目录 <a href="manifest.json">manifest.json</a>。</p>
''' + "\n".join(rows) + "</html>"
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(f"INPUT_PREFLIGHT {len(cards)} cards {check['extracted_and_decoded']} decoded -> {args.out}")


if __name__ == "__main__":
    main()
