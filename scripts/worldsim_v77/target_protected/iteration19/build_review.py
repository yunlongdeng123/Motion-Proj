"""r52 local review: distinguish QA admission, single-image training, and exploration."""
from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(a: argparse.Namespace) -> None:
    root = a.root.resolve()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assets = out / "assets"
    assets.mkdir(exist_ok=True)

    def esc(value: object) -> str:
        return html.escape(str(value))

    def copy(path: Path, label: str) -> str:
        dest = assets / (label + path.suffix)
        shutil.copy2(path, dest)
        return "assets/" + dest.name

    def pic(path: Path, title: str, label: str, note: str = "", pending: bool = False) -> str:
        if not path.exists():
            return (f'<figure><b>{esc(title)}</b><p>结果待同步：{esc(path.relative_to(root))}</p></figure>'
                    if pending else "")
        src = copy(path, label)
        return (f'<figure><a href="{src}"><img loading="lazy" src="{src}"></a>'
                f'<figcaption><b>{esc(title)}</b><p>{esc(note)}</p></figcaption></figure>')

    def evidence_link(path: Path, label: str, title: str) -> str:
        return f'<a href="{copy(path, label)}">{esc(title)}</a>' if path.exists() else ""

    def eval_grid(labels: list[tuple[str, str]], roles: tuple[str, ...],
                  pending: bool = False) -> str:
        blocks = ['<div class="grid">']
        for role in roles:
            for label, title in labels:
                for arm, arm_title in (("R001_official", "官方"), ("R001_r47", "固定 r47")):
                    path = root / "evaluation" / label / arm / (role + ".png")
                    blocks.append(pic(path, f"{title} / {arm_title} / {role}",
                                      f"{label}_{arm}_{role}", pending=pending))
        blocks.append("</div>")
        return "\n".join(blocks)

    def output_review(path: Path, title: str, label: str) -> str:
        review = read_json(path)
        if review is None:
            return f'<p class="notice">{esc(title)}：等待独立 review 文件 {esc(path.relative_to(root))}。</p>'

        scores = []
        aliases = {"before_finetune":"原权重", "after_realRGB_0064":"真实RGB微调64步", "after_best1_0064":"单张伪标签微调64步", "R001_official":"官方接口", "R001_r47":"固定r47接口", "native":"原生输出", "compose":"最终写回", "capacity_320":"320×576容量对照"}

        def collect(node: object, trail: tuple[str, ...] = ()) -> None:
            if isinstance(node, dict):
                score = node.get("ai_score")
                if score is None and "reason" in node:
                    score = node.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool):
                    scores.append((trail, score, node.get("pass"), node.get("reason", "")))
                for key, value in node.items():
                    collect(value, trail + (str(key),))
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    collect(value, trail + (str(index),))

        collect(review)
        blocks = [f'<h3>{esc(title)}</h3>']
        if scores:
            blocks.append('<table><thead><tr><th>条件 / 接口 / 版本</th><th>独立分数</th>'
                          '<th>通过</th><th>简要原因</th></tr></thead><tbody>')
            for trail, score, passed, reason in scores:
                verdict = "未填" if passed is None else ("是" if passed else "否")
                blocks.append(f'<tr><td>{esc(" / ".join(aliases.get(t,t) for t in trail))}</td><td>{score:g}</td>'
                              f'<td>{verdict}</td><td>{esc(reason)}</td></tr>')
            blocks.append('</tbody></table>')
        else:
            blocks.append('<p>独立 review 已存在，但没有可识别的逐输出数值分数；请查看原始记录。</p>')
        finding = next((review[key] for key in ("comparative_finding", "conclusion",
                                                 "summary", "overall_finding", "findings", "comparison")
                        if review.get(key)), None)
        if isinstance(finding, dict):
            for key, value in finding.items():
                if isinstance(value, str):
                    blocks.append(f'<p><b>{esc(key)}：</b>{esc(value)}</p>')
        elif isinstance(finding, str):
            blocks.append(f'<p><b>结论：</b>{esc(finding)}</p>')
        elif isinstance(finding, list):
            for value in finding:
                if isinstance(value, str):
                    blocks.append(f'<p><b>结论：</b>{esc(value)}</p>')
        blocks.append(f'<p>人工 verdict：{esc("null" if review.get("human_verdict") is None else review["human_verdict"])}；'
                      f'{evidence_link(path, label, "独立 review 原始 JSON")}</p>')
        return "\n".join(blocks)

    batch_reviews = []
    for index in range(10):
        frame = f"f{index:02d}"
        qa_path = root / "batch_reviews" / f"R001_{frame}_imagegen.json"
        qa = read_json(qa_path)
        if qa:
            batch_reviews.append((qa, qa_path,
                                  root / "batch_composition" / frame / frame /
                                  "composed_fullframe_registered.png"))
    priority_qa_path = root / "user_priority_composition" / "quality_review.json"
    priority_qa = read_json(priority_qa_path)
    if priority_qa:
        batch_reviews.append((priority_qa, priority_qa_path,
                              root / "user_priority_composition" / "user_priority_roi" /
                              "composed_fullframe_registered.png"))
    selection_path = root / "batch_reviews" / "best1_selection_review.json"
    selection = read_json(selection_path)
    selected_id = selection.get("selected_candidate_id") if selection else None
    admitted = sum(float(qa.get("score", 0)) > 1 for qa, _, _ in batch_reviews)
    selected_count = sum(qa.get("candidate_id") == selected_id for qa, _, _ in batch_reviews)
    selected_frame = selection.get("selected_source_frame_index") if selection else None

    parts = ['''<!doctype html><meta charset="utf-8"><title>r52 · R001 单例审核</title>
<style>
body{background:#111820;color:#e8eef5;font:16px/1.65 system-ui;margin:28px auto;max-width:1600px;padding:0 22px}
h1{font-size:30px}h2{margin-top:40px}a{color:#8bc7ff}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
figure{margin:0;background:#1a2531;padding:12px;border-radius:8px}
img,video{width:100%;height:auto}figcaption{font-size:14px}
p{margin:8px 0}.flow{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.box{border:1px solid #58829c;padding:12px;border-radius:8px}
.notice{border-left:4px solid #e4b660;padding:12px;background:#26303c}
details{margin:18px 0;padding:12px;background:#1a2531;border-radius:8px}
summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border:1px solid #476071;text-align:left;vertical-align:top}
</style>
<h1>r52 · R001 单例学习与伪标签审核</h1>
<p>scene-0290 / CAM_FRONT。只讨论 R001 单例开发与静态单图容量，不能据此声称泛化或连续视频监督。人工 verdict 保持 null。</p>
<div class="flow"><div class="box">真实 RGB + 原 mask<br>同车参考 + 投影几何</div>→
<div class="box">局部候选 → mask 硬合成<br>逐图独立评分</div>→
<div class="box">只选 f04 单张伪标签<br>静态重复窗 + 主 U-Net</div>→
<div class="box">官方 / 固定 r47<br>原生与写回分列</div></div>''']
    parts.append(
        f'<p class="notice"><b>{len(batch_reviews)} 候选，{admitted} 个分数严格 &gt;1，'
        f'历史实验只有 {selected_count} 张实际入训</b>（{esc(selected_id or "待选择")}）。'
        '“分数&gt;1”仅是当时的候选准入，不等于训练用过。该单图实验只用了 f04；'
        '旧 all10 训练保留为历史探索。生成内容不是隐藏背景真实 GT，静态重复窗不评价时序。</p>'
    )
    if len(batch_reviews) != 11 or admitted != 11 or selected_count != 1:
        parts.append('<p class="notice">注意：当前同步的候选/评分或选择记录不完整，请核对原始 JSON。</p>')

    retired = read_json(root / "TRAINING_INPUTS_RETIRED.json")
    if retired:
        parts.append('<p class="notice"><b>本轮数据已停止入训，只保留 failure 证据。</b>道路矩形proxy重复已有背景inpainting任务；矩形拼接伪标签产生接缝与结构错误。单张64步没有过关，全尺寸官方/r47原生0.5→0.2、写回0.6→0.3。不能把主干可学习或同场景输出变化当作新增DELETE监督收益。</p>')
        parts.append('<p>下一步：SAM3真实实例轮廓 → 独立编辑洞 → 原图/缩放/latent一致性检查；GT框仅用于prompt，禁止重新外接矩形。真实轮廓不能自动提供隐藏背景GT，监督来源仍须单独解决。SAM3尚待官方权重授权，本页没有SAM3推理结果。</p>')
    ep = root / "evidence_pack"
    parts.append('<h2>输入证据示例 · f05</h2><div class="grid">')
    for key, title in (("A_original_RGB_ROI", "目标帧真实 RGB"),
                       ("B_instance_control", "独立编辑范围 / O N U / B 投影"),
                       ("C_B_anchor", "真实邻帧参考；未对齐到当前视角")):
        parts.append(pic(ep / (key + ".png"), title, "evidence_" + key))
    parts.append('</div><p>绿色 O 是保留车框投影代理，蓝色 N 是 LiDAR 背景支持，灰色 U 未知；不能把 U 或 N 以外视为已知背景。</p>')

    parts.append('<h2>用户优先候选 · f05</h2>')
    parts.append('<p>用户提供的是独立 ROI 候选，转述 ROI 约2.1分。以下分数针对真实 R001 f05 mask/query 的 registered 硬合成，'
                 '与ROI分数分开；这张图未匹配到已有生成文件，具体模型与prompt尚未核实。</p><div class="grid">')
    priority_dir = root / "user_priority_composition" / "user_priority_roi"
    parts.append(pic(root / "batch_packs" / "f05" / "query_full.jpg",
                     "f05 当前源帧原图", "user_priority_source_f05"))
    parts.append(pic(root / "batch_packs" / "f05" / "model_mask_full.png",
                     "f05 原始目标编辑 mask", "user_priority_mask_f05"))
    parts.append(pic(root / "user_priority_roi" / "candidate.png",
                     "用户提供的原 ROI 候选", "user_priority_candidate"))
    parts.append(pic(priority_dir / "composed_roi_registered.png",
                     "配准后 ROI · 原 mask 硬写回", "user_priority_hard_roi"))
    parts.append(pic(priority_dir / "composed_fullframe_registered.png",
                     "完整原帧 · 原 mask 硬写回", "user_priority_hard_full"))
    parts.append('</div>')
    if priority_qa:
        parts.append(f'<p>独立分数 <b>{esc(priority_qa.get("score"))}</b>；严格 &gt;1，'
                     f'候选准入；实际入训：否。{esc(priority_qa.get("notes", ""))} '
                     f'{evidence_link(priority_qa_path, "user_priority_quality", "原始质量记录")}</p>')
    parts.append('<p>' + evidence_link(root / 'user_priority_roi' / 'provenance.json',
                 'user_priority_provenance', '用户候选来源记录') + '</p>')

    parts.append('<h2>实际选择 · f04 单张伪标签</h2>')
    if selection:
        parts.append(f'<p>选择 <b>{esc(selected_id)}</b>，源帧 f{int(selected_frame):02d}，'
                     f'版本 {esc(selection.get("selected_variant"))}；'
                     f'评审理由：{esc(selection.get("selection_reason", ""))} '
                     f'{evidence_link(selection_path, "best1_selection", "原始选择记录")}</p>')
    parts.append('<div class="grid">')
    parts.append(pic(root / "batch_packs" / "f04" / "query_full.jpg",
                     "f04 当前源帧原图", "selected_f04_source"))
    parts.append(pic(root / "batch_packs" / "f04" / "model_mask_full.png",
                     "f04 原始目标编辑 mask", "selected_f04_mask"))
    parts.append(pic(root / "batch_composition" / "f04" / "f04" /
                     "composed_roi_registered.png", "f04 入选 ROI 硬合成",
                     "selected_f04_roi"))
    parts.append(pic(root / "batch_composition" / "f04" / "f04" /
                     "composed_fullframe_registered.png", "f04 入选完整帧硬合成",
                     "selected_f04_full"))
    parts.append('</div><p>f04 独立分数 1.5；f05 也为 1.5，但其 B 更偏宽、横线黑痕和接缝更明显。'
                 '用户优先图硬合成 1.2，未达到同分或更高分优先条件。以上是单图候选选择，不是隐藏 GT 判定。</p>')
    parts.append('<p>' + evidence_link(root / 'batch_generated/f04/prompt_zh.txt',
                 'selected_f04_prompt', '入选图完整 prompt') + ' · ' +
                 evidence_link(root / 'batch_generated/f04/attempt.json',
                 'selected_f04_source', '生成来源和调用记录') + '</p>')

    parts.append('<details><summary>展开 11 张候选池：分数、准入与实际入训逐张分开</summary><div class="grid">')
    for qa, qa_path, full_path in batch_reviews:
        candidate_id = str(qa.get("candidate_id", "unknown"))
        score = float(qa.get("score", 0))
        admitted_here = score > 1
        chosen_here = candidate_id == selected_id
        note = (f"独立质量分 {score:g}；>1 候选准入：{'是' if admitted_here else '否'}；"
                f"历史Best1实际入训：{'是' if chosen_here else '否'}；当前全部退役；"
                f"人工 verdict：{'null' if qa.get('human_verdict') is None else qa.get('human_verdict')}。"
                f"{qa.get('notes', '')}")
        frame = f'f{int(qa.get("source_frame_index", 0)):02d}'
        parts.append(pic(root / "batch_packs" / frame / "query_full.jpg",
                         candidate_id + " / 当前源帧", "pool_source_" + candidate_id))
        parts.append(pic(root / "batch_packs" / frame / "model_mask_full.png",
                         candidate_id + " / 原始编辑 mask", "pool_mask_" + candidate_id))
        parts.append(pic(full_path, candidate_id, "pool_" + candidate_id, note))
        parts.append(f'<p>{evidence_link(qa_path, "qa_" + candidate_id, candidate_id + " 原始评分 JSON")}</p>')
    parts.append('</div></details>')

    parts.append('<h2>真实 RGB 遮挡对 · 固定 f05 对照</h2>'
                 '<p>原权重与真实 RGB 64 步训练均接官方和固定 r47。'
                 '合成遮挡恢复与真实 R001 去车不是同一监督问题；原生生成和最终写回分开看。</p>')
    parts.append('<div class="grid">')
    parts.append(pic(root / "batch_packs" / "f05" / "query_full.jpg",
                     "真实 R001 DELETE 查询 f05 · 原图", "realrgb_eval_source_f05"))
    parts.append(pic(root / "batch_packs" / "f05" / "model_mask_full.png",
                     "真实 R001 DELETE 查询 f05 · 目标 mask", "realrgb_eval_mask_f05"))
    parts.append('</div>')
    parts.append(eval_grid([("baseline_clip", "原权重"),
                            ("realRGB_0064_clip", "真实 RGB 遮挡对 · 64步")],
                           ("native", "compose")))
    parts.append(output_review(root / "r001_audit" / "realRGB64_output_review.json",
                               "真实 RGB 64步 · 固定 f05 独立输出评分", "realRGB64_output_review"))

    parts.append('<h2>Best1 f04 · 单图静态重复窗</h2>'
                 '<p>训练源只用已选 f04 伪标签，重复为静态 10 帧容量控制。'
                 '显示官方与 r47 的原生生成、最终写回；缺失结果明确标为待同步。</p>')
    parts.append(f'<p>训练目录：<code>{esc(root / "training_pseudo_best1")}</code>'
                 f'（{"已同步" if (root / "training_pseudo_best1").exists() else "待同步"}）。</p>')
    parts.append('<div class="grid">')
    parts.append(pic(root / "batch_packs" / "f04" / "query_full.jpg",
                     "Best1 查询 f04 · 源帧原图", "best1_eval_source_f04"))
    parts.append(pic(root / "batch_packs" / "f04" / "model_mask_full.png",
                     "Best1 查询 f04 · 原始目标 mask", "best1_eval_mask_f04"))
    parts.append('</div>')
    parts.append(eval_grid([("static_f04_base", "f04 原权重"),
                            ("static_f04_best1_64", "f04 Best1 · 64步")],
                           ("native", "compose"), pending=True))
    parts.append(output_review(root / "r001_audit" / "best1_output_review.json",
                               "Best1 单图 64步 · f04 独立输出评分", "best1_output_review"))

    parts.append('<h2>同尺度容量对照 · 320×576</h2>'
                 '<p>两组 X/Y 来源不同，分开展示；真实RGB组使用10个实际时间点，Best1组重复唯一f04，'
                 '不提供真实 R001 删除区的隐藏背景 GT。输出图按原权重与微调并列，像素误差仅辅助。</p>')
    parts.append('<h3>真实 RGB 合成孔 proxy · f05</h3>'
                 '<p>Y 是真实 RGB；X 图显示挖洞前输入，进入模型前先按 hole 置零再缩放。'
                 '此组用于真实 RGB 监督路线，不等同真实去车 GT。</p><div class="grid">')
    real_pair = root / "pair_evidence" / "realRGB_f05"
    for key, title in (("x", "真实 RGB proxy · X 输入"),
                       ("y", "真实 RGB proxy · Y 真实目标"),
                       ("hole", "真实 RGB proxy · 合成孔 mask")):
        parts.append(pic(real_pair / (key + ".png"), title,
                         "capacity_real_pair_" + key, pending=True))
    for label, title in (("capacity_base", "真实 RGB proxy · 原权重写回"),
                         ("capacity_realRGB_0064", "真实 RGB proxy · 微调64步写回")):
        parts.append(pic(root / "evaluation" / label / "proxy_official" / "compose.png",
                         title, label + "_capacity"))
    parts.append('</div>')
    parts.append('<h3>Best1 生成伪标签监督对 · f04</h3>'
                 '<p>X 是带银色 A 的真实 f04 输入；hole 与 f04 原始模型 mask 逐像素相同。'
                 'Y 是已选 imagegen 候选按该 mask 硬合成的<b>生成伪标签</b>，'
                 '不是观测到的隐藏背景，也不是真实 GT。此组与真实 RGB proxy 的 Y 证据等级不同，'
                 '分数不能混作同一真值误差。</p><div class="grid">')
    best_pair = root / "pair_evidence" / "best1_f04"
    parts.append(pic(root / "batch_packs" / "f04" / "query_full.jpg",
                     "Best1 容量对照 · f04 原始源帧", "capacity_best1_source_f04"))
    parts.append(pic(root / "batch_packs" / "f04" / "model_mask_full.png",
                     "Best1 容量对照 · f04 原始删除 mask", "capacity_best1_modelmask_f04"))
    for key, title in (("x", "Best1 proxy · X 输入"),
                       ("y", "Best1 proxy · Y 生成伪标签"),
                       ("hole", "Best1 · f04 原始模型 mask")):
        parts.append(pic(best_pair / (key + ".png"), title,
                         "capacity_best1_pair_" + key, pending=True))
    for label, title in (("capacity_best1_base", "Best1 proxy · 原权重写回"),
                         ("capacity_best1_0064", "Best1 proxy · 微调64步写回")):
        parts.append(pic(root / "evaluation" / label / "proxy_official" / "compose.png",
                         title, label + "_capacity", pending=True))
    parts.append('</div>')

    def add_videos(labels: list[tuple[str, str]]) -> str:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        blocks = ['<div class="grid">']
        for label, title in labels:
            for arm in ("R001_official", "R001_r47"):
                for role in ("native", "compose"):
                    folder = root / "evaluation" / label / arm / (role + "_frames")
                    if not folder.is_dir():
                        continue
                    vid = assets / f"{label}_{arm}_{role}.mp4"
                    subprocess.run([ff, "-v", "error", "-y", "-framerate", "10", "-i",
                                    str(folder / "%05d.png"), "-c:v", "libx264",
                                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                    str(vid)], check=True)
                    subprocess.run([ff, "-v", "error", "-i", str(vid), "-f", "null", "-"],
                                   check=True)
                    blocks.append(f'<figure><video controls loop muted preload="metadata" '
                                  f'src="assets/{vid.name}"></video><figcaption>'
                                  f'{esc(title)} / {esc(arm)} / {role}</figcaption></figure>')
        blocks.append("</div>")
        return "\n".join(blocks)

    if a.video:
        parts.append('<h2>真实 10 帧窗口视频 · 原权重与真实 RGB 路线</h2>')
        parts.append(add_videos([("baseline_clip", "原权重"),
                                 ("realRGB_0064_clip", "真实 RGB · 64步")]))

    parts.append('<details><summary>历史探索：旧 all10 训练、f00 静态窗及早期失败候选</summary>'
                 '<p>旧 all10 将独立生成帧用于探索性训练；其结果不表示这 10 张都在新的 Best1 训练中使用，'
                 '也不构成连续视频真值。Qwen 清理版仅 ROI 曾得 2.1；最终硬合成 0.8，未入训。</p>')
    parts.append('<h3>旧 all10 · 真实 R001 DELETE</h3>')
    parts.append(eval_grid([("pseudo_0064_clip", "旧 all10 伪标签 · 64步")],
                           ("native", "compose")))
    parts.append('<h3>旧 f00 静态结构窗</h3>')
    parts.append(eval_grid([("static_f00_base", "f00 原权重"),
                            ("static_f00_pseudo64", "f00 旧伪标签 · 64步")],
                           ("compose",)))
    parts.append('<h3>早期局部候选</h3><div class="grid">')
    for index, (path, title, qa_path) in enumerate((
        (ep / "imagegen_roi_v2.png", "早期 imagegen 局部版", ep / "imagegen_roi_v2_quality.json"),
        (root / "qwen_roi_v2_native" / "candidate.png", "Qwen 初版：标注污染",
         ep / "qwen_roi_v2_quality.json"),
        (root / "qwen_roi_v2_cleanup" / "candidate.png", "Qwen 清理版",
         ep / "qwen_cleanup_quality.json"),
    )):
        note = qa_path.read_text(encoding="utf-8") if qa_path.exists() else "尚未评分"
        parts.append(pic(path, title, f"early_candidate_{index}", note))
    parts.append("</div>")
    if a.video:
        parts.append(add_videos([("pseudo_0064_clip", "旧 all10 · 64步")]))
    parts.append("</details>")

    closeout = root / "closeout.json"
    if closeout.exists():
        parts.append('<details><summary>运行收口原始记录</summary><pre>' +
                     esc(json.dumps(read_json(closeout), ensure_ascii=False, indent=2)) +
                     "</pre></details>")
    parts.append('<p>所有候选均可按唯一 run r52、源帧、原始评分 JSON 和对应硬合成文件回溯。'
                 '单帧退化试验 base_f05 仍留在运行目录，其彩色异常不作有效视频基线。</p>')
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")
    print(out / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    main(parser.parse_args())
