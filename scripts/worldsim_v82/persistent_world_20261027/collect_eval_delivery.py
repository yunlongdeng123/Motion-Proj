"""收回首批评测证据；只打包报告所需图像/指标，不拷贝模型和大日志。"""
from pathlib import Path
import json
import shutil
import tarfile
from PIL import Image, ImageDraw

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
OUT = ROOT / 'evaluation_delivery_v1'
OUT.mkdir(exist_ok=True)
manifest = json.loads((ROOT / 'evaluation_set_v1/manifest.json').read_text())
shutil.copy2(ROOT / 'evaluation_set_v1/manifest.json', OUT / 'manifest.json')
for source in ['quota_state.json', 'phase2_eval_status.json', 'research_policy.json', 'phase2_reprioritization/receipt.json', 'geometry_smoke/receipt_from100.json']:
    path = ROOT / source
    if path.exists():
        target = OUT / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
for name in ['evaluation_architecture.svg', 'EVALUATION_PROTOCOL.md']:
    shutil.copy2(ROOT / 'code' / name, OUT / name)
for scene in manifest['scenes']:
    key = scene['context']
    raw = scene['selected_window'][24]
    (OUT / 'gallery').mkdir(exist_ok=True)
    image = Image.open(ROOT / 'evaluation_set_v1/scenes' / key / 'gt_front' / f'{raw:03d}.png')
    image.resize((320, 214)).save(OUT / 'gallery' / (key + '.jpg'), quality=90)
summaries = {}
for name in ['scene0017', 'scene0029']:
    folder = ROOT / 'phase2_multiscene' / name
    summary = json.loads((folder / 'summary.json').read_text())
    summaries[name] = summary
    (OUT / name).mkdir(exist_ok=True)
    shutil.copy2(folder / 'summary.json', OUT / name / 'summary.json')
    video = folder / 'delete_all/propainter'
    shutil.copy2(video / 'summary.json', OUT / name / 'video_summary.json')
    shutil.copy2(video / 'video.mp4', OUT / name / 'delete_video.mp4')
    raw = int(sorted((folder / 'noop/native_rgb').glob('*.png'))[24].stem)
    panels = [(operation, folder / operation / 'native_rgb' / f'{raw:03d}.png')
              for operation in ['noop', 'delete_all', 'move_all_modeled', 'insert_all_reused', 'ego_shift', 'repeat_delete']]
    panels.insert(2, ('delete + fixed ProPainter', video / 'model_png' / f'{raw:03d}.png'))
    sheet = Image.new('RGB', (480 * 4, 350 * 2), '#102132')
    draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(panels):
        x, y = (i % 4) * 480, (i // 4) * 350
        draw.text((x + 8, y + 8), name + ' / ' + label + ' / raw' + str(raw), fill='white')
        sheet.paste(Image.open(path).convert('RGB'), (x, y + 27))
    sheet.save(OUT / name / 'contact.jpg', quality=95)
    # 保存每一帧原生与生成输出，后续能复核最差时刻，不仅交付一张选图。
    for operation in ['noop', 'delete_all', 'ego_shift']:
        shutil.copytree(folder / operation / 'native_rgb', OUT / name / operation, dirs_exist_ok=True)
    shutil.copytree(video / 'model_png', OUT / name / 'generated', dirs_exist_ok=True)
repair = ROOT / 'phase2_multiscene/scene0008_repair10k'
if (repair / 'summary.json').exists():
    (OUT / 'scene0008_repair10k').mkdir(exist_ok=True)
    shutil.copy2(repair / 'summary.json', OUT / 'scene0008_repair10k/summary.json')
    shutil.copy2(repair / 'propainter/summary.json', OUT / 'scene0008_repair10k/video_summary.json')
    shutil.copy2(repair / 'propainter/video.mp4', OUT / 'scene0008_repair10k/delete_video.mp4')
metrics = {}
for name, summary in summaries.items():
    rows = [r for r in summary['rows'] if r['operation'] == 'noop']
    metrics[name] = dict(
        native_frames=summary['frames'], native_operations=len(summary['operations']),
        noop_max_difference=max(r['noop_route_max_difference'] for r in rows),
        preview_mean_frame_psnr={domain: sum(r['noop_psnr_png_gt'][domain] for r in rows if r['noop_psnr_png_gt'][domain] is not None) / sum(r['noop_psnr_png_gt'][domain] is not None for r in rows)
                                 for domain in ['full', 'body', 'edge', 'background']},
        repeat_max_difference=max(r['repeat_max_difference'] for r in summary['rows'] if r['repeat_max_difference'] is not None),
        quality_accepted=False)
(OUT / 'summary.json').write_text(json.dumps(dict(dataset=manifest['counts'], frozen_baseline_metrics=metrics,
    executed_native_operation_sequences=12, executed_video_sequences=3,
    all_six_families_complete=False, strict_withdrawal_ready=False,
    selected_object_attribution_solved=False, independent_heldout_results=0,
    quality_accepted=False, human_verdict=None), indent=2))
archive = ROOT / 'evaluation_delivery_v1.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(OUT, arcname='delivery', recursive=True)
print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, metrics=metrics), indent=2))
