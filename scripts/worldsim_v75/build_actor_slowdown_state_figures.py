"""把大误差正例与小误差对照整理成可直接审阅的论文图。"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
EVIDENCE = Path('/root/autodl-tmp/motion_proj/docs/autoresearch/worldsim_v75/actor_trajectory_counterfactual')
LARGE = ROOT/'WS-V75-ACTOR-SLOWDOWN-STATE-GENERATION-01/20260921-r1'
LARGE_G1 = ROOT/'WS-V75-ACTOR-SLOWDOWN-G1-GENERATION-01/20260921-r1'
LARGE_Q = ROOT/'WS-V75-ACTOR-SLOWDOWN-STATE-QUALIFY-01/20260921-r2'
LARGE_BASE = ROOT/'WS-V75-ACTOR-REMOVAL-G1-GENERATION-01/20260921-r1'
SMALL = ROOT/'WS-V75-ACTOR-SLOWDOWN-CONTROL-STATE-01/20260921-r1'
SMALL_G1 = ROOT/'WS-V75-ACTOR-SLOWDOWN-CONTROL-G1-01/20260921-r1'
SMALL_Q = ROOT/'WS-V75-ACTOR-SLOWDOWN-CONTROL-QUALIFY-01/20260921-r1'
SMALL_BASE = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-GENERATION-01/20260921-r1'
ARMS = ['reference', 'dvgt_metric', 'class_prior']
COLORS = {'reference': '#69d2e7', 'dvgt_metric': '#ff6b6b', 'class_prior': '#ffd166'}
LABELS = {'reference': 'reference state', 'dvgt_metric': 'depth-shift state',
          'class_prior': 'depth+shape state'}


def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/{name}', size)


def fit(image, size):
    image = image.copy(); image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', size, '#101923')
    canvas.paste(image, ((size[0]-image.width)//2, (size[1]-image.height)//2))
    return canvas


def boxed_frame(array_path, row, title, frame=85):
    image = Image.fromarray(np.asarray(np.load(array_path, mmap_mode='r')[frame])).copy()
    draw = ImageDraw.Draw(image)
    if row['original_projection'] is not None:
        draw.rectangle(row['original_projection']['bounds'], outline='#ff5148', width=5)
    if row['edited_projection'] is not None:
        draw.rectangle(row['edited_projection']['bounds'], outline='#00e6cf', width=5)
    match = row['edited_match'] or row.get('edited_reference_match')
    if match is not None:
        draw.rectangle(match['box'], outline='#ffe86b', width=5)
    panel = fit(image, (480, 264)); d = ImageDraw.Draw(panel)
    d.rectangle((0, 0, 480, 34), fill=(9, 18, 28, 230)); d.text((10, 7), title, font=font(18, True), fill='white')
    return panel


def initial_panel(array_path, g1_eval, title, subtitle):
    image = Image.fromarray(np.asarray(np.load(array_path, mmap_mode='r')[4])).copy()
    row = next(x for x in g1_eval['rows'] if x['variant'] == 'unedited' and x['frame'] == 4)
    draw = ImageDraw.Draw(image); draw.rectangle(row['original_projection']['bounds'], outline='#ffe86b', width=6)
    panel = fit(image, (480, 264)); d = ImageDraw.Draw(panel)
    d.rectangle((0, 0, 480, 58), fill=(9, 18, 28, 230)); d.text((10, 5), title, font=font(19, True), fill='white')
    d.text((10, 31), subtitle, font=font(15), fill='#d8e6ef')
    return panel


def architecture():
    W, H = 1800, 440; im = Image.new('RGB', (W, H), '#f4f7fa'); d = ImageDraw.Draw(im)
    d.text((55, 28), 'Which input-state errors does OmniDreams propagate into its generated future?', font=font(34, True), fill='#14213d')
    boxes = [
        (45, 125, 260, 270, 'Observed RGB', 'same initial image'),
        (315, 125, 560, 270, 'State variants', 'reference / depth shift / shape'),
        (615, 125, 855, 270, 'Future edit', 'same actor slowdown'),
        (910, 125, 1165, 270, 'Condition raster', 'same map, camera, seed'),
        (1220, 125, 1450, 270, 'OmniDreams', 'MODEL UNDER TEST'),
        (1505, 125, 1755, 270, 'State response', 'track + response error'),
    ]
    for i, (x1, y1, x2, y2, title, sub) in enumerate(boxes):
        fill = '#ffffff' if i not in [1, 2, 4] else ['#ffffff', '#e8f3ff', '#fff2df', '#ffffff', '#eaf8ee', '#ffffff'][i]
        d.rounded_rectangle((x1, y1, x2, y2), 18, fill=fill, outline='#31688e', width=3)
        d.text(((x1+x2)//2, y1+38), title, font=font(24, True), fill='#17324d', anchor='mm')
        d.text(((x1+x2)//2, y1+91), sub, font=font(16), fill='#49657a', anchor='mm')
        if i < len(boxes)-1:
            d.line((x2+10, 197, boxes[i+1][0]-12, 197), fill='#315a78', width=5)
            d.polygon([(boxes[i+1][0]-12, 197), (boxes[i+1][0]-27, 187), (boxes[i+1][0]-27, 207)], fill='#315a78')
    gates = [('G0 state reaches raster', 315, 250), ('G1 OmniDreams follows correct edit', 615, 450),
             ('G2 OmniDreams propagates state error', 1220, 500)]
    for text, x, width in gates:
        d.rounded_rectangle((x, 325, x+width, 385), 14, fill='#14213d')
        d.text((x+width//2, 355), text, font=font(18, True), fill='white', anchor='mm')
    im.save(EVIDENCE/'architecture-components.png')


def response_plot(large_eval, small_eval):
    fig, ax = plt.subplots(figsize=(9.2, 4.5), dpi=170)
    x = np.arange(2); width = .23
    for i, arm in enumerate(ARMS):
        vals = [large_eval['summaries'][arm]['median_pair_response_error_px'], small_eval['summaries'][arm]['median_pair_response_error_px']]
        bars = ax.bar(x+(i-1)*width, vals, width, label=LABELS[arm], color=COLORS[arm], edgecolor='#263746')
        ax.bar_label(bars, labels=[f'{v:.1f}' for v in vals], padding=3, fontsize=9)
    ax.axhline(10, color='#6c7a86', linestyle='--', linewidth=1.2, label='10 px material margin')
    ax.set_xticks(x, ['Large input-state error\ndepth shift 13.27 m', 'Small-error control\ndepth shift 0.38 m'])
    ax.set_ylabel('Median pair-response error (px)'); ax.set_ylim(0, 66)
    ax.grid(axis='y', alpha=.25); ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc='upper right'); fig.tight_layout()
    fig.savefig(EVIDENCE/'response-error-contrast.png', bbox_inches='tight'); plt.close(fig)


def main_figure(large_eval, small_eval):
    W, H = 2020, 1120; im = Image.new('RGB', (W, H), '#f4f7fa'); d = ImageDraw.Draw(im)
    d.text((45, 28), 'OmniDreams propagates a large input-state error into its generated future', font=font(31, True), fill='#14213d')
    d.text((45, 78), 'OmniDreams is the model under test  |  Yellow: output detection  |  Red/Cyan: unedited/slowed state  |  same RGB, map, camera and seed', font=font(18), fill='#49657a')
    configs = [
        ('(a) Large-error source', 'Depth: 41.9m -> 28.6m  |  center error 13.27m',
         LARGE_BASE/'reference-unedited/generated.npy', LARGE_G1/'evaluation.json', large_eval,
         {'reference': LARGE_G1/'reference-edited/generated.npy', 'dvgt_metric': LARGE/'dvgt_metric-edited/generated.npy', 'class_prior': LARGE/'class_prior-edited/generated.npy'}, 130),
        ('(b) Small-error control', 'Depth: 20.3m  |  input center error 0.38m',
         SMALL_BASE/'reference-unedited/generated.npy', SMALL_G1/'evaluation.json', small_eval,
         {'reference': SMALL_G1/'reference-edited/generated.npy', 'dvgt_metric': SMALL/'dvgt_metric-edited/generated.npy', 'class_prior': SMALL/'class_prior-edited/generated.npy'}, 610),
    ]
    for heading, subtitle, initial_path, g1_path, evaluation, paths, y in configs:
        d.text((45, y), heading, font=font(25, True), fill='#14213d')
        g1 = json.loads(g1_path.read_text())
        panels = [initial_panel(initial_path, g1, 'Observed scene', subtitle)]
        for arm in ARMS:
            row = next(x for x in evaluation['rows'] if x['arm'] == arm and x['frame'] == 85)
            err = evaluation['summaries'][arm]['median_pair_response_error_px']
            panels.append(boxed_frame(paths[arm], row, f'OmniDreams · {LABELS[arm]}  |  {err:.1f}px'))
        for i, panel in enumerate(panels): im.paste(panel, (45+i*490, y+52))
        verdict = 'MATERIAL STATE EFFECT' if evaluation['material_reconstruction_state_effect'] else 'NO MATERIAL DEGRADATION'
        color = '#c94242' if evaluation['material_reconstruction_state_effect'] else '#208a5d'
        d.rounded_rectangle((1515, y+337, 1995, y+402), 14, fill=color)
        d.text((1755, y+369), verdict, font=font(20, True), fill='white', anchor='mm')
        if y < 500: d.line((45, 585, 1995, 585), fill='#aebdca', width=2)
    d.rounded_rectangle((45, 1055, 1995, 1100), 12, fill='#14213d')
    d.text((1020, 1077), '13.27 m input depth shift -> wrong future occupancy / actor scale -> +53.5 px OmniDreams response error; 0.38 m control does not degrade response',
           font=font(18, True), fill='white', anchor='mm')
    im.save(EVIDENCE/'main-state-error-figure.png')


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    large = json.loads((LARGE/'evaluation.json').read_text()); small = json.loads((SMALL/'evaluation.json').read_text())
    assert large['material_reconstruction_state_effect'] and not small['material_reconstruction_state_effect']
    architecture(); response_plot(large, small); main_figure(large, small)
    summary = {
        'status': 'complete', 'model_under_test': 'OmniDreams single-view 2B',
        'state_arm_labels': LABELS,
        'large_error_source': {'depth_shift_m': 13.267042594399834,
            'reference_response_error_px': large['summaries']['reference']['median_pair_response_error_px'],
            'depth_shift_response_error_px': large['summaries']['dvgt_metric']['median_pair_response_error_px'],
            'depth_shape_response_error_px': large['summaries']['class_prior']['median_pair_response_error_px'],
            'material': True},
        'small_error_control': {'depth_shift_m': 0.3795239333796184,
            'reference_response_error_px': small['summaries']['reference']['median_pair_response_error_px'],
            'depth_shift_response_error_px': small['summaries']['dvgt_metric']['median_pair_response_error_px'],
            'depth_shape_response_error_px': small['summaries']['class_prior']['median_pair_response_error_px'],
            'material': False},
        'claim_boundary': 'two fixed sources; DVGT-derived range uses known rays and oracle size/yaw; fixed-camera counterfactual, not policy feedback or population prevalence',
        'human_verdict': None, 'failure_ledger_delta': 'none'}
    (EVIDENCE/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
