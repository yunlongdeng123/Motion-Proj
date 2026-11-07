"""Display saved actual K/controls at identical native scale; no scoring or render."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
C49 = ROOT.parent / 'scene008_v44_a5_c49_rgb_20261107'
WORLD = Path('C:/Users/dengyunlong/Documents/Codex/2026-11-05/j-ao-j/outputs/scene008_road_fusion_bake/delivery/repro/red_review/world_masks49_v1')
CASES = [
    ('Delivered 21K components', ROOT / 'evidence/baseline4'),
    ('K before protected-point rollback', ROOT / 'evidence/K_LP_candidate'),
    ('L Cwarm12 rollback / actual 8F', ROOT / 'evidence/L_Cwarm12_candidate'),
]
RAWS = [94, 105, 128, 137]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bbox(mask, padding=6):
    y, x = np.where(mask)
    if not len(x):
        raise ValueError('No registered pixels for display bounding box')
    return (max(0, int(x.min()) - padding), max(0, int(y.min()) - padding),
            min(480, int(x.max()) + padding + 1), min(320, int(y.max()) + padding + 1))


def main():
    figures = ROOT / 'figures'
    figures.mkdir(exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    with np.load(WORLD / '137.npz', allow_pickle=False) as z:
        near137 = bbox(z['near_x_minus'] | z['near_x_plus'])
        lateral137 = bbox(z['lateral_y_minus'] | z['lateral_y_plus'])
    crops = [(94, (190, 245, 345, 320), 'lower-road / former-37 footprint diagnostic'),
             (105, (0, 160, 175, 320), 'a5 roof display crop'),
             (128, (165, 188, 208, 234), 'registered primary128 display bbox'),
             (137, near137, 'registered world near band bbox + 6'),
             (137, lateral137, 'registered world lateral band bbox + 6'), (137, (276,209,325,246), 'fixed protected bicycle-lane diagnostic')]
    originals, outputs = {}, []

    def native(case, raw, state):
        path = case / 'frames' / f'{raw:03d}' / f'{state}.png'
        before = sha(path)
        with Image.open(path) as im:
            if im.size != (480, 320):
                raise ValueError((path, im.size))
            im.load()
            image = im.convert('RGB').copy()
        originals[str(path)] = {'path': str(path), 'sha256': before, 'size': [480, 320]}
        return image

    for state in ['keep', 'delete']:
        board = Image.new('RGB', (1510, 1480), '#f4f7f8')
        draw = ImageDraw.Draw(board)
        draw.text((20, 10), f'L actual four-raw {state.upper()}: 6 pooled + 106 local seam gates pass; full protection gates do not all pass', font=font, fill='#173b4a')
        draw.text((20, 35), 'All cells are original native 480 x 320 PNGs. No image masking, interpolation, recolouring or algorithm synthesis.', font=small, fill='#415863')
        for col, (label, case) in enumerate(CASES):
            draw.text((20 + col * 495, 62), label, font=font, fill='#173b4a')
            for row, raw in enumerate(RAWS):
                top = 112 + row * 338
                draw.text((20 + col * 495, top - 19), f'raw{raw:03d} {state.upper()}', font=small, fill='#173b4a')
                board.paste(native(case, raw, state), (20 + col * 495, top))
        path = figures / f'L_actual4_{state}_direction.png'
        board.save(path)
        outputs.append({'path': str(path), 'sha256': sha(path), 'size': list(board.size), 'native_full_frames': True})

        detail = Image.new('RGB', (1510, 1882), '#f4f7f8')
        d = ImageDraw.Draw(detail)
        d.text((20, 10), f'L {state.upper()} diagnostic crops: fixed source pixels; NEAREST display zoom only', font=font, fill='#173b4a')
        d.text((20, 35), 'Whole-frame RGB and per-bin regressions remain in the CPU audit; these crops do not redefine any scoring domain.', font=small, fill='#415863')
        for col, (label, case) in enumerate(CASES):
            d.text((20 + col * 495, 62), label, font=font, fill='#173b4a')
            for row, (raw, box, label_crop) in enumerate(crops):
                top = 92 + row * 296
                image = native(case, raw, state).crop(box)
                scale = min(6.0, 480 / image.width, 247 / image.height)
                image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.NEAREST)
                d.text((20 + col * 495, top), f'raw{raw:03d}: {label_crop}', font=small, fill='#173b4a')
                d.text((20 + col * 495, top + 18), f'xyxy={box}', font=small, fill='#415863')
                detail.paste(image, (20 + col * 495 + (480 - image.width) // 2, top + 40))
        path = figures / f'L_actual4_{state}_road_details.png'
        detail.save(path)
        outputs.append({'path': str(path), 'sha256': sha(path), 'size': list(detail.size), 'native_full_frames': False})

    for record in originals.values():
        if sha(Path(record['path'])) != record['sha256']:
            raise ValueError('Source PNG changed while building board')
    manifest = {
        'status': 'ACTUAL_FOUR_RAW_L_DIRECTION_BOARDS_ONLY',
        'raws': RAWS, 'states': ['keep', 'delete'],
        'candidate_scope': 'fixed 12-parent Cwarm RGB rollback then actual8F; no new solve/training; no L49',
        'metrics_source': 'L_candidate_fixed4560_cpu.json; pooled seam constraints do not replace full protection or visual review',
        'figures_are_scoring_inputs': False,
        'display_crops': [{'raw': raw, 'xyxy': list(box), 'reason': label} for raw, box, label in crops],
        'sources': list(originals.values()), 'outputs': outputs,
        'source_pngs_verified_unchanged': True,
        'quality_accepted': False, 'human_verdict': None,
    }
    p = ROOT / 'evidence/L_actual4_direction_board_manifest.json'
    p.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'outputs': outputs, 'source_count': len(originals), 'manifest': str(p)}, allow_nan=False))


if __name__ == '__main__':
    main()
