"""固定三帧的直接图像对照；不计算质量分，不改变模型输入。"""
from common import *
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

FRAMES = (0, 5, 9)


def board(rows, destination, crop=None):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    cell_height = 320 if crop else 288
    image = Image.new('RGB', (1536, len(rows) * (cell_height + 28)), '#102232')
    draw = ImageDraw.Draw(image)
    for row, (label, frames) in enumerate(rows):
        for col, (f, rgb) in enumerate(zip(FRAMES, frames)):
            im = Image.fromarray(rgb)
            if crop:
                im = im.crop(crop)
            im = ImageOps.pad(im, (512, cell_height), color='#102232', method=Image.Resampling.LANCZOS)
            top = row * (cell_height + 28)
            draw.text((col * 512 + 7, top + 4), f'{label} / f{f:02}', font=font, fill='white')
            image.paste(im, (col * 512, top + 28))
    image.save(destination)


def pngs(folder, native=False):
    if native:
        folder = folder / 'native'
    return [np.asarray(Image.open(folder / f'{f:05}.png').convert('RGB')) for f in FRAMES]


def main(stage):
    out = O / 'review' / 'direct_review' / stage
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for cid in EVAL_IDS:
        current = O / 'evaluation' / stage / cid / 'correct'
        if not (current / 'result.json').exists():
            continue
        case = cases()[cid]
        req = request(case)
        meta = read(O / 'inputs' / cid / 'routing.json')
        main_owner = meta['identities'].get(meta['main_protected_token'], 0)
        focus = req.edit_mask[list(FRAMES)].any(0)
        if main_owner:
            owner = req.routing_metadata['routing_query_owner'][list(FRAMES), 0]
            focus |= np.repeat(np.repeat((owner == main_owner).any(0), 4, 0), 4, 1)
        yy, xx = np.where(focus)
        crop = (max(0, int(xx.min()) - 48), max(0, int(yy.min()) - 48),
                min(1024, int(xx.max()) + 49), min(576, int(yy.max()) + 49))
        rows = [('Original RGB' if case['kind'] == 'real' else 'Synthetic X / added occluder',
                 [req.target_rgb[f] for f in FRAMES])]
        if case['kind'] != 'real':
            gt = images(case, 'target')
            rows.append(('Real GT / synthetic task only', [gt[f] for f in FRAMES]))
        rows += [(label, pngs(folder)) for label, folder in [
            ('r46 official + complete SAM', PARENT / 'evaluation' / cid / 'baseline'),
            ('r47 RGB + geometry', PARENT / 'evaluation' / cid / 'RGB_and_geometry'),
            ('r48 same-budget 64 steps', CONTROL / 'evaluation/step_0064' / cid),
            ('r49 zero / correct', O / 'evaluation/zero' / cid / 'correct')]]
        if stage == 'step64':
            rows.append(('r49 64 / correct', pngs(current)))
            off = O / 'evaluation/step64' / cid / 'routing_off'
            if (off / 'result.json').exists():
                rows.append(('r49 64 / routing OFF', pngs(off)))
        board(rows, out / f'{cid}_full.png')
        board(rows, out / f'{cid}_focus.png', crop)
        probes = []
        for arm in ('correct', 'wrong', 'no_RGB', 'routing_off'):
            folder = O / 'evaluation' / stage / cid / arm
            if not (folder / 'result.json').exists():
                continue
            probes.append((f'{stage} {arm} / NATIVE', pngs(folder, native=True)))
            probes.append((f'{stage} {arm} / fixed COMPOSE', pngs(folder)))
        board(probes, out / f'{cid}_controls.png', crop)
        records.append({'case_id': cid, 'frames': list(FRAMES), 'focus_crop_xyxy': crop,
                        'boards': [f'{cid}_{k}.png' for k in ('full', 'focus', 'controls')],
                        'human_verdict': None, 'temporal_verdict': None})
    dump(out / 'manifest.json', {'stage': stage, 'records': records,
        'scope': 'Fixed f00/f05/f09 direct image comparisons. No temporal verdict or GT-error ranking.'})
    print(json.dumps({'stage': stage, 'image_cases': len(records), 'folder': str(out)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['zero', 'step64'])
    main(p.parse_args().stage)
