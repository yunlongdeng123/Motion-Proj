"""Contact sheets of existing diagnostic PNGs only; no synthetic/GT substitutions."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

HERE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/component_diagnostic/fixed_pose_temporal32k')
OUT = HERE / 'results'
CHANNELS = ['static_full', 'static_Gaussian_RGB', 'environment_transmission', 'static_alpha', 'static_depth_0_to100m']


def sheet(arm, rows, name):
    width, height = 240, 160
    image = Image.new('RGB', (len(CHANNELS) * width, (height + 24) * len(rows) + 30), '#18212b')
    draw = ImageDraw.Draw(image)
    for column, channel in enumerate(CHANNELS):
        draw.text((column * width + 5, 8), channel, fill='white')
    sources = []
    for index, (mode, raw) in enumerate(rows):
        draw.text((5, 30 + index * (height + 24)), f'{arm} {mode} raw{raw:02d}', fill='white')
        for column, channel in enumerate(CHANNELS):
            source = OUT / arm / mode / f'raw{raw:02d}' / (channel + '.png')
            with Image.open(source) as frame:
                image.paste(frame.convert('RGB').resize((width, height), Image.Resampling.BILINEAR),
                            (column * width, 54 + index * (height + 24)))
            sources.append({'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    target = OUT / 'contact_sheets' / name
    target.parent.mkdir(exist_ok=True)
    image.save(target)
    return {'path': str(target), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'sources': sources,
            'display_only_resize': [width, height], 'new_scene_content_generated': False}


def main():
    assert json.loads((OUT / 'status.json').read_text())['actual_exit_code'] == 0
    records = []
    for arm in ('B1', 'V'):
        records.append(sheet(arm, [('actual', raw) for raw in (0, 10, 20, 30, 48)], f'{arm}_actual_fixed_pose_times.png'))
        rows = [(mode, raw) for mode in ('actual', 'freeze_SHM_position', 'freeze_temporal_opacity_and_support', 'freeze_both') for raw in (0, 48)]
        records.append(sheet(arm, rows, f'{arm}_readonly_counterfactual_endpoints.png'))
    (OUT / 'contact_sheets/receipt.json').write_text(json.dumps({'status': 'existing_PNG_contact_sheets_complete', 'records': records}, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'contact_count': len(records)}))


if __name__ == '__main__':
    main()
