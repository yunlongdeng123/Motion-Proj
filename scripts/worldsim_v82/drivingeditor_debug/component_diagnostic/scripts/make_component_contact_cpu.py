"""Paste actual clamped renderer component PNGs; CPU only, no image replacement."""
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/vgpu_bugfix_20261012/component_diagnostic')
CHANNELS = ('environment_sample', 'static_foreground_contribution', 'static_env_contribution', 'alpha_s')
CASES = ('baseline20', 'baseline30')
WIDTH, HEIGHT, HEADER = 480, 320, 32
canvas = Image.new('RGB', (WIDTH * 4, (HEIGHT + HEADER) * 2), '#15171b')
draw = ImageDraw.Draw(canvas)
report = {'scope': 'Actual raw10 renderer component PNGs, clamped by original save_image only. Float statistics remain unbounded.', 'rows': []}
for row_index, case_id in enumerate(CASES):
    manifest = json.loads((ROOT / 'cases' / case_id / 'case_manifest.json').read_text())
    row = next(r for r in manifest['frames'] if r['frame']['image_name'] == '0017010')
    report['rows'].append({'case_id': case_id, 'frame_id': row['frame']['frame_id'], 'non_sky_outside_actor': row['regions']['non_sky_outside_actor'], 'outputs': {channel: row['outputs'][channel] for channel in CHANNELS}})
    for column, channel in enumerate(CHANNELS):
        image = Image.open(row['outputs'][channel]).convert('RGB')
        if image.size != (WIDTH, HEIGHT):
            raise ValueError('Unexpected source dimensions')
        draw.text((column * WIDTH + 8, row_index * (HEIGHT + HEADER) + 10), case_id + ' / ' + channel, fill='white')
        canvas.paste(image, (column * WIDTH, row_index * (HEIGHT + HEADER) + HEADER))
        if canvas.crop((column * WIDTH, row_index * (HEIGHT + HEADER) + HEADER, (column + 1) * WIDTH, row_index * (HEIGHT + HEADER) + HEADER + HEIGHT)).tobytes() != image.tobytes():
            raise ValueError('Contact pixels differ from source PNG')
destination = ROOT / 'contact_sheets/component_breakdown_raw10.png'
canvas.save(destination)
report['contact_sheet'] = str(destination)
report['source_pixels_preserved_exactly'] = True
(ROOT / 'component_breakdown_raw10.json').write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps({'status': 'CPU_contact_complete', 'contact_sheet': str(destination)}))
