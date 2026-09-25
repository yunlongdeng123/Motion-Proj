import json
from pathlib import Path
from PIL import Image, ImageDraw

root = Path('/root/autodl-tmp/data/v76_vadgs')
gate = root / 'dsine_encoding_gate'
source = root / 'official_000/000'
report = json.loads((gate/'generation_report.json').read_text())
canvas = Image.new('RGB', (1200, 6*249+28), 'white')
draw = ImageDraw.Draw(canvas)
for x, label in zip([0,400,800], ['Official RGB, frame 20', 'Official provided normal prior', 'Regenerated DSINE (native resolution)']):
    draw.text((x+8,8), label, fill='black')
for i, (name, metrics) in enumerate(report['comparisons'].items()):
    for j, path in enumerate([source/'images'/f'{name}.jpg', source/'normal_img'/f'{name}.png', gate/f'{name}.png']):
        canvas.paste(Image.open(path).convert('RGB').resize((400,225)), (j*400, 28+i*249))
    draw.text((8,28+i*249+230), f"Camera {i}; identity mean angle {metrics['identity']['mean_angular_deg']:.2f} deg", fill='black')
canvas.save(gate/'encoding_comparison.png')
