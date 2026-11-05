from pathlib import Path
import numpy as np
from PIL import Image
import sys
campaign = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
p = campaign / 'scene008_instances_20261105_r1/observations/frames'
run = Path(sys.argv[1])
for folder in sorted(run.glob('probe_*')):
    vals = []
    for raw in [105, 113, 130]:
        f = folder / f'raw{raw}.png'
        if not f.is_file():
            continue
        panel = np.asarray(Image.open(f))[320:, 960:, :] / 255.
        with np.load(p / f'{raw:03d}.npz') as data:
            for key in ['road', 'protect']:
                region = data[key]
                vals.append((key, int(region.sum()), panel[region].sum(0)))
    for key in ['road', 'protect']:
        selected = [v for v in vals if v[0] == key]
        if selected:
            print(folder.name, key, np.sum([v[2] for v in selected], axis=0) / sum(v[1] for v in selected))
