"""Collect lightweight, reviewable evidence without copying parent checkpoints."""
from pathlib import Path
import json
import shutil
import tarfile
from PIL import Image

root = Path(__file__).resolve().parent.parent
dest = root / 'delivery'
dest.mkdir(exist_ok=False)
(dest/'frames').mkdir()
(dest/'evidence').mkdir()
(dest/'code').mkdir()
for source in (root/'full49/frames').glob('*.png'):
    Image.open(source).convert('RGB').save(dest/'frames'/source.with_suffix('.jpg').name,
                                         quality=94, subsampling=0)
for source in (root/'source').glob('*.py'):
    shutil.copy2(source, dest/'code'/source.name)
for name in ('full49', 'pilot', 'reference_split'):
    shutil.copy2(root/name/'metrics.json', dest/'evidence'/f'{name}_metrics.json')
for source in (root/'reference_split').glob('*.png'):
    shutil.copy2(source, dest/'evidence'/source.name)
(dest/'evidence/lossless_pilot113').mkdir()
for source in (root/'full49/frames').glob('113_*.png'):
    shutil.copy2(source, dest/'evidence/lossless_pilot113'/source.name)
sources = {
    'native_binding.py': root.parent/'scene008_native_road1k_20261105_r1/source/native_binding.py',
    'repair37.py': root.parent/'scene008_native_road1k_20261105_r1/source/repair37.py',
    'scene_runtime.py': root.parent/'scene008_a6_complete_asset_20261105/source/scene_runtime.py',
    'road_settings.json': root.parent/'scene008_native_road1k_20261105_r1/settings.json',
}
for name, source in sources.items():
    shutil.copy2(source, dest/'code'/name)
inventory = dict(remote_run=str(root), native_updates=0, training_updates=0,
                 native_and_road_frozen=True, GPU_shutdown_authorized=False,
                 full_precision_evidence=[str(root/'full49'), str(root/'reference_split')],
                 sources={name: str(path) for name, path in sources.items()},
                 image_policy='JPEG for interactive gallery; lossless raw113 PNG retained; measurements use float renderer outputs, never JPEG.')
(dest/'evidence/inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
with tarfile.open(root/'delivery.tar.gz', 'w:gz') as archive:
    archive.add(dest, arcname='scene008_local_colour_diagnosis')
print(json.dumps(dict(files=sum(p.is_file() for p in dest.rglob('*')),
                      tar_bytes=(root/'delivery.tar.gz').stat().st_size)))
