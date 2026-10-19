"""Reconstruct review code only; never imports training or calls CUDA."""
from pathlib import Path
import argparse,hashlib,json,shutil
p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['baseline','full','real_only','prior_only','ownership_only'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
if a.output.exists():raise FileExistsError('choose a new empty destination')
shutil.copytree(root/'base',a.output)
if a.experiment!='baseline':shutil.copytree(root/'overlays'/a.experiment,a.output,dirs_exist_ok=True)
expected=json.loads((root/'source_map.json').read_text(encoding='utf-8'))[a.experiment]['files']
for name,e in expected.items():assert hashlib.sha256((a.output/name).read_bytes()).hexdigest()==e['sha256']
print('Verified source files:',len(expected))
