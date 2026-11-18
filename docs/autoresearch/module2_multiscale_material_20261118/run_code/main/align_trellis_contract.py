from pathlib import Path
import json,shutil
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
p=P/'trellis2/execution_contract.json';d=json.loads(p.read_text())
shutil.copy2(p,p.with_suffix('.json.before_aux_path'))
d['aux']='/root/autodl-tmp/models/TRELLIS.2-4B-texturing/aux/dinov3-vitl16-pretrain-lvd1689m'
assert Path(d['aux']).is_dir()
d['official_commit']='75fbf0183001ed9876c8dbb35de6b68552ee08bd'
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print('CPU source paths verified before inference')
