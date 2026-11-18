from pathlib import Path
import json,shutil
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
p=P/'execution_contract.json';d=json.loads(p.read_text())
shutil.copy2(p,p.with_suffix('.json.before_C_known_scope'))
d['qualification']['color_rows']='(A_seen | S_seen | (C_seen & ~original_known)) & material_available & surface/near/world legal & full-packet K-safe; actual native masses from stage0; source/filter validity independent'
d['qualification']['C_only_original_known']='retain readonly prior effective RGB; pseudo C alone cannot overwrite immutable observed-material qualification'
d['qualification']['stage0_92308_scope']='visibility proposal only, not final write scope; derive guarded scientific scope on CPU without extra native calls'
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print('Final guarded scope registered before any optimization')
