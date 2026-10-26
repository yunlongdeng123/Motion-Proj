"""将已有逐来源证据完整登记到持久状态；不从生成图或opacity推断真实观测。"""
import argparse,json
from pathlib import Path
import numpy as np
from world_state import WorldState
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
init=dict(np.load('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_fullclip_cpu_20261026/full199_initialization.npz'))
state=WorldState(root/'world.sqlite','waymo_scene0008_development')
rows=[]
for i in range(len(init['world_xyz'])):
    sources=init['source_raw'][init['source_valid'][:,i]].tolist()
    rows.append((json.dumps(sources),len(sources),'background_preserve_199' if init['eligible'][i] else None,state.scene,i))
with state.db:
    state.db.executemany('UPDATE surface SET source_frames=?,source_count=?,repair_revision=? WHERE scene=? AND id=?',rows)
assert not state.db.execute('SELECT 1 FROM surface WHERE status="observed_candidate" AND source_count<2').fetchone()
summary=dict(state.summary(),source_association='逐格保存原199帧source_valid与source_raw，不依赖生成输出',
    repaired_candidate_cells=int(init['eligible'].sum()),
    repaired_does_not_mean_verified_geometry=True)
state.close();(root/'evidence_registration.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
