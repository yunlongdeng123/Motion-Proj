from pathlib import Path
import sys, json, numpy as np
R4=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_native30k_red_rebuild_20261125_r4')
sys.path.insert(0,str(R4/'code'))
from native30k_adapter import Native30KAdapter
rt=Native30KAdapter(R4/'config/native30k.json')
source=rt.owners(110).cpu().numpy(); semantic=rt.owners(110,semantic=True).cpu().numpy()
packet=rt.packet(110); xyz=packet['means3D'].detach().cpu().numpy()
inv=np.linalg.inv(rt.transform); world=xyz@inv[:3,:3].T+inv[:3,3]
observations=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs/observations/110.json')
boxes=json.loads(observations.read_text())['boxes']; records=[]
for sid in np.unique(source[source>=0]):
    choose=source==sid; row=dict(source_owner=int(sid),rows=int(choose.sum()),semantic_A9=int((choose&(semantic==10)).sum()),semantic_A20=int((choose&(semantic==21)).sum()),boxes=[])
    for box in boxes:
        if box['track'] not in ['NTjq-OhZ_97OMJ4xwJAxXg','bzFbWRrSGOgEaXxnuGV0tw']:continue
        inverse=np.linalg.inv(np.asarray(box['world_from_actor']).reshape(4,4))
        local=world[choose]@inverse[:3,:3].T+inverse[:3,3]
        inside=(abs(local)<=np.asarray(box['dimensions'])/2+.35).all(1)
        row['boxes'].append(dict(track=box['track'],inside_rows=int(inside.sum()),median_local=np.median(local,axis=0).tolist()))
    records.append(row)
out=Path(__file__).parent/'native_lineage_mapping.json';out.write_text(json.dumps(records,indent=2));print(json.dumps(records),flush=True)
