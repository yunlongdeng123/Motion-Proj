from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent.parent
c=np.load(R/'inputs/coarse_envelope.npz');old=np.load(R.parent/'shadow_a5_a37_199_20261103_r4/inputs/road_reference.npz');d=np.load(R/'inputs/views/105.npz')
f=np.load(R.parent/'r15_gn11_fullclip_cpu_20261026/frames/105.npz');inv=np.linalg.inv(f['w2l']);fine=np.load(R/'inputs/road_reference.npz');keyset=set(map(tuple,fine['keys']))
for x,y in [(240,240),(250,248),(270,318)]:
    u=int(d['unit_id'][y,x]);print(json.dumps(dict(x=x,y=y,uid=u,**{k:np.asarray(c[k][u]).tolist() for k in ['weight','mandatory','marking']},
        old_active=bool(old['active'][u]),old_valid=bool(d['valid'][y,x]),sources=int(old['source_valid'][:,u].sum()),keep=float(d['native_keep_weight'][y,x]))))
    ray=np.array([x+.5,y+.5,1.])@np.linalg.inv(f['K']).T;p=ray@inv[:3,:3].T*d['depth_m'][y,x]+inv[:3,3];key=np.floor(p[:2]/.02).astype(int)
    print(json.dumps(dict(point=p.tolist(),coarse_key=old['keys'][u].tolist(),fine_key=key.tolist(),fine_exists=tuple(key) in keyset,coarsepoint=old['local_xyz'][u].tolist())))
