from pathlib import Path
import json,numpy as np
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'backend_restore_r15';m=json.loads((D/'data/manifest.json').read_text());sizes={x:[] for x in m['actor_ids']}
for row in m['frames']:
    if row['split']!='train':continue
    for label in json.loads((Path(m['annotations'])/(row['frame_id']+'.json')).read_text())['laser_labels']:
        if label['id'] in sizes:
            b=label['box'];sizes[label['id']].append([b['length'],b['width'],b['height']])
half=[]
for actor in m['actor_ids']:
    if not sizes[actor]:raise ValueError('No actual training box for '+actor)
    half.append((np.median(sizes[actor],axis=0)/2).tolist())
v=dict(half_sizes_m=half,source='median genuine Waymo dimensions over train46; fixed trajectory, no box-generated mask',weight=1.,loss_coordinates='old normalized coordinate squared via explicit scale .1');p=D/'box_half_sizes.json';p.write_text(json.dumps(v,indent=2));c=json.loads((D/'config.json').read_text());c.update(box_half_sizes_path=str(p),legacy_box_boundary_weight=1.,legacy_visible_mask_weight=100.);(D/'config.json').write_text(json.dumps(c,indent=2));print(json.dumps(dict(actors=len(half),weight=1,visible_mask_weight=100)))
