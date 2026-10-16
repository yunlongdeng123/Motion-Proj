import os,sys,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1');NEW=ROOT/'shadow_sequence_20261016_v2';OLD=ROOT/'shadow_BC_20261016_v1'
sys.path.insert(0,str(OLD/'implementation_v1'));from prepare_e0 import rays_to_plane,sample
frames=json.loads((ROOT/'rigid_post30k/scene0008_089_137/evaluation/scene_evaluation_manifest.json').read_text())['streams']['observed']['frames']
with np.load(NEW/'surface_guarded_data/surface.npz') as f:d={k:f[k] for k in f.files}
rows=[]
for i,points in [(0,[[175,202],[178,202],[181,202],[185,202]]),(12,[[90,240],[110,240],[200,226]])]:
    points=np.array(points);q=rays_to_plane(points,frames[i]['camera'],d['plane'])
    chart=((q-d['origin'])@d['basis']/float(d['world_scale'])-d['lo'])/float(d['step'])
    conf=sample(d['confidence'].reshape(int(d['shape'][1]),int(d['shape'][0]),1),chart)[0][:,0]
    for k,p in enumerate(points):rows.append({'frame':frames[i]['frame_id'],'pixel':p.tolist(),'world_confidence':float(conf[k]),'unselected_box_protected':bool(d['protection'][i,p[1],p[0]])})
    picture=np.repeat(d['protection'][i,:,:,None],3,axis=-1).astype(np.uint8)*255
    Image.fromarray(picture).save(NEW/('guard_'+frames[i]['frame_id']+'.png'))
(NEW/'guard_pixel_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
