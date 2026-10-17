from pathlib import Path
import json,hashlib,re,subprocess
import numpy as np
from PIL import Image
W=Path(__file__).resolve().parent;O=W.parents[2]/'outputs/shadow_rebuild_v3/r14/mask_review'
receipt=json.loads((O/'verification.json').read_text(encoding='utf8'));frames=receipt['frames'];assert len(frames)==49
assert [f['raw'] for f in frames]==list(range(89,138))
cases=0
for f in frames:
    fid=f['frame'];p=O/'arrays'/(fid+'.npz');assert hashlib.sha256(p.read_bytes()).hexdigest()==f['frozen_sha256']
    with np.load(p) as z:mask=z['masks'];valid=z['mask_valid']
    for r in f['actors']:
        a=r['actor'];b=O/'binary'/fid/f'a{a:02d}'
        pos=np.asarray(Image.open(b/'positive.png'));va=np.asarray(Image.open(b/'valid.png'));tri=np.asarray(Image.open(b/'supervision.png'));boundary=np.asarray(Image.open(b/'boundary.png'))>0
        assert pos.shape==va.shape==tri.shape==(320,480)
        assert np.array_equal(pos,mask[...,a].astype(np.uint8)*255)
        assert np.array_equal(va,valid[...,a].astype(np.uint8)*255)
        assert np.array_equal(tri,np.where(valid[...,a],np.where(mask[...,a],255,0),127).astype(np.uint8))
        m=mask[...,a];outside_above=~np.pad(m[:-1],((1,0),(0,0)),constant_values=False)
        outside_below=~np.pad(m[1:],((0,1),(0,0)),constant_values=False)
        outside_left=~np.pad(m[:,:-1],((0,0),(1,0)),constant_values=False)
        outside_right=~np.pad(m[:,1:],((0,0),(0,1)),constant_values=False)
        every_side=m&(outside_above|outside_below|outside_left|outside_right)
        assert boundary[every_side].all()
        assert (O/'actors'/f'{fid}_a{a:02d}.png').exists();cases+=1
    for name in ['rgb','deva','sam3','fused','labels','lidar']:assert Image.open(O/'frames'/fid/(name+'.png')).size==(480,320)
    assert Image.open(O/'reviews'/(fid+'_review.png')).size==(1920,684)
html=(O/'report.html').read_text(encoding='utf8');js=re.search(r'<script>(.*?)</script>',html,re.S).group(1)
script=W/'report_inline_check.js';script.write_text(js,encoding='utf8');subprocess.run(['C:/Program Files/nodejs/node.exe','--check',str(script)],check=True)
result=dict(status='PASS',frames=49,actor_frames=cases,all_frozen_npz_sha256_match=True,all_positive_valid_tristate_pngs_exact=True,upper_lower_left_right_edges_complete=True,report_JS_syntax=True,rendering_only=True,training_data_changed=False)
(O/'local_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
