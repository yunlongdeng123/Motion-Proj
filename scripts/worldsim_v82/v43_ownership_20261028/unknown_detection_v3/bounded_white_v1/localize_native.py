"""原生G/alpha保持不变，只将白底与合成限制到用户局部带，避免天空白块。"""
from pathlib import Path
import json,shutil
import numpy as np
from PIL import Image
H=Path(__file__).resolve().parent
O=H.parents[2]/'outputs/V43_Ownership_20261027/gpu_v43/bounded_white_v1'
def save(p,x):Image.fromarray(np.round(np.clip(x.transpose(1,2,0),0,1)*255).astype('uint8')).save(p)
records=[]
for raw in [98,113,114]:
    r=f'{raw:03d}';src=H/'bounded_native_r1'/r;dst=H/'bounded_native_r2'/r;dst.mkdir(parents=True,exist_ok=True)
    e=dict(np.load(src/'evidence.npz'));inputs=np.load(O/r/'diagnostic_gate.npz');scope=inputs['unknown_display_gate'];base=inputs['source_native_rgb']
    w=e['gaussian_rgb']+(1-e['alpha'])[None];out=base.copy();out[:,scope]=w[:,scope]
    native=base.copy();native[:,scope]=e['native_rgb'][:,scope]
    rec=json.loads((src/'receipt.json').read_text());rec.update(display_scope='user local band, clipped blue; body top minus 8px to blue bottom',
        gaussian_rerender_source='bounded_native_r1',extra_gpu_work=False,outside_display_scope_max_difference=float(np.abs(out[:,~scope]-base[:,~scope]).max()),
        remaining_cross_boundary_support_is_not_reclassified=True,not_all_effect_pixels_are_pure_white=True)
    assert np.array_equal(out[:,~scope],base[:,~scope])
    e.update(white_rgb=out,native_rgb=native,display_scope=scope,baseline_native_rgb=base)
    np.savez_compressed(dst/'evidence.npz',**e);save(dst/'white.png',out);save(dst/'native.png',native)
    for name in ['gaussian.png','alpha.png']:shutil.copy2(src/name,dst/name)
    (dst/'receipt.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');records.append(rec)
(H/'bounded_native_r2/summary.json').write_text(json.dumps(dict(records=records,training_updates=0,source='bounded_native_r1 plus exact component composition on CPU',actual_gaussian_rerender=True),indent=2),encoding='utf-8')
