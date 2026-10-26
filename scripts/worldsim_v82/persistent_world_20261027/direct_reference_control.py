"""强控制：相同真实来源和固定锚点域，去掉学得高斯修复，直接提供参考RGB。"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027/phase1_native')
dest=ROOT/'delete_all_direct_reference';route=ROOT/'delete_all_evidence_routed'
for sub in ['native_rgb','mask']:(dest/sub).mkdir(parents=True,exist_ok=True)
rows=[]
for source in sorted((ROOT/'delete_all_unrepaired/native_rgb').glob('*.png')):
    raw=int(source.stem);image=np.asarray(Image.open(source)).copy()
    evidence=np.asarray(Image.open(route/'evidence_mask'/source.name))>0
    if evidence.any():
        reference=np.load(f'/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026/crossview/{raw}.npz')['rgb']
        image[evidence]=np.round(np.clip(reference[evidence],0,1)*255).astype('uint8')
    Image.fromarray(image).save(dest/'native_rgb'/source.name)
    mask=Image.open(route/'mask'/source.name);mask.save(dest/'mask'/source.name)
    rows.append(dict(raw=raw,reference_anchor_pixels=int(evidence.sum())))
(dest/'contract.json').write_text(json.dumps(dict(
    hypothesis='若直接参考锚点已解释收益，就不把改善归因于高斯联合修复或持久世界记忆',
    input='原静态删车图＋与证据路由相同域的真实跨帧参考RGB',
    fixed_mask_from='delete_all_evidence_routed',backend='相同预训练ProPainter、分辨率、seed与参数',
    optimized_gaussian_repair_used=False,source_input_equalized=True,rows=rows),indent=2))
print('DIRECT_REFERENCE_CONTROL_PREPARED',sum(x['reference_anchor_pixels'] for x in rows))
