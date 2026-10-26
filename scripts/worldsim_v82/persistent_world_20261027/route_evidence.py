"""最小证据路由强控制：固定后端，仅将已支持且原生拟合相符的像素排除生成。"""
import json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
base=ROOT/'phase1_native/delete_all_repaired';dest=ROOT/'phase1_native/delete_all_evidence_routed'
for sub in ['native_rgb','mask','evidence_mask']:(dest/sub).mkdir(parents=True,exist_ok=True)
rows=[]
for frame in sorted((base/'native_rgb').glob('*.png')):
    raw=int(frame.stem);image=np.asarray(Image.open(frame))/255.;original=np.asarray(Image.open(base/'mask'/frame.name))>0
    evidence=np.zeros_like(original)
    # 第一轮只使用闭环前已有的三个参考视图，不偷看新49帧诊断输出。
    source=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026/crossview')/f'{raw}.npz'
    if source.exists():
        d=dict(np.load(source));evidence=d['valid']&(np.abs(image-d['rgb']).mean(2)<=8/255.)&original
    generation=original&~evidence
    if not (dest/'native_rgb'/frame.name).exists():(dest/'native_rgb'/frame.name).symlink_to(frame)
    Image.fromarray((generation*255).astype('uint8')).save(dest/'mask'/frame.name)
    Image.fromarray((evidence*255).astype('uint8')).save(dest/'evidence_mask'/frame.name)
    rows.append(dict(raw=raw,full_mask=int(original.sum()),kept_evidence=int(evidence.sum()),generate=int(generation.sum())))
(dest/'contract.json').write_text(json.dumps(dict(
    method='最简单证据保护强控制，不预先宣称论文核心算法',backend_fixed='ProPainter',
    consistency_rgb_tolerance=8/255.,threshold_not_tuned=True,rows=rows,
    evidence_is_appearance_support_not_geometric_proof=True,
    evaluation_on_preserved_input_pixels_is_not_generalization=True,
    no_new_training_updates=True),indent=2))
print(json.dumps(dict(kept=sum(r['kept_evidence'] for r in rows),target=sum(r['full_mask'] for r in rows))))
