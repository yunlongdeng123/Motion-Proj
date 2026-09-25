"""从已完成的sweep读取图像、几何和actor证据；不改变渲染。"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir',type=Path,required=True)
args = parser.parse_args()
root = args.input_dir
manifest = json.loads((root/'manifest.json').read_text())
fig,axes = plt.subplots(len(manifest['samples']),3,figsize=(15,3*len(manifest['samples'])),squeeze=False)
rows=[]
for i,sample in enumerate(manifest['samples']):
    d=sample['delta_d_m']
    stem=f"frame{sample['frame']:03d}_d{d:+.2f}"
    rgb=np.array(Image.open(root/f'{stem}_rgb.png'))
    acc=np.load(root/f'{stem}_acc.npy').squeeze()
    depth=np.load(root/f'{stem}_depth.npy').squeeze()
    assert np.isfinite(acc).all() and np.isfinite(depth).all()
    assert sample['render_finite']
    covered=acc>.5
    # Rasterizer depth is alpha-weighted. Both raw and normalized statistics
    # are retained; neither gives geometric ground truth in unobserved areas.
    normalized=depth/np.maximum(acc,1e-6)
    actors=sample['actor_world_transform_checks']
    rows.append({'offset_m':d,'delta_s_m':sample['delta_s_m'],'delta_yaw_deg':sample['delta_yaw_deg'],
                 'actor_timestamp':sample['actor_timestamp'],'actor_count':len(actors),
                 'max_actor_translation_delta_m':max((a['translation_delta_max_m'] for a in actors),default=0),
                 'max_actor_quaternion_delta':max((a['quaternion_delta_max'] for a in actors),default=0),
                 'acc_mean':float(acc.mean()),'acc_lt_0_1_fraction':float((acc<.1).mean()),
                 'white_and_low_acc_fraction':float(((rgb>250).all(-1)&(acc<.1)).mean()),
                 'raw_depth_median':float(np.median(depth)),
                 'covered_normalized_depth_median':float(np.median(normalized[covered])) if covered.any() else None})
    axes[i,0].imshow(rgb)
    axes[i,0].set_title(f'd={d:+.2f} m; ds={sample["delta_s_m"]:.1e} m; dyaw={sample["delta_yaw_deg"]:.1e} deg')
    axes[i,1].imshow(acc,cmap='gray',vmin=0,vmax=1)
    axes[i,1].set_title(f'Gaussian acc: mean {acc.mean():.3f}')
    axes[i,2].imshow(np.where(covered,normalized,np.nan),cmap='viridis',vmin=0,vmax=30)
    axes[i,2].set_title('Depth / acc, acc>0.5; display 0-30 m')
    for ax in axes[i]: ax.axis('off')
fig.suptitle(f'{manifest["checkpoint_iteration"]} iterations | camera {manifest["samples"][0]["camera"]}, frame {manifest["samples"][0]["frame"]}\nEgo camera edit; actual actor world transforms checked',fontsize=14)
fig.tight_layout(rect=(0,0,1,.97))
fig.savefig(root/'sweep.png',dpi=110,facecolor='white')
report={'checkpoint_iteration':manifest['checkpoint_iteration'],'config':manifest['config'],
        'samples':rows,'all_finite':True,
        'actor_world_invariant':all(r['max_actor_translation_delta_m']<=1e-5 and r['max_actor_quaternion_delta']<=1e-6 for r in rows),
        'note':'depth diagnostic only; near-object occlusion is not a novel-view quality score',
        'failure_ledger_refs':['V76-F01'],'failure_ledger_delta':'none'}
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
