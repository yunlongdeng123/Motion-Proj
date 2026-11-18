"""CPU制作实际native预览与世界材质参数图；不调用生成模型或修图。"""
from pathlib import Path
import os,sys,json,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import numpy as np
from PIL import Image

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
D=P/'delivery';D.mkdir(exist_ok=True)
manifest=json.loads((P/'qa/final_result_manifest.json').read_text())
alias={'A0_midpoint':'A0_mid','TRELLIS2':'T2_mid'}
encoded=[]
for stage in manifest['stages']:
    name=alias.get(stage['name'],stage['name']);dest=D/'columns'/name;dest.mkdir(parents=True,exist_ok=True)
    for raw in manifest['frames']:
        src=Path(stage['array_template'].format(raw=raw))
        with np.load(src,allow_pickle=False) as z:
            for mode,key in [('road',stage['keys']['P']),('joint',stage['keys']['J'])]:
                rgb=z[key];assert rgb.shape==(3,320,480) and np.isfinite(rgb).all()
                path=dest/f'{raw:03d}_{mode}.webp'
                Image.fromarray(np.clip(np.rint(rgb.transpose(1,2,0)*255),0,255).astype(np.uint8)).save(path,lossless=True)
                encoded.append(dict(model=name,raw=raw,mode=mode,source=str(src),output=str(path.relative_to(D))))

from scipy.ndimage import binary_erosion
domains=D/'domains';domains.mkdir(exist_ok=True)
baseline=manifest['stages'][0]
for raw in manifest['frames']:
    rgba=np.zeros((320,480,4),np.uint8);masks=[]
    if raw in (86,110):
        with np.load(P/f'stage0_native_r1/{raw:03d}_targets.npz') as z:
            masks=[(z['A'],[64,197,231]),(z['C'],[246,174,72]),(z['fullS'],[233,104,178])]
    with np.load(baseline['array_template'].format(raw=raw)) as z:masks.append((z['K'],[92,220,144]))
    for mask,color in masks:
        edge=mask&~binary_erosion(mask,structure=np.ones((3,3),bool),border_value=0)
        rgba[edge,:3]=color;rgba[edge,3]=225
    Image.fromarray(rgba).save(domains/f'{raw:03d}.png')

import torch
torch.set_num_threads(2)
sys.path.insert(0,str(P/'code'))
from world_material_field import blur
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=D/'fields';F.mkdir(exist_ok=True)
saved=torch.load(P/'formal_r1/ABC800/asset/world_field.pt',map_location='cpu',weights_only=False,mmap=True)
s=saved['state'];settings=saved['settings']
with torch.no_grad():
    low=settings['low_bound']*torch.tanh(s['low'])*s['low_parameter_mask']
    raw=torch.tanh(s['high'])*s['high_parameter_mask']
    high=.5*settings['high_bound']*(raw-blur(raw,settings['highpass_sigma_cells']))
    effective=s['mH'].to(high.dtype)*high
with np.load(P/'cpu_exact_r1/A0.npz') as z:a={k:z[k].copy() for k in ['rgb_srgb','valid','mH','grid_world_row']}
with np.load(Path(manifest['stages'][0]['array_template'].format(raw=86)).parent/'asset_guarded/provenance.npz') as z:
    origin=z['world_keys'][a['grid_world_row'][0,0]]
i0,j0=(-99-origin[0],-66-origin[1]);i0,j0=int(i0),int(j0)
yy,xx=np.mgrid[j0:j0+128,i0:i0+128]
uv=np.stack((xx,yy),-1).astype(np.float64)/settings['low_stride']
lv=low.numpy();lo=np.floor(uv).astype(int);t=uv-lo;L=np.zeros((128,128,3))
for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
    x=np.clip(lo[...,0]+dx,0,lv.shape[1]-1);y=np.clip(lo[...,1]+dy,0,lv.shape[0]-1)
    w=(t[...,0] if dx else 1-t[...,0])*(t[...,1] if dy else 1-t[...,1]);L+=w[...,None]*lv[y,x]
H=effective.numpy()[j0:j0+128,i0:i0+128]
A=a['rgb_srgb'][j0:j0+128,i0:i0+128];valid=a['valid'][j0:j0+128,i0:i0+128]
world=np.clip(A+L+H,0,1)
def rgb_plot(value,name):
    value=np.flipud(np.where(valid[...,None],value,.22))
    Image.fromarray(np.rint(value*255).clip(0,255).astype(np.uint8)).resize((512,512),Image.Resampling.NEAREST).save(F/name)
rgb_plot(A,'A0_world.png');rgb_plot(world,'ABC_world_unfiltered.png')
def scalar_plot(value,name,maximum):
    fig,ax=plt.subplots(figsize=(4,4),dpi=140);fig.patch.set_facecolor('#101a2a');ax.set_facecolor('#101a2a')
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#343d48')
    v=np.flipud(np.ma.array(value,mask=~valid))
    im=ax.imshow(v,cmap=cmap,vmin=0,vmax=maximum,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
    bar=fig.colorbar(im,ax=ax,fraction=.045,pad=.03);bar.ax.tick_params(colors='#dbe8f7',labelsize=8)
    fig.tight_layout(pad=.3);fig.savefig(F/name,facecolor=fig.get_facecolor());plt.close(fig)
scalar_plot(np.sqrt(np.mean(L**2,axis=-1)),'low_RGB_RMS.png',.30)
scalar_plot(np.sqrt(np.mean(H**2,axis=-1)),'high_effective_RGB_RMS.png',.04)
scalar_plot(a['mH'][j0:j0+128,i0:i0+128],'high_permission.png',1.)
shutil.copy2(P/'trellis2/world_prior_r1/patch_world_basecolor.png',F/'TRELLIS_world.png')
(D/'preview_receipt.json').write_text(json.dumps(dict(schema='actual_native_and_field_delivery.v1',GPU_operations=0,
 previews=encoded,columns=len(manifest['stages']),frames=manifest['frames'],native_encoding='roundsRGB8+losslessWebP; no image enhancement',
 field_chart=dict(key_min=[-99,-66],shape=[128,128],nominal_extent_m=[2.56,2.56],orientation='imageY=-chart_j',
 world_RGB='unfiltered A0+DeltaL+mHDeltaH before per-row read/legacy overrides; not a native image',
 delta_maps='actual bounded parameter-field RGB_RMS, never latent magnitudes'),complete_model_backup=False),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(native_images=len(encoded),field_images=6,GPU_operations=0)))
