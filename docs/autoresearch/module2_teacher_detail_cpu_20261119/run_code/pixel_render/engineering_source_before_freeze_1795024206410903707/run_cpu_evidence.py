"""生成解析世界oracle/CPU代数证据；不接AutoDL、不运行native、0optimizer更新。"""
import os
for key in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,subprocess,sys,xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.special import ndtr
from PIL import Image,ImageDraw
from pixel_world_material import cpu_front_to_back,shade_pixel_material,look_at_camera
from test_pixel_render_cpu import alias_metrics,runtime_parity,texture

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'validation';OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)


def save_strip(name,arrays,labels):
    w=len(arrays[0]);image=Image.new('RGB',(w*len(arrays),90),'#101a28');draw=ImageDraw.Draw(image)
    for i,(x,label)in enumerate(zip(arrays,labels)):
        band=np.broadcast_to(np.clip(x,0,1)[None,:,None],(64,w,3));image.paste(Image.fromarray(np.round(band*255).astype('uint8')),(i*w,22));draw.text((i*w+4,4),label,fill='white')
    image.save(OUT/name)


def marking(width):
    x=np.linspace(-.3,.3,241);centers=np.arange(-.6,.6001,.02);sigma=.02;pixel=.005
    alpha=.7*np.exp(-.5*((x[None]-centers[:,None])/sigma)**2);alpha[abs(x[None]-centers[:,None])>3*sigma]=0
    ci=.3+.6*(ndtr((width/2-centers)/sigma)-ndtr((-width/2-centers)/sigma))
    basis,_=cpu_front_to_back(alpha[:,None,:],np.repeat(ci[:,None],3,1),np.ones(len(ci),bool),mode='road_only')
    fraction=np.maximum(0,np.minimum(x+pixel/2,width/2)-np.maximum(x-pixel/2,-width/2))/pixel
    target=.3+.6*fraction;world_rgb=np.broadcast_to(target[None,:,None],(1,len(x),3))
    new=shade_pixel_material(basis,world_rgb,np.ones((1,len(x)),bool))
    old=basis.baseline_rgb[0,0].numpy()/basis.eligible_weight[0].numpy();new_color=new[0,0].detach().numpy()/basis.eligible_weight[0].numpy()
    q=abs(x)<width/4
    np.savez_compressed(OUT/f'marking_w{int(width*1000):03d}.npz',world_x=x,target=target,average_color=old,pixel_color=new_color,
        fixed_alpha=basis.alpha.numpy(),centers=centers,sigma_fixture_m=sigma)
    save_strip(f'marking_w{int(width*1000):03d}.png',[target,old,new_color],['analytic world target','CPU avg color + GS weights','CPU eligible pixel shader'])
    return dict(width_m=width,period_phase='1D crosspaint only; curve/dashes separate camera oracle',
        projected_width_pixels=width/pixel,CPU_fixture_sigma_m=sigma,
        average_MAE=float(abs(old-target).mean()),pixel_MAE=float(abs(new_color-target).mean()),
        average_core_contrast=float((old[q].mean()-.3)/.6),pixel_core_contrast=float((new_color[q].mean()-.3)/.6),
        native_scene_result=False,geometry_role='synthetic fixed1D weights; not actual P Gaussian packet',
        target_source='closed-form independent world top-hat pixelbox area; Gaussian ci uses analytic normal CDF, no GS self target')


def world_texture(xyz,width):
    x,y=xyz[...,0],xyz[...,1];R=120.;u=R*np.arctan2(y,R-x);v=R-np.hypot(y,R-x)
    a=np.stack((.32+.015*np.sin(2*np.pi*y/8),.33+.01*np.sin(2*np.pi*y/8),.35+.008*np.cos(2*np.pi*x/2)),-1)
    def smooth(z):z=np.clip(z,0,1);return z*z*(3-2*z)
    cross=smooth((width/2-abs(v))/.02);phase=np.mod(u-.45,3);dash=smooth(phase/.1)*smooth((1.5-phase)/.1)
    paint=cross*dash;return a*(1-paint[...,None])+np.array([.88,.88,.84])*paint[...,None]


def camera_oracles():
    records=[];w,h=128,96;yy,xx=np.indices((h,w));base=np.stack((xx,yy),-1)
    k=(np.arange(4)+.5)/4-.5;dx,dy=np.meshgrid(k,k);offset=np.stack((dx.ravel(),dy.ravel()),-1)
    for name,camera in [('A',look_at_camera([0,-2,1.6],[0,3,0],w,h,100)),('B',look_at_camera([.4,-1.7,1.4],[.2,3.5,0],w,h,100))]:
        X,valid,J=camera.plane_map(pixels=base[...,None,:]+offset)
        central,central_valid,J0=camera.plane_map()
        for width in(.12,.24):
            color=world_texture(X,width).mean(-2);visible=central_valid&(abs(central[...,0])<2)&(central[...,1]>0)&(central[...,1]<8)
            color[~visible]=0
            Image.fromarray(np.round(np.clip(color,0,1)*255).astype('uint8')).save(OUT/f'CPU_ideal_camera{name}_w{int(width*1000):03d}.png')
            normal=np.stack((120-central[...,0],-central[...,1]),-1);normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
            footprint=np.linalg.norm(np.einsum('...i,...ij->...j',normal,J0),axis=-1);pixels=width/np.maximum(footprint,1e-12)
            records.append(dict(camera=name,width_m=width,resolvable_ge3px_pixels=int((visible&(pixels>=3)).sum()),
                visible_pixels=int(visible.sum()),target_source='same analytic world circle120m / dash period3m,phase.45m; independent plane oracle only',
                plane_is_synthetic_fixture_only=True,actual_cached_G_s_maps_are_separate=True))
    return records


def main():
    proc=subprocess.run([sys.executable,'-m','pytest',str(ROOT/'test_pixel_render_cpu.py'),'-q','--junitxml='+str(OUT/'pytest.xml')],capture_output=True,text=True)
    (OUT/'pytest.txt').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    suites=ET.parse(OUT/'pytest.xml').getroot();cases=suites.findall('.//testcase')
    receipt=dict(schema='pixel_world_CPU_reference_acceptance.v1',passed=True,unit_tests=len(cases),
        errors=len(suites.findall('.//error')),failures=len(suites.findall('.//failure')),GPU_executed=False,native_calls=0,optimizer_updates=0,
        CPU_reference_dtype='float64',future_runtime_parameter_dtype='float32',bound_recipe=dict(low_bound=.30,high_bound=.34),
        amplitude_status='central locked commonQ; historicalP .04 not used as new permission bound',
        alias=alias_metrics(),runtime_parity=runtime_parity(),marking=[marking(.12),marking(.24)],camera_oracles=camera_oracles(),
        native_export_entry='native_basis_adapter.export_native_basis; current phase blocked before ctx/import',
        K_gate='joint eligible alphaT exactly0; road-only behind-K weight is diagnostic, not a joint K failure',
        persistence='world texture + renderer_mode + scope + geometry reference; no five-view weight cache as world asset',
        native_validated=False,formal_allowed=False,
        evidence_boundary='CPU analytic/synthetic GS coefficient reference only. Current actual G_s geometry cache is geometry-only. No native binary/actual scene renderer/gradient is claimed validated.')
    p=OUT/'asset_example'
    if not p.exists():texture().save_asset(p,chart=dict(world_from_uv=np.eye(4)),scope_world_rows=[1,4,7],geometry_reference='CPU synthetic plane only',source_ledger='CPU synthetic legalteacher fixture',permission_mode='observed_plus_teacher_H')
    (OUT/'cpu_acceptance.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
