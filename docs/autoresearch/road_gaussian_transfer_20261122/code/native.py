"""高斯路块迁移的实际原生渲染；材质来源不依赖二维生成。"""
from pathlib import Path
import importlib.util,json,time,os,sys
import numpy as np
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=Path(os.environ.get('ROAD_TRANSFER_RUN_ROOT',str(Path(__file__).resolve().parent.parent)))
PARENT=Path(os.environ.get('ROAD_TRANSFER_PARENT_ASSET',str(C/'road_surface_residual_20261121_r2/appearance_fit/asset')))
spec=importlib.util.spec_from_file_location('transfer_native_helpers',C/'road_surface_adaptive_20261121_r1/code/run_adaptive.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.ROOT=ROOT;m.PARENT=PARENT;m.OWNER=ROOT.name
torch=m.torch;load=m.load;read=m.read;write=m.write;save=m.save;tensor=m.tensor
FRAMES=(86,110,134,179,198)

def open_context(asset=PARENT,frames=FRAMES):
    if m.CURRENT is not None:close()
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);m.FRAMES=tuple(frames)
    write(ROOT/'gpu_lease.json',dict(active=True,owner=m.OWNER,gpu_authorized=True,optimizer_updates_allowed=400))
    ctx=m.context(asset);ctx._transfer_opened_at=time.time();return ctx

def render(ctx,raw,colors=None,opacity=None,scales=None,joint=False,subset_world=None,feature_world=None):
    if subset_world is None:
        features=None if feature_world is None else m.active_positions(ctx,np.asarray(feature_world))
        return m.geometry_render(ctx,raw,opacity,scales,colors=colors,joint=joint,feature_ids=features)
    assert not joint
    rows=ctx.base_packet(raw);camera=ctx.camera(raw)
    flags=np.isin(ctx.active_world_rows,np.asarray(subset_world));rows['mask']=rows['mask']&tensor(flags,torch.bool)
    ctx.require_lease();ctx.native_calls+=1
    out=ctx.scene._bindings().raster(camera,ctx.scene.pipe,0,rows,ctx.scene._background)
    return dict(rgb=out['rgb'],alpha=out['alpha'].squeeze())

def pin(parent,asset,record):m.pin(parent,asset,record)
def close():
    if m.CURRENT is not None:
        log=ROOT/'runtime/native_session_log.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:stream.write(json.dumps(dict(PID=os.getpid(),entry=sys.argv,asset=str(m.CURRENT.asset_dir),
            native_context_F=m.CURRENT.native_calls,started_at=getattr(m.CURRENT,'_transfer_opened_at',None),ended_at=time.time(),new_diffusion_calls=0))+'\n')
    m.close_ctx();lease=read(ROOT/'gpu_lease.json');lease['active']=False;write(ROOT/'gpu_lease.json',lease)

def export(ctx,folder,colors=None,opacity=None,scales=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);rows=[]
    with torch.no_grad():
        for raw in FRAMES:
            r=render(ctx,raw,colors,opacity,scales);j=render(ctx,raw,colors,opacity,scales,joint=True)
            d=dict(road_rgb=r['rgb'].cpu().numpy(),road_alpha=r['alpha'].cpu().numpy(),joint_rgb=j['rgb'].cpu().numpy(),joint_alpha=j['alpha'].cpu().numpy(),K=ctx.K_masks[raw])
            np.savez_compressed(folder/f'{raw:03d}.npz',**d)
            for key in ['road','joint']:
                m.Image.fromarray(np.rint(np.clip(d[key+'_rgb'].transpose(1,2,0),0,1)*255).astype(np.uint8)).save(folder/f'{raw:03d}_{key}.png')
            m.Image.fromarray(np.rint(np.clip(d['road_alpha'],0,1)*255).astype(np.uint8)).save(folder/f'{raw:03d}_alpha.png')
            rows.append(dict(raw=raw,packet=str(folder/f'{raw:03d}.npz')))
    write(folder/'receipt.json',dict(complete=True,nativeF=10,rows=rows));return rows
