"""原采集尺寸的只读源块核验；不优化、不修改任何场景资产。"""
from pathlib import Path
import argparse,json,os,sys,time
import numpy as np
from scipy.ndimage import gaussian_filter,binary_erosion
from PIL import Image,ImageDraw

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=C/'road_gaussian_source_quality_20261122_r1'
PRIOR=C/'road_gaussian_transfer_20261122_r2_isometric'
sys.path.insert(0,str(PRIOR/'code'))
import native
native.ROOT=ROOT;native.m.ROOT=ROOT;native.m.OWNER=ROOT.name
torch=native.torch
BANK=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')

def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def image(p,a):
    Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).save(p)

def context(frames):
    from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
    from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
    from road_capacity_native import NativeRoadContext
    asset=native.PARENT
    scene,m,loader=load_persistent_handoff(asset/'persistent/scene.json',asset/'persistent/proof.json')
    for obj in (scene.static,scene.dynamic,scene.environment):
        for p in obj.parameters() if hasattr(obj,'parameters') else ():p.requires_grad_(False)
    data=dict(m.data);data['render_wh']=[1920,1280]
    headers=SceneManifest(data,asset/'persistent').camera_headers();cams={}
    def camera(raw):
        if raw not in cams:cams[raw]=make_native_camera(headers[raw])
        return cams[raw]
    scene.camera_provider=camera
    ROOT.mkdir(exist_ok=True)
    write(ROOT/'gpu_lease.json',dict(active=True,owner=native.m.OWNER,gpu_authorized=True,optimizer_updates_allowed=0))
    ctx=NativeRoadContext(scene,m,loader,asset,frames,ROOT/'gpu_lease.json',native.m.OWNER)
    native.m.CURRENT=ctx;ctx._transfer_opened_at=time.time();return ctx

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frames',nargs='+',type=int,default=[86,110])
    parser.add_argument('--output',default='native1920');args=parser.parse_args()
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest=read(PRIOR/'source_selection/manifest.json');bank={int(v['raw']):v['id'] for v in read(BANK/'receipt.json')['frames']}
    ctx=context(args.frames);results=[];torch.set_num_threads(2)
    try:
        with torch.no_grad():
            for raw in args.frames:
                photo_d=read(BANK/f'observations/{raw:03d}.json')
                photo=np.asarray(Image.open(BANK/photo_d['image_path']).convert('RGB'),np.float32)/255
                assert photo.shape==(1280,1920,3)
                cal={s.split(':')[0]:np.fromstring(s.split(':')[1],sep=' ') for s in (BANK/'calib'/(bank[raw]+'.txt')).read_text().splitlines()}
                K=cal['P0'].reshape(3,4)[:,:3];T=np.eye(4);T[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4)
                T=T@np.linalg.inv(np.asarray(photo_d['ego_world']).reshape(4,4))
                full=native.render(ctx,raw);full_alpha=full['alpha'].cpu().numpy()
                full_rgb=full['rgb'].cpu().numpy().transpose(1,2,0)/np.maximum(full_alpha[:,:,None],1e-8)
                for entry in manifest['candidates']:
                    patch=entry['candidate_id'];s=load(entry['packet'])
                    raster=native.render(ctx,raw,subset_world=s['world_row_ids'])
                    alpha=raster['alpha'].cpu().numpy();rgb=raster['rgb'].cpu().numpy().transpose(1,2,0)/np.maximum(alpha[:,:,None],1e-8)
                    world=np.asarray(entry['world_roi_corners']);q=world@T[:3,:3].T+T[:3,3];p=q@K.T;uv=p[:,:2]/p[:,2:]-.5
                    if (q[:,2]<=0).any():continue
                    mask_img=Image.new('L',(1920,1280));ImageDraw.Draw(mask_img).polygon([tuple(v) for v in uv],fill=255)
                    core=np.asarray(mask_img)>0;inner=binary_erosion(core,iterations=5)
                    valid=inner&(alpha>=.95)
                    lo=np.maximum(0,np.floor(uv.min(0)-12).astype(int));hi=np.minimum([1920,1280],np.ceil(uv.max(0)+13).astype(int))
                    if (hi<=lo).any():continue
                    x0,y0=lo;x1,y1=hi;crop=np.s_[y0:y1,x0:x1];folder=out/patch/f'{raw:03d}';folder.mkdir(parents=True)
                    image(folder/'observation.png',photo[crop]);image(folder/'isolated.png',rgb[crop]);image(folder/'full_road.png',full_rgb[crop])
                    Image.fromarray(np.uint8(core[crop])*255).save(folder/'core_mask.png');Image.fromarray(np.uint8(valid[crop])*255).save(folder/'score_mask.png')
                    image(folder/'isolated_alpha.png',alpha[crop])
                    np.savez_compressed(folder/'core_float.npz',observation=photo[crop],isolated_rgb=rgb[crop],alpha=alpha[crop],
                        full_road_rgb=full_rgb[crop],full_road_alpha=full_alpha[crop],core=core[crop],score_mask=valid[crop],bbox_xyxy=np.r_[lo,hi])
                    a=(photo-gaussian_filter(photo,(1.2,1.2,0)))[valid].ravel()
                    b=(rgb-gaussian_filter(rgb,(1.2,1.2,0)))[valid].ravel()
                    denominator=np.linalg.norm(a)*np.linalg.norm(b)
                    results.append(dict(patch=patch,raw=raw,bbox_xyxy=np.r_[lo,hi].tolist(),crop_wh=(hi-lo).tolist(),
                        core_pixels=int(core.sum()),score_pixels=int(valid.sum()),
                        RGB_MAE=float(abs(photo-rgb)[valid].mean()) if valid.any() else None,
                        HF_cosine=float(a@b/denominator) if denominator>1e-20 else None,
                        photo_HF_RMS=float(np.sqrt(np.mean(a*a))) if len(a) else None,
                        GS_HF_RMS=float(np.sqrt(np.mean(b*b))) if len(b) else None,
                        isolation_to_full_RGB_MAE=float(abs(rgb-full_rgb)[valid].mean()) if valid.any() else None,
                        photo_path=str(BANK/photo_d['image_path']),output=str(folder)))
        write(out/'receipt.json',dict(complete=True,render_wh=[1920,1280],frames=args.frames,rows=results,
            nativeF=ctx.native_calls,nativeB=0,optimizer_updates=0,new_diffusion_calls=0,original_asset_modified=False,
            score_policy='Projected physical source core eroded5px AND isolated alpha>=.95; full capture grid, not upscaled report raster',
            limitation='Current source asset fidelity only; no new source reconstruction or target transfer has been executed.'))
        print(json.dumps(read(out/'receipt.json'),ensure_ascii=False),flush=True)
    finally:native.close()

if __name__=='__main__':main()
