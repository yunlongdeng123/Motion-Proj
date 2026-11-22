"""源照片、持久颜色与原生渲染的分段诊断；不更新几何或目标道路。"""
from pathlib import Path
import argparse, json, sys, time
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT = CAMPAIGN/'road_source_fidelity_20261122_r2'
OLD = CAMPAIGN/'road_gaussian_source_quality_20261122_r1'
sys.path.insert(0, str(OLD/'code'))
import render_source1920 as h
h.ROOT = ROOT
h.native.ROOT = ROOT
h.native.m.ROOT = ROOT
h.native.m.OWNER = ROOT.name
torch = h.torch

def projection(raw):
    observation = h.read(h.BANK/f'observations/{raw:03d}.json')
    identifiers = {int(v['raw']):v['id'] for v in h.read(h.BANK/'receipt.json')['frames']}
    calibration = {s.split(':')[0]:np.fromstring(s.split(':')[1], sep=' ')
                   for s in (h.BANK/'calib'/(identifiers[raw]+'.txt')).read_text().splitlines()}
    K = calibration['P0'].reshape(3,4)[:,:3]
    T = np.eye(4); T[:3] = calibration['Tr_velo_to_cam_0'].reshape(3,4)
    T = T@np.linalg.inv(np.asarray(observation['ego_world']).reshape(4,4))
    photo_path = h.BANK/observation['image_path']
    photo = np.asarray(Image.open(photo_path).convert('RGB'), dtype=np.float32)/255
    def project(xyz):
        camera = np.asarray(xyz)@T[:3,:3].T+T[:3,3]
        q = camera@K.T
        return q[:,:2]/q[:,2:]-.5, camera[:,2]
    return photo, project, str(photo_path), K, T

def spectral(photo, rgb, mask):
    result = []
    for sigma in (1.2, 5/3, 3.):
        a = (photo-gaussian_filter(photo,(sigma,sigma,0)))[mask].ravel()
        b = (rgb-gaussian_filter(rgb,(sigma,sigma,0)))[mask].ravel()
        an,bn = np.linalg.norm(a),np.linalg.norm(b)
        result.append(dict(sigma_native_px=sigma,
            HF_cosine=float(a@b/(an*bn)) if an*bn>1e-20 else None,
            HF_RMS_ratio=float(bn/an) if an>1e-20 else None))
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--frames',nargs='+',type=int,default=[86,110])
    parser.add_argument('--patches',nargs='+',default=['patch_00','patch_02'])
    parser.add_argument('--output',default='chain_probe')
    args=parser.parse_args()
    out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    h.write(out/'contract.json',dict(task='ROAD-SOURCE-FIDELITY-20261122',
        run=ROOT.name,stage=args.output,patches=args.patches,frames=args.frames,
        render_wh=[1920,1280],optimizer_updates=0,new_diffusion_calls=0,
        geometry_opacity_scales_frozen=True,source_selection='Manual diagnostic anchors, not an automatic selection result',
        direct_sampling='PIL RGB/255; bilinear at projected Gaussian center; no gamma conversion, no optimization',
        score_policy='physical patch core eroded5px AND fixed initial isolated alpha>=.95; photo qualification applied separately before fitting',
        target_migration=False,production_promoted=False,human_verdict=None))
    manifest=h.read(h.PRIOR/'source_selection/manifest.json')
    entries={v['candidate_id']:v for v in manifest['candidates']}
    ctx=h.context(args.frames); torch.set_num_threads(2)
    base=ctx.base_rgb.detach().clone(); rows=[]; count=0
    def render(raw, ids, colors):
        nonlocal count
        packet=ctx.base_packet(raw)
        keep=np.isin(ctx.active_world_rows,ids)
        packet['mask']=packet['mask']&torch.tensor(keep,device='cuda',dtype=torch.bool)
        packet['colors_precomp']=colors
        ctx.require_lease();ctx.native_calls+=1;count+=1
        raster=ctx.scene._bindings().raster(ctx.camera(raw),ctx.scene.pipe,0,packet,ctx.scene._background)
        alpha=raster['alpha'].squeeze().detach().cpu().numpy()
        premult=raster['rgb'].detach().cpu().numpy().transpose(1,2,0)
        return premult/np.maximum(alpha[:,:,None],1e-8),alpha,premult
    try:
        with torch.no_grad():
            for patch in args.patches:
                entry=entries[patch]; source=h.load(entry['packet'])
                ids=source['world_row_ids']; positions=np.searchsorted(ctx.active_world_rows,ids)
                assert np.array_equal(ctx.active_world_rows[positions],ids)
                assert np.array_equal(base[positions].cpu().numpy(),source['reference_srgb'])
                core=source['core_mask']; core_pos=torch.tensor(positions[core],device='cuda')
                all_pos=torch.tensor(positions,device='cuda')
                for raw in args.frames:
                    photo,project,photo_path,K,T=projection(raw)
                    uv,depth=project(entry['world_roi_corners'])
                    if (depth<=0).any():continue
                    canvas=Image.new('L',(1920,1280))
                    ImageDraw.Draw(canvas).polygon([tuple(x) for x in uv],fill=255)
                    physical=np.asarray(canvas)>0
                    baseline,alpha,premult=render(raw,ids,base)
                    score=binary_erosion(physical,iterations=5)&(alpha>=.95)
                    if score.sum()<50:continue
                    lo=np.maximum(0,np.floor(uv.min(0)-12).astype(int))
                    hi=np.minimum([1920,1280],np.ceil(uv.max(0)+13).astype(int))
                    x0,y0=lo;x1,y1=hi;crop=np.s_[y0:y1,x0:x1]
                    folder=out/patch/f'{raw:03d}';folder.mkdir(parents=True)
                    xyz_uv,xyz_depth=project(source['world_xyz'])
                    samples=np.stack([map_coordinates(photo[:,:,c],[xyz_uv[:,1],xyz_uv[:,0]],order=1,mode='nearest') for c in range(3)],axis=1)
                    changed=base.index_copy(0,core_pos,torch.tensor(samples[core],device='cuda'))
                    direct,direct_alpha,_=render(raw,ids,changed)
                    assert np.array_equal(alpha,direct_alpha)
                    color=np.asarray([.18,.32,.70],dtype=np.float32)
                    uniform=base.index_copy(0,all_pos,torch.tensor(np.tile(color,(len(ids),1)),device='cuda'))
                    constant,constant_alpha,_=render(raw,ids,uniform)
                    assert np.array_equal(alpha,constant_alpha)
                    repaired={k:v.copy() for k,v in source.items()}
                    repaired['reference_srgb'][core]=samples[core]
                    packet_path=folder/'source_direct_rgb.npz'
                    np.savez_compressed(packet_path,**repaired)
                    loaded=h.load(packet_path)
                    frozen={k:bool(np.array_equal(v,loaded[k])) for k,v in source.items() if k!='reference_srgb'}
                    assert all(frozen.values())
                    saved_delta=float(np.max(np.abs(loaded['reference_srgb'][core]-samples[core])))
                    assert saved_delta==0
                    loaded_colors=base.index_copy(0,all_pos,torch.tensor(loaded['reference_srgb'],device='cuda'))
                    reloaded,reloaded_alpha,_=render(raw,ids,loaded_colors)
                    assert np.array_equal(alpha,reloaded_alpha)
                    for name,a in [('observation',photo),('original',baseline),('direct_sample_render',direct),('reloaded_render',reloaded),('constant_render',constant)]:h.image(folder/(name+'.png'),a[crop])
                    Image.fromarray(np.uint8(physical[crop])*255).save(folder/'core_mask.png')
                    Image.fromarray(np.uint8(score[crop])*255).save(folder/'score_mask.png')
                    h.image(folder/'alpha.png',alpha[crop])
                    np.savez_compressed(folder/'chain_float.npz',observation=photo[crop],original_rgb=baseline[crop],
                        direct_rgb=direct[crop],reloaded_rgb=reloaded[crop],constant_rgb=constant[crop],alpha=alpha[crop],
                        core=physical[crop],score_mask=score[crop],bbox_xyxy=np.r_[lo,hi],
                        sampled_uv=xyz_uv,sampled_rgb=samples,original_row_rgb=source['reference_srgb'],
                        loaded_row_rgb=loaded['reference_srgb'],source_core_rows=core,world_row_ids=ids)
                    metrics={}
                    for name,rgb in [('original',baseline),('direct_sample',direct),('reloaded',reloaded)]:
                        metrics[name]=dict(RGB_MAE=float(np.abs(photo-rgb)[score].mean()),
                            signed_RGB_bias=(rgb-photo)[score].mean(0).tolist(),spectral=spectral(photo,rgb,score))
                    rows.append(dict(patch=patch,raw=raw,score_n=int(score.sum()),
                        photo=photo_path,output=str(folder),bbox_xyxy=np.r_[lo,hi].tolist(),metrics=metrics,
                        stored_to_runtime_base_max_abs=0.,sample_to_saved_RGB_max_abs=saved_delta,
                        direct_to_reloaded_render_max_abs=float(np.max(np.abs(direct-reloaded))),
                        constant_render_max_abs=float(np.max(np.abs(constant[score]-color))),
                        constant_render_mean_abs=float(np.mean(np.abs(constant[score]-color))),
                        alpha_exact=True,non_RGB_fields_exact=frozen,
                        halo_RGB_exact=bool(np.array_equal(loaded['reference_srgb'][~core],source['reference_srgb'][~core]))))
                    print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
        h.write(out/'receipt.json',dict(complete=True,rows=rows,nativeF=count,nativeB=0,optimizer_updates=0,
            new_diffusion_calls=0,original_asset_modified=False,target_migration=False))
    finally:h.native.close()

if __name__=='__main__':main()
