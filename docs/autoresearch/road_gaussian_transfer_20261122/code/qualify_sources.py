"""先核验原参考高斯的实拍外观和兼容旋转ABI，再允许迁移。"""
from native import *
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter,binary_erosion
import os

def main():
    start=time.monotonic();out=ROOT/'source_native';out.mkdir(exist_ok=True)
    ctx=open_context();baseline={};quality=[]
    try:
        with torch.no_grad():
            for raw in FRAMES:
                a=render(ctx,raw);b=render(ctx,raw,joint=True)
                baseline[raw]={k:v.cpu().numpy() for k,v in {'road_rgb':a['rgb'],'road_alpha':a['alpha'],'joint_rgb':b['rgb'],'joint_alpha':b['alpha']}.items()}
            for patch in ['patch_00','patch_02']:
                p=load(ROOT/'source_selection/candidates'/f'{patch}.npz')
                for raw in (86,110):
                    result=render(ctx,raw,subset_world=p['world_row_ids']);rgb=result['rgb'].cpu().numpy().transpose(1,2,0);alpha=result['alpha'].cpu().numpy()
                    camera=load(C/'road_common1152_20261121/maps'/f'{raw:03d}.npz');T=camera['world_to_camera'];K=camera['camera_K'];X=p['source_world_roi_corners']
                    xyz=X@T[:3,:3].T+T[:3,3];uvh=xyz@K.T;uv=uvh[:,:2]/xyz[:,2:]-.5
                    polygon=Image.new('L',(1152,768));ImageDraw.Draw(polygon).polygon([tuple(x) for x in uv],fill=255);mask=np.asarray(polygon)>0
                    obs=np.asarray(Image.open(C/f'road_common1152_20261121/teacher/raw{raw:03d}/observation_1152.png'),np.float32)/255
                    effective=np.divide(rgb,alpha[:,:,None],out=np.zeros_like(rgb),where=alpha[:,:,None]>1e-8)
                    score=binary_erosion(mask,iterations=3)&(alpha>=.95)
                    mae=float(np.abs(effective[score]-obs[score]).mean()) if score.any() else None
                    hf=effective-gaussian_filter(effective,(1,1,0));ht=obs-gaussian_filter(obs,(1,1,0))
                    numerator=float((hf[score]*ht[score]).sum());denominator=float(np.linalg.norm(hf[score])*np.linalg.norm(ht[score]))
                    corr=numerator/denominator if denominator>0 else None
                    yy,xx=np.nonzero(mask);box=[max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(1152,int(xx.max())+13),min(768,int(yy.max())+13)]
                    folder=out/patch;folder.mkdir(exist_ok=True)
                    for name,image in [('observation',obs),('isolated',effective),('full_road',baseline[raw]['road_rgb'].transpose(1,2,0))]:
                        im=Image.fromarray(np.rint(np.clip(image,0,1)*255).astype(np.uint8));im.crop(tuple(box)).save(folder/f'{raw:03d}_{name}_crop.png')
                    quality.append(dict(patch=patch,raw=raw,score_n=int(score.sum()),RGB_MAE=mae,HF_cosine=corr,alpha95_fraction=float((alpha[mask]>=.95).mean()),bbox=box))
        export(ctx,ROOT/'parent/rendered')
        from drivingeditor_dynamic_edit.v44_factory import _legacy_road_world_quaternions
        road=load(PARENT/'road.npz');probe=ROOT/'qualification/zero_override/asset';probe.mkdir(parents=True,exist_ok=True)
        road['gaussian_rotation_world_wxyz']=_legacy_road_world_quaternions(road['normal_world'],road['model_from_world'],road['meter_to_model'])
        road['gaussian_rotation_world_override']=np.zeros(len(road['active']),bool)
        np.savez_compressed(probe/'road.npz',**road)
        if not (probe/'provenance.npz').exists():os.link(PARENT/'provenance.npz',probe/'provenance.npz')
        pin(PARENT,probe,dict(stage='optional_rotation_zero_override_probe',new_diffusion_calls=0))
        close();ctx=open_context(probe);checks=[]
        with torch.no_grad():
            for raw in FRAMES:
                a=render(ctx,raw);b=render(ctx,raw,joint=True)
                d={'road_rgb':a['rgb'],'road_alpha':a['alpha'],'joint_rgb':b['rgb'],'joint_alpha':b['alpha']}
                row={k:bool(np.array_equal(v.cpu().numpy(),baseline[raw][k])) for k,v in d.items()}
                assert all(row.values()),(raw,row);checks.append(dict(raw=raw,exact=row))
        write(ROOT/'qualification/native_rotation_compatibility.json',dict(passed=True,rows=checks,nativeF=20,nativeB=0,optimizer_updates=0))
        assert any(x['score_n']>50 and x['RGB_MAE']<.05 for x in quality)
        write(out/'receipt.json',dict(complete=True,quality=quality,source_isolation_nativeF=4,source_photo_not_generated=True,
            reference_asset=str(PARENT),new_diffusion_calls=0,optimizer_updates=0,seconds=time.monotonic()-start))
        print(json.dumps(read(out/'receipt.json')),flush=True)
    finally:close()
if __name__=='__main__':main()
