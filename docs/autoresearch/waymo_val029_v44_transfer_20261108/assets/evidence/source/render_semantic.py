import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image
from native_runtime import ROOT,load,camera,native_rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--iteration',type=int,required=True);p.add_argument('--ownership',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--asset',type=Path);p.add_argument('--posterior',type=Path);p.add_argument('--raws',type=int,nargs='+',default=[130,155,160,190]);a=p.parse_args();a.output.mkdir(exist_ok=False)
    import torch
    pipe,models,cfg=load(a.model,a.iteration)
    from utils.v4_road_render import raster,merge_rows
    import gaussian_renderer
    static,dynamic,env=models;z=np.load(a.ownership/'semantic_ownership.npz');sids=torch.tensor(z['static_actor_ids'],device='cuda');dids=torch.tensor(z['dynamic_actor_ids'],device='cuda');bg=torch.zeros(3,device='cuda');rows_out=[]
    from drivingeditor_dynamic_edit import actor_ids
    source_ids=actor_ids(dynamic);dids=torch.where(dids>0,dids,source_ids)
    asset=None
    if a.asset:
        import sys
        sys.path.insert(0,str(ROOT.parent/'scene008_a6_asset_controls_20261105/source'))
        sys.path.insert(0,str(ROOT.parent/'scene008_a6_insert_artifacts_20261105/source/scale'))
        from sh_asset import SHAsset
        from surface_lod_asset import SurfaceLODAsset
        base=SHAsset.from_geometry(a.asset).cuda();asset=SurfaceLODAsset(base,a.posterior)
    model_from_world=np.loadtxt(Path(cfg['source_path'])/'transform_matrix.txt')
    for raw in a.raws:
        cam=camera(a.model,cfg,raw);s=native_rows(cam,static,pipe);d=native_rows(cam,dynamic,pipe,True)
        for branch,ids in [(s,sids),(d,dids)]:
            f=branch['features'].new_zeros((len(ids),24));f[:,:2]=branch['features'][:,:2];f[:,2]=(ids==70);f[:,3]=(ids==25);f[:,4]=1-f[:,2]-f[:,3];branch['features']=f
        if asset is not None:
            s['mask'] &= sids!=25;d['mask'] &= (dids!=25)&(source_ids!=25)
        both=merge_rows(s,d);ids=torch.cat((sids,dids))
        if asset is not None:
            obs=json.loads((Path('/root/autodl-tmp/codex_tmp/waymo_val029_cpu_20261108/inputs/observations')/f'{raw:03d}.json').read_text());body=next(b for b in obs['boxes'] if b['track']=='DwjZr_bYUSZOr-XDGregnA');pose=torch.tensor(model_from_world@np.array(body['world_from_actor']).reshape(4,4),device='cuda',dtype=torch.float32)
            ar=asset.rows(pose,cam,pipe,1,precomputed=True);f=ar['features'].new_zeros((len(ar['means3D']),24));f[:,:2]=ar['features'];f[:,3]=1;ar['features']=f;both=merge_rows(both,ar);ids=torch.cat((ids,ids.new_full((len(ar['means3D']),),25)))
        folder=a.output/f'{raw:03d}';folder.mkdir();outputs={}
        with torch.no_grad():
            for name,delete in [('keep',False),('delete',True)]:
                values=dict(both,mask=both['mask'] & ((ids!=70) if delete else torch.ones_like(ids,dtype=torch.bool)))
                package=raster(cam,pipe,0,values,bg);rgb=package['rgb']+(1-package['alpha'])*env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
                out=rgb.cpu().numpy();outputs[name]=out;Image.fromarray(np.uint8(out.transpose(1,2,0).clip(0,1)*255+.5)).save(folder/f'{name}.png');np.savez_compressed(folder/f'{name}.npz',rgb=out,alpha=package['alpha'].cpu().numpy(),depth_sum=package['depth_sum'].cpu().numpy(),group_alpha=package['extra'][:3].cpu().numpy())
                if delete:assert float(package['extra'][0].abs().max())<1e-6
            native=gaussian_renderer.render(cam,static,dynamic,pipe,bg,env_map=env,is_training=False)['render'].cpu().numpy();difference=float(abs(native-outputs['keep']).max())
            if asset is None:assert difference<2e-5,difference
        rows_out.append(dict(raw=raw,keep_native_maxabs=difference,changed_pixels=int((abs(outputs['keep']-outputs['delete']).max(0)>1/255).sum())));print(rows_out[-1],flush=True)
    (a.output/'summary.json').write_text(json.dumps(dict(status='PASS_SEMANTIC_STATIC_DYNAMIC_SHARED_DELETE',frames=rows_out,deleted_rows=dict(static=int((sids==70).sum()),dynamic=int((dids==70).sum())),optimizer_updates=0,quality_accepted=None),indent=2))
if __name__=='__main__':main()
