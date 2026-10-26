"""统一三维操作基线：轨迹框投票绑定、删车/平移/复用插入/相机变轨。

原B未保存逐行actor身份，因此使用可审查的多时刻框投票作为第一版基线；
未决行保留，不把几何候选冒充语义GT。移动/插入后的质量须单独验收。
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp')
from editor_common import *
from world_state import WorldState

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
SOURCE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT));a=p.parse_args();root=Path(a.root)
    dest=root/'phase2_operations';dest.mkdir(exist_ok=True)
    n=Native();meta=read('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json')
    transform=np.asarray(meta['world_to_model'])@np.asarray(meta['coordinate_anchor']);inv=np.linalg.inv(transform)
    bindings=read(SOURCE/'dynamic_multi_actor_preparation.json')['actors'];track_ids=[b['raw_track_id'] for b in bindings]
    votes=torch.zeros((len(n.dynamic.get_xyz),5),dtype=torch.int16,device='cuda')
    active_count=torch.zeros(len(votes),dtype=torch.int16,device='cuda')
    tinv=torch.tensor(inv[:3,:3].T,dtype=torch.float32,device='cuda');offset=torch.tensor(inv[:3,3],dtype=torch.float32,device='cuda')
    local_from_world=np.linalg.inv(np.asarray(meta['coordinate_anchor']))
    started=time.time()
    with torch.no_grad():
        for raw in [89,97,105,113,121,129,137]:
            cam=n.camera(raw);xyz=n.dynamic.get_xyz_SHM(cam.timestamp)@tinv+offset
            active=n.dynamic.get_marginal_t(cam.timestamp)[:,0]>.05;active_count+=active.short()
            record=read(SOURCE/'annotations'/f'0008{raw:03d}.json')
            pose=local_from_world@np.array(record['frame_pose_world_from_vehicle']).reshape(4,4)
            labels={x['id']:x for x in record['laser_labels']}
            for actor,key in enumerate(track_ids):
                box=labels[key]['box'];heading=box['heading'];c,s=np.cos(heading),np.sin(heading)
                axes=pose[:3,:3]@np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                center=pose[:3,:3]@np.array([box['center_x'],box['center_y'],box['center_z']])+pose[:3,3]
                half=torch.tensor([box['length']/2,box['width']/2,box['height']/2],device='cuda')
                q=(xyz-torch.tensor(center,dtype=torch.float32,device='cuda'))@torch.tensor(axes,dtype=torch.float32,device='cuda')
                votes[:,actor]+=((q.abs()<=half).all(1)&active).short()
        top,index=votes.max(1);sorted_votes=votes.sort(1,descending=True).values
        assigned=(active_count>0)&(top.float()>=.6*active_count)&(top>sorted_votes[:,1])
        owners=torch.where(assigned,index,torch.full_like(index,-1))
        np.savez_compressed(dest/'dynamic_binding.npz',owner=cpu(owners),votes=cpu(votes),active_count=cpu(active_count))
        counts={b['description']:int((owners==i).sum()) for i,b in enumerate(bindings)}
        counts['unassigned']=int((owners<0).sum())
        selected=(owners==3)|(owners==4)
        ck=torch.load(str(OUT/'edit_smoke_preserve/full199/checkpoint1000_from0.pth'),weights_only=False)
        road=Road(FULL/'full199_initialization.npz');road.dc.copy_(ck['dc']);road.opacity.copy_(ck['opacity'])
        multiplier=torch.ones(len(n.static.get_xyz),device='cuda').index_copy(0,ck['ids'],ck['gate'].sigmoid())
        delta=torch.tensor(transform[:3,:3]@np.array([0.,1.,0.]),dtype=torch.float32,device='cuda')
        def camera_shift(camera,meters):
            c2w=cpu(camera.c2w).copy();c2w[:3,3]+=transform[:3,:3]@np.array([0.,meters,0.]);w2c=np.linalg.inv(c2w)
            return Camera(camera.colmap_id,w2c[:3,:3].T.copy(),w2c[:3,3].copy(),cx=camera.cx,cy=camera.cy,fx=camera.fx,fy=camera.fy,
                image=torch.zeros(3,320,480),image_name=camera.image_name,resolution=(480,320),timestamp=camera.timestamp)
        def render(camera,operation):
            static=prepare_rasterization(camera,n.static,n.args);static['means2D']=torch.zeros_like(static['means3D'])
            static['opacities']=static['opacities']*multiplier[:,None]
            static['features']=torch.cat((static['features'],torch.zeros_like(static['features'][:,:2])),1)
            dynamic=prepare_rasterization(camera,n.dynamic,n.args,dynamic=True);dynamic['means2D']=torch.zeros_like(dynamic['means3D'])
            if operation in ['delete_selected','camera_revisit']:dynamic['mask']=dynamic['mask']&~selected
            if operation=='move_selected':dynamic['means3D']=dynamic['means3D']+selected[:,None]*delta
            z=torch.cat((dynamic['means3D'],torch.ones_like(dynamic['means3D'][:,:1])),1)@camera.world_view_transform[:,2:3]
            dynamic['features']=torch.cat((z,torch.ones_like(z),torch.ones_like(z),torch.zeros_like(z)),1)
            kwargs=merge_rows(static,dynamic)
            if operation=='insert_reused':
                clone={k:(v.clone() if v is not None else None) for k,v in dynamic.items()}
                clone['mask']=clone['mask']&(owners==4);clone['means3D']=clone['means3D']+2*delta
                z=torch.cat((clone['means3D'],torch.ones_like(clone['means3D'][:,:1])),1)@camera.world_view_transform[:,2:3]
                clone['features']=torch.cat((z,torch.ones_like(z),torch.ones_like(z),torch.zeros_like(z)),1)
                kwargs=merge_rows(kwargs,clone)
            channels=kwargs['shs'].shape[1] if kwargs['shs'] is not None else 1
            rr=road_kwargs(camera,road,n.args,channels,precomputed_color=kwargs['colors_precomp'] is not None)
            rr['features']=torch.cat((rr['features'],torch.zeros_like(rr['features'][:,:1]),torch.ones_like(rr['features'][:,:1])),1)
            kwargs=merge_rows(kwargs,rr);out=raster(camera,n.args,n.static.active_sh_degree,kwargs,n.bg)
            env=n.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            out['rgb']=out['rgb']+(1-out['alpha'])*env
            return out
        operations=['retain_actors','delete_selected','move_selected','insert_reused','camera_shift','camera_revisit']
        for operation in operations:
            for sub in ['native_rgb','mask','float']:(dest/operation/sub).mkdir(parents=True,exist_ok=True)
        rows=[]
        for raw in range(89,138):
            original=n.camera(raw);native=render(original,'retain_actors')
            for operation in operations:
                camera=camera_shift(original,.5) if operation in ['camera_shift','camera_revisit'] else original
                out=render(camera,operation)
                exposure=(native['extra'][0]>.05)&(out['extra'][0]<.05) if camera is original else torch.zeros_like(out['alpha'][0],dtype=torch.bool)
                save(dest/operation/'native_rgb'/f'{raw:03d}.png',out['rgb'])
                Image.fromarray(cpu(exposure).astype('uint8')*255).save(dest/operation/'mask'/f'{raw:03d}.png')
                np.savez_compressed(dest/operation/'float'/f'{raw:03d}.npz',alpha=cpu(out['alpha']),depth_sum=cpu(out['depth_sum']),dynamic_weight=cpu(out['extra'][0]),road_weight=cpu(out['extra'][1]))
                rows.append(dict(raw=raw,operation=operation,exposed_pixels=int(exposure.sum())))
            if raw%10==0:print('OPERATIONS',raw,time.time()-started,flush=True)
        assert n.unchanged()
        summary=dict(status='complete',operations=operations,frames=49,rows=rows,dynamic_binding_counts=counts,
            owner_policy='7时刻真实轨迹框包含投票≥活跃时刻60%，唯一最高；未决行保留；仅为强工程基线',
            object_lateral_translation_m=1.,insert_lateral_translation_m=2.,camera_lateral_translation_m=.5,
            limitations=['已有修复状态只针对原先a37/a6三视图优化','未决动态行可能留下残影','同场景复用插入非外部资产泛化','相机变轨没有GT','尚未学习持续世界一致性'],
            master_files_unchanged=True,new_training_updates=0,seconds=time.time()-started,
            cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),quality_accepted=False,human_verdict=None)
        (dest/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
