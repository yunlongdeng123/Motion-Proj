"""GT对象扫掠 + 去动态LiDAR静态占据；稀疏空白绝不当作自由空间。"""
import argparse,json,pathlib,sys
import numpy as np
from shapely import contains_xy
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from geometry import transform,box_mask
from actor_command_audit import box,footprint,translation_sweep,audit

def supported_voxels(points,scan_ids,origin,voxel_size=.1):
    """重复回波不能冒充多次扫描证据；空输入维持未知。"""
    pts=np.asarray(points,float).reshape(-1,3);scans=np.asarray(scan_ids,int)
    if len(pts)==0:return np.empty((0,3)),np.empty(0,dtype=int)
    cells=np.floor((pts-origin)/voxel_size).astype(np.int32);uniq,inv=np.unique(cells,axis=0,return_inverse=True)
    pairs=np.unique(np.stack([inv,scans],axis=1),axis=0);support=np.bincount(pairs[:,0],minlength=len(uniq))
    sums=np.stack([np.bincount(inv,weights=pts[:,i],minlength=len(uniq)) for i in range(3)],axis=1)
    count=np.bincount(inv,minlength=len(uniq))
    return sums/count[:,None],support

ROOT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-PIPELINE-R2-20260926/r1')
OLD=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,aid,n,ref,length in [('scene_0230','22',50,5,1.),('scene_0255','25',100,20,2.)]:
        data=pathlib.Path('/root/autodl-tmp/data/v76_vadgs')/name;inst=json.loads((data/'instances/instances_info.json').read_text())
        pose,size=box(inst[aid],ref);origin=pose[:3,3].copy();rows=[];scanids=[]
        for f in range(n):
            xyz=np.fromfile(data/'lidar'/f'{f:03d}.bin',np.float32).reshape(-1,4)[:,:3]
            xyz=transform(xyz,np.loadtxt(data/'lidar_pose'/f'{f:03d}.txt'))
            xyz=xyz[(np.linalg.norm(xyz[:,:2]-origin[:2],axis=1)<12)&(np.abs(xyz[:,2]-origin[2])<3)]
            keep=np.ones(len(xyz),bool)
            for row in inst.values():
                b=box(row,f)
                if b is None:continue
                op,os=b
                if np.linalg.norm(op[:2,3]-origin[:2])<18:keep &= ~box_mask(xyz,op,os+.3)
            rows.append(xyz[keep]);scanids.append(np.full(keep.sum(),f))
        pts=np.concatenate(rows);scans=np.concatenate(scanids)
        # Each occupied 10cm voxel stores independent scan support, not raw density.
        vox,support=supported_voxels(pts,scans,origin)
        np.savez_compressed(ROOT/f'{name}_static_voxels.npz',points_world=vox,scan_support=support,origin_world=origin,voxel_size_m=.1)
        olddelta=np.array(json.loads((OLD/name/'actor/placement.json').read_text())['move_delta_world_m'])
        forward=pose[:3,0].copy();forward[2]=0;forward/=np.linalg.norm(forward)
        candidates=[]
        for key,delta in [('old_move',olddelta),(f'forward_{length:g}m',forward*length)]:
            obj=audit(inst,aid,n,delta);frame_rows=[]
            for f in range(n):
                p,s=box(inst[aid],f);source=footprint(p,s);dest=footprint(p,s,delta);sweep=translation_sweep(source,dest)
                bottom=p[2,3]-s[2]/2;top=p[2,3]+s[2]/2
                inxy=contains_xy(sweep,vox[:,0],vox[:,1]);novel=inxy & ~contains_xy(source,vox[:,0],vox[:,1])
                # Points >15 cm over GT bottom are obstacle evidence. Lower curbs,
                # unknown ground, semantic use, and motion feasibility remain open.
                hit=novel & (vox[:,2]>bottom+.15)&(vox[:,2]<top+.15)&(support>=3)
                frame_rows.append({'frame':f,'static_supported_hit_voxels':int(hit.sum()),'static_hit_max_scan_support':int(support[hit].max()) if hit.any() else 0,'hit_points_world':vox[hit].tolist()})
            static_hits=sum(r['static_supported_hit_voxels']>=3 for r in frame_rows)
            gate='rejected_actor_box' if obj['swept_collision_frames'] else 'static_obstacle_evidence' if static_hits else 'unverified_sparse_static_and_map'
            candidates.append({'command':key,'delta_world_m':delta.tolist(),'frames_checked':n,'actor_swept_collision_frames':obj['swept_collision_frames'],'actor_hit_ids':obj['swept_hit_ids'],'minimum_actor_clearance_m':obj['minimum_swept_clearance_m'],'static_evidence_frames':static_hits,'maximum_static_hit_voxels':max(r['static_supported_hit_voxels'] for r in frame_rows),'gate_status':gate,'full_legality_approved':False,'frames':frame_rows})
        result={'scene':name,'actor_id':aid,'background_lidar_points':len(pts),'static_voxels':len(vox),'minimum_scan_support':3,'height_above_gt_bottom_m':.15,'candidate_commands':candidates,'static_scope':'原时间窗GT框外累计LiDAR；正占据可用于拒绝，空白不能证明自由空间；低路缘、地图与行驶运动学未验证','human_verdict':None}
        (ROOT/f'{name}_space_gate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        compact={k:v for k,v in result.items() if k!='candidate_commands'};compact['candidate_commands']=[{k:v for k,v in c.items() if k!='frames'} for c in candidates];results.append(compact)
        print(json.dumps(compact,ensure_ascii=False),flush=True)
    (ROOT/'space_gate_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')

if __name__=="__main__":
    main()
