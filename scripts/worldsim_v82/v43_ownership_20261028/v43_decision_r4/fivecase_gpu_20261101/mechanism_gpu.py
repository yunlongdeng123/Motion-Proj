"""真实五例：实际执行的轨迹/身份接口检查和分量渲染；不修复模型。"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse,json,pathlib,time
import numpy as np
import torch
from run_five import FiveScene,ROOT,old,read,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);a=p.parse_args()
    out=ROOT/'mechanisms_gpu'/a.case;out.mkdir(parents=True,exist_ok=False)
    n=FiveScene(ROOT/'cpu_fullsource'/a.case,out/'restore');S=n.candidate['static_row_count']
    observations=out/'observed';observations.mkdir()
    for fid,cam in n.cameras.items():old.save_rgb(observations/(fid+'.png'),cam.original_image[:3].detach().cpu().numpy())
    probes=[];components=[];rng=np.random.default_rng(20261101)
    with torch.no_grad():
        for fid in n.anchors:
            cam,s,d,e=n.frame_rows(fid)
            target=np.arange(len(d['means3D'])) if n.target_mode=='single_native_actor' else np.asarray(n.target_dynamic_ids or [],np.int64) if n.target_mode=='multi_actor_legal_proxy' else np.empty(0,np.int64)
            group_for={};sample=set()
            for g in n.candidate['groups']:
                rows=g['original_static_rows']
                for j in np.asarray(rows)[np.linspace(0,len(rows)-1,min(12,len(rows))).astype(int)]:sample.add(int(j));group_for[int(j)]=g
            if len(target):sample.update((S+target[np.linspace(0,len(target)-1,min(12,len(target))).astype(int)]).tolist())
            sample.update(np.linspace(0,S-1,12).astype(int).tolist())
            if len(d['means3D']):sample.update((S+np.linspace(0,len(d['means3D'])-1,12).astype(int)).tolist())
            # 还需识别非采样候选行，避免把另一个组的点误当固定上下文。
            for g in n.candidate['groups']:
                selected=sample.intersection(g['original_static_rows'])
                for j in selected:group_for[j]=g
            original=n.helpers.merge_rows(s,d);base=original['means3D'].cpu().numpy().astype(np.float64)
            target_set=set(target.tolist());cur=n.track[fid];C=np.asarray(cur['actor_center_world']);R=np.asarray(cur['actor_rotation_world'])
            for hid in ['h000',n.allrebind_id]:
                route=n.routes[int(hid[1:])];bound,moving,_=n.apply_shared_route(route,fid,s,d)
                for q in old.OPERATIONS:
                    actual=n.execute_query(fid,bound,moving,q)
                    ids=actual['_original_row'].cpu().numpy();positions=actual['means3D'].cpu().numpy()
                    error=0.;compared=omitted=0
                    for uid in sorted(sample):
                        assigned=(uid>=S and uid-S in target_set) or (hid==n.allrebind_id and uid in group_for)
                        # n=1时h000与allrebind相同，但完整来源域每例均有非空三组。
                        expected=base[uid].copy()
                        if hid==n.allrebind_id and uid in group_for:
                            g=group_for[uid];A=np.asarray(g['anchor_actor_center_world']);Ar=np.asarray(g['anchor_actor_rotation_world'])
                            M=n.linear@(R@Ar.T)@n.inverse_linear
                            am=n.linear@A+n.transform[:3,3];cm=n.linear@C+n.transform[:3,3]
                            expected=M@(expected-am)+cm
                        index=np.searchsorted(ids,uid)
                        if assigned and q=='delete':
                            if index<len(ids) and ids[index]==uid:raise ValueError('删除没有移除登记目标行')
                            omitted+=1;continue
                        if assigned and q=='shift':expected+=n.linear@np.asarray([0,1.5,0])
                        if assigned and q=='rotate':
                            z=np.pi/6;Z=np.asarray([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
                            M=n.linear@Z@n.inverse_linear;cm=n.linear@C+n.transform[:3,3];expected=M@(expected-cm)+cm
                        if index==len(ids) or ids[index]!=uid:raise ValueError('上下文或目标行丢失')
                        error=max(error,float(abs(positions[index]-expected).max()));compared+=1
                    probes.append(dict(frame=fid,route=hid,operation=q,compared_sample_rows=compared,expected_deleted_samples=omitted,
                        model_coordinate_max_abs=error,tolerance_model_units=1e-4,technical_transform_pass=bool(error<=1e-4),
                        identity_correctness='not established; matching the registered transform does not prove the binding target is correct'))
            source_ids=sorted({j for g in n.candidate['groups'] for j in g['original_static_rows']})
            for kind,values in [('static_source_candidates_only',old.take_rows(s,torch.as_tensor(source_ids,device='cuda'))),
                                ('native_target_dynamic_only',old.take_rows(d,torch.as_tensor(target,device='cuda')))]:
                if not len(values['means3D']):components.append(dict(frame=fid,component=kind,status='empty_registered_population',rendered=False));continue
                arrays=n.raster(fid,values,torch.zeros_like(e));old.save_arrays(out/kind,fid,arrays)
                components.append(dict(frame=fid,component=kind,status='native_rendered',rows=len(values['means3D']),
                    role='source/associated-component display, not correct-vehicle-owner or intrinsic-color GT'))
    write(out/'technical_and_component_checks.json',dict(case=a.case,technical_motion_rows=probes,
        all_technical_transform_checks_passed=all(r['technical_transform_pass'] for r in probes),
        registered_target_identity_is_correct=None,core_identity_binding_is_in_scope=True,component_renders=components,
        native_restore=read(out/'restore/native_restore_receipt.json'),original_parameters_unchanged=n.unchanged(),training_updates=0))
if __name__=='__main__':main()
