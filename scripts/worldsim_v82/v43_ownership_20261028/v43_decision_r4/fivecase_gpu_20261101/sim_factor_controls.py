"""独立E1机制校验：真实原生光栅中的单因素副本，不修改主五例或训练标签。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import pathlib,sys,json
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parent
CODE=pathlib.Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/v43_ownership_20261028/v43_decision_r4');sys.path.insert(0,str(CODE))
import native_shared_route as old
def main():
    out=ROOT/'sim_factor_controls';out.mkdir(exist_ok=False)
    old.write(out/'pre_control_contract.json',dict(role='E1 known-factor controls on existing procedural-world frozen native representation',
        factors=['pure_binding_wrong','pure_geometry_offset','pure_color_DC','environment_only','timestamp_miswire','missing_front_population'],
        changes='one declared factor per copy; original bundle immutable; not primary H0 methods or repairs',
        coverage_limits=['no physically verified shadow/reflection injection','no independently certified multi-surface single Gaussian','no complete unknown-background factor control'],
        geometry_offset_world_m=[.6,0,.25],color_DC_delta_RGB=[.2,-.1,0],environment_delta=.25,timestamp_delta_model=.1,
        missing_front='remove native target rows above median actor-forward coordinate at fixed middle time; capacity control, not complete-car GT',human_verdict=None))
    cpu=ROOT.parent/'v43_decision_r4_cpu_20261031/run_r2_maskfix/sim_m0_original10k'
    n=old.NativeScene(cpu,out/'restore');fid=n.anchors[1];records=[]
    with torch.no_grad():
        cam,s,d,env=n.frame_rows(fid);base={}
        for q in old.OPERATIONS:
            arrays=n.raster(fid,n.execute_query(fid,s,d,q),env);old.save_arrays(out/'baseline'/q,fid,arrays);base[q]=arrays
        for factor in ['pure_binding_wrong','pure_geometry_offset','pure_color_DC','environment_only','timestamp_miswire','missing_front_population']:
            ss=s;dd=dict(d);ee=env;facts={}
            if factor=='pure_geometry_offset':
                dd['means3D']=d['means3D']+torch.as_tensor(n.linear@np.asarray([.6,0,.25]),device='cuda',dtype=torch.float32);dd=n.update_depth(dd,cam);facts['changed_fields']=['dynamic means3D and derived depth features']
            elif factor=='pure_color_DC':
                dd['shs']=d['shs'].clone();dd['shs'][:,0,:]+=torch.as_tensor([.2,-.1,0],device='cuda')/.28209479177387814;facts['changed_fields']=['dynamic SH DC only']
            elif factor=='environment_only':ee=env+.25;facts['changed_fields']=['environment radiance only']
            elif factor=='timestamp_miswire':
                previous=cam.timestamp;cam.timestamp=float(previous)+.1
                _,ss,dd,ee=n.frame_rows(fid);cam.timestamp=previous;facts['changed_fields']=['query timestamp only; restored before next control'];facts['registered_time_mismatch']=True
            elif factor=='missing_front_population':
                center=np.asarray(n.track[fid]['actor_center_world']);rotation=np.asarray(n.track[fid]['actor_rotation_world']);inv=np.linalg.inv(n.transform)
                world=d['means3D'].cpu().numpy()@inv[:3,:3].T+inv[:3,3];forward=((world-center)@rotation)[:,0]
                keep=torch.as_tensor(forward<=np.median(forward),device='cuda');dd=old.take_rows(d,keep);facts.update(changed_fields=['target row population only'],removed_rows=int((~keep).sum()))
            else:facts['changed_fields']=['binding only: prepared native target rows treated as fixed context for all q']
            same_count=len(dd['means3D'])==len(d['means3D'])
            geometry_delta=float(abs(dd['means3D']-d['means3D']).max()) if same_count else None
            color_delta=float(abs(dd['shs']-d['shs']).max()) if same_count else None
            confounds=[]
            if not same_count:confounds.append('representation_population_changed')
            if geometry_delta is not None and geometry_delta>0:confounds.append('geometry_changed')
            if color_delta is not None and color_delta>0:confounds.append('appearance_parameters_changed')
            if float(abs(ee-env).max())>0:confounds.append('environment_changed')
            if facts.get('registered_time_mismatch'):confounds.append('execution_time_invalid')
            for q in old.OPERATIONS:
                values=n.helpers.merge_rows(ss,dd) if factor=='pure_binding_wrong' else n.execute_query(fid,ss,dd,q)
                arrays=n.raster(fid,values,ee);old.save_arrays(out/factor/q,fid,arrays)
                delta={k:float(abs(arrays[k]-base[q][k]).max()) for k in ['C_G','alpha','E','fill','final','depth_sum_model']}
                records.append(dict(factor=factor,operation=q,field_max_abs_delta=delta,known_control=facts,
                    measured_copy_parameter_deltas=dict(row_count_difference=len(dd['means3D'])-len(d['means3D']),geometry_max_abs=geometry_delta,SH_max_abs=color_delta),
                    naive_RGB_difference_would_flag=bool(delta['final']>1/255),
                    nuisance_gate_exclusions=confounds,
                    may_generate_ownership_label=False,interpretation='known controlled source factor; not learned automatic cause identification'))
        checks=[]
        for r in records:
            f=r['factor'];v=r['field_max_abs_delta'];passed=None
            if f=='pure_color_DC':passed=v['alpha']==0 and v['depth_sum_model']==0 and v['E']==0
            elif f=='environment_only':passed=v['C_G']==0 and v['alpha']==0 and v['depth_sum_model']==0
            elif f=='pure_binding_wrong' and r['operation']=='none':passed=max(v.values())<=2e-6
            if passed is not None:checks.append(dict(factor=f,operation=r['operation'],passed=bool(passed)))
        old.write(out/'summary.json',dict(scene_ancestor='procedural_world_dataset_r2_vehicle0',frame=fid,
            native_control_outputs=28,records=records,invariance_checks=checks,all_invariance_checks_passed=all(x['passed'] for x in checks),
            classification_accuracy=None,controlled_validation_is_partial=True,unvalidated_categories=[6,7,8],
            category11_limit='known population removal tests capacity confounding, not complete physical front-surface validation',
            no_independent_edited_reference_scored=True,no_training_labels_generated=True,original_unchanged=n.unchanged()))
if __name__=='__main__':main()
