"""P1.5 自适应解释：区分等深并列身份变化与实际深度退化，不新增模型。"""
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from audit_worldsim_v74_h2_p15 import load,surface,write,stats,interpolate,describe
from motion_proj.worldsim_v74_h2.first_event_trace import trace

out=Path(sys.argv[1]);source=Path(json.loads((out/'manifest.json').read_text())['source']);teacher=source/'20260912__fit-teacher-r3'/'fit'
trans=json.loads((out/'transitions.json').read_text());ips=json.loads((out/'interpolations.json').read_text());refs=json.loads((out/'references.json').read_text());cases=json.loads((out/'case_counts.json').read_text())
details=[];scan_details=[];tie_checks=[]
for r in trans:
    model,name,step,ray=r['model'],r['case'],r['step'],r['ray'];folder=source/f'20260912__{model}-dagger12-r1'/name
    prev=surface(folder/f'step-{step-1}.npz');nxt=surface(folder/f'step-{step}.npz');obs=load(teacher/name/'supervision.npz');patch=r['winner_before'];d=obs['directions_actor'][ray]
    delta=nxt.center[patch]-prev.center[patch];normal=prev.rotation[patch,:,2];normal_mm=np.dot(delta,normal)*1000
    angle=Rotation.from_matrix(nxt.rotation[patch]@prev.rotation[patch].T).magnitude()*180/np.pi
    actual=r['after_m']-r['before_m'];pred=np.dot(delta,normal)/np.dot(normal,d)
    info=dict(model=model,case=name,step=step,ray=ray,kind=r['kind'],center_mm=float(np.linalg.norm(delta)*1000),normal_translation_mm=float(normal_mm),rotation_deg=float(angle),
              actual_first_depth_delta_mm=actual*1000,fixed_normal_plane_prediction_mm=float(pred*1000),plane_prediction_error_mm=float(abs(actual-pred)*1000),
              previous_threshold_slack_mm=r['hit_threshold_slack_m']*1000,previous_residual_mm=(r['before_m']-r['observed_m'])*1000,next_residual_mm=(r['after_m']-r['observed_m'])*1000)
    if r['kind']=='same_owner_depth':
        center_only=prev.copy();center_only.center=nxt.center.copy();qcenter=trace(center_only,obs)['first'][ray]
        shape_only=prev.copy();shape_only.rotation=nxt.rotation.copy();shape_only.radius=nxt.radius.copy();qshape=trace(shape_only,obs)['first'][ray]
        info.update(center_only_early=bool(qcenter<r['observed_m']-.2),shape_only_early=bool(qshape<r['observed_m']-.2),center_only_first_m=float(qcenter),shape_only_first_m=float(qshape))
    details.append(info)

for ref in refs:
    model,name,step=ref['model'],ref['case'],ref['step'];dest=out/model/name/f'step-{step}';obs=load(teacher/name/'supervision.npz')
    teach=surface(dest/'continuous_reference.npz');end=surface(dest/'learner.npz');tg=describe(teach,obs,trace(teach,obs));z=load(dest/'teacher_learner_scan.npz')
    for ip in [x for x in ips if (x['model'],x['case'],x['step'])==(model,name,step)]:
        ray=ip['ray'];gap=tg['second'][ray]-tg['first'][ray];tol=1e-10*max(1,abs(tg['first'][ray]));own=ip['owner'];early=ip['early']
        if own:
            c=own['cpu_bracket'];tie_checks.append(dict(model=model,case=name,step=step,ray=ray,alpha=own['alpha_hi'],reference_gap_m=float(gap),reference_cofirst=bool(gap<=tol),
                                      owner_before=c[0]['winner'],owner_after=c[1]['winner'],label_before=c[0]['label'],label_after=c[1]['label'],depth_jump_bracket_m=c[1]['first_m']-c[0]['first_m']))
        ix=np.searchsorted(z['alpha'],early['alpha_hi']);ix=min(ix,len(z['alpha'])-1)
        # 首次EARLY区间两端已由CPU硬三角形查询保存；无需把grid身份变化自动当作不连续。
        c=early['cpu_bracket'];before=interpolate(teach,end,[early['alpha_lo']])[0];after=interpolate(teach,end,[early['alpha_hi']])[0]
        bi=describe(before,obs,trace(before,obs));ai=describe(after,obs,trace(after,obs));new=c[1]['winner'];old=c[0]['winner']
        scan_details.append(dict(model=model,case=name,step=step,ray=ray,early_alpha=early['alpha_hi'],winner_changes_at_early=old!=new,
                                  boundary_before_m=float(bi['boundary'][ray,old]),boundary_after_m=float(ai['boundary'][ray,new]),incidence_before=float(bi['incidence'][ray,old]),
                                  depth_bracket_delta_m=c[1]['first_m']-c[0]['first_m'],max_outer_vertex_mm=early['displacement']['max_outer_vertex_mm'],
                                  winner_center_mm=early['displacement']['winner_center_mm'],winner_outer_vertex_mm=early['displacement']['winner_outer_vertex_mm']))

agg={}
for model in ['A','C2']:
    rr=[r for r in details if r['model']==model and r['case'].startswith('grazing_thin')];ss=[r for r in scan_details if r['model']==model and r['case'].startswith('grazing_thin')];tt=[r for r in tie_checks if r['model']==model and r['case'].startswith('grazing_thin')]
    agg[model]=dict(thin_failure_rays=len(rr),rotations_deg=stats([r['rotation_deg'] for r in rr]),normal_translation_mm=stats([r['normal_translation_mm'] for r in rr]),center_mm=stats([r['center_mm'] for r in rr]),
                   depth_delta_mm=stats([r['actual_first_depth_delta_mm'] for r in rr]),plane_prediction_error_mm=stats([r['plane_prediction_error_mm'] for r in rr]),
                   center_only_early=sum(r.get('center_only_early',False) for r in rr),shape_only_early=sum(r.get('shape_only_early',False) for r in rr),previous_threshold_slack_mm=stats([r['previous_threshold_slack_mm'] for r in rr]),
                   teacher_to_learner_early_alpha=stats([r['early_alpha'] for r in ss]),ownership_change_at_early=sum(r['winner_changes_at_early'] for r in ss),
                   vertex_mm_at_early=stats([r['max_outer_vertex_mm'] for r in ss]),winner_center_mm_at_early=stats([r['winner_center_mm'] for r in ss]),winner_vertex_mm_at_early=stats([r['winner_outer_vertex_mm'] for r in ss]),
                   interpolated_owner_changes=len(tt),cofirst_start_owner_changes=sum(r['reference_cofirst'] for r in tt),owner_changes_that_already_make_early=sum(r['label_after']==1 for r in tt),
                   first_owner_alpha=stats([r['alpha'] for r in tt]))
write(out/'depth_decomposition.json',details);write(out/'early_crossing_details.json',scan_details);write(out/'cofirst_diagnostics.json',tie_checks);write(out/'mechanism_summary.json',agg)
print(json.dumps(agg,ensure_ascii=False,indent=2))
