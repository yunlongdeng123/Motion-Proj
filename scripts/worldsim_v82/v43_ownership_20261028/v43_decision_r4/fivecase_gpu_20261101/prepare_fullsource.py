"""原始/显露原生贡献建立完整静态来源候选；不读编辑评分。"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,pathlib,shutil,sys
from copy import deepcopy
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
CODE=pathlib.Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/v43_ownership_20261028/v43_decision_r4')
sys.path.insert(0,str(CODE))
from r4_core import canonical_hash,enumerate_routes,freeze_family,register_tasks
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
FOLDERS={'waymo-train-008_a37':'gpu','waymo-train-212_a0':'gpu_r2','waymo-train-212_a35':'gpu_r2',
         'waymo-train-564_a0':'gpu_564_replay','waymo-train-564_a22':'gpu_564_replay'}
def main():
    out=ROOT/'cpu_fullsource';out.mkdir(exist_ok=False)
    contract=dict(parent_run_contract=read(ROOT/'cpu/run_contract.json'),
        stage='complete-source candidate control after measured-neighbourhood recall limitation was observed',
        inputs='only saved original/target-source-peeled three-anchor native adjoints, fixed masks and trajectories',
        rule='union of per-anchor per-condition body_ratio>=.95 and total>1e-4; all original static rows considered',
        grouping='first qualifying source anchor, disjoint original row IDs, max3 groups,8 shared routes',
        no_edit_score_reference_or_student_read=True,not_formal_admission=True,
        interpretation='contribution candidates may include road/background; not physical owner labels',
        final_candidate_scope_control_this_run=True,human_verdict=None)
    # 保持原loader使用的顶层checkpoint_inputs合同。
    write(out/'run_contract.json',dict(read(ROOT/'cpu/run_contract.json'),fullsource_control=contract))
    summaries=[]
    for case,folder in FOLDERS.items():
        parent=ROOT/'cpu'/case;old=read(parent/'candidate_manifest.json');context=read(parent/'camera_track_context.json')
        n=old['static_row_count'];assigned=np.zeros(n,bool);parent_u=np.zeros(n,bool)
        for g in old['groups']:parent_u[g['original_static_rows']]=True
        anchors=[context['frames'][i] for i in [0,24,48]];groups=[];receipts=[]
        dst=out/case;dst.mkdir()
        mass=oldmass=0.
        for anchor in anchors:
            fid=anchor['frame_id'];original=ROOT/folder/case/'audit49'/(fid+'_row_alphaT.npz')
            peeled=ROOT/'gpu_exposed'/case/'source_peel49'/(fid+'_static_alphaT.npz')
            with np.load(original) as a,np.load(peeled) as b:
                ob=a['body_alphaT'][:n];ok=a['keep_alphaT'][:n];pb=b['body'];pk=b['keep']
                visible=(ob+ok>1e-4)&(ob>=.95*(ob+ok));exposed=(pb+pk>1e-4)&(pb>=.95*(pb+pk))
                selected=visible|exposed;ids=np.flatnonzero(selected&~assigned);assigned[ids]=True
                mass+=float(pb[exposed].sum());oldmass+=float(pb[exposed&parent_u].sum())
                receipts.append(dict(frame=fid,original_source=str(original),peeled_source=str(peeled),
                    all_qualifying_rows=int(selected.sum()),new_disjoint_rows=len(ids),
                    measured_neighbourhood_rows_in_source_pool=int((selected&parent_u).sum())))
                np.savez_compressed(dst/(fid+'_source_selection.npz'),original_static_rows=ids,
                    original_body_alphaT=ob[ids],original_keep_alphaT=ok[ids],peeled_body_alphaT=pb[ids],peeled_keep_alphaT=pk[ids])
            if len(ids):
                groups.append(dict(group_id='g%02d_%s'%(len(groups),fid),original_binding='static',rebind_target='actor_0',
                    anchor_frame_id=fid,anchor_model_time=anchor['model_time'],original_static_rows=ids.tolist(),
                    anchor_actor_center_world=anchor['actor_center_world'],anchor_actor_rotation_world=anchor['actor_rotation_world'],
                    evidence_ref=str(dst/(fid+'_source_selection.npz')),scope_rule='first anchor with original-or-peeled native body contribution'))
        candidate=deepcopy(old);candidate['groups']=groups
        candidate['scope_rule']=dict(kind='full original static row universe; three-anchor original-or-target-source-peel native contribution',
            radius_world_m=None,anchors=[a['frame_id'] for a in anchors],body_ratio=.95,total_alphaT_strict=1e-4,
            first_anchor_disjoint_grouping=True,selected_without_edit_scores=True,threshold_role='development candidate scope only',
            no_physical_identity_or_geometry_truth_claim=True)
        candidate['parent_neighbourhood_candidate_hash']=canonical_hash(old)
        candidate['limitations']=['Source contribution is not ownership; source pool includes possible road, shadows, mixed/large-footprint Gaussians',
            'Three coarse source-anchor groups are not row-level exhaustive routing; no global impossibility claim',
            'First/mid/last anchors only for all8routes; two fixed endpoints get full49frames',
            'Strict global FP32 adjoint mass limitations retained; all eligibility unknown']
        evidence=read(parent/'legal_preedit_evidence.json');frozen=read(parent/'tasks.json')[0]['frozen_scene_context'];routes=enumerate_routes(candidate)
        for name in ['camera_track_context.json','checkpoint_cpu_receipt.json','dynamic_binding.json']:shutil.copy2(parent/name,dst/name)
        for name,value in [('candidate_manifest.json',candidate),('shared_routes.json',routes),('frozen_H0.json',freeze_family(candidate)),
            ('legal_preedit_evidence.json',evidence),('tasks.json',register_tasks(case,candidate,evidence,routes,candidate['checkpoint_static_sha256'],frozen))]:write(dst/name,value)
        row=dict(case=case,parent_neighbourhood_rows=int(parent_u.sum()),full_source_rows=int(assigned.sum()),
            original_static_row_denominator=n,source_anchors=receipts,
            parent_U_fraction_of_high_body_contribution_peeled_mass=oldmass/mass if mass else None,
            candidate_rule_is_not_ownership_or_admission=True)
        summaries.append(row);print(json.dumps(row),flush=True)
    write(out/'source_recall_control.json',dict(rows=summaries,contract=contract))
if __name__=='__main__':main()
