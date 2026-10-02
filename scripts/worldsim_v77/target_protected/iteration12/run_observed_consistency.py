"""r36：同两个已过输入QA的case，单一合法轮廓反证控制；禁止网格补救。"""
from pathlib import Path
import os,sys,json,time,fcntl,traceback
sys.path.insert(0,str(Path(__file__).parent))
import prepare_instance_audit as p
from observed_consistency import consistent_actor_samples,checks
f=p.f;O=f.T/'r36'
BASES={'Q060':'r29','Q046':'r34'}


def prepare():
    assert not (O/'run.json').exists(),'已登记，不覆盖'
    O.mkdir(exist_ok=True)
    plan={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r36','phase':'observed_silhouette_consistency_control',
          'probe_cases':list(BASES),'reference_runs':BASES,'frames':30,'seed':42,'training_steps':0,
          'change':'remove cuboid samples contradicted in an unmasked, unoccluded legal input frame',
          'frozen_parameters':{'outside_same_SAM_halo_px':2,'hole_exclusion_halo_px':2,'front_depth_tolerance_m':.15,'allowed_contradictions':0},
          'method_input':'only saved final-H RGB, observed SAM, GT trajectory/camera, existing window LiDAR; Y QA separately',
          'output_is_condition_not_generated_video':True,'surfel':False,'failure_ledger_refs':['V77-F02'],'human_verdict':None,
          'predeclared_promote_gate':'both cases precision does not drop; wrong-edge neighbor precision improves; each case retains >=90% previous hidden-B coverage; independent QA still required',
          'stop':'single fixed control; if precision/coverage tradeoff fails, retain original and do not tune halo/vote grid',
          'primary_reference':'https://homes.cs.washington.edu/~seitz/papers/kutu-ijcv00.pdf',
          'reference_boundary':'borrows observed-view consistency; not a reimplementation or guarantee for dynamic noisy SAM'}
    f.dump(O/'run.json',plan);jobs=[]
    for cid,base in BASES.items():
        src=f.T/base
        (O/'observed').mkdir(exist_ok=True);(O/'observed'/cid).symlink_to(src/'observed'/cid,target_is_directory=True)
        (O/'observed_masks').mkdir(exist_ok=True)
        for job in f.read(src/'observed_queue.json')['jobs']:
            if job['case_id']!=cid:continue
            jobs.append(job);(O/'observed_masks'/job['job_id']).symlink_to(src/'observed_masks'/job['job_id'],target_is_directory=True)
    f.dump(O/'observed_queue.json',{'jobs':jobs,'source':'unchanged previously built final-H masked input','human_verdict':None})
    f.dump(O/'segmentation_state.json',{'stage':'complete_pending_identity_review','reused':True})


def main():
    if not (O/'run.json').exists():prepare()
    lock=open(O/'controller.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'stage':'consistency_condition_build','pid':os.getpid(),'started':time.time(),'training_steps':0}
    f.dump(O/'controller_state.json',state)
    try:
        checks();f.legacy.O=f.O;f.legacy.ROOT=f.ROOT
        import build_state_probe as b
        b.main(run_root=O,source_root=f.ROOT,geometry_factory=f.legacy.geometry,actor_filter=consistent_actor_samples)
        state.update(stage='complete_pending_paired_evaluation',seconds=time.time()-state['started']);f.dump(O/'controller_state.json',state)
    except Exception as error:
        state.update(stage='engineering_error',error=repr(error),traceback=traceback.format_exc());f.dump(O/'controller_state.json',state);raise


if __name__=='__main__':main()
