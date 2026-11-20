"""New two-arm all-label BUILD export; old outputs and sources never change."""
from pathlib import Path
import argparse,copy,json,subprocess,time,traceback
import numpy as np
from export_build_public_1920 import inspect,execute,B,F0ROOT
from export_public_endpoints import read,write,file_sha

ROOT=B/'all_teacher_build_20261120_r1';ARMS=('G_BUILD_FIELD','G_BUILD_NUMERIC');OWNER='native_capacity_BUILD_public_r1'

def maybe(p):
    try:return read(p)
    except(FileNotFoundError,json.JSONDecodeError):return {}

def bind_F0(new_initial,out):
    old=B/'formal_r1/common_initial_ci.npz';keys=('ci0','active_world_rows','eligible_active')
    with np.load(old,allow_pickle=False)as a,np.load(new_initial,allow_pickle=False)as b:
        exact={k:bool(a[k].dtype==b[k].dtype and np.array_equal(a[k],b[k]))for k in keys}
        difference=float(np.max(abs(a['ci0']-b['ci0'])))
    if not all(exact.values()):raise RuntimeError('BUILD initialci differs from oldF0: '+str(difference))
    evidence=out/'F0_initial_exact.json'
    write(evidence,dict(schema='all_teacher_BUILD_F0_reuse.v1',complete=True,CPU_only=True,new_native_calls=0,
        source_initial=str(old),source_initial_SHA256=file_sha(old),new_initial=str(new_initial),new_initial_SHA256=file_sha(new_initial),
        arrays_exact=exact,ci0_RGB_max=difference,old_F0_manifest=str(F0ROOT/'packets_manifest.json'),
        old_F0_manifest_SHA256=file_sha(F0ROOT/'packets_manifest.json'),training_role='same_initial_no_new_fit',held_generalization_claim=False))
    rows=[]
    for row in read(F0ROOT/'packets_manifest.json')['rows']:
        if row['arm']!='F0':continue
        r=copy.deepcopy(row);r.update(evaluation_role='BUILD_fit_all_qualified'if r['raw']in(86,110)else'diagnostic_only',
            held_generalization_claim=False,reused_old_F0=True,F0_initial_exact_receipt=str(evidence),F0_initial_exact_receipt_SHA256=file_sha(evidence))
        if file_sha(r['path'])!=r['SHA256']:raise RuntimeError('old F0 packet changed')
        rows.append(r)
    return evidence,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--new-initial-ci',default=str(ROOT/'common/initial_ci.npz'));a=p.parse_args()
    phases={'G_BUILD_FIELD':ROOT/'field/formal_r1','G_BUILD_NUMERIC':ROOT/'numerical/formal_r1'}
    deadline=time.monotonic()+7200;print('WAIT_BUILD_FORMALS',flush=True)
    while True:
        ready=all(maybe(d/'receipt.json').get('completed')is True for d in phases.values())and maybe(ROOT/'gpu_lease.json').get('active')is False
        if ready:
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
            if not apps:break
            print('WAIT_COMPUTE_PROCESS',apps,flush=True)
        if time.monotonic()>deadline:raise TimeoutError('BUILD formals not ready, no GPU work launched')
        time.sleep(10)
    out=ROOT/'public1920/export_r1'
    if out.exists():raise FileExistsError('preserve old BUILD export')
    out.mkdir(parents=True);F0evidence,rows=bind_F0(Path(a.new_initial_ci),out)
    plans=[]
    for arm in ARMS:
        d=phases[arm];plans.append(inspect(d/'asset',d/'oracle_colors.pt',d/'receipt.json',arm,d/'field.pt'if arm=='G_BUILD_FIELD'else None))
    auth=out/'authorization.json';lease=out/'gpu_lease.json'
    write(auth,dict(schema='all_teacher_BUILD_public_authorization.v1',GPU_authorized=True,optimizer_updates=0,max_native_forwards=80,
        per_asset_nativeF_cap=40,expected_total_nativeF=40,GPU_keep_on=True,
        root_authorization='Autonomous newall-labelBUILD two endpoints after their formal leases release; no repeated approval',
        role='BUILD_fit_all_qualified',held_generalization_claim=False,F0_initial_exact_receipt=str(F0evidence),plans=plans,
        previous_BUILD_lease=str(ROOT/'gpu_lease.json'),previous_BUILD_lease_inactive=True,previous_BUILD_lease_SHA256=file_sha(ROOT/'gpu_lease.json'),
        source=str(Path(__file__)),source_SHA256=file_sha(__file__)))
    write(lease,dict(gpu_authorized=True,active=True,owner=OWNER,authorization=str(auth),phase='all_teacher_BUILD_public1920',formal_allowed=False))
    start=time.monotonic();receipts=[];visibility=[]
    try:
        print('BUILD_PUBLIC_STARTED',time.time(),flush=True)
        for index,plan in enumerate(plans):
            folder=out/plan['arm'];execute(plan,folder,lease,OWNER,auth,release_lease=False)
            r=read(folder/'receipt.json');receipts.append(dict(arm=plan['arm'],path=str(folder/'receipt.json'),SHA256=file_sha(folder/'receipt.json'),
                native_forward_calls=r['native_forward_calls'],all_K_exact_zero=r['all_K_exact_zero']))
            rows.extend(read(folder/'packets_manifest.json')['rows']);visibility.extend(read(folder/'visibility_manifest_r1.json')['rows'])
            if sum(r['native_forward_calls']for r in receipts)>80:raise RuntimeError('aggregate BUILD80F cap exceeded')
        old_visibility=read(F0ROOT/'visibility_manifest_r1.json')
        visibility.extend(dict(v,evaluation_role='BUILD_fit_all_qualified'if v['raw']in(86,110)else'diagnostic_only',held_generalization_claim=False)for v in old_visibility['rows']if v['arm']=='F0')
        write(out/'packets_manifest.json',dict(schema='native_capacity_public_packets.v1',rows=rows,training_role='BUILD_fit_all_qualified',held_generalization_claim=False))
        write(out/'visibility_manifest_r1.json',dict(schema='native_capacity_visibility_sidecars.v1',complete=True,CPU_only=True,new_native_calls=0,rows=visibility))
        write(out/'receipt.json',dict(schema='all_teacher_BUILD_twoarm_public.v1',complete=True,optimizer_updates=0,
            native_forward_calls=sum(r['native_forward_calls']for r in receipts),subphase_receipts=receipts,F0_nativeF=0,F0_reuse=str(F0evidence),
            role='BUILD_fit_all_qualified',old_held_now_fitting_labels=True,held_generalization_claim=False,
            seconds=time.monotonic()-start,GPU_keep_on=True,production_assets_modified=False,human_verdict=None))
        print('BUILD_PUBLIC_COMPLETED',str(out/'receipt.json'),flush=True)
    except Exception:
        write(out/'failed.json',dict(error=traceback.format_exc(),subphase_receipts=receipts,seconds=time.monotonic()-start));raise
    finally:
        s=read(lease);s.update(active=False,owner=None,ended_at=time.time());write(lease,s)
if __name__=='__main__':main()
