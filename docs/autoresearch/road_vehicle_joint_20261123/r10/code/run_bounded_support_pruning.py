"""Root-authorized, serial support pruning for at most three new children.

Each changed visibility is measured afresh. No colours/centres/scales/opacities
or inherited assets are optimized. Every subprocess command and exit is saved.
"""
from pathlib import Path
import argparse,subprocess,time
from local_gaussian_asset import read,write,sha
from render_new_local_zero import pin


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--r10',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args();R=a.r10;code=R/'code'
    cpu='/root/autodl-tmp/envs/motionproj/bin/python';gpu='/root/autodl-tmp/envs/drivingeditor/bin/python'
    assert a.execute,'This controller requires the explicit root-authorized execution flag'
    assert not a.output.exists();a.output.mkdir(parents=True);started=time.time();stages=[];error=None;final=None
    cfg=R/'local_measure_contract_v2.json';original=R/'timewindow17_expanded_zero_v1';firstmask=R/'timewindow17_footprint_masks_v1'
    def run(tag,python,script,args,receipt=None):
        command=[python,str(code/script),*map(str,args)];log=a.output/(tag+'.log');rec=dict(tag=tag,command=command,started_unix=time.time(),log=str(log))
        stages.append(rec);write(a.output/'running.json',dict(complete=False,stages=stages))
        with log.open('w') as out:result=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT)
        rec.update(exit_code=result.returncode,finished_unix=time.time())
        if receipt and receipt.is_file():rec['actual_receipt']=pin(receipt)
        print(dict(stage=tag,exit_code=result.returncode,actual_receipt=rec.get('actual_receipt')),flush=True)
        assert result.returncode==0,tag+' failed; preserve its log and receipt'
    def fresh(v):
        asset=R/f'timewindow17_asset_support_child_v{v}';zero=R/f'timewindow17_pruned_zero_v{v}'
        mask=R/f'timewindow17_pruned_footprint_masks_v{v}';measure=R/f'timewindow17_pruned_footprint_measure_v{v}'
        args=['--asset',asset,'--measurement-contract',cfg,'--zero',zero,'--surface-review',R/'timewindow17_requalified_surface_review_v2_replay1',
            '--engineering-contract',R/'engineering_contract_v3.json','--masks',mask,'--output',measure]
        run(f'v{v}_fresh_preflight',cpu,'measure_local_edit_footprint.py',args,mask/'receipt.json')
        run(f'v{v}_fresh_actual',gpu,'measure_local_edit_footprint.py',[*args,'--execute'],measure/'receipt.json')
        actual=read(measure/'receipt.json');assert actual['complete'] and actual['nativeF']==actual['nativeB']==5
        return measure
    try:
        # The first fresh measurement is a separately launched/root-recorded
        # actual job. Never wait for or start a duplicate here.
        first=read(R/'timewindow17_pruned_footprint_measure_v1/receipt.json');assert first['complete'] and first['nativeF']==first['nativeB']==5
        for v in [2,3,4]:
            previous=v-1;parent=R/f'timewindow17_asset_support_child_v{previous}'
            measure=R/f'timewindow17_pruned_footprint_measure_v{previous}';audit=R/f'timewindow17_pruned_local_footprint_audit_v{previous}'
            run(f'v{previous}_support_audit',cpu,'inspect_new_local_footprint_cpu.py',['--asset',parent,'--measurement',measure,'--output',audit],audit/'receipt.json')
            child=R/f'timewindow17_asset_support_child_v{v}';zero=R/f'timewindow17_pruned_zero_v{v}'
            run(f'v{v}_child_export',cpu,'export_local_support_child_cpu.py',['--asset',parent,'--audit',audit,'--output',child],child/'active_only_proof.json')
            args=['--asset',child,'--measurement-contract',cfg,'--baseline-measurement',R/'measure_v1','--output',zero,'--measure-row-guard']
            run(f'v{v}_zero_preflight',cpu,'render_new_local_zero.py',args)
            run(f'v{v}_zero_actual',gpu,'render_new_local_zero.py',[*args,'--execute'],zero/'receipt.json')
            review=R/f'timewindow17_pruned_zero_scope_review_v{v}'
            run(f'v{v}_fixed_scope_review',cpu,'review_pruned_zero_cpu.py',['--zero',zero,'--unpruned-zero',original,'--baseline-measurement',R/'measure_v1',
                '--masks',firstmask,'--measurement-contract',cfg,'--output',review],review/'receipt.json')
            decision=read(review/'receipt.json');print(dict(child=v,decision=decision['decision']),flush=True)
            if decision['automatic_fresh_measurement_allowed']:
                final_measure=fresh(v);final=dict(asset=str(child),zero=str(zero),measurement=str(final_measure),review=pin(review/'receipt.json'))
                break
            if v<4:fresh(v)
        if final is None:final=dict(status='STOP_AFTER_THREE_NEW_CHILDREN',quality_accepted=False,scope_accepted=False)
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        actual=[]
        for s in stages:
            if 'actual_receipt' in s:
                r=read(s['actual_receipt']['path'])
                if 'nativeF' in r:actual.append(dict(stage=s['tag'],nativeF=r['nativeF'],nativeB=r.get('nativeB',0)))
        write(a.output/'receipt.json',dict(schema='R10.bounded_active_only_support_pruning_controller.v1',complete=error is None,error=error,
            stages=stages,final=final,maximum_new_child_versions=[2,3,4],actual_GPU_jobs=actual,
            nativeF=sum(r['nativeF'] for r in actual),nativeB=sum(r['nativeB'] for r in actual),updates=0,
            first_measurement_not_counted_here=pin(R/'timewindow17_pruned_footprint_measure_v1/receipt.json'),
            no_old_scope_reused=True,all_original_assets_preserved=True,old_row_retirement_executed=False,whole_joint_quality_accepted=False,seconds=time.time()-started))

if __name__=='__main__':main()
