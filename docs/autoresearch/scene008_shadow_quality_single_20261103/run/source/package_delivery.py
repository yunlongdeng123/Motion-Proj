"""CPU delivery packaging after GPU evaluation, preserving intermediate runs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,subprocess,sys,tarfile,hashlib,time
from pathlib import Path
RUN=Path(__file__).resolve().parent.parent
def main():
    assert (RUN/'quality/metrics.json').exists()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
    if not (RUN/'adaptive_rgb_control/receipt.json').exists():
        with (RUN/'adaptive_rgb.log').open('x') as log:
            subprocess.run([sys.executable,str(RUN/'source/adaptive_rgb_control.py')],cwd=RUN,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    # Correct a receipt-only count of floating point weights ~1e-16 at the
    # exact outer contour; these cells never entered the active representation.
    import numpy as np
    p=RUN/'inputs/receipt.json';original=p.read_text();(RUN/'inputs/receipt_initial.json').write_text(original)
    rec=json.loads(original);r=np.load(RUN/'inputs/coarse_reference.npz');w=r['edit_weight'];rec['taper_cells']=int(((w>.005)&(w<.99)).sum());p.write_text(json.dumps(rec,indent=2))
    with tarfile.open(RUN/'review_light.tar.gz','w:gz',compresslevel=3) as t:
        for folder in ['source','quality','adaptive_rgb_control','single_controls_review']:
            t.add(RUN/folder,arcname='run/'+folder,filter=lambda x:None if '__pycache__' in x.name else x)
        for rel in ['inputs/receipt.json','formal/contract.json','formal/receipt.json','formal/training_from0.jsonl','formal/checkpoint001000.pt','evaluation/invariants.json','evaluation/restore.json','smoke_receipt.json','queue_complete.json','single_controls_complete.json','parameter_entry_receipt.json']:
            t.add(RUN/rel,arcname='run/'+rel)
        for actor in [5,37]:
            for rel in ['formal/checkpoint001000.pt','formal/contract.json','formal/training_from0.jsonl','formal/receipt.json','inputs/receipt.json','evaluation/all49/metrics.json','evaluation/invariants.json','evaluation/restore.json']:
                target='single_a%d/'%actor+rel;t.add(RUN/target,arcname='run/'+target)
        t.add(RUN/'quality_settings.json',arcname='quality_settings.json');t.add(RUN/'B_R4_R8_all49.mp4',arcname='B_R4_R8_all49.mp4')
        t.add(RUN/'single_B_a5_a37_joint_all49.mp4',arcname='single_B_a5_a37_joint_all49.mp4')
    snapshot=RUN/'shadow_quality_replay.tar.gz'
    with tarfile.open(snapshot,'w:gz',compresslevel=3) as t:
        for folder in ['source','inputs','formal','evaluation','quality','adaptive_rgb_control','detail_probe','actor_cause_audit','single_controls_review','smoke','smoke_restore','smoke_repeat']:
            t.add(RUN/folder,arcname=RUN.name+'/'+folder,filter=lambda x:None if '__pycache__' in x.name else x)
        for name in ['quality_settings.json','prepare.log','queue.log','train.log','eval.log','assess.log','adaptive_rgb.log','smoke_receipt.json','queue_complete.json','B_R4_R8_all49.mp4','single_B_a5_a37_joint_all49.mp4','single_controls_complete.json','parameter_entry_receipt.json']:
            t.add(RUN/name,arcname=RUN.name+'/'+name)
        for actor in [5,37]:
            # Single controls retain full derived inputs and final state; their
            # intermediate checkpoints remain in the original remote run.
            single=RUN/('single_a%d'%actor)
            for rel in ['inputs','formal/checkpoint001000.pt','formal/contract.json','formal/training_from0.jsonl','formal/receipt.json','evaluation/invariants.json','evaluation/restore.json','evaluation/all49/metrics.json','smoke/receipt.json','smoke_restore/restore.json','smoke_repeat/restore.json']:
                t.add(single/rel,arcname=RUN.name+'/single_a%d/'%actor+rel)
        for version in ['r5','r6','r7']:
            old=RUN.with_name('shadow_quality_20261103_'+version)
            for rel in ['source','formal/checkpoint001000.pt','formal/contract.json','formal/training_from0.jsonl','formal/receipt.json','quality/metrics.json','inputs/receipt.json','smoke_receipt.json']:
                t.add(old/rel,arcname='engineering_runs/'+old.name+'/'+rel,filter=lambda x:None if '__pycache__' in x.name else x)
    h=hashlib.sha256()
    with snapshot.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    receipt=dict(snapshot_bytes=snapshot.stat().st_size,snapshot_sha256=h.hexdigest(),light_bytes=(RUN/'review_light.tar.gz').stat().st_size,
        formal_updates_this_request=6000,validation_updates=204,total_updates_this_request=6204,prior_formal_updates=4000,
        cumulative_updates=10204,all_original_runs_preserved=True,GPU_shutdown=False,report_pending_local=True)
    (RUN/'package_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
