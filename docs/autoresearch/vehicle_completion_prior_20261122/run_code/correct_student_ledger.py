"""Correct inherited accounting without repeating any optimizer or GPU work."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_completion_prior_20261122_r1/student')

def main():
    original=ROOT/'resource_ledger.json';saved=ROOT/'resource_ledger_original_inherited.json'
    if not saved.exists():shutil.copy2(original,saved)
    raw=json.loads(original.read_text());smoke=json.loads((ROOT/'smoke/proof.json').read_text())
    formal=raw['formal_vehicle_updates']+raw['formal_joint_updates'];smoke_total=smoke['actual_smoke_optimizer_updates']
    assert formal==400 and smoke_total==9
    process={}
    for phase in ['smoke','formal','reload']:
        p=ROOT/(phase+'_resource.json')
        if p.exists():process[phase]=json.loads(p.read_text())
    result=dict(schema='a20.corrected_physical_optimizer_ledger.v1',
        physical_optimizer_updates=dict(formal_vehicle=200,formal_joint=200,formal_total=400,
            smoke_total_including_replay=9,smoke_base=7,replay_within_smoke=2,
            outside_smoke_extra=0,numeric_probe_outside_smoke=0,total=409),
        original_inherited_ledger=dict(path=str(saved),sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
            original_extra_replay_updates=raw.get('extra_replay_updates'),preserved=True),
        per_process_renderer_and_parameter_VJP=process,
        notes=['Inherited extra_replay_updates=2 is included within smoke total 9, not additive.',
               'Formal raster 647 and parameter-gradient backward/VJP 600 cover only that process.',
               'No whole-task raster/backward total is claimed; physical Adam updates are separate.',
               'Preview and fresh reload have zero optimizer updates; no extra numeric probe updates.'],
        current_candidate='diagnostic_rejected_geometry',new_optimizer_updates=0,new_gpu_tasks=0,GPU_keep_on=True)
    (ROOT/'resource_ledger_corrected.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
