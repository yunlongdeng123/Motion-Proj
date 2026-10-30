from pathlib import Path
import json, subprocess
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030')
out={}
for run in ['smoke_r2/fresh','smoke_r2/resumed','smoke_r2/next_stage_entry','formal_r3_10k']:
    p=root/run
    item={}
    for f in ['progress.json','returned.json','restore_verification.json','cuda_memory_peak.json','stop_trigger_timing.json']:
        if (p/f).exists():
            d=json.loads((p/f).read_text())
            if f=='restore_verification.json':d={k:d.get(k) for k in ['iteration','all_passed']}
            item[f]=d
    out[run]=item
for f in ['smoke_r2/smoke_summary.json','smoke_r2/render_entry_execution.json','formal_r3_10k/execution.json']:
    if (root/f).exists():
        d=json.loads((root/f).read_text());out[f]={k:d.get(k) for k in ['status','returncode','ten_minute_contract','approximately_ten_minutes_contract','extra_smoke_updates','extra_entry_updates','formal_updates']}
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip()
print(json.dumps(out,ensure_ascii=False))
