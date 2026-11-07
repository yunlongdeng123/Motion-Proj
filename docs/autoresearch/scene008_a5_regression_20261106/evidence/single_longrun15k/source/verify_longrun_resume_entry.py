from pathlib import Path
from itertools import islice
import json,hashlib
ROOT=Path(__file__).resolve().parent.parent
OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a5_old_single30k_20261106_r1/smoke_retry1_resume2step/paired_steps.jsonl')
NEW=ROOT/'continuation15k/paired_steps.jsonl'
def records(p):
    with p.open() as f:return [json.loads(line) for line in islice(f,2)]
keys=['iteration','frame_id','time_shift','learning_rates']
a=[{key:row[key] for key in keys} for row in records(OLD)]
b=[{key:row[key] for key in keys} for row in records(NEW)]
assert a==b and [row['iteration'] for row in b]==[11978,11979]
result=dict(status='PASS_SAME_NEXT_TWO_NATIVE_ENTRIES_AS_COMPLETED_LIFECYCLE_SMOKE',records=b,
    compared_fields=keys,extra_updates=0,GPU_used=False,loss_or_cuda_bitwise_equality_claimed=False,
    source_paths=[str(OLD),str(NEW)],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT/'resume_entry_schedule_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'])
