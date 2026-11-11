from pathlib import Path
import json
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');O=C/'waymo_val065_handoff_fix_20261111_r1';R=C/'waymo_val065_delete30k_20261109_r1'
load=lambda p:json.loads(p.read_text())
def steps(root):return load(root/'gate32/training.json')+load(root/'gate1000/training.json')
a=steps(R);b=steps(O);assert len(b)==1000;assert [(x['step'],x['key']) for x in a]==[(x['step'],x['key']) for x in b]
times={p.name.removesuffix('_command.json'):load(p)['seconds'] for p in O.glob('*_command.json')}
value=dict(command_wall_seconds=times,formal_gate_updates=len(b),extra_optimization_updates=0,native_gaussian_updates=0,
    old_and_new_sampling_schedule_identical=True,seed=0,learning_rate=.06,loss_weights=[1,2,4,3,2,.0005],initial_gate_logit=2,
    exact_restore=load(O/'gate1000/restore.json')['exact_full_state'],reference_fields_identical=load(O/'reference_equivalence.json')['all_fields_identical'],
    hardware='NVIDIA GeForce RTX 4080 SUPER',automatic_shutdown=False,human_verdict=None)
(O/'resource_and_protocol.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
