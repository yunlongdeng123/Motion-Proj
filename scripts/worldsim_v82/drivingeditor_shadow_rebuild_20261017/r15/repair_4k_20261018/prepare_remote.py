"""Create a fully explicit matched-input repair configuration; no GPU launched."""
from pathlib import Path
import json
Q=Path(__file__).resolve().parent;D=Q.parent
c=json.loads((D/'config.json').read_text())
c.update(repair_actor_geometry=True,actor_drift_bound_m=.25,actor_box_margin_m=.10,
    actor_scale_growth_limit=1.5,actor_max_growth=4.,repair_evidence_audit=True,
    ownership_audit_first=1500,ownership_audit_interval=1000,ownership_measurement_root=str(D),shadow_integrator='factor_fp64')
if (Q/'config.json').exists():raise FileExistsError('Do not replace an experiment configuration')
(Q/'config.json').write_text(json.dumps(c,indent=2))
contract=dict(task_id='WS-V82-R15-REPAIR-4K-20261018',run_id='r15_repair4k_seed0',family='R15',seed=0,
    maximum_updates=4000,learning_rate_schedule_geometry_steps=30000,input=str(D/'data'),
    initialization='same raw measurements, no learned checkpoint',original_masks_unchanged=True,
    background_and_actor_joint_training=True,original_losses_unchanged=True,
    audit_steps=[1500,2500,3500],opacity_reset_steps=[3000],original_checkpoint_preserved=str(D/'paused_checkpoint_004000_20261017/checkpoint.pt'),
    failure_ledger_refs=['V82-F06'],failure_ledger_delta='none',human_verdict=None,
    changed_policies=['FP64 factor shadow integrated','immutable Actor origin, .25m displacement ball',
        '.10m padded box, preserve initial outlier envelope','max sigma <=1.5x original source sigma',
        'max .02 net Actor growth per topology event, total <=4x initial per actor before certified transfers',
        'evidence-unknown Actor points cannot densify','current full-scene alpha*T audit with known positive/negative domains',
        'negative evidence revokes certainty but not owner; repeated actual3D can confirm/change owner',
        'source tag no longer an eligibility gate; road/large support veto protects real background',
        'audit existing geometry; external pool retained, old459-point admission not replayed'],
    interpretation='Combined engineering repair, not a single-factor causal ablation. Box/excess is diagnostic, not ownership truth.',
    acceptance=['finite loss and gradients','strict same-code checkpoint restore','shadow chunk error <=1e-6 in smoke',
        '0 newly violated geometry budgets','actual audit candidate/rejection/transition counts saved',
        'all49 frames x a5,a6,a18,a26,a37 independent RGB/alpha, visible alpha, deletion with/without shadows',
        'body preservation and a6 fragments/a37 residual require visual review; no automatic quality verdict'],
    shutdown='Only after all old comparison, frozen projection, new repair training/evaluation, outputs and Git push are complete')
(Q/'experiment_contract.json').write_text(json.dumps(contract,indent=2))
print(json.dumps(contract))
