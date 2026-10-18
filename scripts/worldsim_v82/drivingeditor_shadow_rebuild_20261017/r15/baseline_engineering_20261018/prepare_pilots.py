"""Pin the actual old5 implementation, then expose two loss coefficients.

The historical implementation, inputs and checkpoints remain read-only. The
four short arms distinguish reproduction from restoring a missing raw-body
constraint and from the 3-to-5 identity-normalization change.
"""
from pathlib import Path
import shutil,json,hashlib,ast,datetime
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
TASK=ROOT/'identity_completion_20261018';OUT=ROOT/'baseline_engineering_20261018'
IMPL=OUT/'implementation'

def main():
    if IMPL.exists():raise FileExistsError('do not replace pinned pilot source')
    IMPL.mkdir();hashes={}
    for p in (TASK/'infra').glob('*.py'):
        hashes[p.name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),mtime_ns=p.stat().st_mtime_ns)
        shutil.copy2(p,IMPL/p.name)
    p=IMPL/'train.py';s=p.read_text()
    def replace(old,new):
        nonlocal s
        assert s.count(old)==1,old;s=s.replace(old,new)
    replace("policy['hybrid_lambda_dynamic_opacity'] = 0.0", "policy['hybrid_lambda_dynamic_opacity'] = args.raw_body_opacity_weight")
    replace("loss = result.total + 100 * mask_loss", "loss = result.total + args.identity_opacity_weight * mask_loss")
    replace("training_policy = resolve_training_policy(vars(args), raw_reference)",
        "training_policy = resolve_training_policy(vars(args), raw_reference)\n        training_policy['adapter_constants'].update(visible_instance_opacity_weight=args.identity_opacity_weight, raw_body_opacity_weight=args.raw_body_opacity_weight)")
    replace("parser.add_argument('--max-features', type=int, default=10)",
        "parser.add_argument('--max-features', type=int, default=10)\n    parser.add_argument('--raw-body-opacity-weight', type=float, default=0.0)\n    parser.add_argument('--identity-opacity-weight', type=float, default=100.0)")
    replace("(args.box_boundary_weight, args.pose_prior_weight)",
        "(args.box_boundary_weight, args.pose_prior_weight, args.raw_body_opacity_weight, args.identity_opacity_weight)")
    replace("'instance_mask_loss': float(mask_loss.detach()),",
        "'raw_body_opacity_loss': float(result.raw['dynamic_opacity'].detach()),\n                          'raw_body_opacity_weight': args.raw_body_opacity_weight,\n                          'identity_opacity_weight': args.identity_opacity_weight,\n                          'instance_mask_loss': float(mask_loss.detach()),")
    # Every legacy loss, schedule, sampler, pose, static/environment policy and
    # topology path is otherwise unchanged. Make this auditable in the snapshot.
    ast.parse(s);p.write_text(s)
    variants=[
        dict(name='three_replay',identities=3,spec=str(ROOT/'rigid_specs/scene0008_089_137.json'),raw_body_weight=0.,ID_weight=100.,role='historical three-ID reproduction'),
        dict(name='five_replay',identities=5,spec=str(TASK/'spec5.json'),raw_body_weight=0.,ID_weight=100.,role='historical five-ID reproduction'),
        dict(name='five_raw_body',identities=5,spec=str(TASK/'spec5.json'),raw_body_weight=100.,ID_weight=100.,role='restore inherited isolated raw-body opacity constraint only'),
        dict(name='five_ID_weight',identities=5,spec=str(TASK/'spec5.json'),raw_body_weight=0.,ID_weight=100.*5/3,role='keep per-ID coefficient equal to historical three-ID arm only')]
    contract=dict(status='PREPARED_NOT_TRAINED',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        variants=variants,updates_each=300,maximum_total_pilot_updates=1200,LR_horizon=30000,
        initialization='same GN11 30000 static/environment and original raw-LiDAR seeds; no trained instances, no R15 weights',
        seed=0,all_49_frames=True,physical_shadows=False,pose_refinement=False,topology='same fixed historical topology',
        R15_parameter_updates=0,full_30000_training_state='BLOCKED_UNTIL_ENGINEERING_AND_PILOT_REVIEW',
        raw_body_term_source='https://github.com/WangXu-xxx/DrivingEditor/blob/560d57c3825ed2bdbc6a6602e0011ad51e8da60e/train.py#L441-L450',
        pending_fidelity_items=['fixed-topology adapter versus native densification/pruning','actual loss-gradient and optimizer/RNG restore receipts','original GN11 prior static artifact provenance'],
        historical_source_hashes=hashes,pilot_train_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        human_verdict=None,algorithm_quality_accepted=False)
    (OUT/'experiment_contract.json').write_text(json.dumps(contract,indent=2))
    print(json.dumps(dict(status=contract['status'],variants=[v['name'] for v in variants],maximum_updates=1200)))

if __name__=='__main__':main()
