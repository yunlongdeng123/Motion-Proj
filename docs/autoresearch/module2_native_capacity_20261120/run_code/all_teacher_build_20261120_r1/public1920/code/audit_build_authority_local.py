"""CPU authority audit with a selective, hashed remote checkpoint extraction.

Local Torch loading avoids loading a second Torch runtime in the remote2GiB
cgroup while the full-resolution CPU evaluator is active. No GPU is called.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);a=p.parse_args();d=Path(a.inputs);out=Path(a.out)
    m=json.loads((d/'source_manifest.json').read_text())
    if sha(d/'arrays.npz')!=m['arrays_SHA256']or sha(d/'oracle_colors.pt')!=m['authority_SHA256']or sha(d/'field_seed_colors.pt')!=m['field_authority_SHA256']:raise RuntimeError('transport binding mismatch')
    with np.load(d/'arrays.npz',allow_pickle=False)as z:
        accepted=z['accepted'];pending=z['pending'];active=z['active_world_rows'];asset=z['asset_effective_rgb_active']
    payload=torch.load(d/'oracle_colors.pt',map_location='cpu',weights_only=False);rgb=payload['rgb'].numpy();eligible=np.asarray(payload['eligible_active']);full=payload['c0'].numpy().copy()
    if not np.array_equal(active,payload['active_world_rows']):raise RuntimeError('active world order differs')
    accepted_exact=np.array_equal(accepted.astype(np.float32).reshape(rgb.shape),rgb)
    full[eligible]=rgb;asset_exact=np.array_equal(full,asset)
    if not accepted_exact or not asset_exact:raise RuntimeError('accepted point/authority/persistent asset mismatch')
    seed=torch.load(d/'field_seed_colors.pt',map_location='cpu',weights_only=False)
    if not np.array_equal(active,seed['active_world_rows'])or not np.array_equal(eligible,seed['eligible_active']):raise RuntimeError('FIELD/NUMERIC scope differs')
    seed_rgb=seed['c0'].numpy().copy();seed_rgb[eligible]=seed['rgb'].numpy()
    readonly_exact=np.array_equal(full[~eligible],seed_rgb[~eligible])
    if not readonly_exact or m['field_query_max']!=0 or m['field_derived_ci_equal']is not True:raise RuntimeError('FIELD seed/query or NUMERIC readonly mismatch')
    if out.exists():raise FileExistsError('preserve old audit')
    value=dict(schema='all_teacher_BUILD_accepted_authority_CPU_audit.v1',complete=True,CPU_only=True,new_native_calls=0,new_updates=0,
        meta_last_accepted_to_authority_float32_exact=True,authority_to_persistent_asset_exact=True,accepted_shape=list(accepted.shape),
        active_rows=len(active),eligible_rows=int(eligible.sum()),pending_to_accepted_Linf=float(np.max(np.abs(pending-accepted))),
        pending_after_float32_equals_authority=np.array_equal(pending.astype(np.float32).reshape(rgb.shape),rgb),
        exported_point='meta_last_accepted_x; not substituted by Fortran pending line-search x',
        numeric_readonly_equals_FIELD_seed_exact=readonly_exact,FIELD_query_and_derived_ci_exact=True,
        data_role=m['data_role'],held_generalization_claim=False,
        reported_training_numeric_converged=m['numeric_training_converged'],strict_negative_certificate=m['strict_numerical_negative_certificate'],
        remote_source_SHA256={m['workspace']:m['workspace_SHA256'],m['authority']:m['authority_SHA256'],m['asset_road']:m['asset_road_SHA256'],m['receipt']:m['receipt_SHA256'],m['field_authority']:m['field_authority_SHA256'],m['field_pt_authority']:m['field_pt_authority_SHA256']},
        transport_arrays_SHA256=m['arrays_SHA256'],local_audit_source=str(Path(__file__).resolve()),local_audit_SHA256=sha(__file__))
    out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');print(json.dumps(value,ensure_ascii=False))
if __name__=='__main__':main()
