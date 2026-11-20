"""CPU proof that exported ci is the accepted point, not pending Fortran x."""
from pathlib import Path
import json
import numpy as np
import torch
from export_public_endpoints import read,write,file_sha

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/gpu_20261119/native_capacity_20261120_r1')
def main():
    receipt=B/'numerical_completion/formal_r1/receipt.json';r=read(receipt);workspace=Path(r['checkpoint']['workspace'])
    if file_sha(workspace)!=r['checkpoint']['workspace_sha256']:raise RuntimeError('checkpoint binding changed')
    with np.load(workspace,allow_pickle=False)as z:accepted=z['meta_last_accepted_x'].copy();pending=z['x'].copy()
    p=torch.load(r['authority'],map_location='cpu',weights_only=False);rgb=p['rgb'].numpy();c0=p['c0'].numpy().copy()
    active=np.asarray(p['active_world_rows']);eligible=np.asarray(p['eligible_active'])
    exact=np.array_equal(accepted.astype(np.float32).reshape(rgb.shape),rgb)
    if not exact:raise RuntimeError('authority is not the last accepted NEW_X')
    c0[eligible]=rgb
    with np.load(Path(r['asset_dir'])/'road.npz',allow_pickle=False)as z:asset=z['reference_srgb'][active]
    if not np.array_equal(c0,asset):raise RuntimeError('accepted authority and persistent asset differ')
    public=B/'public1920/numeric_export_r1';out=public/'authority_checkpoint_audit_r1.json'
    if out.exists():raise FileExistsError('preserve prior audit')
    write(out,dict(schema='G_NUMERIC_accepted_authority_CPU_audit.v1',complete=True,CPU_only=True,new_native_calls=0,new_updates=0,
        meta_last_accepted_to_authority_float32_exact=True,authority_to_persistent_asset_exact=True,
        accepted_shape=list(accepted.shape),active_rows=len(active),eligible_rows=int(eligible.sum()),
        pending_to_accepted_Linf=float(np.max(np.abs(pending-accepted))),pending_Fortran_x_not_used=True,
        source_SHA256={str(p):file_sha(p)for p in [Path(__file__),receipt,workspace,Path(r['authority']),Path(r['asset_dir'])/'road.npz']},
        reported_training_numeric_converged=r['numeric_training_converged'],strict_negative_certificate=r['strict_numerical_negative_certificate']))
    print(str(out))
if __name__=='__main__':main()
