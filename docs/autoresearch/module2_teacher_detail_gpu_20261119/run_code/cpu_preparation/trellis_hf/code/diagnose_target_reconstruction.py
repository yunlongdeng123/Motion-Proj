"""Read-only target reader consistency diagnostic; never invokes a projection solver."""
from pathlib import Path
import argparse, os, json
import numpy as np
from project_h_seed import OutputProjection
from trellis_hf_seed import sha


def quality(actual, target, mask):
    x, y = actual[mask].ravel(), target[mask].ravel()
    energy = max(float(y@y), 1e-30)
    return dict(samples=int(mask.sum()), target_RMS=float(np.sqrt(np.mean(y*y))),
        actual_RMS=float(np.sqrt(np.mean(x*x))), RMSE=float(np.sqrt(np.mean((x-y)**2))),
        abs_max=float(np.abs(x-y).max()), cosine=float((x@y)/max(np.linalg.norm(x)*np.linalg.norm(y), 1e-30)),
        energy_ratio=float((x@x)/energy), explained=float(1-np.sum((x-y)**2)/energy))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU only')
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=Path, required=True)
    args = ap.parse_args()
    out = args.seed.parent/'target_reconstruction_audit.json'
    if out.exists(): raise FileExistsError(out)
    with np.load(args.seed)as z:d={k:z[k]for k in z.files}
    report = json.loads((args.seed.parent/'receipt.json').read_text())
    with np.load(report['permission_path'])as z:realA=z['realA_support']
    mH, legal, target = d['mH_combined'], d['prior_legal'], d['desired_effective_H']
    e = OutputProjection(mH, legal, d['raw_parameter_domain'], prior_window=d['prior_window'],
                         realA=realA, HF_source=d['HF_source'])
    target_unmod = np.divide(target, mH[...,None], out=np.zeros_like(target), where=mH[...,None]>0)
    actual_q = np.stack([e.qM*e.sample(d['unmodulated_H'][...,c])for c in range(3)],-1)
    old_q = np.stack([e.qWindow*e.sample(d['HF_source'][...,c])for c in range(3)],-1)
    correct_q = np.stack([e.qM*e.sample(target_unmod[...,c])for c in range(3)],-1)
    inside = e.qLegal&(e.qM>0)
    geometric = e.sample(legal.astype(float))
    mixed = (geometric>0)&(geometric<1)&inside
    core = (geometric==1)&inside
    data = dict(schema='T2_HF_target_node_vs_query_reconstruction_diagnostic.v1',
        no_solver_or_objective_change=True, seed_sha256=sha(args.seed),
        node_r4_vs_frozen_target=quality(d['actual_effective_H'],target,legal),
        quarter_r4_vs_old_separate_window_target=quality(actual_q,old_q,inside),
        quarter_r4_vs_nodal_reconstructed_target=quality(actual_q,correct_q,inside),
        old_quarter_vs_nodal_target=quality(old_q,correct_q,inside),
        mixed_old_vs_nodal_target=quality(old_q,correct_q,mixed),
        core_old_vs_nodal_target=quality(old_q,correct_q,core),
        frozen_node_source_mean_or_permission_changed=False,
        actual_runtime='bilinear(common_mH) * bilinear(unmodulated_H)',
        actual_target_old='bilinear(prior_window) * bilinear(HF_source)',
        nodal_target_definition='target_i fixed; target_unmodulated_i=target_i/common_mH_i (0 at zero gate); same runtime reconstruction',
        conclusion='Separate prior-window interpolation is a different higher-order field from reconstructing the frozen nodal H target. This diagnostic does not attribute all projection error to it.',
        GPU_operations=0, new_inference_calls=0, optimizer_updates=0)
    out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))


if __name__=='__main__':main()
