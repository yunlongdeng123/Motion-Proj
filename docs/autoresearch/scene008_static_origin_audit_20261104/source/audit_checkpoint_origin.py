"""Read-only CPU audit of the exact B10k checkpoint and saved row attribution.

No rasterization, training, remote connection, or model writes. Attribution
weights are the historical raw113 Old-C2 native raster, not new GPU results.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.append(r'C:/Users/dengyunlong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
BASE = Path(r'C:/Users/dengyunlong/Documents/Codex/2026-11-03/bug-a5-1k-mae-b-a5-2/outputs')


def quantiles(values):
    return dict(zip(('min', 'p25', 'median', 'p75', 'p95', 'max'),
                    np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist()))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    static_path = OUT/'b10k_snapshot/chkpnt10000.pth'
    model, step = torch.load(static_path, map_location='cpu', weights_only=False, mmap=True)
    assert step == 10000 and len(model) == 18
    old = torch.load(ROOT/'outputs/scene008_a37_C2_20261103/checkpoint001000.pt', map_location='cpu', weights_only=False, mmap=True)
    new = torch.load(ROOT/'outputs/scene008_a37_C2_vehicle_20261104/run/formal_r1/checkpoint001000.pt', map_location='cpu', weights_only=False, mmap=True)
    road = torch.load(BASE/'scene008_a37_only_B10k_road_20261103/checkpoint001000.pt', map_location='cpu', weights_only=False, mmap=True)
    attr = np.load(ROOT/'outputs/scene008_a37_C2_vehicle_20261104/run/evidence/fast113/row_attribution.npz')
    geo = np.load(ROOT/'work/c2_a37_20261103/a6_geometry_minimal.npz')
    protected = np.load(ROOT/'outputs/scene008_a37_C2_vehicle_20261104/run/inputs/retained_static_protection.npz')['ids']
    ids, fit = attr['ids'], attr['fit'].astype(float)
    assert ids.min() >= 0 and ids.max() < len(model[1])
    i = int(np.flatnonzero(geo['raw_ids'] == 113)[0])
    inv = np.linalg.inv(geo['model_from_canonical'][i])
    t = float(geo['baseline_model_time'][np.flatnonzero(geo['baseline_raw_ids'] == 113)[0]])
    xyz = (model[1][ids].detach() + model[9][ids].detach()*torch.sin((t-model[7][ids].detach())*2*np.pi/model[16])/(2*np.pi/model[16])).numpy()
    canonical = xyz @ inv[:3, :3].T + inv[:3, 3]
    error = float(np.max(np.abs(canonical-attr['canonical_center'])))
    # Archived GPU float32 inverse vs CPU float64 inverse may differ below 0.5mm.
    assert error < 5e-4, ('B10k row/time correspondence failed', error)
    lineage = model[14]['sgcm_lineage_v1']['lineage_ids'].numpy()
    blocked = model[14]['sgcm_lineage_v1']['clone_blocked'].numpy()
    families, counts = np.unique(lineage, return_counts=True)
    assert families.min() == 0 and families.max() < 591672
    total_by_root = np.bincount(lineage, minlength=591672)
    blocked_by_root = np.bincount(lineage, weights=blocked.astype(int), minlength=591672)
    mixed_blocked_families = int(((blocked_by_root > 0) & (blocked_by_root < total_by_root)).sum())
    assert mixed_blocked_families == 0
    weight_by_root = np.bincount(lineage[ids], weights=fit, minlength=591672)
    top = np.argsort(-weight_by_root)[:20]
    candidates = []
    keep_c2 = old['actor']['gate_raw'].detach().sigmoid().numpy()
    gate_map = np.ones(len(lineage)); gate_map[old['actor']['contributor_ids'].numpy()] = keep_c2
    road_map = np.ones(len(lineage)); road_map[road['repair']['ids'].numpy()] = road['repair']['gate_raw'].detach().sigmoid().numpy()
    longest_sigma = model[4][ids].detach().exp().numpy().max(1)/.1
    temporal_sigma = model[8][ids].detach().exp().numpy().flatten()
    temporal_alpha = np.exp(-.5*((model[7][ids].detach().numpy().flatten()-t)/temporal_sigma)**2)
    for root in top:
        if not weight_by_root[root]: continue
        at = lineage[ids] == root
        candidates.append(dict(initial_lineage_id=int(root), final_family_rows=int(total_by_root[root]),
                               responsible_rows=int(at.sum()), contribution_fraction=float(fit[at].sum()/fit.sum()),
                               canonical_weighted_center_m=np.average(canonical[at], axis=0, weights=fit[at]).tolist()))
    def subset_stats(selected):
        selected = np.isin(ids, selected)
        w = fit[selected]; row_ids = ids[selected]
        return dict(rows=int(selected.sum()), total_saved_attribution=float(w.sum()),
                    initial_lineages=int(len(np.unique(lineage[row_ids]))),
                    SGCM_clone_spent_rows=int(blocked[row_ids].sum()),
                    SGCM_clone_spent_weight_fraction=float(np.average(blocked[row_ids], weights=w)),
                    oldC2_gate_selected_rows=int(np.isin(row_ids,old['actor']['contributor_ids'].numpy()).sum()),
                    oldC2_gate_weighted_multiplier=float(np.average(gate_map[row_ids], weights=w)),
                    road_gate_selected_rows=int(np.isin(row_ids,road['repair']['ids'].numpy()).sum()),
                    road_gate_weighted_multiplier=float(np.average(road_map[row_ids],weights=w)),
                    protected_other_car_rows=int(np.isin(row_ids,protected).sum()),
                    protected_other_car_weight_fraction=float(np.average(np.isin(row_ids,protected),weights=w)))
    runtime = torch.load(OUT/'b10k_snapshot/training_state10000.pth', map_location='cpu', weights_only=False)
    identity_keys = [k for k in old['parent_identity'] if k.endswith('/chkpnt10000.pth')]
    static_hash = digest(static_path)
    parent_record = old['parent_identity'][identity_keys[0]]
    expected = parent_record.get('sha256') if isinstance(parent_record, dict) else parent_record
    result = dict(mode='local CPU only; no new render/optimization/GPU/remote',
                  reference_domain='historical raw113 fast113 fixed PVD-lower projection domain (776px); Old-C2 readout',
                  B10k_step=step, initialized_static_rows=591672, final_static_rows=len(lineage),
                  checkpoint_sha256=static_hash, parent_record=parent_record,
                  exact_parent_sha256_match=(expected==static_hash if isinstance(expected,str) else None),
                  parent_size_matches=(parent_record[0]==static_path.stat().st_size if isinstance(parent_record,(list,tuple)) else None),
                  bad_row_positions_match_exact_B10k_at_raw113_maxabs_m=error,
                  raw113_model_time=t, final_initial_lineages=len(families),
                  original_lineages_with_multiple_surviving_descendants=int((counts>1).sum()),
                  lineage_id_min=int(lineage.min()),lineage_id_max=int(lineage.max()),
                  mixed_clone_spent_flag_families=mixed_blocked_families,
                  all_responsible=subset_stats(ids), top99_responsible=subset_stats(attr['top99_ids']),
                  top_lineage_families=candidates,
                  responsible_longest_sigma_m=quantiles(longest_sigma),
                  responsible_temporal_sigma_modeltime=quantiles(temporal_sigma),
                  responsible_temporal_alpha_weighted=float(np.average(temporal_alpha,weights=fit)),
                  initial_lineage_to_birth_step_available=False,
                  lineage_to_exact_original_actor_available=False,
                  actual_saved_runtime=runtime['model_runtime'],
                  static_body_parameters_modified_after_B10k=False,
                  old_to_new_actor_nonRGB_state_equal={k:bool(torch.equal(old['actor'][k],new['actor'][k])) for k in old['actor'] if k!='colour_logit'},
                  NewLocal_gate_policy='legacy_replacement=native bypasses OldC2 actor gates; r3 gate remains; corridor mask applied',
                  limits=['Lineage IDs identify initialization families, not birth iteration or exact split-vs-SGCM event.',
                          'Clone-spent flag is evidence of SGCM clone allowance usage in a family, not proof SGCM caused the artifact.',
                          'Saved row weights are historical Old-C2 raw113 contributions; do not treat as current Newlocal raw89 attribution.'])
    if isinstance(expected,str) and not result['exact_parent_sha256_match']:
        raise RuntimeError(('Parent checkpoint hash mismatch or unsupported record',parent_record,static_hash))
    (OUT/'checkpoint_origin_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    np.savez_compressed(OUT/'responsible_row_lineage.npz',ids=ids,initial_lineage=lineage[ids],fit=fit,
                        canonical_center_m=canonical,oldC2_gate=gate_map[ids],road_gate=road_map[ids],
                        SGCM_clone_spent=blocked[ids],top99_ids=attr['top99_ids'])
    print(json.dumps({k:v for k,v in result.items() if k!='top_lineage_families'},ensure_ascii=False,indent=2))
    print('TOP_FIVE_FAMILIES',json.dumps(candidates[:5]))


if __name__ == '__main__':
    main()
