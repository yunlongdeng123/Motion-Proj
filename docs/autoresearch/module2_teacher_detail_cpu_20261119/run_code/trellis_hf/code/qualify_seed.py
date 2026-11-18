"""CPU candidate H seed from an independent current teacher permission crop. No GPU entry."""
from pathlib import Path
import os, argparse, json
import numpy as np
from trellis_hf_seed import encode_h_seed, sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only')
    p = argparse.ArgumentParser()
    p.add_argument('--permission', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-bound', type=float, default=.04)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    with np.load(ROOT/'extraction_r1/nodes_2cm.npz') as z:
        source = {k: z[k] for k in z.files}
    with np.load(a.permission) as z:
        ledger = {k: z[k] for k in z.files}
    required = ['world_row_grid', 'teacher_H_allow', 'realA_support', 'guard', 'geometry_support',
                'common_latent_mask', 'common_write_candidate', 'common_material_valid', 'mH_union']
    if any(key not in ledger for key in required):
        raise ValueError('independent common permission/realA/mH schema incomplete')
    grid = ledger['world_row_grid']
    inverse = {int(row): (y, x) for (y, x), row in np.ndenumerate(grid)}
    H, W = grid.shape
    rgb = np.zeros((H, W, 3), np.float64)
    source_valid = np.zeros((H, W), bool)
    for (sy, sx), row in np.ndenumerate(source['source_world_rows']):
        if int(row) not in inverse:
            raise ValueError('source world node absent in independent ledger')
        y, x = inverse[int(row)]
        rgb[y, x] = source['bandlimited_rgb'][sy, sx]
        source_valid[y, x] = source['source_geometry_valid'][sy, sx]
    # realA is the current clean measured support. Never veto teacher S by old source1/known.
    allowed = ledger['teacher_H_allow'].astype(bool) & ~ledger['realA_support'].astype(bool)
    allowed &= ~ledger['guard'].astype(bool) & ledger['geometry_support'].astype(bool)
    allowed &= ledger['common_latent_mask'].astype(bool) & ledger['common_write_candidate'].astype(bool)
    allowed &= ledger['common_material_valid'].astype(bool)
    data, report = encode_h_seed(rgb, allowed, ledger['mH_union'], source_valid=source_valid, model_bound=a.model_bound)
    origin = ledger['crop_origin_yx']
    shape = tuple(ledger['global_chart_shape'])
    full_seed = np.zeros((*shape, 3), np.float32)
    full_seed[origin[0]:origin[0]+H, origin[1]:origin[1]+W] = data['H_seed_latent']
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'H_seed.npz', H_seed_latent=full_seed, H_seed_crop=data['H_seed_latent'],
        crop_origin_yx=origin, global_chart_shape=np.asarray(shape), crop_world_rows=grid,
        world_xyz=ledger['world_xyz'], common_material_valid=ledger['common_material_valid'],
        **{k: v for k, v in data.items() if k != 'H_seed_latent'})
    with np.load(a.output/'H_seed.npz') as saved:
        if not np.array_equal(saved['H_seed_latent'], full_seed):
            raise AssertionError('serialized H seed reload mismatch')
    report.update(schema='TRELLIS_HF_only_H_initialization_candidate.v1',
        status='CPU_SEED_CANDIDATE_NOT_GPU_QUALIFIED', permission_crop_path=str(a.permission), permission_crop_sha256=sha(a.permission),
        parent_ledger_path=str(ledger['source_ledger_path']), parent_ledger_sha256=str(ledger['source_ledger_sha256']),
        common_base_buffer_sha256=str(ledger['common_base_buffer_sha256']),
        source_nodes_sha256=sha(ROOT/'extraction_r1/nodes_2cm.npz'), H_seed_sha256=sha(a.output/'H_seed.npz'),
        source_patch_teacher_legal_nodes=int((allowed & source_valid).sum()),
        source_patch_realA_nodes=int((source_valid & ledger['realA_support']).sum()),
        teacher_legal_old_known_nodes=int((data['prior_legal'] & ledger['original_known']).sum()),
        teacher_legal_old_unknown_nodes=int((data['prior_legal'] & ledger['original_unknown']).sum()),
        teacher_legal_old_source1_nodes=int((data['prior_legal'] & (ledger['original_source'] == 1)).sum()),
        teacher_legal_conflict_diagnostic_nodes=int((data['prior_legal'] & ledger['teacher_conflict']).sum()),
        teacher_legal_RGB_conflict_diagnostic_nodes=int((data['prior_legal'] & ledger['teacher_RGB_conflict']).sum()),
        teacher_legal_HF_conflict_nodes=int((data['prior_legal'] & ledger['teacher_HF_conflict']).sum()),
        patch_meaning='fixed mesh material HF only; mean-plane physical anti-alias and actual 2cm chart nodes',
        no_extra_regularizer_or_teacher_RGB_replacement=True,
        injection='same teacherH+pixel (or explicitly matched average) arm, same A200 complete state; only H seed at B entry, H Adam initial state zero',
        A0_low_A200_and_original_ledger_modified=False, central_contract_locked=False, GPU_entry_permitted=False,
        GPU_operations=0, new_inference_calls=0, optimizer_updates=0)
    (a.output/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
