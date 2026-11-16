"""CPU acceptance and precision note for the frozen candidate, no GPU access."""
from pathlib import Path
import argparse,hashlib,json,os,sys
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def bitwise_equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA_VISIBLE_DEVICES must be empty')
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);freeze=read(args.candidate/'freeze.json');manifest=read(args.candidate/'manifest.json')
    check={};bindings=freeze['source_files']+freeze['code_files']+freeze['target_files']
    check['all_frozen_bindings_unchanged']=all(sha(x['path'])==x['sha256'] for x in bindings)
    check['candidate_not_formal_ready']=manifest['candidate_only'] and not manifest['formal_ready']
    check['GPU_training_render_operations_zero']=all(manifest[x]==0 for x in ['GPU_operations','training_updates','new_render_operations'])
    base=args.candidate.parent.parent;oldroot=base/'road_direct_teacher_20261116_r1'
    for raw in [86,110]:
        old=load(oldroot/'supervision_r3'/f'{raw:03d}.npz');new=load(args.candidate/f'{raw:03d}.npz');prefix=f'raw{raw}_'
        check[prefix+'targets_and_world_geometry_exact']=all(bitwise_equal(new[k],old[k]) for k in ['target_A_srgb','target_C_srgb','F_control_C_srgb','ground_world_xyz','world_row_index','triangle_world_rows','triangle_weights','K','fixed_initial_alpha'])
        check[prefix+'disjoint_A_C_S_K']=not any((a&b).any() for a,b in [(new['A_visible'],new['C_hole']),(new['A_visible'],new['S_target_shadow']),(new['C_hole'],new['S_target_shadow']),((new['A_visible']|new['C_hole']|new['S_target_shadow']),new['K'])])
        check[prefix+'outside_exact_full_authorized_complement']=np.array_equal(new['outside'],~(new['A_visible']|new['C_hole']|new['S_target_shadow']|new['K']))
        check[prefix+'q_S_binary_and_photo_gated']=np.array_equal(new['q_S'],new['S_target_shadow_photo'].astype(np.float32))
        check[prefix+'shadow_boundary_has_no_loss_weight']=not (new['q_S'][new['S_target_shadow_boundary_unknown']]!=0).any()
        check[prefix+'teacher_missing_has_no_loss_weight']=not (new['q_S'][new['S_teacher_missing']]!=0).any()
        check[prefix+'other_shadow_not_erased']=not ((new['S_target_shadow']|new['C_hole'])&new['S_other_shadow_guard']).any()
        check[prefix+'A_has_no_target_shadow_conflict']=not (new['A_mask']&(new['S_target_shadow_possible']|new['S_target_shadow_boundary_unknown'])).any()
        check[prefix+'C_physical_scope_has_no_old_F0_requirement']=np.array_equal(new['C_hole_photo'],new['C_hole']&new['fixed_high_alpha'])
        check[prefix+'legacy_r3_domains_saved_exact']=all(np.array_equal(new['legacy_r3_'+k],old[k]) for k in ['A_mask','C_mask','q_A','q_C','A_visible','C_teacher_hole','coverage_A','coverage_C','coverage_mask','q_A_coverage','q_C_coverage','outside'])
    a=load(oldroot/'direct_C/formal_r1/row_scopes.npz');b=load(oldroot/'matched_F_control/formal_r1/row_scopes.npz')
    precision={}
    for k,x in a.items():
        y=b[k]
        if x.dtype.kind in 'biu':check['old_pair_discrete_'+k+'_exact']=np.array_equal(x,y)
        elif not np.array_equal(x,y):
            precision[k]=dict(different_values=int((x!=y).sum()),max_abs=float(np.abs(x-y).max()),positive_support_identical=bool(np.array_equal(x>0,y>0)))
    check['old_pair_mass_positive_support_exact']=all(x['positive_support_identical'] for x in precision.values())
    check['torch_not_imported']='torch' not in sys.modules
    scope=read(args.candidate/'previous_scope_and_loss_audit.json')['scope']
    check['old_scope_formula_checks_pass']=scope['all_checks_passed']
    check['old_source_matches_formal_binding']=scope['trainer_source_matches_both_formal_bindings']
    result=dict(schema='road_capacity_supervision_cpu_acceptance.v1',status='PASS' if all(check.values()) else 'FAIL',checks=check,
        check_count=len(check),old_pair_attribution_mass_precision=precision,
        note='previous_scope_and_loss_audit matched_Fcontrol_scopes_equal_exact compares the whole NPZ, including nondeterministic floating attribution sums; all discrete scopes and mass>0 eligibility are exact',
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],GPU_operations=0,training_updates=0,render_operations=0,
        formal_ready=False,candidate_freeze_sha256=sha(args.candidate/'freeze.json'))
    write(args.output/'acceptance.json',result);print(json.dumps(result,ensure_ascii=False))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
