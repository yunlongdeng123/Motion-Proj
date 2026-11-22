"""Independent stencil PDE residual, RGB replay, fixed seams and source grain."""
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from audit_r9_shared_field_cpu import load,read,write,pin,stats,lf

WORK=Path(__file__).resolve().parents[2];R9=WORK/'old_raw_teacher_joint_20261123_r9';R7=WORK/'road_shared_appearance_20261122_r7'
OUT=R9/'cpu_review/joint_shared_continuous_v2';OUT.mkdir(parents=True,exist_ok=True)
fp=R9/'joint_shared_continuous_v2/final_atlas.npz';sp=R9/'joint_shared_continuous_v2/solve_state.npz';cp=R9/'continuous_field_contract_v2.json'
f=load(fp);s=load(sp);c=read(cp);producer=read(R9/'joint_shared_continuous_v2/receipt.json')
seedp=R7/'quilt_seed/atlas_seed.npz';seed=load(seedp);real=load(R7/'real_observation_atlas.npz')
v1=load(R9/'joint_shared_v1/solve_state.npz');v1field=load(R9/'joint_shared_v1/final_atlas.npz')
base=seed['atlas_seed_linear'];rgb=f['atlas_linear'];perm=seed['write_domain'];known=seed['same_world_lock'];unknown=perm&~known
material=perm&(f['source_kind']>0);updated=np.any(rgb!=base,-1);delta=s['RGB_residual'].astype(np.float64)
weights=s['data_weights'].astype(np.float64);W=weights.sum(0);data=v1['corrected_RGB'].astype(np.float64)
mu=float(c['seed_residual_prior']);lam=float(c['laplacian_lambda']);rtol=float(c['CG_relative_tolerance'])

def degree(shape):
    d=np.full(shape,4.,np.float64);d[0]-=1;d[-1]-=1;d[:,0]-=1;d[:,-1]-=1;return d
D=degree(W.shape)
def adjacent_sum(x):
    y=np.zeros_like(x);y[1:]+=x[:-1];y[:-1]+=x[1:];y[:,1:]+=x[:,:-1];y[:,:-1]+=x[:,1:];return y
def apply(x):
    return (mu+W)[...,None]*x+lam*(D[...,None]*x-adjacent_sum(x))
def half_ulp(a):return .5*np.spacing(np.abs(a)).astype(np.float64)

rhs=(weights[...,None]*(data-base[None])).sum(0);residual=apply(delta)-rhs
delta_error=half_ulp(s['RGB_residual']);delta_error[~unknown]=0
weight_error=half_ulp(s['data_weights']);eW=weight_error.sum(0)
quant=(mu+W)[...,None]*delta_error+lam*(D[...,None]*delta_error+adjacent_sum(delta_error))
quant+=eW[...,None]*abs(delta)+(weight_error[...,None]*abs(data-base[None])).sum(0)
pde=[]
for channel in range(3):
    norm=float(np.linalg.norm(rhs[...,channel][unknown]));err=float(np.linalg.norm(residual[...,channel][unknown]));bound=float(np.linalg.norm(quant[...,channel][unknown]))
    pde.append({'channel':channel,'relative_residual_from_float32_state':err/max(norm,1e-12),
      'float32_export_error_bound_relative':bound/max(norm,1e-12),'contract_CG_rtol':rtol,
      'within_CG_plus_export_error_bound':bool(err<=bound+1.05*rtol*norm+1e-10)})
influence=s['conditional_teacher_influence'].astype(np.float64)
ires=apply(influence[...,None])[...,0]-W
ie=half_ulp(s['conditional_teacher_influence']);ie[~unknown]=0
ibound=(mu+W)*ie+lam*(D*ie+adjacent_sum(ie))+eW*abs(influence-1)
inorm=float(np.linalg.norm(W[unknown]));ierr=float(np.linalg.norm(ires[unknown]));iq=float(np.linalg.norm(ibound[unknown]))
influence_pde={'relative_residual_from_float32_state':ierr/max(inorm,1e-12),'float32_export_error_bound_relative':iq/max(inorm,1e-12),
  'within_CG_plus_export_error_bound':bool(ierr<=iq+1.05*rtol*inorm+1e-10)}
replay=(base.astype(np.float64)+delta).astype(np.float32);replay[known]=base[known];replay[~perm]=base[~perm]
rgb_error=float(abs(replay-rgb).max())
checks={
  'field_SHA':pin(fp)['sha256']==producer['final_atlas_sha256'],'solve_state_SHA':pin(sp)['sha256']==producer['solve_state_sha256'],
  'contract_SHA':pin(cp)['sha256']==producer['contract_sha256'],'seed_SHA':pin(seedp)['sha256']==producer['seed_sha256'],
  'radiometry_state_SHA':pin(R9/'joint_shared_v1/solve_state.npz')['sha256']==producer['radiometry_state_sha256'],
  'source_receipt_SHA':pin(R9/'world_observations_v2/receipt.json')['sha256']==producer['source_observation_receipt_sha256'],
  'source_sampling_audit_still_PASS':read(R9/'cpu_review/joint_shared_v1/source_sampling_qualification_audit.json')['decision']=='PASS',
  'source_valid_masks_bitexact_v1':bool(np.array_equal(s['valid'],v1['valid'])),
  'real6407':int(known.sum())==6407,'real_lock_mask_exact':bool(np.array_equal(f['real_locks'],known)),
  'real_RGB_byteexact':rgb[known].tobytes()==real['canonical_linear_rgb'][known].tobytes()==base[known].tobytes(),
  'outside_permission_RGB_byteexact':rgb[~perm].tobytes()==base[~perm].tobytes(),
  'permission_exact':bool(np.array_equal(f['permission_mask'],perm)),
  'all_RGB_finite':bool(np.isfinite(rgb).all()),'linear_RGB_valid_without_clipping':bool((rgb>=0).all() and (rgb<=1).all()),
  'residual_Dirichlet_zero':not bool(delta[~unknown].any()),
  'RGB_replay_without_alpha_multiplication':rgb_error<=6e-8,
  'PDE_residuals_match_contract_plus_export_precision':all(x['within_CG_plus_export_error_bound'] for x in pde),
  'influence_PDE_matches_contract_plus_export_precision':influence_pde['within_CG_plus_export_error_bound'],
  'producer_all_CG_within_original_tolerance':all(x['relative_residual']<=rtol for it in producer['IRLS_CG_history'] for x in it['channels']) and producer['influence_CG']['relative_residual']<=rtol,
  'teacher_direct_support_exact':bool(np.array_equal(f['teacher_directly_observed'],s['valid'].any(0))),
  'inferred_label_exact':bool(np.array_equal(f['teacher_inferred_field'],f['teacher_adopted']&~s['valid'].any(0))),
  'source1_exact_real_locks':bool(np.array_equal(f['source_kind']==1,known)),
  'unknown_in_permission_zero':not bool(((f['source_kind']==0)&perm).any()),
  'same_five_view_material_domain':all(x['same_evaluated_domain_bitexact_v1'] for x in read(OUT/'surface_views_receipt.json')['views'])}
final_lf=lf(rgb,material,6.);seed_lf=lf(base,material,6.);boundaries=[]
for name,adopt in [('fixed_v1_teacher_boundary',v1field['teacher_adopted']),('current_v2_source_annotation_boundary',f['teacher_adopted'])]:
    final_values=[];seed_values=[];count=0
    for axis in [0,1]:
        a=(slice(None,-1),slice(None)) if axis==0 else (slice(None),slice(None,-1));b=(slice(1,None),slice(None)) if axis==0 else (slice(None),slice(1,None))
        edge=material[a]&material[b]&(adopt[a]!=adopt[b]);count+=int(edge.sum())
        final_values.append(abs(final_lf[a]-final_lf[b])[edge].ravel());seed_values.append(abs(seed_lf[a]-seed_lf[b])[edge].ravel())
    fs=stats(np.concatenate(final_values));bs=stats(np.concatenate(seed_values));passed=bool(fs['p95']<=.005 or fs['p95']<=bs['p95']+.0025)
    boundaries.append({'domain':name,'fixed_edges':count,'final_LF_jump':fs,'same_edge_seed_LF_jump':bs,'original_contract_pass':passed})
checks['fixed_v1_and_current_boundary_LF_contract_pass']=all(x['original_contract_pass'] for x in boundaries)
grain=[];regions={'complete_actual_updated_domain':updated&unknown,
  'all_teacher_labelled_domain':f['teacher_adopted'],
  'six_texel_material_interior_actual_updated':binary_erosion(material,iterations=6)&updated&unknown}
for sigma in [1.,2.]:
    seed_hf=base-lf(base,material,sigma);final_hf=rgb-lf(rgb,material,sigma);change=final_hf-seed_hf
    for name,mask in regions.items():
        a=seed_hf[mask].ravel();b=final_hf[mask].ravel();d=change[mask].ravel()
        r0=float(np.sqrt(np.mean(a*a)));r1=float(np.sqrt(np.mean(b*b)));rd=float(np.sqrt(np.mean(d*d)))
        grain.append({'domain':name,'domain_texels':int(mask.sum()),'fraction_of_all_actual_updated':float(mask.sum()/max((updated&unknown).sum(),1)),
          'HF_sigma_texel':sigma,'HF_sigma_m':sigma*.02,'seed_HF_RMS':r0,'final_HF_RMS':r1,'HF_delta_RMS':rd,
          'HF_delta_over_seed_HF_RMS':rd/max(r0,1e-12),'HF_amplitude_ratio':r1/max(r0,1e-12),
          'HF_Pearson_correlation':float(np.corrcoef(a,b)[0,1]),'unwritten_grain_threshold_invented':False})
subthreshold=unknown&(f['source_kind']==2)&updated
tail={'source_kind3_threshold':'conditional teacher influence > .001; not a zero-effect boundary',
  'all_actual_RGB_updated_texels':int((updated&unknown).sum()),'kind3_adopted_texels':int(f['teacher_adopted'].sum()),
  'kind2_with_nonzero_continuous_teacher_tail_texels':int(subthreshold.sum()),
  'kind2_tail_RGB_max_abs_change':float(abs(rgb-base)[subthreshold].max()) if subthreshold.any() else 0.,
  'kind2_is_not_claimed_bitexact_seed_by_this_audit':True}
result={'schema':'R9.independent_continuous_shared_field_metrics.v2','complete':True,
  'decision':'LOCAL_NUMERIC_CHECKS_PASS' if all(checks.values()) else 'FAIL','final_atlas':pin(fp),'contract':pin(cp),
  'checks':checks,'stencil_method':'Four-neighbour full-grid Laplacian on residual with exact zero known/outside Dirichlet; does not import producer system/solve.',
  'PDE_channels':pde,'influence_PDE':influence_pde,'float32_export_precision_bound_method':'Half-ULP bounds for saved residual/weights propagated through absolute stencil and RHS; original CG tolerance added unchanged.',
  'RGB_replay_max_abs_from_float32_residual':rgb_error,'boundary_LF':boundaries,
  'source_grain':grain,'lineage_threshold_tail':tail,
  'unchanged_input_candidate_scores':producer['unchanged_fixed_pair_metrics'],
  'input_candidate_scores_are_not_new_field_quality':True,
  'surface_views':pin(OUT/'surface_views_receipt.json'),'fixed1792':read(OUT/'surface_views_receipt.json')['fixed1792'],
  'visual_acceptance':'PENDING independent and root full-resolution review','confidence_is_calibrated_probability':False,
  'CPU_only':True,'GPU_forward':0,'diffusion_calls':0,'native_F':0,'native_B':0,'Gaussian_updates':0}
write(OUT/'independent_metrics.json',result)
print({'decision':result['decision'],'checks':checks,'PDE':pde,'influence_PDE':influence_pde,'boundary':boundaries,'grain':grain,'lineage':tail})
