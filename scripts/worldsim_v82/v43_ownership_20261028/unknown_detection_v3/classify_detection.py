"""Target-related abstention; semantic missingness is never a positive signal.

No optimization and no mask inferred from edited RGB. Thresholds are explicit
proposal operating points, not calibrated probabilities or ground-truth recall.
"""
import numpy as np

CHANNELS=['body','core','boundary','shadow_strict','shadow_photo','background_weak',
          'background_depth_sum','background_depth_squared','target_depth_valid',
          'target_depth_sum','target_depth_squared','candidate_union','total']
REASONS={'body_evidence':1,'boundary_with_body_history':2,'shadow_multiframe':4,
         'shadow_photometric_only':8,'background_support_conflict':16,
         'wide_footprint_mixed':32,'existing_actor_identity':64}
CONFIG={'body_min_mass':.02,'body_min_fraction':.05,'body_repeat_frames':2,
        'single_core_min_mass':.05,'single_core_min_fraction':.20,
        'shadow_min_mass':.01,'shadow_min_fraction':.05,
        'weak_background_min_mass':.05,'background_depth_rms_m':.5,
        'mixed_exterior_fraction':.25,'boundary_fraction':.2}

def classify(w,center_z,body_counts,ns,profile='photo'):
    total=np.maximum(w[:,-1],1e-8);target=w[:,0];core=w[:,1]
    repeat=body_counts>=CONFIG['body_repeat_frames']
    current=(target>CONFIG['body_min_mass'])&(target/total>CONFIG['body_min_fraction'])
    single=(core>CONFIG['single_core_min_mass'])&(core/total>CONFIG['single_core_min_fraction'])
    body=current&(repeat|single)
    boundary=(w[:,2]>.02)&(w[:,2]/total>CONFIG['boundary_fraction'])&repeat&(target>.001)
    strict=(w[:,3]>CONFIG['shadow_min_mass'])&(w[:,3]/total>CONFIG['shadow_min_fraction'])
    photo=(w[:,4]>CONFIG['shadow_min_mass'])&(w[:,4]/total>CONFIG['shadow_min_fraction'])
    sh=strict if profile=='strict' else photo|strict
    m=np.maximum(w[:,5],1e-8);mean=w[:,6]/m;var=np.maximum(w[:,7]/m-mean*mean,0)
    bg_rms=np.sqrt((center_z-mean)**2+var)
    bg=(w[:,5]>CONFIG['weak_background_min_mass'])&(bg_rms<CONFIG['background_depth_rms_m'])
    candidate=body|boundary|sh
    if profile=='shadow':candidate=sh
    # Even a depth-compatible background witness remains weak: it may carry
    # another shadow. Conflicts are retained as mixed unknown, never vetoed.
    # Use one union integral: body, contour and shadow can overlap.
    outside=np.maximum(total-w[:,11],0)/total
    mixed=candidate&(bg|(outside>CONFIG['mixed_exterior_fraction']))
    role=np.zeros(len(w),np.uint8);role[candidate]=2;role[sh]=3;role[mixed]=4
    reasons=np.zeros(len(w),np.uint16)
    for bit,mask in [(1,body),(2,boundary),(4,strict),(8,photo&~strict),(16,bg&candidate),(32,mixed&~bg)]:reasons[mask]|=bit
    # Fixed independently rebuilt LiDAR actor is baseline identity, not a newly
    # learned identity claim. Keep it separate from original static row IDs.
    role[ns:]=1;reasons[ns:]|=64
    return dict(role=role,reason_bits=reasons,bg_weak=bg,bg_rms=bg_rms,
                shadow_effect=sh,body_evidence=body,background_protected=np.zeros(len(w),bool))

def removal(role,mode):
    if mode=='baseline':return np.zeros(len(role),bool)
    if mode=='remove_car':return role==1
    if mode=='remove_u':return role>=2
    if mode=='remove_car_u':return role>=1
    raise ValueError(mode)
