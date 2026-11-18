"""CPU evaluation of saved actual native-basis packets. No Torch, renderer or training import."""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy import ndimage as ndi

FRAMES=(86,110,134,179,198)
EXPECTED=dict(A=16904,C=10830,S=3131,S_full=3187,S_old=2222,S_new=909,S_thin=56,D=38416,K=77266)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def masked_hf(image,mask,sigma):
    weight=ndi.gaussian_filter(mask.astype(np.float64),sigma,mode='constant',cval=0,truncate=3.)
    low=np.stack([ndi.gaussian_filter(c*mask,sigma,mode='constant',cval=0,truncate=3.)for c in image])
    return image-low/np.maximum(weight,1e-12)[None],mask&(weight>=.9),weight


def gradient(image):
    luma=np.einsum('c,chw->hw',np.array([.2126,.7152,.0722]),image)
    return np.stack([ndi.sobel(luma,axis=1,mode='nearest')/8.,ndi.sobel(luma,axis=0,mode='nearest')/8.])


def detail_stats(pred,target,mask):
    """Mask-aware detail; unsupported boundaries remain explicit, never become clean-road GT."""
    result=dict(full_domain_pixels=int(mask.sum()),scales={})
    for sigma in (1.,2.):
        hp,valid,mass=masked_hf(pred,mask,sigma)
        ht,_,_=masked_hf(target,mask,sigma)
        n=int(valid.sum())
        if n:
            x,y=hp[:,valid],ht[:,valid];e=x-y
            nx,ny=np.linalg.norm(x),np.linalg.norm(y)
            value=dict(qualified_pixels=n,unsupported_boundary_pixels=int(mask.sum())-n,
                HF_MAE=float(np.abs(e).mean()),HF_RMSE=float(np.sqrt(np.mean(e*e))),
                HF_error_abs_sum=float(np.abs(e).sum()),HF_error_squared_sum=float((e*e).sum()),
                HF_predicted_squared_sum=float((x*x).sum()),HF_teacher_squared_sum=float((y*y).sum()),HF_dot_sum=float((x*y).sum()),
                predicted_HF_RMS=float(np.sqrt(np.mean(x*x))),teacher_HF_RMS=float(np.sqrt(np.mean(y*y))),
                HF_cosine=float((x*y).sum()/(nx*ny))if nx*ny>1e-15 else None,
                HF_energy_ratio=float((x*x).sum()/max(float((y*y).sum()),1e-30)))
        else:value=dict(qualified_pixels=0,unsupported_boundary_pixels=int(mask.sum()),HF_MAE=None,HF_RMSE=None,HF_cosine=None)
        result['scales'][str(int(sigma))+'px']=value
    gp,gt=gradient(pred),gradient(target)
    good=ndi.binary_erosion(mask,structure=np.ones((3,3)),border_value=0)
    target_mag=np.sqrt((gt*gt).sum(0));pred_mag=np.sqrt((gp*gp).sum(0))
    te=good&(target_mag>=4/255);pe=good&(pred_mag>=4/255)
    matched_p=int((pe&ndi.binary_dilation(te,structure=np.ones((3,3)))).sum())
    matched_t=int((te&ndi.binary_dilation(pe,structure=np.ones((3,3)))).sum())
    nt,npred=int(te.sum()),int(pe.sum())
    precision=matched_p/npred if npred else None;recall=matched_t/nt if nt else None
    F1=2*precision*recall/(precision+recall)if precision is not None and recall is not None and precision+recall>0 else (0. if nt else None)
    hf,flat_valid,_=masked_hf(pred,mask,1.);target_hf,_,_=masked_hf(target,mask,1.)
    flat=flat_valid&(target_mag<=1/255)&(np.abs(target_hf).max(0)<=2/255)
    result['edges']=dict(qualified_pixels=int(good.sum()),unsupported_boundary_pixels=int(mask.sum()-good.sum()),
        threshold_per_pixel=4/255,tolerance_pixels=1,target_edges=nt,predicted_edges=npred,matched_predicted=matched_p,matched_teacher=matched_t,
        precision=precision,recall=recall,F1=F1,
        gradient_L1=float(np.abs(gp[:,good]-gt[:,good]).mean())if good.any()else None,
        teacher_flat_pixels=int(flat.sum()),spurious_HF_RMS_on_teacher_flat=float(np.sqrt(np.mean(hf[:,flat]**2)))if flat.any()else None,
        teacher_flat_HF_RMS=float(np.sqrt(np.mean(target_hf[:,flat]**2)))if flat.any()else None)
    return result


def photo(pred,target,mask):
    n=int(mask.sum())
    if not n:return dict(pixels=0,MAE=None,RMSE=None,abs_sum=0.,squared_sum=0.)
    e=pred[:,mask]-target[:,mask]
    return dict(pixels=n,RGB_denominator=3*n,MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt(np.mean(e*e))),
                abs_sum=float(np.abs(e).sum()),squared_sum=float((e*e).sum()))


def packet_metrics(packet,frozen):
    p=np.asarray(packet['road_rgb']);a=np.asarray(packet['road_alpha']).squeeze();j=np.asarray(packet['joint_rgb'])
    if p.shape!=(3,320,480)or j.shape!=p.shape or a.shape!=(320,480):raise ValueError('native packet CHW/HW shape mismatch')
    if p.dtype.kind!='f'or j.dtype.kind!='f'or a.dtype.kind!='f':raise ValueError('metrics require original float packets, never RGB8 previews')
    if not all(np.isfinite(x).all()for x in [p,a,j]):raise ValueError('nonfinite native packet')
    p,j=p.astype(np.float64),j.astype(np.float64)
    alpha=frozen['baseline_road_alpha'].astype(np.float64)
    K=frozen['K'];delta=np.abs(j-frozen['baseline_joint_rgb'].astype(np.float64)).max(0)
    checks=dict(alpha_exact=np.array_equal(a,frozen['baseline_road_alpha']),
        joint_alpha_present='joint_alpha'in packet,
        joint_alpha_exact='joint_alpha'in packet and np.array_equal(packet['joint_alpha'].squeeze(),frozen['baseline_joint_alpha']),
        K_mask_present='K'in packet,K_mask_exact='K'in packet and np.array_equal(packet['K'],K),
        joint_K_exact=float(delta[K].max())==0)
    out=dict(checks=checks,alpha_max_change=float(np.abs(a-alpha).max()),K_pixels=int(K.sum()),joint_K_RGB_max=float(delta[K].max()),
        road_only_K_RGB_delta_max=float(np.abs(p-frozen['baseline_road_rgb']).max(0)[K].max()),
        D=dict(pixels=int(frozen['D'].sum()),gap=int((frozen['D']&(a<.05)).sum()),
            thin=int((frozen['D']&(a>=.05)&(a<=.9)).sum()),supported=int((frozen['D']&(a>.9)).sum())),
        material={},detail={})
    for region,target_key in [('A','A'),('A_source_available','A'),('A_readonly_floor','A'),('A_native_zero_eligible_weight','A'),('C','C'),('S','S'),
                              ('S_full','S'),('S_old','S'),('S_new','S'),('S_thin','S')]:
        if region not in frozen:continue
        target=alpha[None]*frozen['target_'+target_key].astype(np.float64)
        out['material'][region]=photo(p,target,frozen[region])
        if region in ('A','C','S','S_new'):
            out['detail'][region]=detail_stats(p,target,frozen[region])
    V,H=frozen['stripe_vertical'],frozen['stripe_horizontal'];D=frozen['stripe_domain']
    out['stripe']=dict(vertical_E=float(np.abs(p[:,2:]-2*p[:,1:-1]+p[:,:-2])[:,V].mean()),
        horizontal_E=float(np.abs(p[:,:,2:]-2*p[:,:,1:-1]+p[:,:,:-2])[:,H].mean()),
        fixed_domain_pixels=int(D.sum()),vertical_stencils=int(V.sum()),horizontal_stencils=int(H.sum()),
        RGB_delta_energy=float(np.mean((p-frozen['baseline_road_rgb'])[:,D]**2)))
    for key in ['reader_source_mass','fallback_mask']:
        if key in packet:
            value=packet[key];out[key]=dict(min=float(value.min()),max=float(value.max()),zero_pixels=int((value==0).sum()))
    return out


def evaluate_arm(render_map,frozen_manifest,output_dir=None,*,arm_id='M11',stage='C800',public_reload=None):
    """evaluate_arm({86: float_npz,...}, manifest, outdir, arm_id=..., stage=...)->JSON dict."""
    manifest=read(frozen_manifest)if not isinstance(frozen_manifest,dict)else frozen_manifest
    mapping=read(render_map)if isinstance(render_map,(str,Path))else render_map
    rows=[];counts={k:0 for k in EXPECTED};checks={};frozen={};packets={}
    for raw in FRAMES:
        item=manifest['frames'][str(raw)];f=load(item['path']);frozen[raw]=f
        if sha(item['path'])!=item['sha256']:raise ValueError('frozen evaluation capsule changed')
        path=Path(mapping.get(str(raw),mapping.get(f'{raw:03d}',mapping.get(raw))))
        z=load(path);packets[raw]=z
        row=packet_metrics(z,f);row.update(raw=raw,view_role='teacher_source'if raw in (86,110)else'diagnostic_not_blind',source=str(path),source_sha256=sha(path))
        rows.append(row)
        checks.update({f'{raw}:{k}':v for k,v in row['checks'].items()})
        for k in counts:counts[k]+=int(f[k].sum())if k in f else 0
    if counts!=EXPECTED:raise ValueError('fixed photo/coverage/K denominators changed: '+str(counts))
    pooled={}
    for key in rows[0]['material']:
        values=[r['material'][key]for r in rows if key in r['material']]
        n=sum(v['pixels']for v in values)
        pooled[key]=dict(pixels=n,RGB_denominator=3*n,MAE=sum(v['abs_sum']for v in values)/(3*n)if n else None,
            RMSE=float(np.sqrt(sum(v['squared_sum']for v in values)/(3*n)))if n else None)
    detail={}
    for region in ['A','C','S','S_new']:
        detail[region]={}
        for scale in ['1px','2px']:
            values=[r['detail'][region]['scales'][scale]for r in rows if region in r['detail']]
            n=sum(v['qualified_pixels']for v in values)
            detail[region][scale]=dict(qualified_pixels=n,unsupported_boundary_pixels=sum(v['unsupported_boundary_pixels']for v in values),
                HF_MAE=sum(v.get('HF_error_abs_sum',0)for v in values)/(3*n)if n else None,
                HF_RMSE=float(np.sqrt(sum(v.get('HF_error_squared_sum',0)for v in values)/(3*n)))if n else None)
            px=sum(v.get('HF_predicted_squared_sum',0)for v in values);ty=sum(v.get('HF_teacher_squared_sum',0)for v in values)
            detail[region][scale].update(HF_cosine=sum(v.get('HF_dot_sum',0)for v in values)/np.sqrt(px*ty)if px*ty>1e-30 else None,
                HF_energy_ratio=px/ty if ty>1e-30 else None,predicted_HF_RMS=float(np.sqrt(px/(3*n)))if n else None,teacher_HF_RMS=float(np.sqrt(ty/(3*n)))if n else None)
        e=[r['detail'][region]['edges']for r in rows if region in r['detail']]
        nt=sum(v['target_edges']for v in e);npred=sum(v['predicted_edges']for v in e)
        mp=sum(v['matched_predicted']for v in e);mt=sum(v['matched_teacher']for v in e)
        precision=mp/npred if npred else None;recall=mt/nt if nt else None
        detail[region]['edges']=dict(target_edges=nt,predicted_edges=npred,precision=precision,recall=recall,
            F1=2*precision*recall/(precision+recall)if precision is not None and recall is not None and precision+recall>0 else(0. if nt else None),
            qualified_pixels=sum(v['qualified_pixels']for v in e),unsupported_boundary_pixels=sum(v['unsupported_boundary_pixels']for v in e))
        flat_n=sum(v['teacher_flat_pixels']for v in e)
        detail[region]['edges'].update(teacher_flat_pixels=flat_n,
            spurious_HF_RMS_on_teacher_flat=float(np.sqrt(sum((v['spurious_HF_RMS_on_teacher_flat']or 0)**2*v['teacher_flat_pixels']for v in e)/flat_n))if flat_n else None,
            teacher_flat_HF_RMS=float(np.sqrt(sum((v['teacher_flat_HF_RMS']or 0)**2*v['teacher_flat_pixels']for v in e)/flat_n))if flat_n else None)
    result=dict(schema='teacher_detail_saved_native_CPU_metrics.v1',arm_id=arm_id,stage=stage,rows=rows,pooled=pooled,detail=detail,
        evaluator_source_sha256=sha(__file__),frozen_manifest_sha256=sha(frozen_manifest)if not isinstance(frozen_manifest,dict)else None,
        fixed_denominators=counts,mechanical_checks=checks,mechanical_pass=all(checks.values()),
        metric_space='native premult sRGB P versus FIXED baseline alpha*T; q=1, binary-mask/channel pooled means; no P/alpha photo or HF metric',
        detail_policy='per-domain mask-aware Gaussian sigma1/2px with localmass>=.9; unsupported counts retained; edge/F1 and teacher-flat noise reported separately',
        teacher_roles='A measured visible; C/S frozen pseudo, not hidden-road GT',
        diagnostic_views=[134,179,198],inherited199BUILD_not_blind=True,
        evaluator_GPU_operations=0,evaluator_native_calls=0,evaluator_training_updates=0,
        public_reload=read(public_reload)if isinstance(public_reload,(str,Path))else public_reload,
        reload_status='separate_native_receipt'if public_reload is not None else'pending_not_inferred_from_image_metrics',
        actual_joint_road_visibility='requires actual eligible joint alphaT basis; road-only alpha is not joint visibility',
        human_verdict=None,production_promoted=False)
    if manifest.get('crossview'):
        result['crossview']=evaluate_crossview(packets,frozen,manifest['crossview'])
    if output_dir is not None:
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
        if(out/'metrics.json').exists():raise FileExistsError('preserve prior stage metrics')
        write(out/'metrics.json',result)
    return result


def evaluate_crossview(packets,frozen,spec):
    # The actual fixed reprojection implementation is installed by the frozen preparer, never queried from a learned field.
    from .crossview import evaluate_crossview_packets
    return evaluate_crossview_packets(packets,frozen,spec)
