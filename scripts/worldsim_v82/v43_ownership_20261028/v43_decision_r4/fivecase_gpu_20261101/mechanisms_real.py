"""真实五例八类机制的实际证据与禁用域。未知不记作隔离完成，不生成好坏标签。"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,pathlib,sys,time
import numpy as np
import torch
from PIL import Image,ImageFilter,ImageDraw
ROOT=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def world_camera(camera,T):
    scale=np.linalg.norm(T[:3,0]);inv=np.linalg.inv(T);m=np.eye(4)
    m[:3,:3]=inv[:3,:3]@np.asarray(camera['rotation'])*scale;m[:3,3]=inv[:3,:3]@camera['position']+inv[:3,3]
    return m,np.asarray([camera['fx']*480/camera['width'],camera['fy']*320/camera['height'],camera['cx']*480/camera['width'],camera['cy']*320/camera['height']])
def project(world,camera,K):
    xyz=(world-camera[:3,3])@camera[:3,:3];valid=np.isfinite(xyz).all(1)&(xyz[:,2]>.05)
    uv=np.zeros((len(world),2));uv[valid]=xyz[valid,:2]/xyz[valid,2,None]*K[:2]+K[2:]
    pix=np.floor(uv).astype(np.int64);valid&=(pix[:,0]>=0)&(pix[:,0]<480)&(pix[:,1]>=0)&(pix[:,1]<320)
    return xyz,pix,valid
def hidden_by_boxes(points,origin,centers,rotations,dimensions):
    blocked=np.zeros(len(points),bool)
    for C,R,D in zip(centers,rotations,dimensions):
        o=(origin-C)@R;v=(points-origin)@R
        denominator=np.where(abs(v)>1e-12,v,1e-12)
        lo=(-D/2-o)/denominator;hi=(D/2-o)/denominator
        near=np.minimum(lo,hi).max(1);far=np.maximum(lo,hi).min(1)
        blocked|=(far>=np.maximum(near,0))&(near<.999)&(far>0)
    return blocked
def request_mask(frame,camera,K,q):
    C=np.asarray(frame['actor_center_world']);R=np.asarray(frame['actor_rotation_world']);D=np.asarray(frame['dimensions'])
    if q=='shift':C=C+np.asarray([0,1.5,0])
    if q=='rotate':
        a=np.pi/6;R=np.asarray([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])@R
    signs=np.asarray([[1 if i&(1<<j) else -1 for j in range(3)] for i in range(8)])
    xyz,pix,valid=project(signs*D/2@R.T+C,camera,K);image=Image.new('L',(480,320))
    points=[tuple(p) for p in pix[xyz[:,2]>.05]]
    if len(points)>=3:
        from scipy.spatial import ConvexHull
        points=np.asarray(points);hull=ConvexHull(points);ImageDraw.Draw(image).polygon([tuple(x) for x in points[hull.vertices]],fill=255)
    return np.asarray(image)>0
FOLDERS={'waymo-train-008_a37':'gpu','waymo-train-212_a0':'gpu_r2','waymo-train-212_a35':'gpu_r2','waymo-train-564_a0':'gpu_564_replay','waymo-train-564_a22':'gpu_564_replay'}

def source_data(event,context):
    src=pathlib.Path(event['source_path']);T=np.loadtxt(src/'transform_matrix.txt');scale=np.linalg.norm(T[:3,0])
    cameras={x['img_name']:x for x in read(pathlib.Path(event['model_path'])/'cameras.json')}
    state=torch.load(event['checkpoints']['training_state']['path'],map_location='cpu',weights_only=False)
    plane=np.asarray(state['sgcm']['plane_model'],dtype=float)
    ctx=read(ROOT/'sensor_inputs/manifest.json');context_id=next(x['context'] for x in ctx if x['case']==context['scene_key']+'_'+context['target_alias'])
    boxes=np.load(ROOT/'mechanism_context'/(context_id+'.npz'))
    cache={};ground=[];ground_frame=[]
    for index,f in enumerate(context['frames']):
        fid=f['frame_id'];raw=f['raw'];camera,K=world_camera(cameras[fid],T)
        cloud=np.fromfile(src/'velodyne'/(fid+'.bin'),np.float32).reshape(-1,6)
        pose=np.loadtxt(src/'pose'/(fid+'.txt'));world=cloud[:,:3].astype(np.float64)@pose[:3,:3].T+pose[:3,3]
        cv,pix,valid=project(world,camera,K);ids=np.flatnonzero(valid)
        order=ids[np.argsort(cv[ids,2],kind='stable')];flat=pix[order,1]*480+pix[order,0]
        _,first=np.unique(flat,return_index=True);ids=order[first];flat=pix[ids,1]*480+pix[ids,0]
        depth=np.full(320*480,np.nan);depth[flat]=cv[ids,2]
        model=world[ids]@T[:3,:3].T+T[:3,3]
        close=np.abs(model@plane[:3]+plane[3])/np.linalg.norm(plane[:3])/scale<=.10
        candidates=world[ids][close]
        b=boxes['raw']==raw
        if len(candidates):
            unobstructed=~hidden_by_boxes(candidates,camera[:3,3],boxes['center'][b],boxes['rotation'][b],boxes['dimensions'][b]);candidates=candidates[unobstructed]
        ground.append(candidates);ground_frame.append(np.full(len(candidates),index,np.int16))
        cache[fid]=dict(depth=depth.reshape(320,480),world=world[ids],flat=flat,camera=camera,K=K,raw=raw,index=index)
    points=np.concatenate(ground);frames=np.concatenate(ground_frame)
    if not len(points):
        return dict(frames=cache,ground=np.empty((0,3)),ground_source_frames=np.empty(0,np.int16),T=T,scale=scale,
            background_support_role='no observed ground witnesses produced; no dense background conclusion',
            limitations=['Registered49frames only; absence is unknown, not proof of never observed background'])
    voxel=np.floor(points/.10).astype(np.int64);_,inverse=np.unique(voxel,axis=0,return_inverse=True)
    distinct=np.unique(np.stack([inverse,frames],1),axis=0);count=np.bincount(distinct[:,0],minlength=int(inverse.max())+1)
    keep=count[inverse]>=2
    # 体素内保留首、末观测，查询时排除同帧；不从未见部分外推颜色。
    ids=np.flatnonzero(keep);_,a=np.unique(inverse[ids],return_index=True);_,z=np.unique(inverse[ids][::-1],return_index=True)
    representatives=np.unique(np.r_[ids[a],ids[::-1][z]])
    return dict(frames=cache,ground=points[representatives],ground_source_frames=frames[representatives],T=T,scale=scale,
        background_support_role='sparse repeated ground-plane proxy, vehicle-box ray-occlusion checked; not certified dense road/radiance GT',
        limitations=['Only the registered49source frames; lack of witness is unknown, not never observed in all199frames or the physical world',
            'Ground plane is inherited measured-data fit, not road semantics GT; pedestrians/unlabelled occluders, rolling shutter and radiance stability unresolved'])

def main():
    out=ROOT/'mechanisms_real';out.mkdir(exist_ok=False);cases=[];ledger=[];cache={}
    prior={r['case']:r for r in read(ROOT/'mechanism_context/prior_source_audit.json')}
    specifications={s['id']:s for s in read(ROOT/'cpu/run_contract.json')['checkpoint_inputs']}
    for case,folder in FOLDERS.items():
        context=read(ROOT/'cpu'/case/'camera_track_context.json');context['target_alias']=case.rsplit('_',1)[1]
        event=read(specifications[case]['event']);key=event['source_path']
        if key not in cache:cache[key]=source_data(event,context)
        source=cache[key];dst=out/case;dst.mkdir();maskdir=dst/'regions';maskdir.mkdir()
        technical=read(ROOT/'mechanisms_gpu'/case/'technical_and_component_checks.json')
        original=ROOT/folder/case/'audit49/original_none';metrics=[];envmetrics=[]
        for f in context['frames']:
            fid=f['frame_id'];obs=source['frames'][fid];body=np.asarray(Image.open(f['mask_ref']).convert('L').resize((480,320),Image.Resampling.NEAREST))>0
            with np.load(original/(fid+'.npz')) as a:
                alpha=a['alpha'][0];native_depth=np.divide(a['depth_sum_model'][0],alpha,out=np.full_like(alpha,np.nan),where=alpha>.01)/source['scale']
                rgb=a['final'].copy();cg=a['C_G'];fill=a['fill'];E=a['E']
                closure=float(abs(a['final']-cg-fill).max())
            point=obs['world'];local=(point-np.asarray(f['actor_center_world']))@np.asarray(f['actor_rotation_world'])
            in_box=(abs(local)<=np.asarray(f['dimensions'])/2).all(1)
            target=np.zeros(320*480,bool);target[obs['flat'][in_box]]=True;target=target.reshape(320,480)
            measured=body&target&np.isfinite(obs['depth']);valid=measured&np.isfinite(native_depth)
            difference=np.abs(native_depth-obs['depth']);agree=valid&(difference<=.30);disagree=valid&(difference>.30)
            observed=np.asarray(Image.open(ROOT/'mechanisms_gpu'/case/'observed'/(fid+'.png')),dtype=float)/255
            color=np.abs(np.moveaxis(rgb,0,-1)-observed).mean(2);appearance=agree&(color>10/255)
            points=source['ground'];which=source['ground_source_frames']!=obs['index'];points=points[which]
            cv,pix,v=project(points,obs['camera'],obs['K'])
            # 只登记通过当前目标盒后方的历史地面实测点；不是从图像补全的道路。
            ids=np.flatnonzero(v);behind=hidden_by_boxes(points[ids],obs['camera'][:3,3],np.asarray([f['actor_center_world']]),np.asarray([f['actor_rotation_world']]),np.asarray([f['dimensions']]))
            ids=ids[behind];witness=np.zeros((320,480),bool);witness[pix[ids,1],pix[ids,0]]=True;witness&=body
            unknown_background=body&~witness
            outer=np.asarray(Image.fromarray(body.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(31)))>0;outer&=~body
            rotate=request_mask(f,obs['camera'],obs['K'],'rotate');shift=request_mask(f,obs['camera'],obs['K'],'shift')
            np.savez_compressed(maskdir/(fid+'.npz'),observed_target_depth_proxy=measured,mean_depth_disagreement_proxy=disagree,
                appearance_error_with_mean_depth_agreement=appearance,historical_ground_witness_proxy=witness,
                background_without_verified_witness=unknown_background,shadow_reflection_review_envelope=outer,
                requested_rotate_envelope=rotate,requested_shift_envelope=shift,environment_fill_nonzero=(abs(fill).max(0)>1/255),
                ownership_training_label_allowed=np.zeros((320,480),bool))
            metrics.append(dict(frame=fid,raw=f['raw'],body_pixels=int(body.sum()),target_depth_proxy_pixels=int(measured.sum()),
                native_mean_depth_comparable_pixels=int(valid.sum()),mean_depth_abs_error_median_m=float(np.median(difference[valid])) if valid.any() else None,
                mean_depth_abs_error_p90_m=float(np.quantile(difference[valid],.9)) if valid.any() else None,
                depth_agreement_proxy_pixels=int(agree.sum()),appearance_error_in_agreement_proxy_pixels=int(appearance.sum()),
                mean_RGB_error_on_depth_agreement=float(color[agree].mean()) if agree.any() else None,
                historical_ground_witness_pixels=int(witness.sum()),old_body_without_verified_witness=int(unknown_background.sum()),
                dense_background_GT_available=False,ownership_training_label_pixels=0))
            envmetrics.append(dict(frame=fid,none_closure_max_abs=closure,none_body_CG_abs_mean=float(abs(cg[:,body]).mean()),none_body_fill_abs_mean=float(abs(fill[:,body]).mean())))
        # 原生删除三锚：实际拆开点贡献与环境填充，绝不把残留颜色统一当车体。
        gpu=ROOT/'gpu_fullsource'/case/'source_enum';manifest=read(gpu/'gpu_manifest.json');delete_records=[]
        for row in manifest['rows']:
            if row['edit']!='delete' or row['action_id'] not in ['h000',manifest['allrebind_action_id']]:continue
            body=np.asarray(Image.open(next(f['mask_ref'] for f in context['frames'] if f['frame_id']==row['frame'])).convert('L').resize((480,320),Image.Resampling.NEAREST))>0
            with np.load(row['fp32']) as a,np.load(original/(row['frame']+'.npz')) as b:
                delete_records.append(dict(frame=row['frame'],route=row['action_id'],body_pixels=int(body.sum()),
                    CG_abs_mean=float(abs(a['C_G'][:,body]).mean()),fill_abs_mean=float(abs(a['fill'][:,body]).mean()),
                    environment_E_vs_original_max_abs=float(abs(a['E']-b['E']).max()),closure_max_abs=float(abs(a['final']-a['C_G']-a['fill']).max()),
                    semantic_ghost_attribution='not established by mean amplitude'))
                for field in ['final','C_G','fill','E']:
                    im=np.rint(np.clip(np.moveaxis(a[field],0,-1),0,1)*255).astype(np.uint8)
                    Image.fromarray(im).save(dst/(row['frame']+'_'+row['action_id']+'_delete_'+field+'.png'))
        responsibility=np.load(ROOT/folder/case/'audit49/full49_row_responsibility.npz');b=responsibility['body_alphaT'];k=responsibility['keep_alphaT']
        dual=int(((b>1e-4)&(k>1e-4)).sum())
        meta=prior[case]
        evidence=dict(case=case,frames=metrics,component_checks=envmetrics,delete_components=delete_records,
            dual_body_outside_rows=dual,dual_domain_is_not_multisurface_proof=True,
            source_view_geometry={k:meta.get(k) for k in ['calibrated_original_view_pair_max_deg','rotate30_worldZ_nearest_original_direction_deg','shift_worldY1p5m_nearest_original_direction_deg']},
            technical_motion=technical['all_technical_transform_checks_passed'],
            historical_ground_observation_proxy=dict(points=len(source['ground']),role=source['background_support_role'],limitations=source['limitations']),
            thresholds_role='0.30m and10/255 are descriptive inspection groups only; no P threshold or ownership labels',
            shape_color_causally_separated_in_real=False,complete_mechanism_isolation=False,training_labels_generated=0)
        write(dst/'evidence.json',evidence);cases.append(evidence)
        descriptions={
          4:('partially_checked','原始none的稀疏目标深度与原生平均深度逐帧对照；不是首交面、完整形状或几何因果鉴定','深度不一致/无可信深度的车体区不能作为归属好坏标签'),
          5:('partially_checked','在平均深度近似一致的观测像素另记RGB误差；仍可能有光照、反射、遮挡与SH外观影响','颜色残差不能单独生成错误归属或换绑定能修色的标签'),
          6:('projective_conflict_checked_physical_cause_unresolved','实际原生伴随证明有行同时影响车身与其外域；未证明两独立物理表面','跨域行及其影响区保持歧义，不能从二元带走/留下失效断言选择器选错'),
          7:('not_causally_tested','未完成真实阴影/反射来源识别，也无物理受控光照验证','车身外暗影/反射待核包围域不生成漏删车身标签'),
          8:('partially_checked','实际跨时地面实测点、注册车辆盒射线遮挡与稀疏历史见证；非稠密道路或辐射真值','无经核实历史见证的显露区禁用于正确/错误删车奖励；有地面代理见证也未获完整RGB监督'),
          9:('components_numerically_separated_semantics_unresolved','三维C_G、环境E、透射fill逐项拆开并核对合成；未自动鉴别车形语义','环境填充造成的差异不算归属错误；点贡献内的残留仍需进一步辨因'),
          11:('view_support_checked_surface_completeness_unresolved','实际49帧观察方向与请求方向距离已测；无完整新面的独立证据','旋转/平移新显露面无可靠支持时，不把缺面/外观错误用于归属好坏标签'),
          12:('technical_execution_checked_binding_correctness_in_scope','实际样本行的跨帧/跨操作变换、删除及上下文保留核对；未证明登记身份本身正确','只有坐标/时间接线无效才作工程排除；同批点跟错车辆仍属本轮核心责任，不能免责')}
        for category,(state,checked,excluded) in descriptions.items():
            ledger.append(dict(case=case,category=category,check_state=state,checks_actually_executed=checked,
                unresolved='见逐例原始证据与本类状态；不能把unknown记作已完成',excluded_from_ownership_judgement=excluded,
                evidence=str(dst/'evidence.json'),region_masks=str(maskdir),ownership_training_label_permission=False,
                repair_required_this_round=False))
        print(case,'mechanism_evidence_complete','ground_witness_pixels',sum(x['historical_ground_witness_pixels'] for x in metrics),flush=True)
    sim=read(ROOT/'sim_factor_controls/summary.json')
    write(out/'mechanism_ledger.json',dict(rows=ledger,case_count=5,category_count=8,record_count=40,
        all_mechanisms_isolated=False,controlled_simulation_validation=sim,
        gate=dict(training_good_bad_answers_allowed=False,reason='uncalibrated validity domains, unresolved real confounds and incomplete controlled validation',
            generated_training_preference_pairs=0,generated_ownership_labels=0),human_verdict=None))
    write(out/'summary.json',dict(cases=[dict(case=x['case'],measured_depth_pixels=sum(f['target_depth_proxy_pixels'] for f in x['frames']),
        depth_disagreement_pixels=sum(f['native_mean_depth_comparable_pixels']-f['depth_agreement_proxy_pixels'] for f in x['frames']),
        appearance_confounded_pixels=sum(f['appearance_error_in_agreement_proxy_pixels'] for f in x['frames']),
        sparse_historical_ground_witness_pixels=sum(f['historical_ground_witness_pixels'] for f in x['frames']),
        dual_body_outside_rows=x['dual_body_outside_rows'],technical_motion=x['technical_motion']) for x in cases],
        all_mechanisms_isolated=False,training_labels_generated=0))
if __name__=='__main__':main()
