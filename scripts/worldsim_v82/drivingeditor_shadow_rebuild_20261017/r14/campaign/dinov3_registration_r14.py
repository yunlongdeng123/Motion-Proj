"""Independent real DINOv3/official IDSplat RANSAC. Never overwrites GT/points."""
from pathlib import Path
import os,json,time,sys,importlib.util,hashlib
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch,numpy as np
from PIL import Image
from transformers import AutoImageProcessor,AutoModel
torch.set_num_threads(2);torch.manual_seed(0)
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');S=R/'ownership_rebuild_r14';D=S/'idsplat_frontend';O=D/'dinov3_validation'
O.mkdir(exist_ok=False);(O/'features').mkdir();(O/'matches').mkdir()
spec=importlib.util.spec_from_file_location('official_ransac','/root/autodl-tmp/idsplat/nerfstudio/utils/ransac_utils_torch_batched.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
m=json.loads((S/'data_A_box/manifest.json').read_text());tr=np.load(S/'data_A_box/trajectory.npz');targets=[5,6,12,18,25,26,37]
before=hashlib.sha256((D/'data_IDSplat/trajectory.npz').read_bytes()).hexdigest();start=time.monotonic();bank={a:{} for a in targets}
sources={a:np.load(D/'pointsets'/f'a{a:02d}_all_candidates.npz') for a in targets}
def status(phase,**kw):(O/'status.json').write_text(json.dumps(dict(phase=phase,seconds=time.monotonic()-start,scene_training_updates=0,GT_modified=False,**kw),indent=2))
modelpath='/root/autodl-tmp/models/idsplat/dinov3-vitb16';processor=AutoImageProcessor.from_pretrained(modelpath,local_files_only=True,do_resize=False,do_rescale=True);model=AutoModel.from_pretrained(modelpath,local_files_only=True).eval().cuda()
for fi,row in enumerate(m['frames']):
    if row['split']!='train':continue
    image=Image.open(S/'data_A_box'/row['rgb']).convert('RGB');inputs={k:v.cuda() for k,v in processor(images=image,return_tensors='pt').items()}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(**inputs,output_hidden_states=True)
    f=out.hidden_states[7][:,5:,:].float();assert tuple(f.shape)==(1,600,768) and bool(torch.isfinite(f).all());np.save(O/'features'/(row['frame_id']+'.npy'),f[0].cpu().numpy().astype(np.float16))
    grid=f.transpose(1,2).reshape(1,768,20,30)
    for a in targets:
        src=sources[a];ids=np.flatnonzero(src['accepted']&(src['source_frame']==fi))
        if len(ids)<5:continue
        # Equal deterministic measured-surface sampling, without GT matching.
        _,keep=np.unique(np.floor(src['xyz'][ids]/.04).astype(np.int32),axis=0,return_index=True);ids=ids[keep]
        if len(ids)>512:ids=ids[np.random.default_rng(a*1000+fi).choice(len(ids),512,replace=False)]
        pixel=src['pixel'][ids].astype(np.float32);uv=torch.tensor((pixel+.5)/[480,320]*2-1,device='cuda',dtype=torch.float32).reshape(1,1,-1,2)
        feats=torch.nn.functional.grid_sample(grid,uv,mode='bilinear',padding_mode='border',align_corners=False).squeeze(0).squeeze(1).T
        bank[a][fi]={'ids':ids,'world':src['source_world'][ids],'feats':feats.cpu().numpy()}
    status('REAL_DINOV3_TRAIN46',frame_index=fi,frame_id=row['frame_id'])
del model,out,grid,inputs;torch.cuda.empty_cache();records=[]
for a in targets:
    frames=bank[a]
    if len(frames)<2:
        records.append(dict(actor=a,status='INSUFFICIENT_VISIBLE_BUILD_FRAMES'));continue
    canonical=max(frames,key=lambda fi:len(frames[fi]['ids']));others=sorted(fi for fi in frames if fi!=canonical)
    chosen=np.unique(np.linspace(0,len(others)-1,min(8,len(others))).round().astype(int));src=frames[canonical]
    for k in chosen:
        fi=others[k];tgt=frames[fi];P=torch.tensor(src['world'],device='cuda');Q=torch.tensor(tgt['world'],device='cuda');F=torch.tensor(src['feats'],device='cuda');G=torch.tensor(tgt['feats'],device='cuda')
        matches=official.find_feature_matches_torch(F,G,mutual=False);rec=dict(actor=a,source_frame=canonical,target_frame=fi,source_frame_id=m['frames'][canonical]['frame_id'],target_frame_id=m['frames'][fi]['frame_id'],source_points=len(P),target_points=len(Q),cosine_matches=len(matches),unique_source_matches=int(matches[:,0].unique().numel()) if len(matches) else 0)
        if rec['unique_source_matches']<3:
            rec['status']='INSUFFICIENT_DISTINCT_FEATURE_MATCHES';records.append(rec);continue
        torch.manual_seed(a*10000+fi)
        rot,trans,inliers=official.ransac_rigid_registration_torch_batched(P,Q,F,G,num_hypotheses=100000,batch_size=512,distance_threshold=.2,mutual=False,refine=True,device=torch.device('cuda'))
        if rot is None or trans is None:
            rec['status']='RANSAC_NO_TRANSFORM';records.append(rec);continue
        fitness=official.compute_registration_fitness(P,Q,rot,trans,threshold=.1,mode='tgt_coverage')
        T=np.eye(4);T[:3,:3]=rot.cpu().numpy();T[:3,3]=trans.cpu().numpy();GT=tr['poses'][fi,a]@np.linalg.inv(tr['poses'][canonical,a]);diff=T[:3,:3]@GT[:3,:3].T
        angle=float(np.degrees(np.arccos(np.clip((np.trace(diff)-1)/2,-1,1))))
        center=tr['poses'][canonical,a,:3,3];center_error=float(np.linalg.norm(T[:3,:3]@center+T[:3,3]-(GT[:3,:3]@center+GT[:3,3])))
        measured=src['world'];transfer_error=np.linalg.norm(measured@T[:3,:3].T+T[:3,3]-(measured@GT[:3,:3].T+GT[:3,3]),axis=1)
        mi=matches[inliers].cpu().numpy();pp=src['world'][mi[:,0]];qq=tgt['world'][mi[:,1]]
        def local(x,pose):return (x-pose[:3,3])@pose[:3,:3]
        world_error=np.linalg.norm(pp-qq,axis=1);actor_error=np.linalg.norm(local(pp,tr['poses'][canonical,a])-local(qq,tr['poses'][fi,a]),axis=1)
        rec.update(status='INDEPENDENT_REGISTRATION_MEASURED',target_coverage10cm=fitness,matched_inliers20cm=len(mi),rotation_vs_GT_deg=angle,vehicle_center_vs_GT_m=center_error,surface_transfer_vs_GT_median_m=float(np.median(transfer_error)),matched_world_residual_median_m=float(np.median(world_error)),matched_GT_local_residual_median_m=float(np.median(actor_error)),RANSAC_transform_world_src_to_tgt=T.tolist(),fixed_GT_world_src_to_tgt=GT.tolist(),fitness_passed_official_half=fitness>=.5)
        np.savez_compressed(O/'matches'/f'a{a:02d}_{canonical:02d}_{fi:02d}.npz',source_world=src['world'],target_world=tgt['world'],source_raw_rows=sources[a]['source_raw_row'][src['ids']],target_raw_rows=sources[a]['source_raw_row'][tgt['ids']],cosine_matches=matches.cpu().numpy(),inlier_match_indices=inliers.cpu().numpy(),RANSAC_transform=T,GT_transform=GT)
        records.append(rec);(O/'partial_results.json').write_text(json.dumps(records,indent=2));status('OFFICIAL_RANSAC_INDEPENDENT_VALIDATION',actor=a,completed_pairs=len(records));print(json.dumps({k:v for k,v in rec.items() if 'transform' not in k}),flush=True)
after=hashlib.sha256((D/'data_IDSplat/trajectory.npz').read_bytes()).hexdigest();assert before==after
report=dict(status='INDEPENDENT_VALIDATION_COMPLETE_NOT_TRAJECTORY_REPLACEMENT',records=records,seconds=time.monotonic()-start,train_feature_frames=46,QUERY_features_extracted=False,descriptor='real DINOv3 B16 layer7 minus5prefix, native bilinear no independent PCA',official_RANSAC_source='/root/autodl-tmp/idsplat/nerfstudio/utils/ransac_utils_torch_batched.py',official_parameters=dict(hypotheses=100000,batch_size=512,threshold_m=.2,mutual=False,fitness_target_radius_m=.1),GT_before_sha256=before,GT_after_sha256=after,points_and_ownership_unchanged=True,scene_training_updates=0,
            limitations='Feature correspondences may repeat on low-texture surfaces. A high target coverage alone is not proof of a correct rigid transform; compare GT rotation/center/surface errors. Near-stationary/occluded/conflicting cases remain uncertain. No bound-Gaussian motion is used.')
(O/'result.json').write_text(json.dumps(report,indent=2));status('COMPLETE',pairs=len(records));print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
