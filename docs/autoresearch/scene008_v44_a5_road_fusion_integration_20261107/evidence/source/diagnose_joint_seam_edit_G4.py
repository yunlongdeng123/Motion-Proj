"""Signed fixed-bin colour operator and DELETE-background protection; no fitting."""
from pathlib import Path
import hashlib,json,os,sys,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_v44_a5_road_fusion_integration_20261107_r1';S=B/'scene008_v44_full5_integration_20261107_r1'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');OUT=R/'G4_joint_seam_edit_operator';OBS=B/'scene008_instances_20261105_r1/observations/frames'
WORLD=B/'scene008_road_fusion_bake_20261107_r1/red_review/world_masks49_v1';RAWS=[94,105,128,137];NS=1042776;NR=143248;C=203932;EPS=5e-7
QUAL=R/'F4_bounded_appearance_qualification/qualification_manifest.json';CONTRACT=R/'contract/joint_seam_edit_safe_feasibility_v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 import numpy as np
 start=time.time();assert not OUT.exists();OUT.mkdir();(OUT/'batch_mass').mkdir()
 assert sha(CONTRACT)=='fa4b173749c5f1a9ff795b4ce5f5993fbab924b2d1bd5929b0cb86e6f1f7ea60'
 sys.path[:0]=[str(R/'source'),str(B/'scene008_v44_rgb_asset_20261107_r1/source'),str(REPO)]
 from registered_seam_pairs import seam_pairs
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.bounded_road_appearance import read_bounded_qualification
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.baseline_road_appearance import baseline_appearance_binding
 q,ids,wfull,_,_,matrix_path=read_bounded_qualification(QUAL)
 assert len(ids)==C and sha(QUAL)=='c79ad887a9f6e6dbccf77a98575c37a7f601d4e8fee6de452250d9b7ac6c4a96'
 pairs={raw:seam_pairs(raw,WORLD) for raw in RAWS};assert [len(pairs[r]) for r in RAWS]==[0,14,12,27]
 targetroot=B/'scene008_road_fusion_bake_20261107_r1/targets_edgeblend_v1';targets={};target_paths=[]
 for raw in RAWS:
  folder='projected_BUILD40' if raw in [94,128] else 'projected_HELD9';p=targetroot/folder/f'raw{raw:03d}.npz'
  with np.load(p,allow_pickle=False) as z:targets[raw]=z['edgeblend_rgb'].copy()
  target_paths.append(p)
 target_manifest=B/'scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1/target_manifest.json'
 assert sha(target_manifest)=='d7cca4167e2bbb68347f63babae356320a1b0ddd60e580bf746e966f2ac8acbe'
 registered={x['raw']:x for x in json.loads(target_manifest.read_text())['rows']}
 for raw,p in zip(RAWS,target_paths):assert sha(p)==registered[raw]['sha256'] and str(p).endswith(registered[raw]['file'])
 assert sha(R/'source/registered_seam_pairs.py')=='2f8598ccc0997e8a522316cddbb2290431253e5318c6cafc702c163c0a63b528'
 paths=[CONTRACT,QUAL,matrix_path,Path(__file__),target_manifest,R/'source/registered_seam_pairs.py',WORLD/'registry.json',WORLD/'primary_raw128/fixed_masks_raw128.npz']
 paths += [WORLD/f'{r:03d}.npz' for r in RAWS]+[OBS/f'{r:03d}.npz' for r in RAWS]+target_paths
 paths += [R/f'baseline4/frames/{r:03d}/{st}_float.npz' for r in RAWS for st in ['keep','delete']]
 inputs={str(p):ident(p) for p in paths}
 write(OUT/'operator_protocol.json',{'status':'SEALED_BEFORE_G_OPERATOR_NATIVE_CALLS','contract_sha256':sha(CONTRACT),'inputs':inputs,'candidate_rows':C,
  'raw_ids':RAWS,'states':['keep','delete'],'candidate_order':'F exact global-branch order; native_static then oldroad',
  'features':'six candidate-only ones at18:24 -> extra16:22','signed_loss':'mean(plus)-mean(minus); signed gradients are legal',
  'DELETE_loss':'SUM on actor_labels==5 OR frozen baselineKEEP group1>0.5','planned_native_forwards':22,'planned_dummy_backwards':22,
  'optimizer_updates':0,'automatic_fit':False,'automatic49':False,'shutdown':False,'reset_card':False})
 import torch
 from v44_runtime import V44Runtime
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
 torch.set_num_threads(1);assert torch.cuda.is_available()
 base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json')
 rt=SemanticActor5V44Integration(base,semantic_sidecar_path=S/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=R/'contract/full5_transform_matrix.txt')
 binding=baseline_appearance_binding(rt);assert binding==q['model_binding'];write(OUT/'model_binding_before.json',binding)
 sid=torch.as_tensor(ids[ids<NS],device='cuda');rid=torch.as_tensor(ids[ids>=NS]-NS,device='cuda')
 signed=[];j0=[];jt=[];seam_records=[];delete_records=[];delete_mass=[];batch_records=[];prepare_records=[];minimum=None;calls=0;backs=0
 for raw in RAWS:
  with np.load(OBS/f'{raw:03d}.npz',allow_pickle=False) as z:full37=np.squeeze(z['actor_labels']==5).astype(bool)
  with np.load(R/f'baseline4/frames/{raw:03d}/keep_float.npz',allow_pickle=False) as z:keep37=np.asarray(z['group_alpha_T'][1])>.5
  delete_mask=full37|keep37
  for state in ['keep','delete']:
   scene=rt.scene(raw,delete=state=='delete');original={k:v.detach() if torch.is_tensor(v) else v for k,v in scene['rows'].items()}
   a=torch.where(original['features'][:,5]==1)[0];b=torch.where(original['features'][:,4]==1)[0];assert len(a)==NS and len(b)==NR
   idx=torch.cat([a[sid],b[rid]]);assert len(idx)==C and torch.unique(idx).numel()==C
   colours=original['colors_precomp'][idx].detach().cpu().numpy().copy();assert colours.shape==(C,3) and np.isfinite(colours).all() and colours.min()>=0
   minimum=colours if minimum is None else np.minimum(minimum,colours)
   prepare_records.append({'raw':raw,'state':state,'native_forwards':0,'candidate_colours_min':float(colours.min()),'candidate_RGB_sha256':hashlib.sha256(colours.tobytes()).hexdigest()})
   with np.load(R/f'baseline4/frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:baseline={k:z[k].copy() for k in ['rgb','alpha','depth_m','group_alpha_T']}
   heads=[]
   for pair in pairs[raw]:
    info={k:v for k,v in pair.items() if k not in ['minus_mask','plus_mask']};info.update(raw=raw,state=state,operator_row=len(seam_records))
    minus=pair['minus_mask'];plus=pair['plus_mask'];assert minus.sum()==pair['minus_pixels'] and plus.sum()==pair['plus_pixels']
    heads.append({'kind':'signed_seam','pair':pair,'record':info});seam_records.append(info)
    j0.append((baseline['rgb'][:,plus].mean(1,dtype=np.float64)-baseline['rgb'][:,minus].mean(1,dtype=np.float64)))
    target=targets[raw].transpose(2,0,1);jt.append(target[:,plus].mean(1,dtype=np.float64)-target[:,minus].mean(1,dtype=np.float64))
   chunks=[heads[i:i+6] for i in range(0,len(heads),6)]
   if state=='delete':
    record={'raw':raw,'state':state,'pixels':int(delete_mask.sum()),'mask_sha256':hashlib.sha256(delete_mask.tobytes()).hexdigest(),
     'full37_pixels':int(full37.sum()),'baselineKEEP37_gt_half_pixels':int(keep37.sum()),'source':'actor_labels==5 OR baselineKEEP group1>0.5'}
    delete_records.append(record);extra={'kind':'DELETE_positive_mass','mask':delete_mask,'record':record}
    if chunks and len(chunks[-1])<6:chunks[-1].append(extra)
    else:chunks.append([extra])
   for batch,headlist in enumerate(chunks):
    kw=dict(original);assert not bool((kw['features'][:,18:24]!=0).any())
    probe=torch.ones((C,6),device=idx.device,dtype=kw['features'].dtype,requires_grad=True)
    full=torch.zeros((len(kw['features']),6),device=idx.device,dtype=probe.dtype).index_copy(0,idx,probe)
    kw['features']=torch.cat([kw['features'][:,:18],full],1)
    assert all(not v.requires_grad for k,v in kw.items() if torch.is_tensor(v) and k!='features')
    out=base.raster(scene['cam'],base.pipe,0,kw,scene['bg']);calls+=1
    write(OUT/'kernel_progress.json',{'raw':raw,'state':state,'batch':batch,'actual_native_forwards':calls,'actual_dummy_backwards':backs,'optimizer_updates':0})
    readout=out['extra'][16:22];pairerror=max(float((readout[i].detach()-readout[0].detach()).abs().max()) for i in range(1,6));assert pairerror==0
    loss=readout[0].sum()*0.;expected=[]
    for h,head in enumerate(headlist):
     if head['kind']=='signed_seam':
      plus=torch.as_tensor(head['pair']['plus_mask'],device=idx.device);minus=torch.as_tensor(head['pair']['minus_mask'],device=idx.device)
      loss=loss+readout[h][plus].mean()-readout[h][minus].mean()
      view=readout[h].detach().cpu().numpy().astype(np.float64);expected.append(float(view[head['pair']['plus_mask']].mean()-view[head['pair']['minus_mask']].mean()))
     else:
      mask=torch.as_tensor(head['mask'],device=idx.device);loss=loss+readout[h][mask].sum()
      view=readout[h].detach().cpu().numpy().astype(np.float64);expected.append(float(view[head['mask']].sum()))
    mass=torch.autograd.grad(loss,probe,retain_graph=False,create_graph=False)[0].detach().cpu().numpy();backs+=1
    write(OUT/'kernel_progress.json',{'raw':raw,'state':state,'batch':batch,'actual_native_forwards':calls,'actual_dummy_backwards':backs,'optimizer_updates':0})
    assert mass.shape==(C,6) and np.isfinite(mass).all()
    conservation=[]
    for h,head in enumerate(headlist):
     vector=mass[:,h].astype(np.float64);error=abs(float(vector.sum())-expected[h]);relative=error/max(abs(expected[h]),1e-30)
     if head['kind']=='signed_seam':assert error<=1e-6;signed.append((head['record']['operator_row'],vector))
     else:
      assert vector.min()>=0 and (relative<=1e-5 if expected[h] else vector.sum()==0);delete_mass.append((raw,vector))
     conservation.append({'kind':head['kind'],'absolute_error':error,'relative_error':relative,'expected_sum':expected[h],'measured_sum':float(vector.sum()),'negative_coefficients':int((vector<0).sum())})
    alpha=out['alpha'].detach().cpu().numpy()[0];rgb=(out['rgb']+(1-out['alpha'])*scene['env'].detach()).detach().cpu().numpy();depth=out['depth_sum'].detach().cpu().numpy()[0]/np.maximum(alpha,1e-8)/base.scale
    parity={}
    for name,value in [('rgb',rgb),('alpha',alpha),('depth_m',depth),('group_alpha_T',out['extra'][:7].detach().cpu().numpy())]:
     parity[name]=float(np.abs(value-baseline[name]).max());assert np.array_equal(value,baseline[name]),(raw,state,batch,name,parity[name])
    file=OUT/f'batch_mass/{raw:03d}_{state}_{batch:02d}.npz';np.savez_compressed(file,mass=mass)
    info={'raw':raw,'state':state,'batch':batch,'heads':[{'kind':h['kind'],'record':h['record']} for h in headlist],
     'file':str(file.relative_to(OUT)),'sha256':sha(file),'conservation':conservation,'ones_readout_pair_max_abs':pairerror,'baseline_parity_max_abs':parity}
    batch_records.append(info);write(OUT/'progress.json',{'actual_native_forwards':calls,'actual_dummy_backwards':backs,'batches':batch_records});print(json.dumps({'raw':raw,'state':state,'batch':batch,'heads':len(headlist),'calls':calls,'backs':backs,'parity':parity}),flush=True)
    del out,probe,full,kw,loss,readout;torch.cuda.empty_cache()
   rt.assert_unchanged();del scene,original
  base._data.clear()
 assert calls==backs==22 and len(signed)==len(seam_records)==len(j0)==len(jt)==106
 signed.sort(key=lambda x:x[0]);g=np.stack([x[1] for x in signed]);delete_mass.sort(key=lambda x:x[0]);Adelete=np.stack([x[1] for x in delete_mass])
 assert [x[0] for x in delete_mass]==RAWS and len(delete_records)==4
 counts=np.asarray([x['pixels'] for x in delete_records],np.float64);assert (counts>0).all()
 weights_G=np.maximum(wfull[ids],np.max(Adelete/(3.0*counts[:,None]*EPS),axis=0));lower=np.maximum(np.float32(-1),-minimum).astype(np.float32)
 assert g.shape==(106,C) and minimum.shape==lower.shape==(C,3) and np.isfinite(g).all() and np.isfinite(weights_G).all() and (weights_G>=wfull[ids]).all()
 arrays=OUT/'joint_seam_operator.npz';np.savez_compressed(arrays,candidate_row_ids=ids,signed_g=g,baseline_J=np.stack(j0),target_J=np.stack(jt),delete_mass=Adelete,weights_G=weights_G,weights_F=wfull[ids],minimum_base_RGB=minimum,delta_lower_RGB=lower)
 rt.assert_unchanged(full=True);assert binding==baseline_appearance_binding(rt)
 for path,value in inputs.items():assert ident(path)==value,path
 manifest={'schema':'joint-seam-edit-safe-operator-v1','contract_sha256':sha(CONTRACT),'raw_ids':RAWS,'states':['keep','delete'],'candidate_rows':C,
  'F_qualification_manifest_sha256':sha(QUAL),'F_matrix_sha256':sha(matrix_path),'model_binding':binding,'operator_file':arrays.name,'operator_sha256':sha(arrays),
  'seam_records':seam_records,'DELETE_protection_records':delete_records,'batch_records':batch_records,'colour_preparations':prepare_records,
  'signed_g_definition':'mean plus alphaT - mean minus alphaT; candidate order matches original F','linear_prediction':'baseline_J + signed_g @ delta',
  'weights_G_definition':'max(original F candidate weights, max DELETE_mass/(3*Npixels*5e-7))','delta_lower_definition':'maximum(-1, -minimum evaluated candidate RGB across all eight view/states)',
  'optimizer_updates':0,'original_model_updates':0,'quality_candidate':False,'automatic_fit':False,'automatic49':False}
 write(OUT/'operator_manifest.json',manifest)
 result={'status':'PASS_G_SIGNED_SEAM_AND_DELETE_PROTECTION_OPERATOR_NO_FIT','actual_native_forwards':calls,'actual_dummy_backwards':backs,'optimizer_updates':0,'original_model_updates':0,
  'candidate_rows':C,'signed_seam_rows':106,'DELETE_protection_domains':4,'weights_increased_rows':int((weights_G>wfull[ids]).sum()),'zero_weight_rows':int((weights_G==0).sum()),
  'colour_preparation_calls_without_native_forward':len(prepare_records),'operator_manifest_sha256':sha(OUT/'operator_manifest.json'),'source_inputs_and_model_binding_unchanged':True,
  'seconds':time.time()-start,'max_reserved_cuda_bytes':int(torch.cuda.max_memory_reserved()),'synthetic_actual8F_not_executed_by_this_worker':True,'automatic_fit':False,'automatic49':False,'shutdown':False,'reset_card':False}
 write(OUT/'receipt.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
