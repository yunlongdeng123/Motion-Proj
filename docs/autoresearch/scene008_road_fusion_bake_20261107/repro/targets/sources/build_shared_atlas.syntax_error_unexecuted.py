"""一次固定世界chart screened-Poisson目标；纯CPU，不修改几何或旧场景。

prepare先写合同再汇集190个非HELD实拍见证。finalize只读共享基底缓存，
构建可靠外环硬边界、解一次三通道线性系统，最后输出同域A/B双状态共享目标。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='1'
from pathlib import Path
import argparse,hashlib,json,time,warnings
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates,distance_transform_edt
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign');RUN=CAM/'scene008_road_fusion_bake_20261107_r1';OUT=RUN/'targets'
OLD=CAM/'scene008_road_r0_v44_20261107_r1';PREV=CAM/'scene008_road_seam_contact_20261107_r1'
SOURCE=OLD/'surface/frontfill_v2/road_reference_199.npz'
CORE=PREV/'engineer/inputs/new_road_canonical_candidate.npz'
BASE=RUN/'engineer/material_fit_v1/baseline_cache'
HELD=(89,95,101,105,113,121,125,130,137);BUILD=tuple(x for x in range(89,138) if x not in HELD)
CELL=.08;HALO=.4;SCREEN=(CELL/HALO)**2;THRESH=.03

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(2**20),b''):h.update(a)
 return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def load(p):
 with np.load(p,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}
def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def srgb(x):return np.where(x<=.0031308,x*12.92,1.055*np.maximum(x,0)**(1/2.4)-.055)
def stats(x):
 x=np.asarray(x,dtype=float)
 return dict(count=int(x.size),mean=float(x.mean()) if x.size else None,p95=float(np.quantile(x,.95)) if x.size else None,max=float(x.max()) if x.size else None)
def sample(a,uv):
 coords=np.stack([uv[:,1],uv[:,0]])
 if a.ndim==3:return np.stack([map_coordinates(ch,coords,order=1,mode='nearest') for ch in a],1)
 return map_coordinates(a,coords,order=1,mode='nearest')

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'atlas_contract.json').exists():raise FileExistsError('preserve first frozen atlas contract')
 contract=dict(schema='WORLD_SHARED_ROAD_ATLAS_POISSON_V1',task_run=RUN.name,
  roles={'source':'Immutable geometrically qualified original source_valid/source_rgb,190 raws=0..198 excludingHELD9; no source model/render RGB used for interior.',
   'core':'Frozen parent10098 roadcell keys and measured heights; n2+AA geometry remains unchanged.',
   'boundary':'KEEP shared baseline render only on qualified outside-core halo, screened against actual RGB; never blind-trust dark native static.',
   'A':'Unfused shared190-source chart projected to cameras. Outside common replacement domain, actual rawRGB.',
   'B':'Same source/geometry/domain, one screened-Poisson fused chart projected to cameras; outside domain same actual rawRGB.',
   'observation_rgb':'Original real rawRGB for evaluation, not a hidden deleted-road label.'},
  BUILD40=list(BUILD),HELD9=list(HELD),atlas_source_raws=[r for r in range(199) if r not in HELD],
  HELD_policy='HELD9 not used in atlas statistics, gradients or rendered boundary. Project/evaluate only. Historical original199 proxy was previously seen; it is not reused as chart colour.',
  geometry='Existing8cm measured-support candidate grid; core plus<=.4m Euclidean outside halo. No inferred/optimized depth, no road motion.',
  RGB='encoded sRGB[0,1], matching native fitting space; shared apparent radiance, not intrinsic albedo/PBR.',
  source_rule='Reuse established fixed linear-light q75 cluster [-.025,+.06], median; accept maxRGB linear residual<.035,>=3 agreeing raw ids with span>=6. No old finite brightness exclusion, preserving real bright markings. Unsupported values never become measured GT.',
  gradients='4-neighbour edge: component median of encoded-sRGB differences from same raw accepted at both endpoints when>=2 shared witnesses. Otherwise source-chart gradient (or0 if missing), weakweight.25 and marked synthetic. Observed edge weight=min(shared_count/5,1). No per-frame free colours.',
  boundary_rule={'outside_core_only':True,'halo_m':HALO,'source_agree_current_BUILD':True,'projected_box_and_vehicle_guards_clear':True,
    'newroad_alphaT_max':.05,'maxchannel_render_vs_true_source_RGB':THRESH,'maxchannel_source_vs_shared_chart_RGB':THRESH,
    'minimum_distinct_BUILD_views':3,'minimum_raw_span':6,'target':'component median of accepted KEEP full-render RGB; hard Dirichlet on reliable exterior nodes'},
  Poisson={'screen_lambda':SCREEN,'confidence':'min(agree_count/5,1)*agree_count/max(valid_count,1),0 if unsupported',
    'objective':'sum_edge w||Tj-Ti-gij||² + .04 sum_node c||Ti-Si||²; reliable outer nodes hard=Tboundary',
    'solver':'Sparse symmetric system once per3channels; no parameter search.',
    'no_anchor_component':'No source or boundary support: unavailable, not a zero observation.',
    'range_policy':'Record raw solution range/residual; deterministic clip to[0,1] if needed, count clip and recompute equation/gradient residual. No silent clipping.'},
  projection='One deterministic bilinear sampler on shared chart, actual camera transforms, iterative ray intersection with existing measured grid height. Four supported corners required; reject available measured-depth disagreement>.3m; original fixed fitmask remains unchanged.',
  image_interface={'BUILD':'projected_BUILD40/rawNNN.npz','HELD':'projected_HELD9/rawNNN.npz',
    'keys':['direct_rgb','fusion_rgb','observation_rgb','fit_mask','protect_mask','protect_rgb','fusion_replace_mask','atlas_support_mask','source_count_map','source_confidence_map','imputed_mask','fusion_ready'],
    'state_axis':False,'KEEP_DELETE_identical':True,'fit_mask':'Exact old714546 BUILD40 true-road/guard pixels; HELD same rule for evaluation only.',
    'projection_target_semantics':'Desired complete joint-image RGB on legal observed-road pixels; not roadRGBT, no division by newroadW, not hiddenGT.'},
  development_boundary='Same scene development; raw128 is BUILD; no blind/independent/held generalization claim.',
  seed='deterministic, no RNG',GPU_extra_forwards=0,baseline_forwards='reuse engineer prequal80F, no duplication',
  failure_ledger_refs=['V82-F13'],failure_ledger_delta='pending',human_verdict=None,
  immutable_sources={str(SOURCE):sha(SOURCE),str(CORE):sha(CORE),str(Path(__file__).resolve()):sha(__file__)})
 write(OUT/'atlas_contract.json',contract)
 a=load(SOURCE);core=load(CORE);keys=a['keys'].astype(int);lookup={tuple(k):i for i,k in enumerate(keys)}
 coreids=np.array([lookup[tuple(k)] for k in core['keys']],int);distance=cKDTree(core['local_xyz'][:,:2]).query(a['local_xyz'][:,:2],k=1)[0]
 chosen=a['geometry_measured_support'].astype(bool)&(distance<=HALO+1e-6);chosen[coreids]=True;ids=np.flatnonzero(chosen)
 keys=keys[ids];xyz=a['local_xyz'][ids];coremask=np.isin(ids,coreids);raws=a['source_raw'].astype(int);take=np.array([i for i,r in enumerate(raws) if r not in HELD]);source_raws=raws[take]
 valid=a['source_valid'][take][:,ids].astype(bool);rgb=a['source_rgb'][take][:,ids].astype(np.float32)
 observed=linear(np.where(valid[:,:,None],rgb,0));luma=observed@np.array([.2126,.7152,.0722],np.float32)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  q=np.nanquantile(np.where(valid,luma,np.nan),.75,axis=0)
  upper=valid&(luma>=q[None]-.025)&(luma<=q[None]+.06)
  proto=np.nanmedian(np.where(upper[:,:,None],observed,np.nan),axis=0)
 finite=np.isfinite(proto).all(1);proto=np.nan_to_num(proto,nan=0)
 agree=valid&(abs(observed-proto[None]).max(2)<.035)
 count=agree.sum(0);vcount=valid.sum(0);first=np.where(agree,source_raws[:,None],999).min(0);last=np.where(agree,source_raws[:,None],-1).max(0)
 supported=finite&(count>=3)&(last-first>=6);confidence=np.minimum(count/5,1)*count/np.maximum(vcount,1);confidence[~supported]=0
 S=srgb(proto).astype(np.float32);S[~finite]=0
 local_lookup={tuple(k):i for i,k in enumerate(keys)};edges=[]
 for i,k in enumerate(keys):
  for delta in [(1,0),(0,1)]:
   j=local_lookup.get((k[0]+delta[0],k[1]+delta[1]))
   if j is not None:edges.append((i,j))
 edges=np.asarray(edges,int);G=(S[edges[:,1]]-S[edges[:,0]]).astype(np.float64);ecount=np.zeros(len(edges),np.int16)
 for start in range(0,len(edges),2048):
  e=edges[start:start+2048];both=agree[:,e[:,0]]&agree[:,e[:,1]];n=both.sum(0);ecount[start:start+len(e)]=n
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning)
   med=np.nanmedian(np.where(both[:,:,None],rgb[:,e[:,1]]-rgb[:,e[:,0]],np.nan),axis=0)
  use=n>=2;G[start:start+len(e)][use]=med[use]
 ew=np.where(ecount>=2,np.minimum(ecount/5,1),.25)
 build_rows=np.array([int(np.flatnonzero(source_raws==raw)[0]) for raw in BUILD])
 np.savez_compressed(OUT/'atlas_inputs.npz',candidate_ids=ids,keys=keys,local_xyz=xyz,world_xyz=a['world_xyz'][ids],core_mask=coremask,
    source_chart_rgb=S,source_supported=supported,source_valid_count=vcount,source_agree_count=count,source_confidence=confidence,
    source_first_raw=first,source_last_raw=last,source_raws=source_raws,source_agree=agree,
    build_raws=np.array(BUILD),build_source_rgb=rgb[build_rows],build_source_agree=agree[build_rows],build_source_valid=valid[build_rows],
    edges=edges,gradient_rgb=G,gradient_weight=ew,same_frame_gradient_count=ecount,coordinate_anchor=a['coordinate_anchor'])
 receipt=dict(status='SOURCE_CHART_AND_CONTRACT_READY',atlas_nodes=len(ids),core_nodes=int(coremask.sum()),outside_halo_nodes=int((~coremask).sum()),
  supported_nodes=int(supported.sum()),supported_core=int((supported&coremask).sum()),valid_count_zero=int((vcount==0).sum()),agree_count_0_1_2_ge3=[int((count==0).sum()),int((count==1).sum()),int((count==2).sum()),int((count>=3).sum())],
  neighbour_edges=len(edges),same_frame_true_gradient_edges=int((ecount>=2).sum()),fallback_gradient_edges=int((ecount<2).sum()),
  source_raws=source_raws.tolist(),atlas_contract_sha256=sha(OUT/'atlas_contract.json'),atlas_inputs_sha256=sha(OUT/'atlas_inputs.npz'))
 write(OUT/'source_chart_receipt.json',receipt);print(json.dumps(receipt),flush=True)

def build_boundaries(p):
 n=len(p['keys']);values=np.full((40,n,3),np.nan,np.float32);before=np.zeros((40,n),bool);rejected_static=np.zeros_like(before);records=[]
 exterior=~p['core_mask'];world=p['world_xyz'];source=p['source_chart_rgb']
 for bi,raw in enumerate(BUILD):
  path=BASE/f'raw{raw:03d}_keep.npz'
  if not path.exists():raise FileNotFoundError('Wait for complete engineer baseline80 cache: '+str(path))
  d=load(path);obs=np.load(CAM/f'scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz');view=np.load(CAM/f'scene008_blue_global_road1k_20261105_r1/inputs/views/{raw:03d}.npz')
  q=world@obs['world_to_camera'][:3,:3].T+obs['world_to_camera'][:3,3];uv=(q@obs['K'].T)[:,:2]/q[:,2:]-.5
  inimage=(q[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)
  ui=np.clip(np.rint(uv[:,0]).astype(int),0,479);vi=np.clip(np.rint(uv[:,1]).astype(int),0,319)
  guard=view['other_vehicle_guard'].astype(bool)|obs['visible'].any(0)|obs['visible_reviewed'].any(0)|obs['box_candidate'].any(0)
  R=sample(d['rgb'],uv);W=sample(d['newroad_alpha_T'],uv);static=sample(d['group_alpha_T'][3],uv)
  actual=p['build_source_rgb'][bi];agree=p['build_source_agree'][bi]
  pre=exterior&p['source_supported']&agree&inimage&~guard[vi,ui]&(W<=.05)
  residual=abs(R-actual).max(1);cohort_residual=abs(actual-source).max(1)
  accept=pre&(residual<=THRESH)&(cohort_residual<=THRESH)
  values[bi,accept]=R[accept];before[bi]=pre
  rejected_static[bi]=pre&(static>.5)&(residual>THRESH)
  records.append(dict(raw=raw,qualified_pre_RGB=int(pre.sum()),accepted=int(accept.sum()),RGB_reject=int((pre&(residual>THRESH)).sum()),
    cohort_reject=int((pre&(cohort_residual>THRESH)).sum()),dark_static_RGB_reject=int(rejected_static[bi].sum()),
    pre_RGB_residual=stats(residual[pre]),accepted_RGB_residual=stats(residual[accept]),baseline_sha256=sha(path)))
 count=np.isfinite(values[:,:,0]).sum(0);first=np.where(np.isfinite(values[:,:,0]),np.array(BUILD)[:,None],999).min(0);last=np.where(np.isfinite(values[:,:,0]),np.array(BUILD)[:,None],-1).max(0)
 boundary=exterior&(count>=3)&(last-first>=6)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);K=np.nanmedian(values,axis=0)
 K=np.nan_to_num(K,nan=0)
 np.savez_compressed(OUT/'boundary_evidence.npz',boundary_mask=boundary,boundary_rgb=K,boundary_view_count=count,
   accepted_render_rgb=values,pre_RGB_eligible=before,rejected_static_RGB=rejected_static)
 receipt=dict(status='RELIABLE_EXTERIOR_BOUNDARY_AUDITED',nodes=int(boundary.sum()),records=records,
   rejected_static_node_union=int(rejected_static.any(0).sum()),rejected_static_view_node_records=int(rejected_static.sum()),
   no_boundary_warning=bool(not boundary.any()),boundary_vs_source_chart_RGB=stats(abs(K[boundary]-source[boundary])),
   boundary_evidence_sha256=sha(OUT/'boundary_evidence.npz'),baseline_manifest_sha256=sha(BASE/'manifest.json'))
 write(OUT/'boundary_receipt.json',receipt);return boundary,K,receipt

def solve(p,boundary,K):
 n=len(p['keys']);e=p['edges'];w=p['gradient_weight'];g=p['gradient_rgb'];c=SCREEN*p['source_confidence']
 i,j=e[:,0],e[:,1]
 L=coo_matrix((np.r_[w,w,-w,-w],(np.r_[i,j,i,j],np.r_[i,j,j,i])),shape=(n,n)).tocsr()+diags(c)
 rhs=c[:,None]*p['source_chart_rgb']
 np.add.at(rhs,i,-w[:,None]*g);np.add.at(rhs,j,w[:,None]*g)
 # Unsupported disconnected components get no invented zero datum. A tiny numeric
 # ridge is unnecessary if supported chart is connected; explicitly audit components.
 from scipy.sparse.csgraph import connected_components
 components,labels=connected_components(L, directed=False);available=np.ones(n,bool)
 for label in range(components):
  m=labels==label
  if not (p['source_confidence'][m]>0).any() and not boundary[m].any():available[m]=False
 fixed=boundary&available;free=available&~fixed
 result=np.full((n,3),np.nan);result[fixed]=K[fixed]
 A=L[free][:,free];b=rhs[free]-L[free][:,fixed]@K[fixed]
 result[free]=spsolve(A,b)
 raw=result.copy();clipped=np.clip(result,0,1);changed=np.isfinite(raw)&(raw!=clipped)
 residual_raw=A@raw[free]-b;residual_clip=A@clipped[free]-b
 eraw=(raw[j]-raw[i])-g;eclip=(clipped[j]-clipped[i])-g
 receipt=dict(status='ONE_FIXED_SCREENED_POISSON_SOLVE',available_nodes=int(available.sum()),unsupported_component_nodes=int((~available).sum()),
  connected_components=int(components),boundary_nodes=int(fixed.sum()),screen_lambda=SCREEN,
  raw_solution_range=[float(np.nanmin(raw)),float(np.nanmax(raw))],clipped_channels=int(changed.sum()),clipped_nodes=int(changed.any(1).sum()),
  equation_residual_before_clip=stats(abs(residual_raw)),equation_residual_after_clip=stats(abs(residual_clip)),
  edge_gradient_residual_before_clip=stats(abs(eraw[np.isfinite(eraw).all(1)])),edge_gradient_residual_after_clip=stats(abs(eclip[np.isfinite(eclip).all(1)])),
  fused_vs_source_RGB=stats(abs(clipped[p['source_supported']]-p['source_chart_rgb'][p['source_supported'])),
  measured_edge_gradient_drift=stats(abs(eclip[(p['same_frame_gradient_count']>=2)&np.isfinite(eclip).all(1)])))
 write(OUT/'poisson_receipt.json',receipt)
 return clipped.astype(np.float32),available,receipt

def chart_grid(p,T,available):
 keys=p['keys'];low=keys.min(0);shape=keys.max(0)-low+1;grid=np.full(tuple(shape),-1,int);loc=keys-low;grid[loc[:,0],loc[:,1]]=np.arange(len(keys))
 z=np.full(tuple(shape),np.nan);z[loc[:,0],loc[:,1]]=p['local_xyz'][:,2]
 nearest=distance_transform_edt(~np.isfinite(z),return_distances=False,return_indices=True);zfill=z[tuple(nearest)]
 offset=np.median(p['local_xyz'][:,:2]-keys*CELL,axis=0)
 return grid,zfill,low,offset

def project(raw,p,T,available,grid,z,low,offset):
 folder=OUT/('projected_BUILD40' if raw in BUILD else 'projected_HELD9');path=folder/f'raw{raw:03d}.npz';old=load(path)
 if bool(old['fusion_ready']):raise RuntimeError('Final target is immutable: '+str(path))
 obs=np.load(CAM/f'scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz');h,w=old['fit_mask'].shape;yy,xx=np.mgrid[:h,:w]
 w2l=obs['world_to_camera']@p['coordinate_anchor'];c2l=np.linalg.inv(w2l)
 rays=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)@np.linalg.inv(obs['K']).T;ray=rays@c2l[:3,:3].T;origin=c2l[:3,3]
 dz=ray[...,2];validray=abs(dz)>1e-8;t=(float(np.median(p['local_xyz'][:,2]))-origin[2])/np.where(validray,dz,1)
 for _ in range(12):
  xyz=origin+t[...,None]*ray;uv=(xyz[...,:2]-offset)/CELL-low
  height=map_coordinates(z,[uv[...,0],uv[...,1]],order=1,mode='nearest')
  nxt=(height-origin[2])/np.where(validray,dz,1);converged=abs(nxt-t)*np.linalg.norm(ray,axis=-1)<.02;t=nxt
 xyz=origin+t[...,None]*ray;uv=(xyz[...,:2]-offset)/CELL-low
 fx=np.floor(uv[...,0]).astype(int);fy=np.floor(uv[...,1]).astype(int);ix=np.clip(fx,0,grid.shape[0]-2);iy=np.clip(fy,0,grid.shape[1]-2)
 ids=np.stack([grid[ix,iy],grid[ix+1,iy],grid[ix,iy+1],grid[ix+1,iy+1]],-1);safe=np.maximum(ids,0)
 geom=validray&(t>0)&(t<100)&converged&(fx>=0)&(fy>=0)&(fx<grid.shape[0]-1)&(fy<grid.shape[1]-1)&(ids>=0).all(-1)
 supported=geom&p['source_supported'][safe].all(-1)&available[safe].all(-1)
 measured=obs['scene_depth_m']>0;depthconflict=measured&(abs(t-obs['scene_depth_m'])>.3)
 replace=old['fit_mask']&supported&~depthconflict
 dx=np.clip(uv[...,0]-fx,0,1);dy=np.clip(uv[...,1]-fy,0,1)
 weights=np.stack([(1-dx)*(1-dy),dx*(1-dy),(1-dx)*dy,dx*dy],-1)
 S=np.sum(p['source_chart_rgb'][safe]*weights[...,None],axis=-2);F=np.sum(np.nan_to_num(T[safe])*weights[...,None],axis=-2)
 observation=old['direct_rgb'].copy();A=observation.copy();B=observation.copy();A[replace]=S[replace];B[replace]=F[replace]
 source_count=np.where(geom,p['source_agree_count'][safe].min(-1),0).astype(np.int16)
 confidence=np.where(geom,np.sum(p['source_confidence'][safe]*weights,axis=-1),0).astype(np.float32)
 np.savez_compressed(path,raw=old['raw'],BUILD40=old['BUILD40'],fusion_ready=np.array(True),direct_rgb=A.astype(np.float32),fusion_rgb=B.astype(np.float32),
   observation_rgb=observation,fit_mask=old['fit_mask'],protect_mask=old['protect_mask'],protect_rgb=old['protect_rgb'],
   fusion_replace_mask=replace,atlas_support_mask=supported,source_count_map=source_count,source_confidence_map=confidence,
   imputed_mask=geom&~supported,direct_scene_depth_agree_mask=replace&measured,
   atlas_local_xy_m=xyz[...,:2].astype(np.float32))
 return dict(raw=raw,role='BUILD_FIT' if raw in BUILD else 'HELD_PROJECT_EVAL_ONLY',file=str(path.relative_to(OUT)),sha256=sha(path),fit_pixels=int(old['fit_mask'].sum()),
  replace_pixels=int(replace.sum()),depth_conflict_in_fit=int((old['fit_mask']&geom&depthconflict).sum()),
  no_atlas_support_in_fit=int((old['fit_mask']&~supported).sum()),imputed_not_fit=int((old['fit_mask']&geom&~supported).sum()),
  direct_vs_true_on_replace=stats(abs(A[replace]-observation[replace])),fusion_vs_true_on_replace=stats(abs(B[replace]-observation[replace])),
  fusion_vs_direct_on_replace=stats(abs(B[replace]-A[replace])),outside_replace_equal_true=bool(np.array_equal(A[~replace],observation[~replace]) and np.array_equal(B[~replace],observation[~replace])),
  mask_array_sha256=hashlib.sha256(np.ascontiguousarray(old['fit_mask']).tobytes()).hexdigest())

def finalize():
 start=time.time()
 if (OUT/'target_manifest.json').exists():raise FileExistsError('preserve immutable final target manifest')
 c=json.loads((OUT/'atlas_contract.json').read_text())
 for path,digest in c['immutable_sources'].items():assert sha(path)==digest,path
 p=load(OUT/'atlas_inputs.npz');boundary,K,br=build_boundaries(p);T,available,pr=solve(p,boundary,K)
 np.savez_compressed(OUT/'world_atlas_targets.npz',keys=p['keys'],local_xyz=p['local_xyz'],coordinate_anchor=p['coordinate_anchor'],
    source_chart_rgb=p['source_chart_rgb'],fusion_chart_rgb=T,core_mask=p['core_mask'],source_supported=p['source_supported'],source_count=p['source_agree_count'],
    source_confidence=p['source_confidence'],available=available,boundary_mask=boundary,boundary_rgb=K,
    edges=p['edges'],same_frame_gradient_rgb=p['gradient_rgb'],same_frame_gradient_count=p['same_frame_gradient_count'])
 grid,z,low,offset=chart_grid(p,T,available);rows=[]
 for raw in range(89,138):
  rows.append(project(raw,p,T,available,grid,z,low,offset))
  print(json.dumps(dict(raw=raw,replace=rows[-1]['replace_pixels'],seconds=time.time()-start)),flush=True)
 direct=json.loads((OUT/'direct_manifest.json').read_text());initial={r['raw']:r for r in direct['rows']}
 assert all(r['mask_array_sha256']==initial[r['raw']]['mask_array_sha256'] for r in rows)
 manifest=dict(schema='IMMUTABLE_SHARED_WORLD_FUSION_TARGETS_V1',status='FINAL_SHARED_TARGETS_READY',BUILD40=list(BUILD),HELD9=list(HELD),rows=rows,
  direct_label='Unfused same190-source atlas projection',fusion_label='One fixed screened-Poisson fused same-source atlas projection',
  observation_label='Original actual RGB, separate evaluation reference; not hidden deleted-road GT',
  RGB_encoding='encoded sRGB[0,1], HWC float32, desired complete joint-image supervision',state_axis=False,KEEP_DELETE_same_target_mask=True,
  fit_pixels_unique_BUILD40=sum(r['fit_pixels'] for r in rows if r['raw'] in BUILD),replace_pixels_BUILD40=sum(r['replace_pixels'] for r in rows if r['raw'] in BUILD),
  all_fit_masks_exact_initial=True,HELD9_only_projected=True,contract_sha256=sha(OUT/'atlas_contract.json'),
  source_chart_receipt_sha256=sha(OUT/'source_chart_receipt.json'),world_atlas_sha256=sha(OUT/'world_atlas_targets.npz'),
  boundary_receipt=br,Poisson_receipt=pr,extra_GPU_forwards=0,seconds_finalize=time.time()-start,
  limitations=['Shared apparent colour; no intrinsic/shadow decomposition','Target smoothing/continuity is not true-image accuracy',
              'Atlas is synthetic construction from real witnesses, not hiddenRGB GT','Same scene already open; no independent generalization claim'],
  failure_ledger_refs=['V82-F13'],failure_ledger_delta='pending',human_verdict=None)
 write(OUT/'target_manifest.json',manifest);print(json.dumps(dict(status=manifest['status'],BUILD_replace=manifest['replace_pixels_BUILD40'],seconds=manifest['seconds_finalize'])),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','finalize']);args=ap.parse_args()
 prepare() if args.stage=='prepare' else finalize()
