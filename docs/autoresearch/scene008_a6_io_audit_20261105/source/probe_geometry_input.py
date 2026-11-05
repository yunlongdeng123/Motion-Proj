"""Pre-registered PVD I/O sensitivity probes. Frozen weights; no adoption.

R0 exactly replays the original input. RY changes only the proper yaw
convention. RV uses real confirmed visible points from one BUILD frame.
None of these probes estimates unobserved shape accuracy or RGB capability.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib,types,argparse
from pathlib import Path
import numpy as np

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=CAM/'scene008_a37_C2_20261103_r1'
PVD=ROOT/'source/prior/pvd'
OLD=ROOT/'source/prior/proposals/pvd_seed000'
RUN=Path(__file__).resolve().parent.parent
DEST=RUN/'geometry_probes_v3'
OBS=CAM/'scene008_instances_20261105_r1/observations/a6.npz'
HELD=(89,95,101,105,113,121,125,130,137)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):return dict(np.load(p,allow_pickle=False))

def prepare():
    DEST.mkdir(exist_ok=False)
    source=load(OLD/'fixed_partial_input.npz');norm=json.loads((PVD/'normalization_contract.json').read_text())
    observation=load(OBS);rf=observation['sample_raw_frame'];valid=observation['sample_rgb_valid']
    counts=sorted([(int((valid&(rf==r)).sum()),int(r)) for r in range(89,138) if r not in HELD],key=lambda v:(-v[0],v[1]))
    n,raw=counts[0]
    if n<200:raise ValueError(f'No one BUILD frame has >=200 reviewed/depth verified samples: {counts[:5]}')
    eligible=np.flatnonzero(valid&(rf==raw));ix=np.random.default_rng(0).choice(eligible,200,replace=False)
    M=np.asarray(norm['pvd_to_actor_native_4x4'],np.float64)
    # Change the car's front convention about its canonical whole-box centre.
    # Both length and lateral axes flip (proper yaw, not a reflection).
    # Rotate the FULL affine inverse, including its nonzero stats offset.
    S=np.diag([-1.,-1.,1.,1.]);yawM=S@M
    arms={
      'R0':(source['points_actor_native_m'],source['points_pvd_normalized'],M,
          dict(kind='Original 200-row exact replay',source_cloud_row=source['raw_row_indices'].tolist())),
      'R0_repeat':(source['points_actor_native_m'],source['points_pvd_normalized'],M,
          dict(kind='Identical R0 input/RNG repeated to measure backend numerical variation',source_cloud_row=source['raw_row_indices'].tolist())),
      'RY':(source['points_actor_native_m'],None,yawM,
          dict(kind='Proper yaw180 convention sensitivity ONLY',source_cloud_row=source['raw_row_indices'].tolist())),
      'RV':(observation['sample_xyz_local_m'][ix],None,M,
          dict(kind='Single-view confirmed-visible real BUILD input',raw_frame=raw,eligible_points=n,
             observation_rows=ix.tolist(),source_cloud_rows=observation['sample_source_cloud_row'][ix].tolist()))}
    contract=dict(status='PRE_REGISTERED_BEFORE_INFERENCE',seed=0,steps=1000,generated=1848,conditioning=200,
        model_sha256=sha(PVD/'weights/car_completion_3199.pth'),normalization_sha256=sha(PVD/'normalization_contract.json'),
        paired_initial_state_sha256=sha(OLD/'trajectory/initial_or_resumed.pt'),
        paired_noise='Exact original 1848 noise rows and saved CPU/CUDA RNG reused in every arm',
        original_output_sha256=sha(OLD/'candidate_points_actor_native.npz'),observation_sha256=sha(OBS),
        selection='RV: max confirmed-visible point count among BUILD40, tie lowest raw; rng0 uniform 200 without replacement',
        changed_factors=dict(R0='none',R0_repeat='none; reproducibility control',RY='proper yaw convention only; fixed 200 rows',RV='point selection only; one true visible frame'),
        precision_gate='8 exact state/RNG reloads; next-step maxabs<=1e-4 normalized; report limited precision if not bitwise. Same-input full R0 repeat measures accumulated numeric variation.',
        decision='Diagnostic only; NEVER pick front sign or adopt a scene asset based on partial-cloud score or visual preference.',
        limits=['RV changes purity, time/view distribution and spatial sampling jointly; cannot isolate a denoising gain.',
          'Original R0 conditioning contains old HELD points; all measured-distance scores are self-consistency, not independent accuracy.',
          'No hidden-shape GT or RGB material is available; point-to-partial distances cannot evaluate unknown generated surfaces.',
          'Whole-box anisotropic adaptation and true front convention remain adaptation assumptions, not exact ShapeNet benchmark I/O.'],
        arms={})
    for name,(native,q,mat,meta) in arms.items():
        folder=DEST/name;folder.mkdir();inverse=np.linalg.inv(mat)
        q=(native@inverse[:3,:3].T+inverse[:3,3]).astype(np.float32) if q is None else q.astype(np.float32)
        error=float(abs(q.astype(np.float64)@mat[:3,:3].T+mat[:3,3]-native).max())
        assert error<2e-6 and np.linalg.det(mat[:3,:3])>0
        np.savez_compressed(folder/'input.npz',native=native,normalized=q,pvd_to_native=mat)
        contract['arms'][name]=dict(meta,roundtrip_maxabs_m=error,input_sha256=sha(folder/'input.npz'))
    write(DEST/'contract.json',contract);print(json.dumps(contract),flush=True)

def model_and_torch():
    import torch
    sys.path.insert(0,str(PVD/'vendor'));sys.path.insert(0,str(PVD))
    from inference_core import Model,get_betas
    torch.set_num_threads(8);torch.cuda.set_device(0)
    assert sha(PVD/'weights/car_completion_3199.pth')=='567ec9e6f0c3b453e128d916363bb8f1ddf6a432eef13d10035ee006d732dc1a'
    opt=types.SimpleNamespace(nc=3,svpoints=200,embed_dim=64,attention=True,dropout=.1)
    model=Model(opt,get_betas('linear',.0001,.02,1000),'mse','eps','fixedsmall')
    ckpt=torch.load(PVD/'weights/car_completion_3199.pth',map_location='cpu')
    model.load_state_dict({k.replace('model.module.','model.',1):v for k,v in ckpt['model_state'].items()},strict=True)
    model.requires_grad_(False);model.model.eval();model.training=False
    return model.cuda(),torch

def initial(torch,q):
    contract=json.loads((DEST/'contract.json').read_text())
    assert sha(OLD/'trajectory/initial_or_resumed.pt')==contract['paired_initial_state_sha256']
    saved=torch.load(OLD/'trajectory/initial_or_resumed.pt',map_location='cpu')
    original=load(OLD/'fixed_partial_input.npz')['points_pvd_normalized']
    assert saved['next_t']==999 and saved['points_pvd_normalized'].shape==(1,3,2048)
    assert np.array_equal(saved['points_pvd_normalized'][0,:,:200].T.numpy(),original)
    torch.set_rng_state(saved['torch_cpu_rng_state']);torch.cuda.set_rng_state(saved['torch_cuda_rng_state'])
    p=torch.from_numpy(q.T[None].copy()).cuda()
    current=saved['points_pvd_normalized'].cuda();current[:,:,:200]=p
    return p,current
def next_state(model,torch,current,t):
    return model.diffusion.p_sample(model._denoise,current,torch.full((1,),t,device='cuda',dtype=torch.int64),
        noise_fn=torch.randn,clip_denoised=False,return_pred_xstart=False)

def smoke():
    start=time.time();model,torch=model_and_torch();q=load(DEST/'R0/input.npz')['normalized']
    with torch.no_grad():
        partial,current=initial(torch,q)
        for t in range(999,967,-1):current=next_state(model,torch,current,t)
        assert torch.isfinite(current).all() and torch.equal(current[:,:,:200],partial)
        state=dict(current=current.cpu(),next_t=967,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            contract_sha256=sha(DEST/'contract.json'),model_sha256=sha(PVD/'weights/car_completion_3199.pth'))
        torch.save(state,DEST/'smoke_state.pt');restored=torch.load(DEST/'smoke_state.pt',map_location='cpu')
        assert torch.equal(restored['current'],state['current']) and all(torch.equal(restored[k],state[k]) for k in ('cpu_rng','cuda_rng'))
        replays=[]
        for _ in range(8):
            torch.set_rng_state(restored['cpu_rng']);torch.cuda.set_rng_state(restored['cuda_rng'])
            replays.append(next_state(model,torch,restored['current'].cuda(),restored['next_t']))
        ref=replays[0];errors=[float((v-ref).abs().max()) for v in replays]
        means=[float((v-ref).abs().double().mean()) for v in replays]
        maxerror=max(errors);exact=all(torch.equal(v,ref) for v in replays)
        assert maxerror<=1e-4,dict(next_step_maxabs=maxerror,errors=errors)
        # Next-stage export/inverse geometry wiring, independent of sample quality.
        M=load(DEST/'R0/input.npz')['pvd_to_native'];test=ref[0].T.cpu().numpy().astype(np.float64)@M[:3,:3].T+M[:3,3]
        np.savez_compressed(DEST/'smoke_inverse.npz',native=test)
        assert np.array_equal(load(DEST/'smoke_inverse.npz')['native'],test)
    seconds=time.time()-start;projected=seconds/32*4000
    status='REQUIRES_REAL_10MIN_SMOKE' if projected>=3600 else 'PASS_LIMITED_PRECISION_DIAGNOSTIC'
    write(DEST/'smoke_receipt.json',dict(status=status,inference_only=True,backward='Not applicable: frozen no-grad sampling',
       contract_sha256=sha(DEST/'contract.json'),source_sha256=sha(__file__),
       extra_sampling_steps=40,finite=True,conditioning_exact=True,checkpoint_save=True,full_state_and_rng_bytes_restored_exact=True,
       next_step_bitwise_equal=exact,next_step_maxabs_normalized=maxerror,next_step_repeat_maxabs=errors,next_step_repeat_meanabs=means,
       bitwise_sampling_claim=False,backend='Original PVCNN contains floating-point atomicAdd; no kernel/method replacement made',
       next_stage_inverse_export_reload=True,seconds=seconds,
       projected_formal_total_seconds=projected,training_steps=0))
    if projected>=3600:raise RuntimeError('Long run: real 10min smoke required')

def run():
    from scipy.spatial import cKDTree
    receipt=json.loads((DEST/'smoke_receipt.json').read_text())
    assert receipt['status']=='PASS_LIMITED_PRECISION_DIAGNOSTIC' and receipt['projected_formal_total_seconds']<3600
    assert receipt['source_sha256']==sha(__file__) and receipt['contract_sha256']==sha(DEST/'contract.json')
    contract=json.loads((DEST/'contract.json').read_text());assert contract['status']=='PRE_REGISTERED_BEFORE_INFERENCE'
    model,torch=model_and_torch();original=load(OLD/'candidate_points_actor_native.npz')
    real=load(ROOT/'inputs/a6_geometry_minimal.npz')['canonical_xyz'];reports=[]
    original_hashes={str(p):sha(p) for p in (OLD/'candidate_points_actor_native.npz',PVD/'weights/car_completion_3199.pth')}
    outputs={};snapshots={}
    for arm in ('R0','R0_repeat','RY','RV'):
        start=time.time();folder=DEST/arm;data=load(folder/'input.npz');assert sha(folder/'input.npz')==contract['arms'][arm]['input_sha256']
        trace_t=[];trace_xyz=[]
        with torch.no_grad():
            partial,current=initial(torch,data['normalized'])
            for t in range(999,-1,-1):
                current=next_state(model,torch,current,t)
                if t%100==0:
                    assert torch.isfinite(current).all() and torch.equal(current[:,:,:200],partial)
                    M=data['pvd_to_native'];sample=current[0].T.cpu().numpy().astype(np.float64)@M[:3,:3].T+M[:3,3]
                    trace_t.append(t);trace_xyz.append(sample)
                if t==500:
                    torch.save(dict(points=current.cpu(),next_t=t-1,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                        contract_sha256=sha(DEST/'contract.json'),arm=arm),folder/'midpoint_state.pt')
            normalized=current[0].T.cpu().numpy();M=data['pvd_to_native']
            native=normalized.astype(np.float64)@M[:3,:3].T+M[:3,3]
            assert np.max(abs(native[:200]-data['native']))<2e-6
            np.savez_compressed(folder/'output.npz',points_actor_native_m=native.astype(np.float32),points_pvd_normalized=normalized,
                generated_mask=np.arange(2048)>=200,conditioning_native=data['native'],pvd_to_native=M)
            np.savez_compressed(folder/'sampling_snapshots.npz',timestep=np.asarray(trace_t),native=np.asarray(trace_xyz))
        dist=cKDTree(native).query(real,k=1)[0]
        row=dict(arm=arm,seconds=time.time()-start,output_sha256=sha(folder/'output.npz'),finite=True,fixed_partial_exact=True,
            torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            points=2048,below_real_lowest_z=int((native[:,2]<real[:,2].min()).sum()),
            partial_only_distance_NOT_hidden_accuracy=dict(mean_m=float(dist.mean()),p50_m=float(np.median(dist)),p90_m=float(np.quantile(dist,.9)),
                within4cm=float((dist<=.04).mean()),within8cm=float((dist<=.08).mean())),
            bounds_m=[native.min(0).tolist(),native.max(0).tolist()],training_steps=0,adopted=False)
        if arm=='R0':
            row['replay_normalized_maxabs']=float(abs(normalized-original['points_pvd_normalized']).max())
            row['replay_native_maxabs_m']=float(abs(native.astype(np.float32)-original['points_actor_native_m']).max())
            row['replay_normalized_bitwise_equal']=bool(np.array_equal(normalized,original['points_pvd_normalized']))
        outputs[arm]=native.copy()
        snapshots[arm]=np.asarray(trace_xyz)
        if arm!='R0':
            delta=np.linalg.norm(native[200:]-outputs['R0'][200:],axis=1)
            row['paired_generated_row_shift_vs_R0_m']=dict(mean=float(delta.mean()),p90=float(np.quantile(delta,.9)),max=float(delta.max()))
            d1=cKDTree(native[200:]).query(outputs['R0'][200:])[0];d2=cKDTree(outputs['R0'][200:]).query(native[200:])[0]
            row['symmetric_generated_set_distance_vs_R0_m']=float((d1.mean()+d2.mean())/2)
            drift=np.linalg.norm(snapshots[arm][:,200:]-snapshots['R0'][:,200:],axis=2)
            row['paired_trace_vs_R0']=[dict(timestep=t,mean_m=float(v.mean()),max_m=float(v.max())) for t,v in zip(trace_t,drift)]
        reports.append(row);write(folder/'receipt.json',row);print(json.dumps(row),flush=True)
    assert all(sha(p)==v for p,v in original_hashes.items())
    write(DEST/'summary.json',dict(status='DIAGNOSTICS_ONLY_NOT_EXACT_REPLAY_CERTIFIED',arms=reports,contract_sha256=sha(DEST/'contract.json'),
        formal_sampling_steps=4000,smoke_sampling_steps_extra=40,training_steps=0,assets_adopted=0,
        conclusion='Sensitivity and replay test; no front-sign truth, hidden surface accuracy, or material completion conclusion.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','smoke','run']);args=p.parse_args()
    globals()[args.stage]()
