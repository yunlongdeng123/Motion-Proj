"""CPU evidence/owner/lineage audit. Never equates a point label with pixel truth."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import sys,base64
import torch
torch.set_num_threads(2)
from identity_common import *

OUT=TASK/'tristate'
CODE=ROOT/'shadow_rebuild_20261017_v3/implementation_r15_train_v3/shadow_rebuild_v3'
sys.path.insert(0,str(CODE))
from shadow_rebuild.math3d import quaternion_matrix

def counts(a):
    u,n=np.unique(a,return_counts=True)
    return {str(int(k)):int(v) for k,v in zip(u,n)}

def quant(a):
    a=np.asarray(a)
    return None if not len(a) else dict(zip(['min','p50','p90','p95','p99','max'],map(float,np.quantile(a,[0,.5,.9,.95,.99,1]))))

def key(a):
    t=np.empty(len(a['owner']),dtype=[('frame','<i8'),('row','<i8'),('candidate','<i8')])
    for dest,src in [('frame','source_frame'),('row','source_raw_row'),('candidate','source_candidate_owner')]:t[dest]=a[src]
    return t

def compact(a,ids,xyz=None):
    # Little-endian Float32 12-column records. A deterministic sample; not all points.
    p=a['identity_probability'][ids]
    xyz=a['xyz'][ids] if xyz is None else xyz
    columns=np.column_stack((xyz,a['identity_evidence_status'][ids],a['owner'][ids],
        np.max(a['scales'][ids],axis=-1),a['source_frame'][ids],a['source_raw_row'][ids],
        a['source_candidate_owner'][ids],p)).astype('<f4')
    return dict(rows=len(ids),stride=12,data=base64.b64encode(columns.tobytes()).decode())

def sample(ids,n,rng):return np.sort(rng.choice(ids,n,replace=False)) if len(ids)>n else ids

def main():
    if (OUT/'cpu_complete.json').exists():raise FileExistsError('CPU audit already complete')
    OUT.mkdir(exist_ok=True)
    seed=dict(np.load(R15/'data/seed.npz',allow_pickle=False))
    ck=R15/'paused_checkpoint_004000_20261017/checkpoint.pt';before=stat(ck)
    saved=torch.load(ck,map_location='cpu',weights_only=False)
    cur={k[6:]:v.numpy() for k,v in saved['state'].items() if k.startswith('field.')}
    cfg=saved['config'];cur['scales']=np.exp(cur['log_scales']).clip(cfg['scale_min_m'],cfg['scale_max_m'])
    seed['scales']=seed['scales'].clip(cfg['scale_min_m'],cfg['scale_max_m'])
    n=len(seed['owner']);line=cur['sgcm_lineage'];eligible=(line>=0)&(line<n)
    seedkeys=key(seed);curkeys=key(cur)
    equal=np.zeros(len(line),bool);equal[eligible]=seedkeys[line[eligible]]==curkeys[eligible]
    # A lineage link is used only when the immutable measurement triple agrees.
    parent=np.full(len(line),-1,np.int64);parent[equal]=line[equal]
    unmatched=(parent<0)&eligible
    if unmatched.any():raise ValueError('In-range lineage contradicts original source triple; cannot silently match')
    matrix=np.zeros((3,3),np.int64);owner_changes=0;prob_changes=0
    np.add.at(matrix,(seed['identity_evidence_status'][parent[equal]],cur['identity_evidence_status'][equal]),1)
    owner_changes=int(np.count_nonzero(seed['owner'][parent[equal]]!=cur['owner'][equal]))
    prob_changes=int(np.count_nonzero(np.any(seed['identity_probability'][parent[equal]]!=cur['identity_probability'][equal],axis=1)))
    surviving=np.unique(parent[equal]);seed_has=np.zeros(n,bool);seed_has[surviving]=True
    groups=[]
    for status in [2,1,0]:
        pids=np.flatnonzero(seed['identity_evidence_status']==status)
        rows=equal&(seed['identity_evidence_status'][np.maximum(parent,0)]==status)
        groups.append(dict(initial_status=status,initial_parents=len(pids),surviving_initial_parents=int(seed_has[pids].sum()),
            no_surviving_descendant=int((~seed_has[pids]).sum()),current_descendant_rows=int(rows.sum())))
    report=dict(task_id=TASK_ID,checkpoint=before,checkpoint_step=int(saved['global_step']),
        raw_status_encoding={'0':'unknown evidence','1':'definite Actor evidence','2':'definite static evidence'},
        display_encoding={'0':'raw status 2: definite static evidence','1':'raw status 1: definite Actor evidence','U':'raw status 0: unknown evidence'},
        identity_probability_channels=['static','Actor','unknown'],
        identity_status_and_probability='registered buffers; not Adam parameters; explicit audit interventions only',
        initial_count=n,current_count=len(line),initial_status_counts=counts(seed['identity_evidence_status']),
        current_status_counts=counts(cur['identity_evidence_status']),
        initial_unknown_hard_owner=counts(seed['owner'][seed['identity_evidence_status']==0]),
        current_unknown_hard_owner=counts(cur['owner'][cur['identity_evidence_status']==0]),
        lineage=dict(method='sgcm_lineage checked against (source_frame,source_raw_row,source_candidate_owner)',
            matched_current_rows=int(equal.sum()),new_lineage_current_rows=int((~eligible).sum()),
            in_range_source_key_mismatches=int(unmatched.sum()),
            initial_to_current_status_matrix=matrix.tolist(),matrix_axis_order_raw=[0,1,2],
            matched_owner_changed_current_rows=owner_changes,matched_probability_changed_current_rows=prob_changes,
            matched_unique_surviving_seed_parents=len(surviving),status_parent_groups=groups,
            new_lineage_status_counts=counts(cur['identity_evidence_status'][~eligible]),
            new_lineage_unique_parents=len(np.unique(line[~eligible])),
            caution='matrix counts current descendants, not unique point transitions; pruning is not relabelling'),
        actors=[],pixel_masks=[],external_pools=[],human_verdict=None,
        limitations=['Evidence scores are not owner ground truth.','A known-Actor fragment can arise from a bad seed or later geometry/scale growth.',
            'Centers do not visualize full Gaussian support; use the raster panels for support/occlusion.'])
    half=np.asarray(read(cfg['box_half_sizes_path'])['half_sizes_m'])
    trajectories=dict(np.load(R15/'data/trajectory.npz'));poses=trajectories['poses'][24]
    rng=np.random.default_rng(20261018)
    viewer=dict(columns=['x','y','z','raw_status','hard_owner','max_sigma_m','source_frame_index','source_raw_row','source_candidate_owner','p_static','p_Actor','p_unknown'],
        reference_frame='0008113',panels=[],actors=ACTORS,display_legend=report['display_encoding'])
    for stage,a in [('init',seed),('4k',cur)]:
        state=a['identity_evidence_status'];owner=a['owner']
        worlds=a['xyz'].copy()
        for actor in range(43):
            take=owner==actor
            worlds[take]=a['xyz'][take]@poses[actor,:3,:3].T+poses[actor,:3,3]
        ids=np.concatenate([sample(np.flatnonzero(state==s),12000,rng) for s in [2,1,0]])
        viewer['panels'].append(dict(stage=stage,scope='all',coordinates='world@0008113',total=len(owner),
            counts=counts(state),points=compact(a,ids,worlds[ids])))
        for actor in ACTORS:
            take=owner==actor;rows=np.flatnonzero(take);x=a['xyz'][take];sc=a['scales'][take]
            q=quaternion_matrix(torch.from_numpy(a['quaternion'][take])).numpy()
            sigma_axis=np.sqrt((q*q*sc[:,None,:]**2).sum(-1))
            outside=(np.abs(x)>half[actor]).any(-1)
            support=(np.abs(x)+3*sigma_axis>half[actor]).any(-1)
            item=dict(stage=stage,actor=actor,count=int(take.sum()),status_counts=counts(state[take]),
                half_box_m=half[actor].tolist(),centers_outside_box=int(outside.sum()),
                three_sigma_axis_bound_crosses_box=int(support.sum()),
                max_principal_sigma_m=quant(sc.max(-1)),
                caveat='3sigma axis bounds / OBB crossing are diagnostics, not proof of road geometry')
            if stage=='4k':
                known=parent[rows]>=0;pr=parent[rows[known]]
                item.update(matched_seed_descendants=int(known.sum()),
                    new_lineage_rows=int((~known).sum()),unique_initial_parents=int(len(np.unique(pr))),
                    distance_to_original_seed_m=quant(np.linalg.norm(x[known]-seed['xyz'][pr],axis=-1)),
                    sigma_ratio_to_original_seed=quant(sc[known].max(-1)/seed['scales'][pr].max(-1)))
            report['actors'].append(item)
            ids=sample(rows,24000,rng)
            viewer['panels'].append(dict(stage=stage,scope='a'+str(actor),coordinates='Actor-local meters',
                total=len(rows),counts=counts(state[take]),half_box_m=half[actor].tolist(),points=compact(a,ids)))
            # Local context contains actual other owners, with Actor poses applied.
            inv=poses[actor,:3,:3];local=(worlds-poses[actor,:3,3])@inv
            context=(np.abs(local)<np.array([8,8,4])).all(-1)&~take
            ids=sample(np.flatnonzero(context),16000,rng)
            viewer['panels'].append(dict(stage=stage,scope='a'+str(actor)+'context',coordinates='reference Actor-local meters @0008113; other owners included',
                total=int(context.sum()),counts=counts(state[context]),half_box_m=half[actor].tolist(),points=compact(a,ids,local[ids])))
    for label,path in [('initial measured pool',R15.parent/'ownership_rebuild_r14/uncertain_candidate_pool.npz'),
                       ('post3k retained measured pool',R15/'gates/003000/owner/unresolved_candidate_pool.npz')]:
        pool=dict(np.load(path,allow_pickle=False))
        report['external_pools'].append(dict(name=label,path=str(path),count=len(pool['xyz']),
            status_counts=counts(pool['identity_evidence_status']),candidate_owner_counts=counts(pool['source_candidate_owner']),
            rendered_Gaussian=False))
        for actor in [6,37]:
            rows=np.flatnonzero(pool['source_candidate_owner']==actor)
            ids=sample(rows,12000,rng)
            # pool.xyz is raw measured world, not an Actor-owned local Gaussian.
            local=(pool['source_world'][ids]-poses[actor,:3,3])@poses[actor,:3,:3]
            tmp=dict(pool,owner=np.full(len(pool['xyz']),-999),scales=np.zeros_like(pool['xyz']))
            viewer['panels'].append(dict(stage='pool_init' if 'initial' in label else 'pool_3k',scope='a'+str(actor),
                coordinates='measured source_world in reference Actor frame; NOT Gaussian',total=len(rows),
                counts=counts(pool['identity_evidence_status'][rows]),half_box_m=half[actor].tolist(),points=compact(tmp,ids,local)))
    manifest=read(R15/'data/manifest.json')
    pixelout=OUT/'pixels';pixelout.mkdir(exist_ok=True)
    from PIL import Image
    palette=np.array([[56,165,240],[38,230,140],[255,182,39]],np.uint8)
    for row in manifest['frames']:
        z=dict(np.load(R15/'data'/row['supervision'],allow_pickle=False))
        for actor in ACTORS:
            valid=z['mask_valid'][...,actor]&z['mask_known'][actor]
            positive=z['masks'][...,actor]&valid
            cls=np.where(valid,np.where(positive,1,0),2)
            Image.fromarray(palette[cls]).save(pixelout/(row['frame_id']+'_a%02d.png'%actor))
            report['pixel_masks'].append(dict(frame_id=row['frame_id'],actor=actor,
                negative=int((cls==0).sum()),positive=int((cls==1).sum()),unknown=int((cls==2).sum())))
    report['optimizer_group_names']=[g.get('name') for g in saved['optimizer']['param_groups']]
    assert before==stat(ck)
    write(OUT/'evidence_audit.json',report)
    write(OUT/'viewer_data.json',viewer)
    write(OUT/'cpu_complete.json',dict(status='CPU_AUDIT_COMPLETE_GPU_PROJECTION_PENDING',checkpoint_unchanged=True))
    print(json.dumps({k:report[k] for k in ['initial_count','current_count','lineage','external_pools']},indent=2),flush=True)

if __name__=='__main__':main()
