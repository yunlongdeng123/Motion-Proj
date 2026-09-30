"""冻结有限候选格：先物理/视角，再真实mask遮挡；不按生成模型结果挑seed。"""
import argparse,json,itertools
from pathlib import Path
from collections import Counter,defaultdict
import cv2
import numpy as np
from scipy.ndimage import median_filter
from geometry_factory import Geometry,read,dump,source_masks,view_angles,wrap

OFFSETS=[(z,x) for z in [-6.,-4.,-2.,0.,2.,4.,5.5,7.,9.] for x in [-6.,-4.5,-3.,-1.8,-1.2,0.,1.2,1.8,3.,4.5,6.]]
def runlen(bools):
    best=cur=0
    for x in bools:cur=cur+1 if x else 0;best=max(best,cur)
    return best
def warp_matrix(dframe,pose,contact_fraction):
    db=np.array(dframe['actors'][0]['projection']['box_xyxy']);bb=np.array(pose['box']);scale=(bb[2:]-bb[:2])/(db[2:]-db[:2])
    # 保留GT尺度比，以真实mask下沿作接地点；只改变新增A，不改Y。
    sx,sy=scale;ty=bb[3]-sy*(db[1]+contact_fraction*(db[3]-db[1]));tx=bb[0]-sx*db[0]
    return np.float32([[sx,0,tx],[0,sy,ty]])
def actor_contact(mask,db):
    rows=np.flatnonzero(mask.sum(1)>=max(3,int((db[2]-db[0])*.015)))
    return (float(rows[-1])-db[1])/(db[3]-db[1])

def evaluate(geo,traj,dmasks,contacts,protected,diagnose_unreviewed=False,ego_bottom_guard_px=0):
    sid=traj['source_id'];c=geo.sources[sid];d=geo.sources[traj['donor_source_id']]
    known=set(protected);overlaps={tok:[] for tok in known};hole_overlaps={tok:[] for tok in known};contact_errors=[];actual_sizes=[];unverified_hits=[]
    # 上界含1px插值支持、1px窄feather以及最多4px洞边。
    for i,(f,df,pose) in enumerate(zip(c['frames'],d['frames'],traj['frames'])):
        T=warp_matrix(df,pose,contacts[i]);a=cv2.warpAffine(dmasks[i].astype(np.uint8),T,(1024,576),flags=cv2.INTER_NEAREST)>0
        if a.sum()<200:return None,'empty_warp'
        yy,xx=np.where(a);wh=[int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)];actual_sizes.append(wh)
        if wh[0]<72 or wh[1]<40:return None,'actual_actor_too_small'
        H=cv2.dilate(a.astype(np.uint8),np.ones((13,13),np.uint8))>0
        if ego_bottom_guard_px and H[576-ego_bottom_guard_px:].any():return None,'conservative_ego_image_band'
        bb=pose['box'];rows=np.flatnonzero(a.sum(1)>=3);err=abs(rows[-1]-bb[3]);contact_errors.append(float(err))
        if err>3:return None,'contact_anchor_jitter'
        for tok,masks in protected.items():
            pm=masks[i];overlaps[tok].append(float((a&pm).sum()/max(1,pm.sum())))
            hole_overlaps[tok].append(float((H&pm).sum()/max(1,pm.sum())))
        for ob in geo.obstacles[sid][i]:
            p=ob['_projection'];tok=ob['instance_token']
            if p is None:continue
            # 对未有精确mask的对象，投影包络仅作保守排除，不能据交叠断言分割错。
            x0,y0,x1,y1=p['box'];x0=max(0,int(np.floor(x0)));x1=min(1024,int(np.ceil(x1)));y0=max(0,int(np.floor(y0)));y1=min(576,int(np.ceil(y1)))
            if x1<=x0 or y1<=y0:continue
            hit=int(H[y0:y1,x0:x1].sum())
            if tok not in known and hit>max(12,.01*int(H.sum())):
                if not diagnose_unreviewed:return None,'unreviewed_actor_envelope_overlap'
                unverified_hits.append({'frame':i,'instance_token':tok,'category':ob['category'],'hole_fraction':hit/int(H.sum()),'A_in_front_of_envelope':pose['depth_interval'][1]<=.3+p['near_depth']})
            if tok in known and overlaps[tok][-1]>.01 and pose['depth_interval'][1]>.3+p['near_depth']:return None,'depth_order_uncertain'
    active=[tok for tok,v in overlaps.items() if max(v)>.01]
    if not active:typ='background'
    elif len(active)==1:
        tok=active[0];v=np.array(overlaps[tok]);h=np.array(hole_overlaps[tok])
        if runlen((v>=.3)&(v<=.8))<10 or max(v)>.85 or max(h)>.85:return None,'single_visibility_ratio'
        typ='single_actor'
    elif len(active)>=2:
        v=np.array([overlaps[t] for t in active]);h=np.array([hole_overlaps[t] for t in active])
        if runlen(np.all((v>=.2)&(v<=.7),axis=0))<10 or v.max()>.8 or h.max()>.85:return None,'dense_visibility_ratio'
        typ='dense_actors'
    else:return None,'unknown'
    return {'type':typ,'protected_instances':active,'occlusion_fraction':overlaps,'model_hole_occlusion_upper_bound':hole_overlaps,
            'contact_errors_px':contact_errors,'contact_fractions':contacts.tolist(),'minimum_actual_actor_size_px':np.min(actual_sizes,axis=0).tolist(),
            'unverified_actor_envelope_hits':unverified_hits,'render_allowed':not unverified_hits},None

def main(root,same_log=False,prefix='Q',mode='world_offset',allow_downsample=False,max_donors=None,dense_refine=False,diagnose_unreviewed=False,ego_bottom_guard_px=0,run_id='r2'):
    cv2.setNumThreads(1);geo=Geometry(root)
    sq={r['source_id']:r for r in read(root/'subagent_source_reviews.json')['clips']}
    mq={r['source_id']:r for r in read(root/'mask_review/independent_mask_reviews.json')['clips']}
    numeric={r['source_id']:r['numeric_gate_pass'] for r in read(root/'mask_review/mask_audit.json')['clips']}
    receivers=sorted(s for s in mq if numeric.get(s) and sq[s]['receiver_status']=='pass' and mq[s]['receiver_mask_status']=='pass')
    donors=sorted(s for s in mq if numeric.get(s) and sq[s]['donor_status']=='pass' and mq[s]['donor_mask_status']=='pass')
    masks={s:source_masks(root,s) for s in set(receivers)|set(donors)};contacts={}
    from iteration2.donor_gate import check_masks,load_policy
    donor_gate={s:check_masks(geo.sources[s]['actors'][0]['instance_token'],masks[s]) for s in donors}
    donors=[s for s in donors if donor_gate[s]['eligible_for_pairing']]
    dump(root/'donor_admission.json',{'policy':load_policy(),'sources':donor_gate})
    for s in donors:
        arr=[actor_contact(m,np.array(f['actors'][0]['projection']['box_xyxy'])) for m,f in zip(masks[s],geo.sources[s]['frames'])]
        contacts[s]=median_filter(np.array(arr),size=5,mode='nearest')
    secondary=read(root/'mask_review_secondary/independent_secondary_mask_reviews.json')['clips']
    protected={s:{geo.sources[s]['actors'][0]['instance_token']:masks[s]} for s in receivers}
    for r in secondary:
        if r['protected_mask_status']=='pass' and r['source_id'] in protected:
            protected[r['source_id']][r['instance_token']]=source_masks(root,r['source_id'],r['job_id'])
    offsets=OFFSETS if mode=='world_offset' else [(z,x) for z in [-6.,0.,4.5,6.,8.] for x in [-6.,-3.,-1.5,0.,1.5,3.,6.]]
    if dense_refine:
        receivers=[s for s in receivers if len(protected[s])>=2]
        offsets=[(z,x) for z in [1.,2.,3.,4.,5.,6.,7.] for x in [-4.5,-3.,-2.,-1.,0.,1.,2.,3.,4.5]]
    config={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':run_id,'donor_policy':load_policy()['version'],'offsets_m':offsets,'placement_mode':mode,
            'target_counts':{'background':18,'single_actor':20,'dense_actors':12},'max_proposals_per_receiver_type':4,
            'different_donor_log':not same_log,'different_donor_scene':True,'different_donor_track':True,'model_inference_selection':False,'model_hole_dilation_upper_bound_px':6,
            'proposal_revision':4 if allow_downsample else 3,'proposal_change_reason':'explicit actual-size downsample contrast; source provenance and independent synthesis review required',
            'max_ranked_donors_per_receiver':max_donors,'frame_counts':sorted({len(c['frames']) for c in geo.sources.values()}),
            'dense_refine':dense_refine,'image_bottom_guard_px':ego_bottom_guard_px or (64 if dense_refine else None),
            'diagnose_unreviewed_envelopes_only':diagnose_unreviewed,
            'actual_warped_actor_bbox_min_px':[72,40],'actual_size_reason':'P004 was visually uncertain despite projected GT size pass; reject tiny actual visible cutouts prospectively',
            'minimum_scale_ratio':0 if allow_downsample else .65,'maximum_scale_ratio':1.5,
            'downsample_policy':'actual alpha size >=72x40 plus visual QA replaces old .65 lower proxy' if allow_downsample else 'old .65 lower proxy retained',
            'ground_support_distance_max_m':2.5,'clearance_min_m':.3,'seed':42,'GT_generated':False}
    dump(root/'synthesis_config.json',config)
    accepted=[];rejections=Counter();summary=[]
    for sid in receivers:
        meta=geo.prepare(sid)
        if not meta['pass']:
            summary.append({'source_id':sid,'ground_pass':False,'candidate_counts':{}});continue
        c=geo.sources[sid];pools=defaultdict(list)
        ranked=[]
        for donor in donors:
            d=geo.sources[donor]
            if d['scene']==c['scene'] or d['actors'][0]['instance_token']==c['actors'][0]['instance_token']:continue
            if not same_log and d['log_token']==c['log_token']:continue
            if len(c['frames'])!=len(d['frames']):continue
            n=len(c['frames']);yaw=np.mean([abs(wrap(view_angles(c['frames'][i]['actors'][0],c['frames'][i])[0]-view_angles(d['frames'][i]['actors'][0],d['frames'][i])[0])) for i in [0,n//2,n-1]])
            if yaw>35:continue
            ranked.append((yaw,donor))
        # 各source有限格搜索，不根据模型生成图或人工分数调整。
        for _,donor in sorted(ranked)[:max_donors]:
            for lo,la in offsets:
                traj,why=geo.trajectory(sid,lo,la,donor,mode,0 if allow_downsample else .65)
                if traj is None:rejections[why]+=1;continue
                if dense_refine and max(f['box'][3] for f in traj['frames'])+6>=512:rejections['conservative_ego_image_band']+=1;continue
                q,why=evaluate(geo,traj,masks[donor],contacts[donor],protected[sid],diagnose_unreviewed,ego_bottom_guard_px)
                if q is None:rejections[why]+=1;continue
                typ=q['type']
                if dense_refine and typ!='dense_actors':rejections['not_dense_after_real_masks']+=1;continue
                if len(pools[typ])>=4:continue
                if any(x['donor_source_id']==donor and abs(x['offset_lateral_m']-la)<1.3 and abs(x['offset_longitudinal_m']-lo)<2 for x in pools[typ]):continue
                pools[typ].append(traj|q)
            if len(pools['background'])>=4 and len(pools['single_actor'])>=4 and (len(protected[sid])<2 or len(pools['dense_actors'])>=4):break
        for typ,arr in pools.items():accepted.extend(arr)
        row={'source_id':sid,'ground_pass':True,'candidate_counts':{k:len(v) for k,v in pools.items()}}
        summary.append(row);print('CANDIDATES',row,flush=True)
        dump(root/'pair_search_progress.json',{'completed_sources':summary,'candidate_count':len(accepted),'rejection_counts':dict(rejections)})
    selected=[]
    # 先轮转scene，每source每type最多4，避免前几个场景耗尽名额。
    for typ,limit in config['target_counts'].items():
        pool=[x for x in accepted if x['type']==typ]
        bysource=defaultdict(list)
        for x in pool:bysource[x['source_id']].append(x)
        count=0
        for rank in range(4):
            for sid in sorted(bysource):
                if rank<len(bysource[sid]) and count<limit:
                    x=bysource[sid][rank];x['case_id']=f'{prefix}{len(selected)+1:03}';selected.append(x);count+=1
    for i,pair in enumerate(selected):
        pair['edge_mode']=['near_hard','feather_05','feather_10'][i%3]
        pair['extra_motion_blur_px']=[0,.4,.8][i%3]
        pair['mask_dilation_px']=[2,3,4][i%3]
        pair['synthetic_quality']='pending_render_and_independent_review';pair['human_verdict']=None
    dump(root/'pair_candidates.json',{'config':config,'summary':summary,'rejection_counts':dict(rejections),'candidate_count':len(accepted),'selected':selected,
                                     'selected_counts':dict(Counter(p['type'] for p in selected)),'qualified_synthetic_count':0})
    print('SELECTED',len(selected),dict(Counter(p['type'] for p in selected)),dict(rejections),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--same-log',action='store_true');p.add_argument('--prefix',default='Q');p.add_argument('--mode',choices=['world_offset','donor_camera_replay'],default='world_offset');p.add_argument('--allow-downsample',action='store_true');p.add_argument('--max-donors',type=int);p.add_argument('--dense-refine',action='store_true');p.add_argument('--diagnose-unreviewed',action='store_true');a=p.parse_args();main(a.root,a.same_log,a.prefix,a.mode,a.allow_downsample,a.max_donors,a.dense_refine,a.diagnose_unreviewed)
