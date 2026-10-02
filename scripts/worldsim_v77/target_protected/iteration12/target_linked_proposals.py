"""r33：由指定清楚B的图像边缘反解静止A位置，替代与主体脱节的车道采样。

每个已有主B只解f0/15/29的左右半遮两种位置，最多6解。原RGB、空间、
连续性、H、证据门槛不变；完整Y/SAM仅用于离线数据规划及质检。
"""
from pathlib import Path
import sys,os,copy,math,time,json
from collections import Counter
sys.path.insert(0,str(Path(__file__).parent))
import reveal_factory as f
from recover_instance_audit import overlapping
from prepare_instance_audit import clean
from scipy.optimize import brentq
from pyquaternion import Quaternion
import numpy as np
from PIL import Image,ImageDraw
import cv2
O=f.T/'r33'


def pose(xy,yaw,size,plane):
    R=f.ground_orientation(yaw,plane)
    center=np.r_[xy,np.dot(np.r_[xy,1.],plane)]+R[:,2]*size[2]/2
    return {'translation':center.tolist(),'rotation':Quaternion(matrix=R).elements.tolist(),'size':size}


def solve(g,c,tok,frame_index,side,size,mask):
    frame=c['frames'][frame_index]
    b=next(a for a in frame['actors'] if a['instance_token']==tok)
    plane=np.array(g.ground[c['source_id']]['plane']);rotation=Quaternion(b['rotation']).rotation_matrix
    yaw=math.atan2(rotation[1,0],rotation[0,0]);R=f.ground_orientation(yaw,plane)
    yy,xx=np.where(mask)
    if not len(xx) or np.ptp(xx)+1<72 or np.ptp(yy)+1<40:return None,'primary_B_not_well_observed_at_anchor'
    prb=f.projection(b,frame)
    if prb is None:return None,'B_projection_missing'
    # 固定A最远深度在B最近深度前0.6m；余下一维平移求遮挡边缘位置。
    m=frame['_w2c'];zero=pose(np.zeros(2),yaw,size,plane)
    corners=np.array([[sx*size[1]/2,sy*size[0]/2,sz*size[2]/2]
                      for sx in [-1,1] for sy in [-1,1] for sz in [-1,1]])@R.T
    zextent=float((corners@m[2,:3]).max())
    coeff=m[2,:2]+m[2,2]*plane[:2]
    const=float(m[2,:3]@np.array(zero['translation'])+m[2,3])
    wanted=prb['near_depth']-.6-zextent-const
    if np.linalg.norm(coeff)<1e-6:return None,'degenerate_depth_plane'
    anchor=np.array(b['translation'])[:2]
    base=anchor+coeff*(wanted-np.dot(coeff,anchor))/np.dot(coeff,coeff)
    lateral=np.array([-coeff[1],coeff[0]])/np.linalg.norm(coeff)
    # 外扩H的方向常数仅用于求初值；最终H始终从实际网格轮廓重新算。
    desired=float(np.median(xx))+(-12 if side=='left' else 11)
    edge=2 if side=='left' else 0
    def delta(s):
        pr=f.projection(pose(base+lateral*s,yaw,size,plane),frame)
        if pr is None:raise ValueError('A crosses camera plane')
        return float(pr['box'][edge]-desired)
    try:
        left,right=delta(-12.),delta(12.)
        if left*right>0:return None,'edge_constraint_no_bracket'
        shift=brentq(delta,-12.,12.,xtol=1e-4,maxiter=32)
    except ValueError:return None,'projection_equation_invalid'
    xy=base+lateral*shift
    q={'token':'view_edge_solution_not_a_lane_token','s':0.,'path':np.array([[*xy,yaw],[*(xy+.01*R[:2,0]),yaw]]),
       'distances':np.array([0.,.01])}
    return q,None


def reference_quality(g,c,tr,masks,base_pm,label_cache):
    """沿用r28完整实例核验；缺标签就隔离，不把缺标签当空背景。"""
    sid=c['source_id'];holes=[f.prepare_masks(a)[0]['model_mask'] for a in masks]
    original=g.obstacles[sid];obs=copy.deepcopy(original);aug=copy.deepcopy(c);pm=dict(base_pm)
    hits=overlapping(holes,obs);labels={};missing=[];identity=[];scope=[];retained=[]
    for tok,hit in hits.items():
        jid=sid+'_'+tok[:8];path=f.T/'r28/quality_labels/observed_masks'/jid
        if tok in base_pm:mm=base_pm[tok]
        elif (path/'result.json').is_file():
            if jid not in label_cache:label_cache[jid]=[np.asarray(Image.open(path/f'{i:05}.png'))>0 for i in range(30)]
            mm=label_cache[jid]
        else:missing.append({'instance':tok,'category':hit['category']});continue
        annotations=[next((ob for ob in oo if ob['instance_token']==tok),None) for oo in original]
        labels[tok]=mm
        if not any(m.any() for m in mm):identity.append(tok);continue
        full=all(ob is not None and not ob.get('interpolation_uncertain',False) for ob in annotations)
        if hit['category'].startswith('vehicle.') and full:
            pm[tok]=mm
            for i,fr in enumerate(aug['frames']):
                if not any(a['instance_token']==tok for a in fr['actors']):fr['actors'].append(annotations[i])
        elif hit['category'] in {'movable_object.barrier','movable_object.trafficcone','static_object.bicycle_rack'}:pass
        elif any((h&m).any() for h,m in zip(holes,mm)):scope.append(tok)
        for i,objects in enumerate(obs):
            if (holes[i]&mm[i]).any():continue
            for ob in objects:
                if ob['instance_token']==tok:ob['_projection']=None
        retained.append({'instance_token':tok,'category':hit['category'],'annotations':clean(annotations),
            'boxes':[clean(ob['_projection']['box']) if ob is not None and ob['_projection'] is not None else None for ob in annotations],
            'Y_quality_job':jid,'quality_input_available':True,'curated_source_actor':tok in base_pm})
    keys=sorted(labels)
    for j,a in enumerate(keys):
        for b in keys[j+1:]:
            if sum((ma&mb).sum()/max(1,(ma|mb).sum())>.35 for ma,mb in zip(labels[a],labels[b]))>=3:identity.append([a,b])
    detail={'missing_reference':missing,'identity_flags':identity,'scope_flags':scope,'retained_instances':retained}
    if missing:return None,'reference_labels_missing',detail
    if identity:return None,'instance_identity_or_visibility_uncertain',detail
    if scope:return None,'nonvehicle_or_incomplete_protection',detail
    g.obstacles[sid]=obs
    try:q,why=f.exact(g,aug,tr,masks,pm,reveal_policy='primary_plus_preserved')
    finally:g.obstacles[sid]=original
    return q,why,detail


def save_candidate(c,tr,masks,quality,detail,cid):
    dest=O/'observed'/cid
    for role in ['rgb','H','influence','proposal_H']:(dest/role).mkdir(parents=True,exist_ok=True)
    source=O/'quality_only'/c['source_id'];source.mkdir(parents=True,exist_ok=True)
    frames=[]
    for i,(fr,a) in enumerate(zip(c['frames'],masks)):
        y=np.asarray(Image.open(f.ROOT/'rgb'/fr['filename']).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS))
        contract,_=f.prepare_masks(a);h=contract['model_mask'];guard=contract['write_mask'];x=y.copy();x[guard]=127
        if not (source/f'{i:05}.png').exists():Image.fromarray(y).save(source/f'{i:05}.png')
        Image.fromarray(x).save(dest/'rgb'/f'{i:05}.png')
        for role,arr in [('H',guard),('proposal_H',h),('influence',a)]:Image.fromarray(arr.astype('uint8')*255).save(dest/role/f'{i:05}.png')
        assert h[a].all() and not np.any(x[~h]!=y[~h]) and not h[512:].any()
        if i in [0,15,29]:
            label=y.copy();label[a]=np.rint(label[a]*.3+np.array([255,195,30])*.7).astype('uint8')
            for tok in quality['reveal_instances']:
                original=next((v for v in fr['actors'] if v['instance_token']==tok),None)
                if original is not None:bb=original['projection']['box_xyxy']
                else:
                    other=next(v for v in detail['retained_instances'] if v['instance_token']==tok)
                    bb=other['boxes'][i]
                if bb is None:continue
                x0,y0,x1,y1=np.rint(bb).astype(int)
                cv2.rectangle(label,(x0,y0),(x1,y1),(35,255,130),2)
            model=y.copy();model[h]=127;frames.append((i,[y,label,model]))
    row={'case_id':cid,'source_id':c['source_id'],'scene':c['scene'],'split':c['source_split'],
         'asset':tr['asset'],'trajectory':tr,'frames':clean(c['frames']),'camera':c['camera'],
         'source_Y_quality_only':str(source),'observed_folder':str(dest),'retained_instances':detail['retained_instances'],
         'quality':quality,'input_contract':{'frames':30,'GT_untouched':True,'synthetic_RGB_leak':0},
         'training_admission':False,'human_verdict':None}
    row=clean(row);f.dump(dest/'case.json',row)
    sheet=Image.new('RGB',(1536,966),(18,24,32));draw=ImageDraw.Draw(sheet)
    for r,(i,images) in enumerate(frames):
        for k,(name,im) in enumerate(zip(['real Y QA only','A yellow / intended B box green','final masked input'],images)):
            draw.text((512*k+6,322*r+7),f'{cid} f{i} '+name,fill='white');sheet.paste(Image.fromarray(im).resize((512,288)),(512*k,322*r+28))
    (O/'contacts').mkdir(exist_ok=True);sheet.save(O/'contacts'/f'{cid}.jpg',quality=95)
    return row


def main():
    cv2.setNumThreads(1);O.mkdir(exist_ok=True)
    f.legacy.O=f.O;f.legacy.ROOT=f.ROOT;g=f.legacy.geometry()
    plan={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r33',
        'question':'can linking stationary occluder proposals to the intended clear B produce actual reveal without changing input/quality rules?',
        'sources':sorted(g.sources),'source_splits_unchanged':True,'frames':30,'anchor_frames':[0,15,29],
        'sides':['left','right'],'max_proposals_per_source_actor':6,'A_motion':'world stationary',
        'yaw':'copy real B heading at anchor, then ground-align and hold constant',
        'solution':'A far depth = B near depth - 0.6m; projected H side at median visible B x',
        'solve_lateral_bracket_m':[-12,12],'main_B_anchor_min_visible_size_px':[72,40],
        'main_B_must_be_original_curated_instance':True,'space_mask_evidence_thresholds_unchanged':True,
        'GT_camera_tracks_LiDAR':'declared offline POC auxiliary','Y_RGB_SAM':'offline quality only; never method condition',
        'unknown_reference_action':'quarantine and count, do not assume empty','training_steps':0,'surfel':False,
        'stop_rule':'one fixed six-solution pass per source actor; no offsets, velocities or source-window grid afterwards',
        'failure_ledger_refs':['V77-F02'],'human_verdict':None}
    if (O/'run.json').exists():assert f.read(O/'run.json')==plan,'不能更改已登记方案后静默续跑'
    else:f.dump(O/'run.json',plan)
    rows=f.read(O/'proposal_audit.json')['rows'] if (O/'proposal_audit.json').exists() else []
    accepted=f.read(O/'prepared.json')['cases'] if (O/'prepared.json').exists() else []
    done={r['source_id'] for r in rows};cache={};started=time.time()
    assets={a:dict(np.load(f.T/'r8/assets'/f'{a}.npz')) for a in ['sedan','suv']}
    for sid in sorted(g.sources):
        if sid in done:continue
        c=g.sources[sid];ground=g.prepare(sid);pm=f.legacy.protections(g,sid);local=[];keep=0;seen_poses=[]
        if not ground['pass'] or not pm:continue
        f.legacy.support(g,sid);asset=f.POLICY['split_shape'][c['source_split']];mesh=assets[asset]
        size=[1.85,4.5,1.5] if asset=='sedan' else [1.9,4.6,1.7]
        for tok in sorted(pm):
            for anchor in [0,15,29]:
                for side in ['left','right']:
                    record={'source_id':sid,'scene':c['scene'],'split':c['source_split'],'intended_B':tok,'anchor':anchor,'side':side}
                    q,why=solve(g,c,tok,anchor,side,size,pm[tok][anchor])
                    if q is not None:
                        tr,why=f.trajectory(g,c,q,0.,asset)
                        if tr is not None:
                            xyz=np.array(tr['frames'][0]['actor']['translation'])
                            if any(np.linalg.norm(xyz-prior)<.001 for prior in seen_poses):
                                record['result']='duplicate_static_position';local.append(record);rows.append(record);continue
                            seen_poses.append(xyz)
                            tr.pop('lane_token');tr.pop('lane_midpoint_distance_m')
                            tr.update(proposal_kind='target_linked_depth_and_image_edge',intended_B=tok,anchor_frame=anchor,side=side)
                            masks=[f.legacy.old.silhouette(mesh['vertices'],mesh['faces'],a['actor'],fr) for a,fr in zip(tr['frames'],c['frames'])]
                            quality,why,detail=reference_quality(g,c,tr,masks,pm,cache)
                            record['reference_detail']={k:v for k,v in detail.items() if k!='retained_instances'}
                            if quality is not None and tok not in quality.get('reveal_instances',[]):quality=None;why='intended_B_does_not_reveal'
                            if quality is not None:
                                if keep>=2:why='source_quota_full'
                                else:
                                    cid=f'P{len(accepted)+1:03}';accepted.append(save_candidate(c,tr,masks,quality,detail,cid));keep+=1;why='technical_candidate';record['case_id']=cid
                    record['result']=why;local.append(record);rows.append(record)
        f.dump(O/'proposal_audit.json',{'rows':rows,'counts':dict(Counter(r['result'] for r in rows))})
        f.dump(O/'prepared.json',{'cases':accepted,'training_admission':0})
        f.dump(O/'controller_state.json',{'stage':'planning','pid':os.getpid(),'last_source':sid,'attempts':len(rows),
            'candidates':len(accepted),'seconds':time.time()-started})
        print('TARGET_LINKED',sid,len(accepted),dict(Counter(r['result'] for r in local)),flush=True)
    f.dump(O/'controller_state.json',{'stage':'complete_pending_candidate_QA','pid':os.getpid(),'attempts':len(rows),
        'candidates':len(accepted),'seconds':time.time()-started,'training_steps':0})
    print('TARGET_LINKED_DONE',len(accepted),dict(Counter(r['result'] for r in rows)),flush=True)

if __name__=='__main__':main()
