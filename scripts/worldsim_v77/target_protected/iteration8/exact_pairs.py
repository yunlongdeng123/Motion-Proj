"""精确实例轮廓审查候选，scene均衡选择；质量缺额保留，不放宽门槛。"""
from pathlib import Path
import sys,os,copy
from collections import Counter,defaultdict
from functools import lru_cache
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
import numpy as np,cv2
from scipy.ndimage import median_filter
from geometry_factory import Geometry,read,dump,source_masks
from build_pairs import actor_contact,evaluate,warp_matrix
from iteration2.donor_gate import check_masks
from iteration5.planning import normalized_masks
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r8';F=O/'factory'
def main():
    cv2.setNumThreads(1)
    assert read(O/'sam_state.json')['stage']=='complete_pending_QA'
    geo=Geometry(F);valscenes=set(read(O/'source_split.json')['synthetic_validation_scenes']);audit={};secondary=[]
    @lru_cache(maxsize=16)
    def primary(sid):
        c=geo.sources[sid];m=source_masks(F,sid);stats=normalized_masks(m,[f['actors'][0]['projection']['box_xyxy'] for f in c['frames']])
        gate=check_masks(c['actors'][0]['instance_token'],m);good=gate['eligible_for_pairing'] and all(s['pixels'] and s['normalized_iou']>=.8 and .85<=s['normalized_area_ratio']<=1.15 for s in stats)
        audit[sid]={'primary_pass':bool(good),'stats':stats,'donor_gate':gate}
        if not good:return None,None
        contacts=median_filter(np.array([actor_contact(mm,np.array(f['actors'][0]['projection']['box_xyxy'])) for mm,f in zip(m,c['frames'])]),size=5,mode='nearest')
        return m,contacts
    @lru_cache(maxsize=8)
    def protections(sid):
        c=geo.sources[sid];m,_=primary(sid)
        if m is None:return None
        protected={c['actors'][0]['instance_token']:m}
        for a in c['actors'][1:]:
            tok=a['instance_token'];jid=sid+'_'+tok[:8]
            if not (F/'segmented_secondary'/jid/'sam2_raw').exists():continue
            masks=source_masks(F,sid,jid);stats=normalized_masks(masks,[next(a for a in f['actors'] if a['instance_token']==tok)['projection']['box_xyxy'] for f in c['frames']])
            good=all(s['pixels'] and s['normalized_iou']>=.8 and .85<=s['normalized_area_ratio']<=1.15 for s in stats)
            audit[sid].setdefault('secondary',{})[tok]={'pass':bool(good),'stats':stats}
            if good:protected[tok]=masks;secondary.append({'source_id':sid,'instance_token':tok,'job_id':jid,'protected_mask_status':'pass','basis':'numeric gate; all final masks independently reviewed with synthetic case before training'})
        return protected
    pools=[];total=Counter();rows=[];(O/'exact').mkdir(exist_ok=True)
    for sid,c in geo.sources.items():
        if not sid.startswith('N'):continue
        selfpath=O/'preflight'/f'{sid}.json';crosspath=O/'cross_preflight'/f'{sid}.json'
        plans=read(selfpath)['plans']+(read(crosspath)['plans'] if crosspath.exists() else [])
        if not plans:continue
        saved=O/'exact'/f'{sid}.json'
        # 不缓存mask准入，保证每次严格复查全帧；候选输出只有完整结束才可使用。
        protected=protections(sid)
        if protected is None:rows.append({'source_id':sid,'rejected':'primary_mask_gate'});continue
        geo.prepare(sid);reject=Counter();bytype=defaultdict(list)
        for tr in plans:
            ds=tr['donor_source_id'];d=geo.sources[ds]
            if c['scene'] in valscenes and d['scene']!=c['scene']:continue
            if c['scene'] not in valscenes and d['scene'] in valscenes:continue
            dm,contacts=primary(ds)
            if dm is None:reject['template_mask_gate']+=1;continue
            # nuScenes的barrier/cone/rack是待恢复真实背景，不能误算为未知保护车。
            # 只在新增A的整个深度区间位于其前方时解除未知包络拒绝；行人/车辆不解除。
            static_ok=set();static_bad=False
            kinds={'movable_object.barrier','movable_object.trafficcone','static_object.bicycle_rack'}
            for i,(df,pose,mm,ct) in enumerate(zip(d['frames'],tr['frames'],dm,contacts)):
                aa=cv2.warpAffine(mm.astype('uint8'),warp_matrix(df,pose,ct),(1024,576),flags=cv2.INTER_NEAREST)>0
                hh=cv2.dilate(aa.astype('uint8'),np.ones((13,13),'uint8'))>0
                for ob in geo.obstacles[sid][i]:
                    pr=ob['_projection']
                    if ob['category'] not in kinds or pr is None:continue
                    bb=pr['box'];x0,y0=np.maximum(np.floor(bb[:2]).astype(int),0);x1,y1=np.minimum(np.ceil(bb[2:]).astype(int),[1024,576])
                    if x1<=x0 or y1<=y0 or hh[y0:y1,x0:x1].sum()<=max(12,.01*hh.sum()):continue
                    if ob.get('interpolation_uncertain') or pose['depth_interval'][1]>.3+pr['near_depth']:static_bad=True;break
                    static_ok.add(ob['instance_token'])
                if static_bad:break
            if static_bad:reject['static_foreground_or_depth_unknown']+=1;continue
            checking=dict(protected)
            for tok in static_ok:checking[tok]=[np.zeros((576,1024),dtype=bool)]*10
            q,why=evaluate(geo,tr,dm,contacts,checking,ego_bottom_guard_px=64)
            if q is None:reject[why]+=1;continue
            for field in ['occlusion_fraction','model_hole_occlusion_upper_bound']:
                q[field]={tok:v for tok,v in q[field].items() if tok in protected}
            kind=q['type']
            if len(bytype[kind])>=3:continue
            if any(p['donor_source_id']==ds and abs(p['offset_lateral_m']-tr['offset_lateral_m'])<1.3 and abs(p['offset_longitudinal_m']-tr['offset_longitudinal_m'])<2 for p in bytype[kind]):continue
            # 最终连续alpha也归一化复查；不能用来源轮廓稳定代替投影稳定。
            warped=[cv2.warpAffine(m.astype('uint8'),warp_matrix(df,p,ct),(1024,576),flags=cv2.INTER_NEAREST)>0 for m,df,p,ct in zip(dm,d['frames'],tr['frames'],contacts)]
            stats=normalized_masks(warped,[p['box'] for p in tr['frames']])
            if any(s['normalized_iou']<.8 or not .85<=s['normalized_area_ratio']<=1.15 for s in stats):reject['projected_silhouette_continuity']+=1;continue
            p=tr|q|{'scene':c['scene'],'donor_scene':d['scene'],'candidate_role':'validation' if c['scene'] in valscenes else 'train',
                       'static_background_annotations_behind_A':sorted(static_ok),
                       'silhouette_temporal_stats':stats,'template_semantics':'duplicated receiver instance at different safe 3D pose; all template RGB erased' if ds==sid else 'previous independently admitted donor silhouette'}
            bytype[kind].append(p)
        pool=[p for ar in bytype.values() for p in ar];pools+=pool;total.update(reject)
        row={'source_id':sid,'scene':c['scene'],'candidate_counts':{k:len(v) for k,v in bytype.items()},'rejections':dict(reject)};rows.append(row)
        dump(saved,{'summary':row,'candidates':pool});print('EXACT',sid,row['candidate_counts'],flush=True)
    dump(O/'mask_numeric_audit.json',{'sources':audit})
    dump(F/'mask_review_secondary/independent_secondary_mask_reviews.json',{'clips':list({r['job_id']:r for r in secondary}.values()),'scope':'numeric staging only, not independent visual QA; final case reviews override training admission'})
    dump(O/'all_exact_candidates.json',{'candidates':pools,'rows':rows,'counts':dict(Counter(p['type'] for p in pools)),'rejections':dict(total)})
    # 先类型稀缺（dense），再轮转scene；保留同场景最多3例与一个来源窗最多2例。
    selected=[];scene_counts=Counter();window_counts=Counter();type_counts=Counter()
    for role,limits in [('validation',{'dense_actors':4,'single_actor':6,'background':5}),('train',{'dense_actors':14,'single_actor':20,'background':16})]:
        for kind,limit in limits.items():
            pp=[p for p in pools if p['candidate_role']==role and p['type']==kind]
            for rank in range(3):
                for scene in sorted({p['scene'] for p in pp}):
                    choices=[p for p in pp if p['scene']==scene and p not in selected and window_counts[p['source_id']]<2]
                    if not choices or scene_counts[scene]>=3 or type_counts[role,kind]>=limit:continue
                    p=choices[0];selected.append(p);scene_counts[scene]+=1;window_counts[p['source_id']]+=1;type_counts[role,kind]+=1
    for i,p in enumerate(selected):
        p.update(case_id=f'Q{i+1:03}',edge_mode=['near_hard','feather_05','feather_10'][i%3],extra_motion_blur_px=[0,.4,.8][i%3],mask_dilation_px=[2,3,4][i%3],human_verdict=None,quality_status='pending_actual_render_and_independent_QA')
    summary={'total_selected':len(selected),'role_type_counts':{f'{r}:{t}':n for (r,t),n in type_counts.items()},'training_scenes':len({p['scene'] for p in selected if p['candidate_role']=='train'}),'validation_scenes':len({p['scene'] for p in selected if p['candidate_role']=='validation'}),'max_per_scene':max(scene_counts.values()) if scene_counts else 0}
    dump(F/'pair_candidates.json',{'config':{'run_id':'r8','image_bottom_guard_px':64},'selected':selected,'summary':summary,'training_ready':False});print('SELECTED',summary,flush=True)
if __name__=='__main__':main()
