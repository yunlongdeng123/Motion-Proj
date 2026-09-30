"""全帧CPU训练合同：RGB泄漏和latent mask覆盖分别报告，不能混为同一失败。"""
import argparse,json,sys
from pathlib import Path
from collections import Counter
import numpy as np,cv2
from PIL import Image
from pyquaternion import Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_pairs import warp_matrix
from geometry_factory import projection

def read(p):return json.loads(Path(p).read_text())
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def mask(p):return np.asarray(Image.open(p))>0
def audit(root,old):
    cv2.setNumThreads(1);reviews=read(root/'ai_case_reviews.json')['reviews'];wanted={r['case_id'] for r in reviews if r['score']==1}
    rows=[];total=0
    for branch in ['', 'expanded_factory','native_window_factory']:
        factory=old/branch;manifest=read(factory/'synthetic_review/synthetic_manifest.json');sources={c['source_id']:c for c in read(factory/'source_manifest.json')['clips']}
        for c in manifest['clips']:
            cid=c['case_id']
            if cid not in wanted:continue
            case=factory/'synthetic'/cid;frames=[];previous=None;previous_pose=None;bad=[];maxd=0;maxyaw=0;minnormiou=1;maxscale=1
            for i,f in enumerate(c['frames']):
                y=np.asarray(Image.open(case/'Y'/f'{i:03}.png').convert('RGB'));x=np.asarray(Image.open(case/'X'/f'{i:03}.png').convert('RGB'))
                h=mask(case/'model_hole'/f'{i:03}.png');influence=mask(case/'influence'/f'{i:03}.png')
                changed=np.any(x!=y,axis=-1);leak=int((changed&~h).sum());uncovered=int((influence&~h).sum())
                cx=x.astype(np.float32)/127.5-1;cy=y.astype(np.float32)/127.5-1;cx[h]=0;cy[h]=0
                assert leak==0 and uncovered==0 and np.array_equal(cx,cy)
                # CLIP首帧仍先masked再resize；这里验证像素因果性，不模拟权重前向。
                resized_equal=np.array_equal(cv2.resize(cx,(224,224),interpolation=cv2.INTER_CUBIC),cv2.resize(cy,(224,224),interpolation=cv2.INTER_CUBIC));assert resized_equal
                latent=cv2.resize(h.astype(np.uint8),(128,72),interpolation=cv2.INTER_NEAREST)>0
                up=cv2.resize(latent.astype(np.uint8),(1024,576),interpolation=cv2.INTER_NEAREST)>0
                coarse_uncovered=int((influence&~up).sum())
                # 安全latent洞候选：任何受合成影响的8x8 cell均为洞；输入RGB洞不改。
                latent_any=h.reshape(72,8,128,8).any(axis=(1,3));up_any=np.repeat(np.repeat(latent_any,8,0),8,1)
                assert not np.any(influence&~up_any)
                clear=~h;remaining={};collateral={};lowres_extra={};depth_order={}
                sf=sources[c['source_id']]['frames'][i];sf['_w2c']=np.linalg.inv(sf['camera_to_world']);bytoken={a['instance_token']:a for a in sf['actors']}
                for tok in c['protected_instances']:
                    pm=mask(case/'protected'/f'{i:03}_{tok}.png')
                    remaining[tok]=float((pm&clear).sum()/max(1,pm.sum()));collateral[tok]=float((pm&h&~influence).sum()/max(1,pm.sum()))
                    lowres_extra[tok]=float((pm&up_any&~h).sum()/max(1,pm.sum()))
                    if tok in bytoken:
                        bp=projection(bytoken[tok],sf);depth_order[tok]=bool(bp is not None and f['depth_interval'][1]<=bp['near_depth']+.3)
                    else:depth_order[tok]=None
                box=np.asarray(f['box']);x0,y0,x1,y1=np.rint(box).astype(int);cut=h[max(y0,0):min(y1+1,576),max(x0,0):min(x1+1,1024)]
                norm=cv2.resize(cut.astype(np.uint8),(128,128),interpolation=cv2.INTER_NEAREST)>0
                if previous is not None:
                    dt=(f['timestamp']-previous_pose['timestamp'])/1e6;assert dt>0
                    dist=np.linalg.norm(np.array(f['actor']['translation'])-np.array(previous_pose['actor']['translation']))*.1/dt;maxd=max(maxd,float(dist))
                    dot=abs(float(np.dot(Quaternion(f['actor']['rotation']).elements,Quaternion(previous_pose['actor']['rotation']).elements)));yaw=2*np.arccos(np.clip(dot,0,1))*180/np.pi*.1/dt;maxyaw=max(maxyaw,float(yaw))
                    prevbox=np.asarray(previous_pose['box']);ratio=(box[2:]-box[:2])/(prevbox[2:]-prevbox[:2]);maxscale=max(maxscale,float(np.max(np.maximum(ratio,1/ratio)**(.1/dt))))
                    iou=float((norm&previous).sum()/max(1,(norm|previous).sum()));minnormiou=min(minnormiou,iou)
                previous=norm;previous_pose=f
                yy,xx=np.where(h);frames.append({'frame':i,'timestamp_us':f['timestamp'],'RGB_leak_pixels':leak,'uncovered_influence_pixels':uncovered,
                    'masked_X_equals_masked_Y':True,'post_mask_resize_equal':resized_equal,'latent_nearest_upsample_uncovered_influence_px':coarse_uncovered,
                    'latent_any_cell_uncovered':0,'latent_any_extra_area_px':int((up_any&~h).sum()),
                    'latent_any_extra_protected_fraction':lowres_extra,'remaining_protected_fraction':remaining,
                    'A_far_before_B_near_with_0_3m_tolerance':depth_order,
                    'dilation_collateral_protected_fraction':collateral,'hole_bottom_px':int(yy.max())})
                total+=1
            if c['min_GT_clearance_m']<.3:bad.append('GT_clearance_below_0.3m')
            if maxd>2.001:bad.append('trajectory_step_over_2m_per_100ms')
            if maxyaw>5.001:bad.append('yaw_step_over_5deg_per_100ms')
            if maxscale>1.18+1e-4:bad.append('scale_jump')
            if minnormiou<.8:bad.append('normalized_hole_continuity_flag')
            # 保留先前明确真实图证据，不把底64px代理当已证实ego交叠。
            known_bad=cid in ('D006','D009')
            if known_bad:bad.append('known_actual_ego_hood_overwrite')
            if any(min(f['remaining_protected_fraction'].values(),default=1)<.15 for f in frames):bad.append('protected_visible_fraction_below_0.15')
            if any(v is not True for f in frames for v in f['A_far_before_B_near_with_0_3m_tolerance'].values()):bad.append('depth_order_missing_or_failed')
            row={'case_id':cid,'frame_count':len(frames),'geometry_record':{k:c[k] for k in ['type','min_GT_clearance_m','max_ground_support_distance_m','max_view_yaw_delta_deg','max_view_pitch_delta_deg','ground']},
                'max_translation_step_m_per_100ms':maxd,'max_rotation_step_deg_per_100ms':maxyaw,'max_scale_ratio_per_100ms':maxscale,
                'min_bbox_normalized_hole_IoU':minnormiou,'RGB_leak_pixels':0,'full_resolution_coverage_pass':True,
                'latent_nearest_hole_alias_frames':sum(f['latent_nearest_upsample_uncovered_influence_px']>0 for f in frames),
                'latent_mask_alias_is_RGB_leak':False,'engineering_flags':bad,'engineering_pass':not bad,'frames':frames,
                'spatial_limit':'GT/LiDAR/placement record supports placement; real silhouette ground-contact and ego visibility still require sampled visual check',
                'human_verdict':None}
            rows.append(row);print(cid,'frames',len(frames),'flags',bad,'latent_alias',row['latent_nearest_hole_alias_frames'],flush=True)
    assert {r['case_id'] for r in rows}==wanted
    result={'cases':rows,'case_count':len(rows),'frames':total,'RGB_leak_pixels':0,'engineering_flagged_cases':sum(not r['engineering_pass'] for r in rows),
        'latent_nearest_alias_cases':sum(r['latent_nearest_hole_alias_frames']>0 for r in rows),
        'processing_order':'X final1024x576 -> normalize -> zero full-resolution H -> conditional VAE; first masked frame -> CLIP resize. H separately nearest-downsamples to latent mask. No raw X object reference.',
        'latent_proposal':'any-pixel max pool would fully cover affected latent cells; keep as documented optional candidate, not silently modify frozen loader/input. It can remove additional protected spatial evidence.',
        'scope':'all AI=1 cases, CPU all frames; no VAE/UNet forward, no regenerated outputs, temporal metrics do not certify all-frame visual identity'}
    dump(root/'training_input_audit.json',result);print({k:v for k,v in result.items() if k!='cases'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old',type=Path,required=True);a=p.parse_args();audit(a.root,a.old)
