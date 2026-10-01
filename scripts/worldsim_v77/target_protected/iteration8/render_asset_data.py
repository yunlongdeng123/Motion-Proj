"""完整X只诊断；精确真实Y、车辆洞H和受保护B/C保存为训练合同。"""
from pathlib import Path
import sys,os
from collections import Counter
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,F,ParkingGeometry,silhouette,exact,read,dump,source_masks
from render_pairs import label,save,encode,font
from data_contract import assert_pair_pixels,masked_condition_from_x
from iteration5.planning import normalized_masks
import numpy as np,cv2
from PIL import Image,ImageDraw
def main():
    geo=ParkingGeometry(F);out=O/'data_review';out.mkdir(exist_ok=True);(out/'contacts').mkdir(exist_ok=True)
    selected=read(O/'asset_selected.json')['selected'];rows=[]
    for p in selected:
        cid=p['case_id'];c=geo.sources[p['source_id']];geo.prepare(c['source_id']);dest=F/'synthetic'/cid;preview=out/'assets'/cid
        dest.mkdir(exist_ok=True,parents=True);preview.mkdir(exist_ok=True,parents=True)
        for role in ['Y','X','alpha','influence','model_hole','protected','condition_preview']:(dest/role).mkdir(exist_ok=True)
        data=dict(np.load(O/'assets'/(p['asset']+'.npz')));alphas=[silhouette(data['vertices'],data['faces'],pose['actor'],f) for pose,f in zip(p['frames'],c['frames'])]
        protected={}
        for j,a in enumerate(c['actors']):
            jid=c['source_id'] if j==0 else c['source_id']+'_'+a['instance_token'][:8]
            folder=F/('segmented' if j==0 else 'segmented_secondary')/jid/'sam2_raw'
            if not folder.exists():continue
            tok=a['instance_token'];mm=source_masks(F,c['source_id'],None if j==0 else jid);stats=normalized_masks(mm,[next(aa for aa in f['actors'] if aa['instance_token']==tok)['projection']['box_xyxy'] for f in c['frames']])
            if all(s['pixels']>0 and s['normalized_iou']>=.8 and .85<=s['normalized_area_ratio']<=1.15 for s in stats):protected[a['instance_token']]=mm
        q,why=exact(geo,p,alphas,protected);assert q is not None,why;assert q['type']==p['type']
        metrics=[];panels=[]
        for i,(f,aa) in enumerate(zip(c['frames'],alphas)):
            with Image.open(F/'rgb'/f['filename']) as im:y=np.asarray(im.convert('RGB').resize((1024,576),Image.Resampling.LANCZOS)).copy()
            alpha=aa.astype('float32');edge={'near_hard':0,'feather_05':.5,'feather_10':1}[p['edge_mode']]
            if edge:alpha=cv2.GaussianBlur(alpha,(3,3),edge)
            influence=alpha>1/65535;alpha=np.where(influence,alpha,0)
            # 灰色A只是轮廓/位置诊断，不作为光照或材质质量；条件编码前全擦除。
            tint=np.array([96,110,127],dtype='float32')
            x=np.rint(y.astype('float32')*(1-alpha[...,None])+tint*alpha[...,None]).clip(0,255).astype('uint8')
            r=p['mask_dilation_px'];hole=cv2.dilate(influence.astype('uint8'),np.ones((2*r+1,2*r+1),'uint8'))>0
            assert not hole[512:].any();assert_pair_pixels(y,x,influence,hole)
            cond=masked_condition_from_x(x[None],hole[None])[0];assert np.array_equal(cond,masked_condition_from_x(y[None],hole[None])[0])
            cp=np.rint((cond+1)*127.5).clip(0,255).astype('uint8');pm={tok:m[i] for tok,m in protected.items()}
            ann=label(x,aa,pm);ratios={tok:float((hole&m).sum()/max(1,m.sum())) for tok,m in pm.items()}
            assert max(ratios.values(),default=0)<=.85
            for role,arr in [('Y',y),('X',x),('alpha',np.rint(alpha*255).astype('uint8')),('influence',influence.astype('uint8')*255),('model_hole',hole.astype('uint8')*255),('condition_preview',cp)]:save(dest/role/f'{i:03}.png',arr)
            for tok,m in pm.items():save(dest/'protected'/f'{i:03}_{tok}.png',m.astype('uint8')*255)
            panel={'gt':y,'input':x,'labels':ann,'condition':cp};panels.append(panel)
            for role,arr in panel.items():Image.fromarray(arr).save(preview/f'{i:03}_{role}.jpg',quality=94)
            metrics.append({'frame':i,'hole_fraction':float(hole.mean()),'protected_hidden_canvas_fraction':float((hole&np.logical_or.reduce(list(pm.values()))).mean()) if pm else 0,'hole_overlap':ratios,'A_in_H':True,'masked_X_equals_masked_Y':True,'GT_exact_real_redecode':True})
        index=max(range(10),key=lambda i:max((v[i] for v in p['occlusion_fraction'].values()),default=0)) if p['type']!='background' else 5
        box=np.array(p['frames'][index]['box']);boxes=[box]
        for tok in p['protected_instances']:
            yy,xx=np.where(protected[tok][index]);boxes.append(np.array([xx.min(),yy.min(),xx.max(),yy.max()]))
        boxes=np.array(boxes);b=np.r_[boxes[:,:2].min(0)-25,boxes[:,2:].max(0)+25].astype(int);b[[0,2]]=b[[0,2]].clip(0,1024);b[[1,3]]=b[[1,3]].clip(0,576)
        sheet=Image.new('RGB',(1600,900),(14,21,31));draw=ImageDraw.Draw(sheet);draw.text((12,8),f'{cid} | {c["scene"]} | {p["type"]} | f{index} | {p["candidate_role"]}',font=font(22),fill='white')
        for j,(role,a) in enumerate(panels[index].items()):
            draw.text((j*400+8,43),role,font=font(),fill='white');im=Image.fromarray(a);sheet.paste(im.resize((400,225)),(j*400,72));crop=im.crop(tuple(b));crop.thumbnail((396,540));sheet.paste(crop,(j*400+(400-crop.width)//2,330))
        sheet.save(out/'contacts'/f'{cid}_f{index:02}.jpg',quality=96)
        for role in ['gt','input','labels','condition']:encode(preview,role)
        row=p|{'receiver_scene':c['scene'],'camera':c['camera'],'folder':str(dest),'frame_count':10,'review_frame':index,'review_contact':f'contacts/{cid}_f{index:02}.jpg','protected_tokens':list(protected),'pixel_metrics':metrics,'quality_status':'pending_independent_QA','human_verdict':None,'videos':{role:f'assets/{cid}/{role}.mp4' for role in ['gt','input','labels','condition']}}
        dump(dest/'pair_manifest.json',row);rows.append(row);dump(out/'synthetic_manifest.json',{'clips':rows,'summary':read(O/'asset_selected.json')['summary']});print('RENDER',cid,p['type'],flush=True)
    print('RENDERED',len(rows),dict(Counter(c['type'] for c in rows)),flush=True)
if __name__=='__main__':main()
