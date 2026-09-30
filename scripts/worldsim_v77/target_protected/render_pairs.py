"""真实Y/连续donor合成X。保存无损合同与独立审核图，不调用生成模型。"""
import argparse,json,subprocess
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from geometry_factory import read,dump,source_masks
from build_pairs import warp_matrix
from data_contract import assert_pair_pixels,masked_condition_from_x
from iteration2.donor_gate import assert_donor

def font(n=18):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
def rgb(root,f):
    with Image.open(root/'rgb'/f['filename']) as im:return np.asarray(im.convert('RGB').resize((1024,576),Image.Resampling.LANCZOS)).copy()
def save(p,a):Image.fromarray(a).save(p,compress_level=1)
def label(x,a,protected):
    out=x.copy();colors=[(42,234,172),(54,175,255),(197,112,255)]
    cv2.drawContours(out,cv2.findContours(a.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(255,84,92),2)
    for j,(tok,p) in enumerate(protected.items()):
        color=colors[j%len(colors)];cv2.drawContours(out,cv2.findContours(p.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,color,2)
        yy,xx=np.where(p)
        if len(xx):cv2.putText(out,f'{chr(66+j)} {tok[:7]}',(int(xx.min()),max(22,int(yy.min())-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
    yy,xx=np.where(a)
    if len(xx):cv2.putText(out,'A DELETE',(int(xx.min()),max(22,int(yy.min())-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,84,92),2,cv2.LINE_AA)
    return out
def encode(folder,kind):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-framerate','10','-i',str(folder/f'%03d_{kind}.jpg'),'-c:v','libx264','-threads','2','-preset','medium','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/f'{kind}.mp4')],check=True)

def render(root,out,candidates,limit=None,shard_index=0,shard_count=1):
    cv2.setNumThreads(1);out.mkdir(parents=True,exist_ok=True);(out/'contacts').mkdir(exist_ok=True)
    sources={c['source_id']:c for c in read(root/'source_manifest.json')['clips']}
    secondary=read(root/'mask_review_secondary/independent_secondary_mask_reviews.json')['clips']
    selected=read(candidates)['selected'];selected=selected[:limit] if limit else selected
    assert 0<=shard_index<shard_count
    selected=selected[shard_index::shard_count]
    manifest_name='synthetic_manifest.json' if shard_count==1 else f'synthetic_manifest.part{shard_index}.json'
    cache={};maskcache={};rows=[]
    def frames(sid):
        if sid not in cache:cache[sid]=[rgb(root,f) for f in sources[sid]['frames']]
        return cache[sid]
    for pair in selected:
        if pair.get('render_allowed') is False or pair.get('unverified_actor_envelope_hits'):raise ValueError('diagnostic candidate is not admitted to rendering')
        cid=pair['case_id'];sid=pair['source_id'];dsid=pair['donor_source_id'];c=sources[sid];d=sources[dsid]
        dest=root/'synthetic'/cid;preview=out/'assets'/cid;dest.mkdir(parents=True,exist_ok=True);preview.mkdir(parents=True,exist_ok=True)
        for role in ['Y','X','alpha','influence','model_hole','protected','condition_preview']:(dest/role).mkdir(exist_ok=True)
        for s in [sid,dsid]:
            if s not in maskcache:maskcache[s]=source_masks(root,s)
        # 渲染入口再查，防止历史提案或人工选case绕过新供体规则。
        assert_donor(d,maskcache[dsid])
        protected={c['actors'][0]['instance_token']:maskcache[sid]}
        for r in secondary:
            if r['source_id']==sid and r['protected_mask_status']=='pass':protected[r['instance_token']]=source_masks(root,sid,r['job_id'])
        metrics=[];reviewframes=[];framesmeta=[]
        for i,pose in enumerate(pair['frames']):
            y=frames(sid)[i];donor=frames(dsid)[i];mask=maskcache[dsid][i].astype(np.float32)
            T=warp_matrix(d['frames'][i],pose,pair['contact_fractions'][i])
            alpha=cv2.warpAffine(mask,T,(1024,576),flags=cv2.INTER_LINEAR)
            premul=cv2.warpAffine(donor.astype(np.float32)*mask[...,None],T,(1024,576),flags=cv2.INTER_LINEAR)
            # RGB内部轻微增加模糊，保持真实donor已有模糊；不扩张alpha影响域。
            sigma=pair['extra_motion_blur_px']
            if sigma:
                premul=cv2.GaussianBlur(premul,(3,3),sigma);denom=cv2.GaussianBlur(alpha,(3,3),sigma)
            else:denom=alpha
            color=premul/np.maximum(denom[...,None],1e-6)
            # 3x3窄feather半径1，最大hole半径4，匹配候选5px上界。
            edge={'near_hard':0,'feather_05':.5,'feather_10':1}[pair['edge_mode']]
            if edge:
                # 外围颜色由premultiplied邻域传播，避免黑色halo。
                pa=cv2.GaussianBlur(color*alpha[...,None],(3,3),edge)
                alpha=cv2.GaussianBlur(alpha,(3,3),edge);color=pa/np.maximum(alpha[...,None],1e-6)
            influence=alpha>1/65535;alpha=np.where(influence,alpha,0)
            x=np.rint(y.astype(np.float32)*(1-alpha[...,None])+color*alpha[...,None]).clip(0,255).astype(np.uint8)
            radius=pair['mask_dilation_px'];hole=cv2.dilate(influence.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8))>0
            assert_pair_pixels(y,x,influence,hole)
            cond=masked_condition_from_x(x[None],hole[None])[0]
            assert np.array_equal(cond,masked_condition_from_x(y[None],hole[None])[0]),'A fully masked contract'
            # condition数据合同洞内是归一化0；RGB预览相应为中灰，非纯黑/生成结果。
            cp=np.rint((cond+1)*127.5).clip(0,255).astype(np.uint8)
            pm={tok:m[i] for tok,m in protected.items()};ann=label(x,alpha>=.5,pm)
            ratios={tok:float((hole&m).sum()/max(1,m.sum())) for tok,m in pm.items()}
            if any(v>.85 for v in ratios.values()):raise ValueError(f'{cid} f{i}: model_hole_protected_visibility')
            n=f'{i:03}'
            for key,arr in [('Y',y),('X',x),('alpha',np.rint(alpha*255).astype(np.uint8)),('influence',influence.astype(np.uint8)*255),('model_hole',hole.astype(np.uint8)*255),('condition_preview',cp)]:save(dest/key/f'{n}.png',arr)
            for tok,m in pm.items():save(dest/'protected'/f'{n}_{tok}.png',m.astype(np.uint8)*255)
            panels={'gt':y,'input':x,'labels':ann,'condition':cp}
            for kind,arr in panels.items():Image.fromarray(arr).save(preview/f'{n}_{kind}.jpg',quality=93)
            reviewframes.append(panels)
            metrics.append({'frame':i,'hole_fraction':float(hole.mean()),'remaining_protected_fraction':{t:1-v for t,v in ratios.items()},'pixel_contract':True,'condition_equals_masked_Y':True,'outside_hole_Y_only':True})
            framesmeta.append({'frame':i,'timestamp_us':c['frames'][i]['timestamp'],'source_filename':c['frames'][i]['filename'],**{k:f'assets/{cid}/{n}_{k}.jpg' for k in panels}})
        nframes=len(pair['frames'])
        maxocc=max(range(nframes),key=lambda i:max((v[i] for v in pair['occlusion_fraction'].values()),default=0))
        worst=max(range(nframes),key=lambda i:pair['contact_errors_px'][i])
        indices=list(dict.fromkeys([0,nframes//2,nframes-1,maxocc,worst]))
        # 单行四张完整画面+裁切，以便审核每个指定实际帧。
        for i in indices:
            sheet=Image.new('RGB',(1536,940),(13,19,29));draw=ImageDraw.Draw(sheet)
            draw.text((10,7),f'{cid} {sid} <- {dsid} | {pair["type"]} | frame {i}',font=font(21),fill='white')
            box=np.array(pair['frames'][i]['box']);boxes=[box]
            for tok in pair['protected_instances']:
                yy,xx=np.where(protected[tok][i]);boxes.append(np.array([xx.min(),yy.min(),xx.max(),yy.max()]))
            boxes=np.array(boxes);b=np.r_[boxes[:,:2].min(0)-25,boxes[:,2:].max(0)+25].astype(int);b[[0,2]]=b[[0,2]].clip(0,1024);b[[1,3]]=b[[1,3]].clip(0,576)
            for j,(kind,arr) in enumerate(reviewframes[i].items()):
                x0=j*384;draw.text((x0+8,42),kind,font=font(),fill='white')
                im=Image.fromarray(arr);sheet.paste(im.resize((384,216)),(x0,72))
                crop=im.crop(tuple(b));crop.thumbnail((378,570));sheet.paste(crop,(x0+(384-crop.width)//2,330))
            sheet.save(out/'contacts'/f'{cid}_f{i:02}.jpg',quality=96)
        for kind in ['gt','input','labels','condition']:encode(preview,kind)
        row=pair|{'scene':c['scene'],'donor_scene':d['scene'],'camera':c['camera'],'receiver_primary':c['actors'][0]['instance_token'],
                  'donor_instance':d['actors'][0]['instance_token'],'review_frames':indices,'contacts':[f'contacts/{cid}_f{i:02}.jpg' for i in indices],
                  'preview_frames':framesmeta,'pixel_metrics':metrics,'videos':{k:f'assets/{cid}/{k}.mp4' for k in ['gt','input','labels','condition']},
                  'quality_status':'pending_independent_review','human_verdict':None,'training_ready':False,'frame_count':nframes,
                  'source_window':c.get('window_provenance'),'donor_window':d.get('window_provenance'),
                  'condition_scope':'actual masked X contract; no DriveEditor forward; no Y hidden-region condition',
                  'synthetic_method':'real donor 2.5D affine cutout, camera/metric trajectory/LiDAR ground gated; no generated GT or neural relighting',
                  'extra_blur_kind':'clip-constant 3x3 isotropic Gaussian, preserving donor real video blur; no synthetic directional motion blur claim'}
        dump(dest/'pair_manifest.json',row);rows.append(row);dump(out/manifest_name,{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r2','stage':'synthetic_pending_independent_QA','clips':rows})
        print('RENDER',cid,pair['type'],'frames',len(metrics),flush=True)
    print('RENDERED',len(rows),dict(Counter(r['type'] for r in rows)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--candidates',type=Path);p.add_argument('--limit',type=int);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shard-count',type=int,default=1);a=p.parse_args();render(a.root,a.out,a.candidates or a.root/'pair_candidates.json',a.limit,a.shard_index,a.shard_count)
