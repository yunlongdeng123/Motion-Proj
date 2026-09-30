"""冻结SAM2从已检查的中间帧mask双向传播，逐帧审核连续轮廓。"""
import argparse,json,sys,time,fcntl
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'iteration2'))
from geometry_factory import read,dump
from iteration2.donor_gate import check_masks
from iteration2.segment_multiview import CHECKPOINT,CONFIG

def main(root,folder):
    out=root/folder;sources=read(out/'source_manifest.json')['clips'];qa={r['view_id']:r for r in read(root/'still_mask_reviews.json')['reviews']}
    clips=[c for c in sources if c['geometry_pass'] and qa[c['seed_view']['view_id']]['status']=='pass']
    assert not read(out/'preparation.json')['missing_RGB']
    import torch
    assert torch.cuda.is_available();torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42)
    lock=open(out/'sam2.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
    from sam2.build_sam import build_sam2_video_predictor
    predictor=build_sam2_video_predictor(CONFIG,CHECKPOINT,device='cuda');reviews=[];qadir=out/'mask_review';qadir.mkdir(exist_ok=True)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for c in clips:
            sid=c['source_id'];dest=out/'segmented'/sid;rgbdir=dest/'rgb';mdir=dest/'sam2_raw';rgbdir.mkdir(parents=True,exist_ok=True);mdir.mkdir(exist_ok=True)
            for f in c['frames']:
                im=Image.open(out/'rgb'/f['filename']).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS);im.save(rgbdir/f"{f['frame']:05}.jpg",quality=96)
            prompt=c.get('prompt_frame',5);assert c['frames'][prompt]['filename']==c['seed_view']['filename']
            seed=root/'multiview/segmented'/c['seed_view']['view_id'];seedrow=read(seed/'mask_manifest.json');mask=np.asarray(Image.open(seed/seedrow['mask_files'][seedrow['primary_candidate']]))>0
            state=predictor.init_state(video_path=str(rgbdir),offload_video_to_cpu=True,offload_state_to_cpu=True)
            predictor.add_new_mask(state,frame_idx=prompt,obj_id=1,mask=mask)
            raw={};start=time.monotonic()
            for reverse in [False,True]:
                for fid,_,logits in predictor.propagate_in_video(state,start_frame_idx=prompt,reverse=reverse):raw[int(fid)]=(logits[0,0]>0).cpu().numpy()
            assert set(raw)==set(range(10));metrics=[];prev=None
            for i,m in sorted(raw.items()):
                Image.fromarray(m.astype(np.uint8)*255).save(mdir/f'{i:05}.png');b=np.array(c['frames'][i]['actors'][0]['projection']['box_xyxy']);s=128/(b[2:]-b[:2]);T=np.float32([[s[0],0,-b[0]*s[0]],[0,s[1],-b[1]*s[1]]]);norm=cv2.warpAffine(m.astype(np.uint8),T,(128,128),flags=cv2.INTER_NEAREST)>0
                metrics.append({'frame':i,'pixels':int(m.sum()),'normalized_iou':float((norm&prev).sum()/max(1,(norm|prev).sum())) if prev is not None else 1.,'normalized_area_ratio':float(norm.sum()/max(1,prev.sum())) if prev is not None else 1.});prev=norm
            gate=check_masks(c['actors'][0]['instance_token'],[raw[i] for i in range(10)]);bad=[r for r in metrics if not r['pixels'] or r['normalized_iou']<.8 or not .85<=r['normalized_area_ratio']<=1.15]
            fids=list(dict.fromkeys([0,5,9,min(metrics,key=lambda r:r['normalized_iou'])['frame']]))
            sheet=Image.new('RGB',(1400,390*len(fids)),(20,25,33));dr=ImageDraw.Draw(sheet)
            for j,i in enumerate(fids):
                a=np.asarray(Image.open(rgbdir/f'{i:05}.jpg')).copy();overlay=a.copy();overlay[raw[i]]=(a[raw[i]]*.55+np.array([40,230,140])*.45).astype(np.uint8)
                b=np.array(c['frames'][i]['actors'][0]['projection']['box_xyxy']);crop=(max(0,int(b[0])-18),max(0,int(b[1])-18),min(1024,int(b[2])+18),min(576,int(b[3])+18));cut=Image.fromarray(np.where(raw[i][...,None],a,127).astype(np.uint8)).crop(crop);cut.thumbnail((690,345))
                y=j*390;dr.text((10,y+8),f'{sid} {c["camera"]} f{i} IoU={metrics[i]["normalized_iou"]:.3f}',fill='white');sheet.paste(Image.fromarray(overlay).resize((680,382)),(0,y+25));sheet.paste(cut,(700,y+32))
            sheet.save(qadir/f'{sid}.jpg',quality=95)
            record={'source_id':sid,'seed_view':c['seed_view']['view_id'],'instance_token':c['actors'][0]['instance_token'],'camera':c['camera'],'frames':10,'prompt_frame':prompt,'review_frames':fids,'numeric_gate_pass':not bad and gate['eligible_for_pairing'],'metrics':metrics,'mechanical_gate':gate,'seconds':time.monotonic()-start,'human_verdict':None,'quality_status':'pending_independent_temporal_QA','checkpoint':CHECKPOINT,'mask_prompt':'checked still mask at original seed exposure; forward and reverse official propagation; no GT clipping'}
            dump(dest/'mask_manifest.json',record);reviews.append(record);dump(out/'segmentation_state.json',{'state':'running','clips':reviews})
            predictor.reset_state(state);del state;torch.cuda.empty_cache()
    dump(qadir/'mask_audit.json',{'clips':reviews});dump(out/'segmentation_state.json',{'state':'complete_pending_QA','clips':reviews})
    print('TEMPORAL_MASKS',len(reviews),'NUMERIC_PASS',sum(r['numeric_gate_pass'] for r in reviews),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--folder',default='temporal_windows');a=p.parse_args();main(a.root,a.folder)
