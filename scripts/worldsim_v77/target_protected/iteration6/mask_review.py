"""r4新保护实例的原SAM2轮廓与连续性证据；不裁框或自动修补。"""
import argparse,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import read,dump
from iteration5.planning import normalized_masks
from iteration4.segment_receivers import reusable
def main(parent,root):
    cv2.setNumThreads(1);out=root/'mask_review';out.mkdir(parents=True,exist_ok=True)
    src={c['source_id']:c for c in read(parent/'native10_factory/source_manifest.json')['clips']};rows=[]
    for job in read(parent/'gpu_queue.json')['jobs']:
        dest=parent/'segmented_receivers'/job['job_id'];assert reusable(dest,job)
        masks=[np.asarray(Image.open(dest/'sam2_raw'/f'{i:05}.png'))>0 for i in range(10)]
        metrics=normalized_masks(masks,job['box_prompts']);bad=[r for r in metrics if not r['pixels'] or r['normalized_iou']<.8 or not .85<=r['normalized_area_ratio']<=1.15]
        worst=min(metrics,key=lambda x:x['normalized_iou'])['frame'];ids=list(dict.fromkeys([0,5,9,worst]))
        sheet=Image.new('RGB',(1600,380*len(ids)),'#102030');draw=ImageDraw.Draw(sheet)
        for j,i in enumerate(ids):
            f=src[job['source_id']]['frames'][i]
            with Image.open(parent/'native10_factory/rgb'/f['filename']) as im:rgb=np.asarray(im.convert('RGB').resize((1024,576),Image.Resampling.LANCZOS))
            m=masks[i];overlay=rgb.copy();overlay[m]=(.55*rgb[m]+.45*np.array([50,225,140])).astype('uint8')
            isolated=np.where(m[...,None],rgb,np.uint8(25));ov=Image.fromarray(overlay);ImageDraw.Draw(ov).rectangle(job['box_prompts'][i],outline='#ffc050',width=2)
            y=j*380;sheet.paste(ov.resize((640,360)),(0,y+20));sheet.paste(Image.fromarray(isolated).resize((640,360)),(640,y+20))
            box=job['box_prompts'][i];crop=Image.fromarray(isolated).crop((max(0,box[0]-15),max(0,box[1]-15),min(1024,box[2]+15),min(576,box[3]+15)));crop.thumbnail((310,345));sheet.paste(crop,(1280,y+25))
            draw.text((5,y+2),f'{job["job_id"]} frame {i} / raw SAM2 overlay / isolated / crop; IoU={metrics[i]["normalized_iou"]:.3f}',fill='white')
        path=out/f'{job["job_id"]}.jpg';sheet.save(path,quality=95)
        rows.append(job|{'numeric_gate_pass':not bad,'bad_frames':bad,'metrics':metrics,'review_frames':ids,'image':path.name,'mask_status':'pending'})
    dump(out/'mask_audit.json',{'jobs':rows,'human_verdict':None});print('MASK_REVIEW',len(rows),'numeric_pass',sum(r['numeric_gate_pass'] for r in rows))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.parent,a.root)
