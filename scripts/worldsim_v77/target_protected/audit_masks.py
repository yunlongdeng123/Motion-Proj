"""真实SAM2 mask连续性统计、首中末/最弱过渡可视化；不裁框修mask。"""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont

def read(p):return json.loads(Path(p).read_text())
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def crop(im,b,scale=1.3):
    x0,y0,x1,y1=b;cx=(x0+x1)/2;cy=(y0+y1)/2;w=(x1-x0)*scale;h=(y1-y0)*scale
    return im.crop((max(0,int(cx-w/2)),max(0,int(cy-h/2)),min(1024,int(cx+w/2)),min(576,int(cy+h/2))))
def fit(im,size):
    out=Image.new('RGB',size,(22,29,40));im=im.copy();im.thumbnail(size,Image.Resampling.LANCZOS);out.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return out
def main(root,secondary=False):
    cv2.setNumThreads(1)
    src={c['source_id']:c for c in read(root/'source_manifest.json')['clips']}
    queue=read(root/('secondary_segmentation_queue.json' if secondary else 'segmentation_queue.json'))['jobs'];out=root/('mask_review_secondary' if secondary else 'mask_review');out.mkdir(exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    rows=[]
    for job in queue:
        sid=job['source_id'];jid=job.get('job_id',sid);c=src[sid];dest=root/('segmented_secondary' if secondary else 'segmented')/jid
        if not (dest/'mask_manifest.json').exists():continue
        masks=[];norms=[];metrics=[]
        for f in c['frames']:
            fid=f['frame'];m=np.array(Image.open(dest/'sam2_raw'/f'{fid:05}.png'))>0;masks.append(m)
            a=next(a for a in f['actors'] if a['instance_token']==job['instance_token']);b=np.array(a['projection']['box_xyxy'])
            sx=128/(b[2]-b[0]);sy=128/(b[3]-b[1]);T=np.float32([[sx,0,-b[0]*sx],[0,sy,-b[1]*sy]])
            norm=cv2.warpAffine(m.astype(np.uint8),T,(128,128),flags=cv2.INTER_NEAREST)>0;norms.append(norm)
            hull=np.zeros(m.shape,np.uint8);cv2.fillConvexPoly(hull,np.int32(a['projection']['hull']),1)
            iou=1. if fid==0 else np.count_nonzero(norm&norms[-2])/max(1,np.count_nonzero(norm|norms[-2]))
            area_ratio=1. if fid==0 else norm.sum()/max(1,norms[-2].sum())
            metrics.append({'frame':fid,'pixels':int(m.sum()),'normalized_iou':float(iou),'normalized_area_ratio':float(area_ratio),
                            'outside_GT_hull_fraction':float((m&~hull.astype(bool)).sum()/max(1,m.sum()))})
        bad=[r for r in metrics if r['pixels']==0 or r['normalized_iou']<.8 or not .85<=r['normalized_area_ratio']<=1.15]
        worst=min(metrics,key=lambda r:r['normalized_iou'])['frame'];fids=list(dict.fromkeys([0,15,29,worst]))
        sheet=Image.new('RGB',(1400,760),(15,21,31));dr=ImageDraw.Draw(sheet)
        for j,fid in enumerate(fids):
            f=c['frames'][fid];a=next(a for a in f['actors'] if a['instance_token']==job['instance_token']);m=masks[fid]
            rgb=np.array(Image.open(root/'rgb'/f['filename']).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS))
            overlay=rgb.copy();overlay[m]=(.58*rgb[m]+.42*np.array([55,220,155])).astype(np.uint8)
            im=Image.fromarray(overlay);d=ImageDraw.Draw(im);d.rectangle(a['projection']['box_xyxy'],outline=(255,212,55),width=2)
            yy,xx=np.indices(m.shape);checker=np.where(((xx//12+yy//12)%2)[...,None],np.array([60,65,75]),np.array([25,30,40])).astype(np.uint8)
            cut=Image.fromarray(np.where(m[...,None],rgb,checker))
            x=(j%2)*700;y=(j//2)*380
            dr.text((x+7,y+5),f'{jid} f{fid:02} | SAM2 overlay + isolated pixels',font=font,fill='white')
            sheet.paste(im.resize((480,270),Image.Resampling.LANCZOS),(x,y+32))
            sheet.paste(fit(crop(cut,a['projection']['box_xyxy']),(212,270)),(x+486,y+32))
            r=metrics[fid];dr.text((x+7,y+310),f'norm IoU={r["normalized_iou"]:.3f} areaRatio={r["normalized_area_ratio"]:.3f} px={r["pixels"]}',font=font,fill=(175,199,215))
            dr.text((x+7,y+335),f'{c["scene"]} {c["camera"]} | {job["instance_token"][:12]}',font=font,fill=(175,199,215))
        sheet.save(out/f'{jid}.jpg',quality=94)
        rows.append({'source_id':sid,'job_id':jid,'instance_token':job['instance_token'],'reviewed_frame_proposal':fids,
                     'numeric_gate_pass':not bad,'bad_frames':bad,'metrics':metrics,'independent_mask_review':'pending',
                     'sam2_source':'official_defaults_no_custom_cleanup','human_verdict':None})
    dump(out/'mask_audit.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r1','clips':rows,
        'counts':{'completed':len(rows),'numeric_pass':sum(r['numeric_gate_pass'] for r in rows)}})
    print('MASK_AUDIT',len(rows),'NUMERIC_PASS',sum(r['numeric_gate_pass'] for r in rows),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--secondary',action='store_true');a=p.parse_args();main(a.root,a.secondary)
