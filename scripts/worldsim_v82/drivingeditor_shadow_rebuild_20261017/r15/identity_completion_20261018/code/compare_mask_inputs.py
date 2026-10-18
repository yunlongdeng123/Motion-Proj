"""CPU: compare two actual supervision inputs, without claiming SAM is truth."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from PIL import Image,ImageFilter
from identity_common import *

def main():
    out=TASK/'tristate/mask_comparison';out.mkdir(exist_ok=True)
    manifest=read(R15/'data/manifest.json');records=[]
    palette=np.array([[56,165,240],[38,230,140],[255,182,39]],np.uint8)
    for row in manifest['frames']:
        fid=row['frame_id'];s=dict(np.load(R15/'data'/row['supervision']))
        native=dict(np.load(TASK/'native_masks_for_review'/(fid+'.npz')))
        rgb=np.asarray(Image.open(R15/'data'/row['rgb']).convert('RGB'))
        for j,actor in enumerate(NEW_ACTORS):
            n=native['masks'][...,j];valid=s['mask_valid'][...,actor]&s['mask_known'][actor]
            cls=np.where(valid,np.where(s['masks'][...,actor],1,0),2)
            assert native['mask_valid'][...,j].all() and native['mask_known'][j]
            # White outline is full native SAM proposal, not a reference annotation.
            pil=Image.fromarray(n.astype(np.uint8)*255)
            edge=np.asarray(pil.filter(ImageFilter.MaxFilter(3)))!=np.asarray(pil.filter(ImageFilter.MinFilter(3)))
            overlay=np.rint(.55*rgb+.45*palette[cls]).astype(np.uint8);overlay[edge]=[255,255,255]
            Image.fromarray(overlay).save(out/(fid+'_a%02d.png'%actor))
            yy,xx=np.where(n);px=max(20,(xx.max()-xx.min())*.5);py=max(20,(yy.max()-yy.min())*.6)
            crop=(max(0,int(xx.min()-px)),max(0,int(yy.min()-py)),min(rgb.shape[1],int(xx.max()+px+1)),min(rgb.shape[0],int(yy.max()+py+1)))
            for suffix,img in [('overlay_crop',overlay),('state_crop',palette[cls]),('native_crop',n.astype(np.uint8)*255),('rgb_crop',rgb)]:
                Image.fromarray(img).crop(crop).save(out/(fid+'_a%02d_'%actor+suffix+'.png'))
            records.append(dict(frame_id=fid,actor=actor,native_positive=int(n.sum()),
                diagnostic_crop_xyxy=list(crop),crop_not_used_to_change_render_or_train=True,
                R15_positive=int((cls==1).sum()),R15_negative=int((cls==0).sum()),R15_unknown=int((cls==2).sum()),
                native_positive_in_R15_positive=int((n&(cls==1)).sum()),
                native_positive_in_R15_negative=int((n&(cls==0)).sum()),
                native_positive_in_R15_unknown=int((n&(cls==2)).sum())))
    summary={}
    for actor in NEW_ACTORS:
        rows=[r for r in records if r['actor']==actor];keys=[k for k in rows[0] if k not in ['frame_id','actor','diagnostic_crop_xyxy','crop_not_used_to_change_render_or_train']]
        totals={k:sum(r[k] for r in rows) for k in keys}
        summary[str(actor)]=dict(totals=totals,
            native_positive_R15_domain_fraction={k:totals['native_positive_in_R15_'+k]/totals['native_positive'] for k in ['positive','negative','unknown']})
    write(out/'comparison.json',dict(status='ACTUAL_INPUT_COMPARISON_NOT_GT_VALIDATION',records=records,summary=summary,
        original_masks_unchanged=True,white_outline='native full SAM proposal; no ground truth boundary claim',
        colors={'blue':'R15 valid negative0','green':'R15 valid positive1','orange':'R15 invalid/unknown'},
        caution='Different masks can disagree for segmentation, occlusion and fusion reasons; do not infer Gaussian owner error from pixel disagreement alone.'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
