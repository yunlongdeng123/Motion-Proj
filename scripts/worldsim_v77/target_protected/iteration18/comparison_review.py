"""Encode r50+r51 comparisons as they finish; no changes to inference pixels."""
from pathlib import Path
import os, sys, json, time, argparse, shutil, subprocess, html
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps

cv2.setNumThreads(1)
REPO=Path('/root/autodl-tmp/motion_proj_v77')
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
O=T/'r51/r47_comparison'; DEST=O/'review'
sys.path.insert(0,str(REPO/'scripts/worldsim_v77/target_protected/iteration17'))
from common import read,dump,ffmpeg_binary
from review import annotate,generate_assets
import review as old_review

def fit(im,size):
    out=Image.new('RGB',size,(18,24,34));im=ImageOps.contain(im,size)
    out.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2));return out

def encode(frames, dest):
    if dest.exists():return
    arr=np.stack(frames).astype('uint8');h,w=arr.shape[1:3]
    cmd=[ffmpeg_binary(),'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
         '-r','10','-i','-','-an','-c:v','libx264','-threads','1','-preset','veryfast','-crf','18',
         '-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
    subprocess.run(cmd,input=arr.tobytes(),check=True)

def images(folder,extension='png'):
    return [Image.open(folder/f'{i:05}.{extension}').convert('RGB') for i in range(10)]

def assets(c,orig):
    cid=c['case_id'];d=DEST/cid
    if (d/'asset_check.json').exists():return
    d.mkdir(parents=True,exist_ok=True)
    folder=Path(c['folder']);out=O/'evaluation'/cid;cond=O/'inputs'/cid
    old_review.O=T/c['source_batch'];generate_assets(orig,False)
    old=T/c['source_batch']/'review'/cid
    for name in ['original.mp4','model_input.mp4','native.mp4','delete.mp4','mask_review.jpg','inputs_f00_f05_f09.jpg']:
        shutil.copy2(old/name,d/name)
    raw=images(out/'native');final=images(out/'compose')
    encode(raw,d/'r47_native.mp4');encode(final,d/'r47_delete.mp4')
    model_mask=np.array(Image.open(folder/'model_mask/00005.png'))>0
    yy,xx=np.where(model_mask);x0,y0,x1,y1=orig['frames'][5]['target']['box_xyxy']
    roi=(max(0,min(int(x0),int(xx.min()))-64),max(0,min(int(y0),int(yy.min()))-64),
         min(1024,max(int(x1),int(xx.max())+1)+64),min(576,max(int(y1),int(yy.max())+1)+64))
    panels=[annotate(Image.open(folder/'rgb/00005.jpg').convert('RGB'),orig,5),
            annotate(Image.open(old/'mask_overlay/00005.jpg').convert('RGB'),orig,5),
            Image.open(folder/'native/00005.png').convert('RGB'),Image.open(folder/'compose/00005.png').convert('RGB'),raw[5],final[5]]
    names=['ORIGINAL A=yellow B=green','SAM contour / actual HOLE','OFFICIAL native DELETE','r46 fixed writeback','r47 native DELETE','r47 fixed writeback']
    board=Image.new('RGB',(3072,650),(18,24,34));draw=ImageDraw.Draw(board)
    for k,(im,name) in enumerate(zip(panels,names)):
        draw.text((512*k+6,5),f'{cid} f05 {name}',fill='white');board.paste(im.resize((512,288)),(512*k,27))
        draw.text((512*k+6,325),'SAME ROI / no sharpening',fill='white');board.paste(fit(im.crop(roi),(512,288)),(512*k,348))
    board.save(d/'comparison_f05.jpg',quality=96)
    # Native comparison is also available at a larger per-panel size for adjudication.
    native_board=Image.new('RGB',(2304,900),(18,24,34));nd=ImageDraw.Draw(native_board)
    for j,k in enumerate([0,2,4]):
        nd.text((768*j+6,4),f'{cid} f05 {names[k]}',fill='white')
        native_board.paste(panels[k].resize((768,432)),(768*j,25))
        native_board.paste(fit(panels[k].crop(roi),(768,432)),(768*j,464))
    native_board.save(d/'native_f05.jpg',quality=96)
    p=dict(np.load(cond/'condition.npz'));refs=read(cond/'references.json')['references']
    condition_frames=[];coverage=[]
    for i in range(10):
        g=p['geometry'][i];b=p['bev'][i];hole=p['hole'][i].astype(bool)
        color=np.full((144,256,3),45,'uint8');color[g[0]>0]=[50,195,115];color[g[1]>0]=[65,150,255]
        # The yellow outline identifies the independent edit hole, without overwriting its classes.
        cv2.drawContours(color,cv2.findContours(hole.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(255,220,40),1)
        occ=Image.fromarray(color).resize((1024,576),Image.Resampling.NEAREST)
        bev=np.full((128,128,3),45,'uint8');bev[b[5]>0]=[65,150,255];bev[b[0]>0]=[50,195,115];bev[b[1]>0]=[180,140,70]
        bv=Image.fromarray(bev[::-1]).resize((576,576),Image.Resampling.NEAREST)
        frame=Image.new('RGB',(1600,614),(18,24,34));dr=ImageDraw.Draw(frame)
        dr.text((8,8),f'{cid} f{i:02} 2D OCC proxy: green retained vehicle / blue LiDAR / gray unknown / yellow hole',fill='white')
        dr.text((1030,8),'BEV 80m x 80m (world axes, not ego-up)',fill='white')
        frame.paste(occ,(0,38));frame.paste(bv,(1024,38));condition_frames.append(frame)
        coverage.append({'frame':i,**{label:float(g[ch][hole].mean()) for label,ch in [('O',0),('N',1),('U',2)]}})
    encode(condition_frames,d/'occ.mp4');condition_frames[5].save(d/'occ_f05.jpg',quality=95)
    refboard=Image.new('RGB',(1536,324),(18,24,34));rd=ImageDraw.Draw(refboard)
    ref_info=[]
    for i,r in enumerate(refs):
        valid=float(p['reference_valid'][i].mean());rel=(r['timestamp']-c['frames'][0]['timestamp'])/1e6
        refboard.paste(Image.fromarray(p['references'][i]),(i*256,58))
        label=f"slot{i} {'B crop' if r['token'] else 'context'} {rel:+.2f}s"
        rd.text((i*256+3,4),label,fill='white');rd.text((i*256+3,21),f"{r['camera']} valid={valid:.0%}",fill='white')
        rd.text((i*256+3,38),('DISABLED' if r.get('disabled') or r.get('padding') else 'consumed by r47 VAE'),fill='white')
        ref_info.append({'slot':i,'role':r['role'],'camera':r['camera'],'relative_seconds':rel,
                         'valid_fraction':valid,'disabled':r.get('disabled',False),'padding':r.get('padding',False),'instance_token':r['token']})
    refboard.save(d/'references.jpg',quality=96)
    # Check every delivered video, not just metadata or link existence.
    decoded={}
    for video in d.glob('*.mp4'):
        cap=cv2.VideoCapture(str(video));n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.ndim==3;n+=1
        cap.release();assert n==10,(video,n);decoded[video.name]=n
    record={'case_id':cid,'source_batch':c['source_batch'],'scene':c['scene'],'camera':c['camera'],
            'instance_token':c['target_token'],'roi_f05':roi,'decoded_video_frames':decoded,
            'O_N_U_hole_coverage':coverage,'references':ref_info,'input_quality':orig.get('input_quality_review'),
            'input_difficulty':orig.get('input_difficulty'),'human_verdict':None,'temporal_verdict':None}
    dump(d/'asset_check.json',record);print('ASSETS',cid,flush=True)

def collect():
    plan=read(O/'manifest.json');originals={}
    for batch in ['r50','r51']:
        originals.update({c['case_id']:c for c in read(T/batch/'manifest.json')['cases']})
    for c in plan['cases']:
        if (O/'evaluation'/c['case_id']/'result.json').exists():assets(c,originals[c['case_id']])
    rows=[read(DEST/c['case_id']/'asset_check.json') for c in plan['cases'] if (DEST/c['case_id']/'asset_check.json').exists()]
    dump(DEST/'delivery_index.json',{'completed':len(rows),'expected':len(plan['cases']),'cases':rows})
    return len(rows),len(plan['cases'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    while True:
        n,total=collect()
        if not args.watch or n==total:break
        chain=read(O/'chain_state.json') if (O/'chain_state.json').exists() else {}
        if chain.get('stage')=='error':raise RuntimeError(chain)
        time.sleep(30)
