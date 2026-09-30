"""CPU空间预案预览：原RGB、A洞、俯视关系；不冒充SAM2/合成准入结果。"""
import argparse,html,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import median_filter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import read,dump,footprint
from build_pairs import actor_contact,warp_matrix

def build(root,donor_root):
    cv2.setNumThreads(1);factory=root/'native10_factory';out=root/'review';out.mkdir(exist_ok=True);(out/'assets').mkdir(exist_ok=True)
    plan=read(root/('selected_pair_plan.json' if (root/'selected_pair_plan.json').exists() else 'pair_plan.json'));src={c['source_id']:c for c in read(factory/'source_manifest.json')['clips']};donors={c['source_id']:c for c in read(donor_root/'source_manifest.json')['clips']};reviews=[];queue=[]
    for p in plan['plans']:
        sid=p['source_id'];c=src[sid];d=donors[p['donor_source_id']];i=5;f=c['frames'][i];df=d['frames'][i]
        raw=Image.open(factory/'rgb'/f['filename']).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS)
        masks=[cv2.imread(str(x),0)>0 for x in sorted((donor_root/'segmented'/d['source_id']/'sam2_raw').glob('*.png'))]
        contact=median_filter(np.array([actor_contact(m,np.array(fr['actors'][0]['projection']['box_xyxy'])) for m,fr in zip(masks,d['frames'])]),size=5,mode='nearest')
        a=cv2.warpAffine(masks[i].astype('uint8'),warp_matrix(df,p['frames'][i],contact[i]),(1024,576),flags=cv2.INTER_NEAREST)>0;hole=cv2.dilate(a.astype('uint8'),np.ones((13,13),'uint8'))>0
        labelled=raw.copy();draw=ImageDraw.Draw(labelled);palette=['#35e69b','#7fc3ff','#e87dff','#ffc040']
        for j,ob in enumerate(f['actors']):
            b=ob['projection']['box_xyxy'];draw.rectangle(b,outline=palette[j%4],width=3);draw.text((b[0],max(0,b[1]-15)),f'Protected {chr(66+j)} {ob["instance_token"][:7]}',fill=palette[j%4])
        overlay=np.array(raw).copy();overlay[hole]=(.5*overlay[hole]+.5*np.array([255,180,30])).astype('uint8');ov=Image.fromarray(overlay);od=ImageDraw.Draw(ov);od.rectangle(p['frames'][i]['box'],outline='#ffbd30',width=2)
        conditional=np.array(raw).copy();conditional[hole]=0;conditional=Image.fromarray(conditional)
        bev=Image.new('RGB',(1024,576),'#152237');bd=ImageDraw.Draw(bev);center=np.array(p['frames'][i]['actor']['translation'])[:2];cam=np.array(f['camera_to_world'])[:2,3]
        def xy(q):return (512+(float(q[0])-center[0])*14,288-(float(q[1])-center[1])*14)
        for x in range(8,1024,70):bd.line([(x,0),(x,576)],fill='#334258')
        for y in range(8,576,70):bd.line([(0,y),(1024,y)],fill='#334258')
        for j,ob in enumerate(f['actors']):
            poly=[xy(q) for q in footprint(ob).exterior.coords];bd.polygon(poly,outline=palette[j%4],width=3);bd.text(xy(ob['translation']),chr(66+j),fill=palette[j%4])
        bd.polygon([xy(q) for q in footprint(p['frames'][i]['actor']).exterior.coords],outline='#ffbd30',width=4);bd.text(xy(center),'A candidate',fill='#ffbd30')
        pc=xy(cam);bd.ellipse((pc[0]-5,pc[1]-5,pc[0]+5,pc[1]+5),fill='white');bd.text(pc,'camera',fill='white');bd.line([pc,xy(center)],fill='#eee5ad',width=2)
        panels=[labelled,ov,conditional,bev];labels=['Real Y / protected actor envelopes','Proposed A silhouette + 6px hole bound','Planned masked Y (NOT a training pair)','World XY / 5m grid / selected A and protected B,C']
        sheet=Image.new('RGB',(1600,944),'#080d18');sd=ImageDraw.Draw(sheet)
        for j,(im,label) in enumerate(zip(panels,labels)):
            x=(j%2)*800;y=(j//2)*472;sheet.paste(im.resize((800,450),Image.Resampling.LANCZOS),(x,y+22));sd.text((x+6,y+5),f'{sid} {label}',fill='white')
        sheet.save(out/'assets'/f'{sid}.jpg',quality=93)
        row={'case_id':sid,'scene':c['scene'],'camera':c['camera'],'planned_type':p['planned_type'],'donor':d['source_id'],'frame':i,
             'image':f'assets/{sid}.jpg','protected_tokens':p['required_protected_instances'],'min_clearance_m':p['min_GT_clearance_m'],
             'max_ground_support_m':p['max_ground_support_distance_m'],'max_view_delta_deg':p['max_view_yaw_delta_deg'],
             'human_verdict':None,'training_ready':False,'status':'pending_independent_source_review_and_GPU_masks','planning_control':p.get('planning_control','original_top_six_windows')}
        reviews.append(row)
        for tok in p['required_protected_instances']:
            queue.append({'job_id':sid+'_'+tok[:8],'source_id':sid,'instance_token':tok,'role':'protected_receiver',
                          'source_frames':[f['filename'] for f in c['frames']],'prompt_frame':5,
                          'box_prompts':[next(ob['projection']['box_xyxy'] for ob in f['actors'] if ob['instance_token']==tok) for f in c['frames']],
                          'enabled':False,'blockers':['independent_source_review','explicit_GPU_authorization'],
                          'followup':'SAM2 true masks, original full-frame overlap/visibility/conditioning checks, independent synthetic QA; not box deletion'})
    dump(out/'preflight_manifest.json',{'run_id':'r4','sampling':'one fixed frame 5 per CPU preflight case; no temporal visual certification','cases':reviews})
    dump(root/'gpu_queue.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r4','jobs':queue,'executed':0,'enabled':False,'architecture_changes':False,'training_ready':0})
    print('PREVIEWS',len(reviews),'GPU_JOBS_NOT_STARTED',len(queue),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--donor-root',type=Path,required=True);a=p.parse_args();build(a.root,a.donor_root)
