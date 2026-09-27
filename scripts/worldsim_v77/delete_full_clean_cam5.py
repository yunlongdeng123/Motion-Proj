import pathlib,json,shutil,cv2,numpy as np
from PIL import Image,ImageDraw
from delete_full_common import ROOT,dump
regpath=ROOT/'registration.json';base=ROOT/'scene_0230/cam5';backup=ROOT/'prepare_backup_cam5';backup.mkdir(exist_ok=False)
assert not (ROOT/'drive_state.json').exists()
shutil.copy2(regpath,backup/'registration.json')
for sub in ['mask','target_sam']:shutil.copytree(base/sub,backup/sub)
rows=[]
for f in range(50):
    sam=cv2.imread(str(base/'target_sam'/f'{f:05}.png'),0)>0;before=int(sam.sum())
    if sam.any():
        n,labels,stats,_=cv2.connectedComponentsWithStats(sam.astype('uint8'),8);sam=labels==(1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])))
    mask=np.zeros((576,1024),np.uint8)
    if sam.any():
        y,x=np.where(sam);rect=[max(0,int(x.min())-8),max(0,int(y.min())-8),min(1024,int(x.max())+9),min(576,int(y.max())+25)];x0,y0,x1,y1=rect;mask[y0:y1,x0:x1]=255
    cv2.imwrite(str(base/'target_sam'/f'{f:05}.png'),sam.astype('uint8')*255);cv2.imwrite(str(base/'mask'/f'{f:05}.png'),mask)
    rows.append({'frame':f,'before':before,'after':int(sam.sum()),'removed':before-int(sam.sum()),'edit_pixels':int((mask>0).sum())})
reg=json.loads(regpath.read_text());reg['sam_cleanup']={'scene':'scene_0230','camera':5,'method':'largest connected component only in this visually evidenced noisy stream','reason':'f17旧SAM把建筑/道路离散小块当车；原mask完整备份','rows':rows}
for s in reg['scenes']:
    if s['name']=='scene_0230':
        for st in s['streams']:
            if st['camera']==5:
                for r,new in zip(st['mask_rows'],rows):r['sam_pixels']=new['after'];r['edit_pixels']=new['edit_pixels'];r['sam_cleanup_removed']=new['removed']
dump(regpath,reg);dump(ROOT/'scene_0230/cam5_cleanup.json',reg['sam_cleanup'])
f=17;im=cv2.cvtColor(cv2.imread(str(base/'rgb'/f'{f:05}.png')),cv2.COLOR_BGR2RGB);mask=cv2.imread(str(base/'mask'/f'{f:05}.png'),0)>0;sam=cv2.imread(str(base/'target_sam'/f'{f:05}.png'),0)>0
im[mask]=(.7*im[mask]+.3*np.array([255,150,0])).astype('uint8');cs,_=cv2.findContours(sam.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cs,-1,(0,255,240),2);Image.fromarray(im).save(ROOT/'scene_0230/input_cam5_clean.jpg')
print(json.dumps({'changed_frames':sum(r['removed']>0 for r in rows),'f17':rows[17]}),flush=True)
