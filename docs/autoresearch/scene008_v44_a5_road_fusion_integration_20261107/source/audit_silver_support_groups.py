"""Read saved baseline/C alpha_T groups for silver remnants versus C2 (CPU)."""
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from PIL import Image, ImageDraw

R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
OUT=R/'review/silver_support_group_audit_v1'
OUT.mkdir(exist_ok=True)
NAMES=['C2_asset_a6','native_a37','oldroad','native_static','semantic_a5','unused_a18','unused_a26','newroad']
COLORS=np.array([[224,72,72],[142,70,190],[244,161,37],[43,128,205],[36,164,81],[170,170,170],[110,110,110],[21,200,191]],np.uint8)
ROIS={128:(210,202,302,242),137:(207,207,274,227)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def snap(p):
    p=Path(p);s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':sha(p)}


def arr(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def image(a):
    return Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8),'RGB')


def group_metrics(g,alpha):
    g=np.asarray(g,np.float64);winner=g.argmax(0)
    return {'joint_alpha_mean':float(alpha.mean()),'joint_alpha_min':float(alpha.min()),
            'group_sum_joint_alpha_max_abs':float(abs(g.sum(0)-alpha).max()),
            'alpha_zero_pixels':int((alpha==0).sum()),
            'group_alpha_T_mean':dict(zip(NAMES,g.mean((1,2)).tolist())),
            'group_alpha_T_max':dict(zip(NAMES,g.max((1,2)).tolist())),
            'dominant_group_pixels':dict(zip(NAMES,np.bincount(winner.ravel(),minlength=8).tolist()))}


paths=[]
for raw in ROIS:
    for case in ['baseline4','C4_retry1']:
        for state in ['keep','delete']:paths.append(R/f'{case}/frames/{raw:03d}/{state}_float.npz')
before=[snap(p) for p in paths]
report={'status':'PASS_READONLY_SAVED_GROUP_ATTRIBUTION_NOT_QUALITY_ACCEPTANCE','script_sha256':sha(__file__),
        'actual_gpu_forwards':0,'actual_backwards':0,'actual_optimizer_updates':0,
        'group_order':NAMES,'ROI_provenance':{'128':'Root-specified silver ROI; fixed before this audit; not a score domain',
            '137':'Manual post-result rectangular silver road remnant from actually viewed baseline/C DELETE PNG; not a score domain'},
        'limits':['alpha_T is native transmittance-weighted contribution, not per-pixel semantic truth or RGB decomposition.',
            'Old group-specific RGB was not saved; no per-group colour is fabricated.',
            'The large128 ROI contains both road/static representations and real C2 contributions, so cannot be retired wholesale.',
            'Original fixed wheel rays remain a separate test; a road remnant being covered does not prove tires preserved.'],
        'records':[],'figures':[]}
for raw,box in ROIS.items():
    x0,y0,x1,y1=box
    for state in ['keep','delete']:
        bs=arr(R/f'baseline4/frames/{raw:03d}/{state}_float.npz')
        cs=arr(R/f'C4_retry1/frames/{raw:03d}/{state}_float.npz')
        bg=bs['group_alpha_T'];bg=np.concatenate([bg,np.zeros_like(bg[:1])],0)
        cg=cs['group_alpha_T'];assert cg.shape==(8,320,480)
        bg=bg[:,y0:y1,x0:x1];cg=cg[:,y0:y1,x0:x1]
        ba=bs['alpha'][y0:y1,x0:x1];ca=cs['alpha'][y0:y1,x0:x1]
        drop=bg[0]-cg[0]
        flat=np.argsort(-drop.ravel(),kind='stable')[:20]
        top=[]
        for index in flat:
            yy,xx=np.unravel_index(int(index),drop.shape);x,y=int(x0+xx),int(y0+yy)
            top.append({'x':x,'y':y,'C2_alpha_T_baseline':float(bg[0,yy,xx]),'C2_alpha_T_C':float(cg[0,yy,xx]),
                        'C2_alpha_T_drop':float(drop[yy,xx]),'newroad_alpha_T_C':float(cg[7,yy,xx]),
                        'baseline_group_alpha_T':bg[:,yy,xx].tolist(),'C_group_alpha_T':cg[:,yy,xx].tolist()})
        rec={'raw':raw,'state':state,'ROI_xyxy':list(box),'pixels':(x1-x0)*(y1-y0),
             'baseline':group_metrics(bg,ba),'C':group_metrics(cg,ca),
             'oldroad_alpha_T_exact_in_ROI':bool(np.array_equal(bg[2],cg[2])),
             'C2_alpha_T_max_drop':float(drop.max()),'C2_alpha_T_drop_gt1e_6_pixels':int((drop>1e-6).sum()),
             'top20_C2_alpha_T_drops':top}
        report['records'].append(rec)
        scale=5;w=(x1-x0)*scale;h=(y1-y0)*scale
        board=Image.new('RGB',(max(3*w,900),2*h+155),'white');d=ImageDraw.Draw(board)
        d.text((8,5),f'raw{raw} {state.upper()} ROI {box}; nearest x5; alpha_T attribution, not semantic/RGB truth',fill='black')
        panels=[('Latest native RGB',image(bs['rgb'].transpose(1,2,0)).crop(box)),
                ('C native RGB',image(cs['rgb'].transpose(1,2,0)).crop(box)),
                ('Latest dominant alpha_T group',Image.fromarray(COLORS[bg.argmax(0)])),
                ('C dominant alpha_T group',Image.fromarray(COLORS[cg.argmax(0)])),
                ('Latest C2 alpha_T (gray 0..1)',image(np.repeat(bg[0,:,:,None],3,2))),
                ('C C2 alpha_T (gray 0..1)',image(np.repeat(cg[0,:,:,None],3,2)))]
        for index,(name,im) in enumerate(panels):
            col=index%3;row=index//3;ox=col*w;oy=35+row*(h+35)
            d.text((ox+4,oy),name,fill='black');board.paste(im.resize((w,h),Image.Resampling.NEAREST if hasattr(Image,'Resampling') else Image.NEAREST),(ox,oy+24))
        ly=2*h+110
        for index,name in enumerate(NAMES):
            x=8+(index%4)*220;y=ly+(index//4)*20
            d.rectangle((x,y,x+12,y+12),fill=tuple(int(v) for v in COLORS[index]));d.text((x+16,y),name,fill='black')
        p=OUT/f'raw{raw:03d}_{state}_silver_groups.png';board.save(p)
        report['figures'].append({'path':str(p),'sha256':sha(p),'size':list(board.size)})
after=[snap(p) for p in paths];assert before==after
report['source_integrity']={'status':'PASS_SIZE_MTIME_SHA_UNCHANGED','files':before}
p=R/'review/silver_support_group_audit_v1.json';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(p),'sha256':sha(p),'records':len(report['records'])}))
