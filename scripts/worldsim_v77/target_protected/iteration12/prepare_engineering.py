"""CPU另建真实入口与去重目录；旧输入、目录和结果只读保留。"""
from pathlib import Path
import json,sys,os,copy
import numpy as np
import cv2
from PIL import Image,ImageDraw
P=Path('/root/autodl-tmp/motion_proj_v77');S=P/'scripts/worldsim_v77'
sys.path.insert(0,str(S/'delete_audit'))
from mask_contract import prepare_masks,verify
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r21'


def read(p):return json.loads(p.read_text())
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def hull(points):
    m=np.zeros((576,1024),np.uint8)
    if points:cv2.fillConvexPoly(m,np.asarray(points,np.int32),1)
    return m>0


def main():
    cv2.setNumThreads(4);checks=verify()
    audit=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-AUDIT-20260928/r1')
    geom={c['clip_id']:c for c in read(audit/'clip_geometry.json')['clips']}
    cases=[c for c in read(T/'r14/evaluation_plan.json')['cases'] if c['kind']=='real_development']
    assert len(cases)==8
    data=[];contacts=O/'mask_contacts';contacts.mkdir(exist_ok=True)
    for c in cases:
        old=Path(c['folder']);out=O/'real_inputs'/c['clip_id'];out.mkdir(parents=True,exist_ok=True);rows=[]
        for role in ['rgb','sam','core','write_mask','model_mask','protect','alpha']:(out/role).mkdir(exist_ok=True)
        frames=geom[c['clip_id']]['frames']
        for fr in frames:
            i=fr['frame'];sam=np.asarray(Image.open(old/'sam'/f'{i:05}.png'))>0
            target=hull(fr['target']['hull'] if fr['target'] else [])
            neighbors=np.zeros_like(target)
            for n in fr['neighbors']:neighbors|=hull(n['hull'])
            ar,info=prepare_masks(sam,target,neighbors);info['frame']=i;rows.append(info)
            for role,val in ar.items():Image.fromarray(np.rint(val*255).astype('uint8')).save(out/role/f'{i:05}.png')
            for role,extension in [('rgb','jpg'),('sam','png')]:
                dst=out/role/f'{i:05}.{extension}'
                if not dst.exists():os.link(old/role/dst.name,dst)
        assert all(r['target_outside_model']==0 and r['target_alpha_not_one']==0 for r in rows)
        dump(out/'mask_stats_v2.json',rows)
        # 固定首/中/尾及提示帧。只审核目标身份，不根据生成效果挑帧。
        ids=sorted(set([0,12,25,geom[c['clip_id']]['prompt_frame']]))
        canvas=Image.new('RGB',(1024*3,610*len(ids)),(20,24,30));draw=ImageDraw.Draw(canvas)
        for j,i in enumerate(ids):
            rgb=np.asarray(Image.open(old/'rgb'/f'{i:05}.jpg'));sam=np.asarray(Image.open(old/'sam'/f'{i:05}.png'))>0;new=np.asarray(Image.open(out/'model_mask'/f'{i:05}.png'))>0;prior=np.asarray(Image.open(old/'model_mask'/f'{i:05}.png'))>0
            for k,(title,m) in enumerate([('Original / target SAM',sam),('OLD model H',prior),('NEW H complete SAM',new)]):
                im=rgb.copy();im[m]=np.rint(im[m]*.55+np.array([255,180,20] if k==0 else [30,160,250])*.45).astype('uint8');canvas.paste(Image.fromarray(im),(k*1024,j*610+30));draw.text((k*1024+8,j*610+5),f'{c["eval_id"]} f{i} {title}',fill='white')
        canvas.save(contacts/(c['clip_id']+'.jpg'),quality=92)
        rec=copy.deepcopy(c);rec['folder']=str(out);rec['old_folder']=str(old);rec['mask_policy']='sam_full_v2';rec['identity_review_pending']=True;data.append(rec)
    dump(O/'real_input_plan.json',{'cases':data,'same_source_RGB':True,'old_inputs_untouched':True,'target_coverage_contract':checks,'model_quality_claim':False})
    # 使用r19实际像素逐值比较的结果，不用文件名或仅scene统计代替去重。
    d=read(T/'r19/process_and_duplicates.json')
    dump(O/'dedup_audit_source.json',d)
    dump(O/'input_fix_summary.json',{'real_cases':8,'actual_frames':208,'SAM_outside_H':0,'SAM_alpha_not_one':0,'identity_review_pending':True,'unit_contracts':checks,'old_70_cases_preserved':True})
    print('REAL_MASK_FIX',8,208,checks,flush=True)
    print('DEDUP_KEYS',list(d),flush=True)


if __name__=='__main__':main()
