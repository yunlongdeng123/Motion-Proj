"""0230扩大mask产生车形残影后的单项范围控制；不扫参、不重训。"""
import argparse,json,pathlib,shutil,os
import numpy as np
from PIL import Image
import v77_driveeditor_compare as de

OLD=de.ROOT
NEW=OLD.parent/'r2'
SCENE='scene_0230'
p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','drive']);a=p.parse_args()
if a.action=='prepare':
    assert json.loads((OLD/SCENE/'C/result.json').read_text())['status']=='complete'
    NEW.mkdir(exist_ok=False)
    base=NEW/SCENE;base.mkdir()
    for name in ['rgb','target_sam']:
        shutil.copytree(OLD/SCENE/name,base/name)
    (base/'mask_b').mkdir()
    rows=[]
    for i in range(10):
        sam=np.array(Image.open(base/'target_sam'/f'{i:05}.png'))>0
        yy,xx=np.where(sam)
        # 固定1024x576像素边距：左右/上8px，下24px覆盖近车底区域。
        # 不使用生成结果选范围；矩形保留官方训练的mask形状。
        x0=max(0,int(xx.min())-8);x1=min(1024,int(xx.max())+1+8)
        y0=max(0,int(yy.min())-8);y1=min(576,int(yy.max())+1+24)
        mask=np.zeros((576,1024),dtype=np.uint8);mask[y0:y1,x0:x1]=255
        Image.fromarray(mask).save(base/'mask_b'/f'{i:05}.png')
        old=np.array(Image.open(OLD/SCENE/'mask_b'/f'{i:05}.png'))>0
        assert (mask[sam]>0).all()
        rows.append({'frame':i,'rect_xyxy':[x0,y0,x1,y1],'target_coverage':float((mask[sam]>0).mean()),'mask_fraction':float((mask>0).mean()),'old_fraction':float(old.mean())})
    de.dump(NEW/'registration.json',{'task_id':'WS-V77-DRIVEEDITOR-COMPARE-20260927','run_id':'r2','scene':SCENE,'actor':22,'camera':2,'source_frames':list(range(10)),'question':'r1扩大矩形伴随0230车形残影；只缩小操作范围，判断是否足以改善。','trigger_evidence':str(OLD/SCENE/'C/native'),'input_roles':'原RGB和SAM2目标mask；SAM外包矩形+固定8/8/8/24px边距，保留矩形形状；无隐藏背景GT。该两scene是开发例。','only_change':'deletion mask；范围与中心不再随机；不是单独区分面积和随机偏移的因果实验','seed':42,'steps':25,'decoding_t':1,'frames':10,'size':[1024,576],'weights':'same official trained checkpoint frozen','training_steps':0,'resource':'one RTX3090;CPU4;timeout1800s','stop_rule':'只跑此一个范围控制；仍有残影则停止当前mask路线，不扫种子或边距。','mask_rows':rows,'failure_ledger_refs':['V77-F02'],'human_verdict':None})
    print('PREPARED_R2',flush=True)
else:
    assert (NEW/'registration.json').is_file()
    assert not (NEW/'state.json').exists(),'检查旧状态，不重复启动'
    de.ROOT=NEW
    de.dump(NEW/'state.json',{'state':'running','pid':os.getpid(),'training_steps':0,'human_verdict':None})
    try:
        de.drive(SCENE)
        de.dump(NEW/'state.json',{'state':'complete','training_steps':0,'human_verdict':None})
    except Exception as exc:
        de.dump(NEW/'state.json',{'state':'failed','error':str(exc),'human_verdict':None})
        raise
