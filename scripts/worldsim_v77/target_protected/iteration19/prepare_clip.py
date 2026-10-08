"""同一R001真实10帧配对；不把重复单帧当视频，也不使用生成GT。"""
from pathlib import Path
import cv2,json
import numpy as np
from PIL import Image,ImageDraw

T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
root=T/'r52';source=T/'r50/inputs/R001';dest=root/'pairs/R001_realRGB_clip_v2'
dest.mkdir(exist_ok=False)
for d in ['x','y','hole']:(dest/d).mkdir()
cards=[];rows=[]
template=np.asarray(Image.open(source/'model_mask/00005.png'))>0
hole=cv2.warpAffine(template.astype('uint8'),np.float32([[1,0,-195],[0,1,38]]),(1024,576),flags=cv2.INTER_NEAREST)>0
for i,p in enumerate(sorted((source/'rgb').glob('*.jpg'))):
    im=np.asarray(Image.open(p).convert('RGB'))
    old=np.asarray(Image.open(source/'model_mask'/f'{i:05}.png'))>0
    assert hole.sum()==template.sum()
    x=im.copy();x[hole]=127
    for role,val in [('x',x),('y',im),('hole',hole.astype('uint8')*255)]:Image.fromarray(val).save(dest/role/f'{i:05}.png')
    yy,xx=np.where(hole);bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
    roi=[max(0,bbox[0]-12),max(0,bbox[1]-12),min(1024,bbox[2]+12),min(576,bbox[3]+12)]
    card=Image.new('RGB',(440,190),'#111111');draw=ImageDraw.Draw(card);draw.text((5,2),f'f{i:02} REAL Y / MASK OVERLAY',fill='white')
    overlay=im.copy();overlay[hole]=(overlay[hole]*.5+np.array([30,140,255])*.5).astype('uint8')
    for j,v in enumerate([im,overlay]):card.paste(Image.fromarray(v).crop(roi).resize((216,166)),(j*220,22))
    cards.append(card);rows.append({'frame':i,'bbox':bbox,'source':str(p),'hole_pixels':int(hole.sum()),'overlap_original_target_envelope_pixels':int((hole&old).sum())})
assert len(cards)==10
board=Image.new('RGB',(880,950),'#111111')
for i,c in enumerate(cards):board.paste(c,((i%2)*440,(i//2)*190))
board.save(dest/'review_all10.jpg',quality=94)
pair={'case_id':'R001_realRGB_clip_v2','source_case':'R001','scene':'scene-0290','frame_index':5,'num_frames':10,
      'supervision_kind':'real_rgb_synthetic_occlusion','target_is_real_RGB':True,
      'x':str(dest/'x'),'y':str(dest/'y'),'hole':str(dest/'hole'),'qa_pass':False,
      'qa_record':str(dest/'quality_review.json'),'human_verdict':None,'hole_translation_xy':[-195,38],
      'mask_policy':'固定f05实际模型洞平移，所有10帧同一洞；仅容量诊断，不宣称3D actor轨迹',
      'frames':rows,'limits':'仅同景真实RGB的合成遮挡恢复对；不是真实SUV删除后的hidden GT；真实10帧非重复静帧'}
(dest/'pair.json').write_text(json.dumps(pair,ensure_ascii=False,indent=2)+'\n')
print('PREPARED_REAL_CLIP',len(cards))
