"""工程诊断视频与接触图；旧输出复用，新零步输出单列。"""
from pathlib import Path
import sys,os
S=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected');sys.path.insert(0,str(S));sys.path.insert(0,str(S/'iteration9'))
from temporal_factory import T,read,dump
from render_pairs import font,encode
import numpy as np,cv2
from PIL import Image,ImageDraw

def main():
 assert read(T/'r20/state.json')['stage']=='complete_pending_visual_review'
 O=T/'r19/review';O.mkdir(exist_ok=True);(O/'contacts').mkdir(exist_ok=True);cases=[]
 roles=['target','condition','base','r7_round_only','r7','r14_round_only','r14'];arms=['base','r7_round_only','r7','r14_round_only','r14']
 for c in read(T/'r20/evaluation_plan.json')['cases']:
  cid=c['eval_id'];src=T/'r20/evaluation'/cid;old=T/'r14/review/effects'/cid;out=O/'effects'/cid;out.mkdir(parents=True,exist_ok=True);panels=[]
  for role in ['target','base','r7','r14','base_native','r7_native','r14_native']:
   p=old/(role+'.mp4')
   if not (out/p.name).exists():os.link(p,out/p.name)
  for i in range(10):
   pp={}
   for role in ['target','base','r7','r14']:
    p=old/f'{i:03}_{role}.jpg';pp[role]=Image.open(p).convert('RGB')
    if not (out/p.name).exists():os.link(p,out/p.name)
   for role in ['condition','r7_round_only','r14_round_only','r7_round_only_native','r14_round_only_native']:
    im=Image.open(src/role/f'{i:05}.png').convert('RGB');pp[role]=im;im.save(out/f'{i:03}_{role}.jpg',quality=96)
   panels.append(pp)
  for role in ['condition','r7_round_only','r14_round_only','r7_round_only_native','r14_round_only_native']:
   if not (out/(role+'.mp4')).exists():encode(out,role)
  for name,ids in [('f5',[5]),('first5',list(range(5))),('last5',list(range(5,10)))]:
   sheet=Image.new('RGB',(2800,42+len(ids)*300),(16,23,34));d=ImageDraw.Draw(sheet);d.text((8,7),cid+' | zero-step rounding / trained patch | '+str(ids),font=font(21),fill='white')
   for row,i in enumerate(ids):
    h=np.asarray(Image.open(src/'mask'/f'{i:05}.png'))>0;yy,xx=np.where(h);box=(max(0,int(xx.min())-30),max(0,int(yy.min())-30),min(1024,int(xx.max())+31),min(576,int(yy.max())+31))
    for j,role in enumerate(roles):
     im=panels[i][role].crop(box);scale=min(394/im.width,256/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS);d.text((j*400+6,45+row*300),f'f{i} {role}',font=font(18),fill='white');sheet.paste(im,(j*400+(400-im.width)//2,74+row*300))
   sheet.save(O/'contacts'/f'{cid}_{name}.jpg',quality=96)
  metrics={a:read(src/f'{a}_metrics.json') for a in ['r7_round_only','r14_round_only']}
  cases.append({'id':cid,'scene':c.get('receiver_scene',c.get('scene')),'videos':{r:f'effects/{cid}/{r}.mp4' for r in roles},'native':{a:f'effects/{cid}/{a}_native.mp4' for a in arms},'frame_pattern':f'effects/{cid}/{{i}}_{{role}}.jpg','contact':f'contacts/{cid}_f5.jpg','metrics':metrics,'human_verdict':None})
 dump(O/'diagnostic_manifest.json',{'cases':cases,'roles':roles,'arms':arms,'training_steps':0,'new_generated_windows':4,'reused_windows':6})
 # 原SAM不是像素真值；这些图只用于定位残留条件，不能自动判定漏目标。
 audit=read(T/'r19/audit_result.json');plan=read(T/'r14/evaluation_plan.json');sheet=Image.new('RGB',(1536,1050),(16,23,34));d=ImageDraw.Draw(sheet)
 for row,cid in enumerate(['A022','A042','A041_w10']):
  c=next(c for c in plan['cases'] if c['eval_id']==cid);r=next(r for r in audit['real'] if r['eval_id']==cid);q=max(r['mask_checks'],key=lambda x:x['SAM_pixels_outside_model_H']);j=q['frame'];folder=Path(c['folder']);rgb=np.asarray(Image.open(sorted((folder/'rgb').glob('*.jpg'))[j]).convert('RGB')).copy();h=np.asarray(Image.open(sorted((folder/'model_mask').glob('*.png'))[j]))>0;sam=np.asarray(Image.open(sorted((folder/'sam').glob('*.png'))[j]))>0
  overlay=rgb.copy();overlay[h]=(.55*overlay[h]+.45*np.array([20,120,255])).astype('uint8');overlay[sam&~h]=[255,35,35];condition=rgb.copy();condition[h]=127
  d.text((8,row*350+5),f'{cid} input frame {j}; SAM outside H={q["SAM_pixels_outside_model_H"]}; red != verified target leak',font=font(21),fill='white')
  for col,(name,im) in enumerate([('source RGB',rgb),('blue=H red=SAM outside H',overlay),('actual masked condition',condition)]):d.text((col*512+8,row*350+32),name,font=font(18),fill='white');sheet.paste(Image.fromarray(im).resize((512,288)),(col*512,row*350+60))
 sheet.save(O/'contacts/real_mask_trace.jpg',quality=96)
 print('DIAGNOSTIC_ASSETS',len(cases),'24 videos / 140 referenced frames')
if __name__=='__main__':main()
