"""r3：生成条件与保护写回分离。固定矩形上下文，不改权重/seed。"""
import sys,os,time,gc,random,shutil,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_diffueraser as deio
from hybrid_common import read,dump,rgb,mask,write_mask,W,H,HW,Image,np,cv2
import torch
R2=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927/r2')
ROOT=R2.parent/'r3';SOURCE=deio.SOURCE;WEIGHTS=deio.WEIGHTS

def prepare():
 assert not (ROOT/'registration.json').exists();ROOT.mkdir(exist_ok=True)
 cfg=read(R2/'registration.json');cfg.update(run_id='r3',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hypothesis='vehicle silhouette/shadow and fragmented protected holes anchor completion; separate model mask from exact protected writeback',arms=['rect_condition'],fallback='if vehicle prior remains, same mask with noise initialization as diagnostic only',model_mask='target bounding rectangle: left/right/top16px bottom48px; no protect holes; observed outside allowed model edits locked on output',writeback='rectangle minus protect, factual observed locked',source_run='r2',current_user_instruction='continue diagnosis and iteration until clean temporally stable completion; do not stop at failure report',human_verdict=None)
 dump(ROOT/'registration.json',cfg)
 for s in cfg['scenes']:
  old=R2/s['name'];out=ROOT/s['name'];out.mkdir()
  for key in ['condition','write','input','masked','observed']:(out/key).mkdir()
  stats=[]
  for i in range(30):
   original=rgb(old/'rgb'/f'{i:05}.png');core=mask(old/'core'/f'{i:05}.png');protect=mask(old/'protect'/f'{i:05}.png');y,x=np.where(core)
   rect=np.zeros(HW,bool);box=[max(0,int(x.min())-16),max(0,int(y.min())-16),min(W,int(x.max())+17),min(H,int(y.max())+49)];rect[box[1]:box[3],box[0]:box[2]]=True
   observed=mask(old/'observed'/f'{i:05}.png')&rect&~protect
   # 条件中一并遮挡设施；最终精确恢复设施，不把其重画结果写入输出。
   source=original.copy();e=rgb(old/'evidence'/f'{i:05}.png');source[observed]=e[observed]
   for key,a in [('condition',rect),('write',rect&~protect),('observed',observed)]:write_mask(out/key/f'{i:05}.png',a)
   Image.fromarray(source).save(out/'input'/f'{i:05}.png');Image.fromarray(np.where(rect[...,None],0,source).astype('uint8')).save(out/'masked'/f'{i:05}.png')
   stats.append(dict(frame=s['source_frames'][i],box=box,condition_pixels=int(rect.sum()),protected_pixels_hidden_in_model=int((rect&protect).sum()),write_pixels=int((rect&~protect).sum()),observed_pixels=int(observed.sum())))
  dump(out/'input_stats.json',stats)
  sheet=Image.new('RGB',(W*2,H*4))
  for j,i in enumerate([0,7,15,29]):
   im=rgb(old/'rgb'/f'{i:05}.png');p=mask(old/'protect'/f'{i:05}.png');c=mask(out/'condition'/f'{i:05}.png');a=im.copy();a[c]=(.45*a[c]+.55*np.array([70,100,250])).astype('uint8');a[p]=(.5*a[p]+.5*np.array([0,230,220])).astype('uint8')
   sheet.paste(Image.fromarray(a),(0,j*H));sheet.paste(Image.open(out/'masked'/f'{i:05}.png'),(W,j*H))
  sheet.resize((1280,1430)).save(out/'input_review.jpg',quality=94)
 backup=ROOT/'repo_backups';backup.mkdir();status=deio.REPO/'docs/RESEARCH_STATUS.md';shutil.copy2(status,backup/status.name)
 status.write_text('''# 当前研究状态

更新：2026-09-27，v77。用户明确要求继续排查与迭代，当前目标是干净、时序稳定的补景；上一轮失败报告不等于目标完成。`WS-V77-HYBRID-BG-20260927/r3`继续固定两scene、原权重/seed，在真实图片/视频上检查。

当前工程假设：r2模型mask保留车形边界/阴影及围栏孔洞，会留下不利补景线索。r3区分连贯模型条件区域与精确输出保护区，覆盖车身+阴影，生成后恢复protect与observed。先各30帧矩形条件控制；若prior仍有车形，仅预注册同mask的noise初始化诊断，不机械换seed。输出仍须视觉与语义gate，不因检测为零自动准入。

GPU实际已恢复且启动前空闲。旧r2、DriveEditor FULL默认、Ω/GLB及全部失败保留；无定时任务、新MOVE或训练。主要路径`/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927/r3`。原HTML`outputs/v77-hybrid-delete/index.html`保留，将追加新对照；human_verdict始终null。
''')
 print('PREPARED',flush=True)

def run():
 assert not (ROOT/'state.json').exists();cfg=read(ROOT/'registration.json');state=dict(state='running',pid=os.getpid(),phase='prior',started=time.time(),scenes={},human_verdict=None);dump(ROOT/'state.json',state)
 pmod=deio.module(SOURCE/'propainter/inference.py','r3_prior',[('        ##save composed video##','        capture(comp_frames, output_path, "prior_png")\n        ##save composed video##')]);pmod.read_frame_from_videos=lambda p,l:(deio.images(p),10.,(W,H),'r3',30);pmod.read_mask=deio.exact_prior_mask
 prior=pmod.Propainter(str(WEIGHTS/'propainter'),device=torch.device('cuda'))
 for s in cfg['scenes']:
  out=ROOT/s['name'];random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  prior.forward(str(out/'input'),str(out/'condition'),str(out/'prior_native.mp4'),video_length=3,mask_dilation=0,save_fps=10,resize_ratio=.6,neighbor_length=10,subvideo_length=50)
 del prior;gc.collect();torch.cuda.empty_cache()
 dmod=deio.module(SOURCE/'diffueraser/diffueraser.py','r3_diffusion',[('        ################ Compose ################','        capture(images, output_path, "native_png")\n        ################ Compose ################')]);dmod.read_video=deio.exact_video;dmod.read_mask=deio.exact_diffusion_mask;dmod.read_priori=lambda p,fps,n,size:deio.images(p,size)
 d=dmod.DiffuEraser(torch.device('cuda'),str(WEIGHTS/'stable-diffusion-v1-5'),str(WEIGHTS/'sd-vae-ft-mse'),str(WEIGHTS/'diffuEraser'),ckpt='2-Step')
 for s in cfg['scenes']:
  old=R2/s['name'];out=ROOT/s['name'];state.update(phase='diffusion',scene=s['name']);dump(ROOT/'state.json',state)
  random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  d.forward(str(out/'input'),str(out/'condition'),str(out/'prior_png'),str(out/'diffueraser_native.mp4'),max_img_size=960,video_length=3,mask_dilation_iter=0,seed=42,guidance_scale=0,blended=False)
  (out/'final').mkdir();checks=[]
  for i,f in enumerate(s['source_frames']):
   original=rgb(old/'rgb'/f'{i:05}.png');final=original.copy();wr=mask(out/'write'/f'{i:05}.png');obs=mask(out/'observed'/f'{i:05}.png');native=rgb(out/'native_png'/f'{i:05}.png');e=rgb(out/'input'/f'{i:05}.png');protect=mask(old/'protect'/f'{i:05}.png')
   final[wr]=native[wr];final[obs]=e[obs];assert np.array_equal(final[protect],original[protect]);assert np.array_equal(final[~wr],original[~wr]);Image.fromarray(final).save(out/'final'/f'{i:05}.png');checks.append(dict(frame=f,protect_changes=0,outside_changes=0))
  dump(out/'pixel_checks.json',checks);state['scenes'][s['name']]=dict(frames=30,complete=True);dump(ROOT/'state.json',state)
  sheet=Image.new('RGB',(420*4,236*4))
  from PIL import ImageDraw
  for j,i in enumerate([0,7,15,29]):
   m=mask(old/'core'/f'{i:05}.png');yy,xx=np.where(m);cw=480 if s['name']=='scene_0230' else 400;ch=round(cw*H/W);left=round(max(0,min(W-cw,(xx.min()+xx.max()-cw)/2)));top=round(max(0,min(H-ch,(yy.min()+yy.max()-ch)/2)))
   for k,(name,path) in enumerate([('Original',old/'rgb'),('r2',old/'final'),('rect prior',out/'prior_png'),('r3 rect',out/'final')]):
    im=Image.fromarray(rgb(path/f'{i:05}.png')).crop((left,top,left+cw,top+ch)).resize((420,236));ImageDraw.Draw(im).text((5,5),f'{name} f{s["source_frames"][i]}',fill='yellow');sheet.paste(im,(k*420,j*236))
  sheet.save(out/'quicklook.jpg',quality=94);print('COMPLETE',s['name'],flush=True)
 state.update(state='complete',elapsed_seconds=time.time()-state['started']);dump(ROOT/'state.json',state)
if __name__=='__main__':
 if sys.argv[-1]=='prepare':prepare()
 elif sys.argv[-1]=='run':run()
 else:raise ValueError('prepare or run required')
