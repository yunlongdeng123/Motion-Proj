"""r17：仪器化重放r15 neighbor一次，保存已有模型SV3D中间输出。

只定位第一次身份退化，不改条件，不把重放计作新候选收益。
"""
import os,sys,time,datetime,signal
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/work')
from hybrid_neighbor_condition import Engine,BASE,DE,set_seed
from hybrid_common import read,dump
import torch,numpy as np
from PIL import Image,ImageDraw
ROOT=BASE/'r17'

def prepare():
 assert not ROOT.exists();ROOT.mkdir()
 dump(ROOT/'registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r17',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_run='r15/neighbor',
  hypothesis='Inspect whether identity changes already appear in decoded SV3D views or first emerge during 2D video integration; preserve full 21-view prior.',
  operation='One exact-condition instrumentation replay, checkpoint/seed/mask/pose/reference unchanged. Save sampler 2D and 3D latents and decode the existing 3D output; compare all 10 2D outputs to preserved r15 PNG.',
  inputs='Same r15 original actor52 reference and GT52 conditions; no new data, training, model family or download',fixed=dict(seed=42,steps=25,num_frames=10,num_views=21,sequential_cfg=True,decode_chunk=1),
  stop_rule='One replay only. If 2D pixels fail reproduction, do not use intermediate output to causally explain old r15 without resolving discrepancy.',resources=dict(gpu='one RTX3090',cpu_threads=4,timeout_seconds=600),failure_ledger_refs=['V77-F02'],human_verdict=None,background_input_dir=None))
 print('R17_PREPARED',flush=True)

def run():
 assert (ROOT/'registration.json').exists() and not (ROOT/'state.json').exists();os.chdir(DE);state=dict(state='loading',pid=os.getpid(),human_verdict=None);dump(ROOT/'state.json',state)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('r17 replay exceeded 600s')))
 try:
  e=Engine(True);e.arm='neighbor';e.im_result=[];sampler=e.model.sampler;saved={}
  def capture(*args,**kwargs):
   video,views=sampler(*args,**kwargs);saved['video']=video.detach().cpu();saved['views']=views.detach().cpu();torch.save(saved,ROOT/'sampler_latents.pt');return video,views
  e.model.sampler=capture;set_seed(42);torch.cuda.reset_peak_memory_stats();start=time.time();signal.alarm(600);state.update(state='running');dump(ROOT/'state.json',state)
  e.predict(1,False,'Deletion');checks=[];(ROOT/'video_native').mkdir()
  for i,im in enumerate(e.im_result):
   prev=np.array(Image.open(BASE/'r15/neighbor/native'/f'{i:05}.png'));err=np.abs(im.astype('int16')-prev.astype('int16'));checks.append(dict(frame=65+i,max_pixel_difference=int(err.max()),changed_pixels=int((err>0).any(-1).sum())));Image.fromarray(im).save(ROOT/'video_native'/f'{i:05}.png')
  assert all(r['max_pixel_difference']==0 for r in checks),'Instrumentation replay does not reproduce r15 pixels'
  with torch.no_grad():views=e.model.decode_first_stage(saved['views'].to('cuda'));views=((views+1)/2).clamp(0,1);views=(views.permute(0,2,3,1).cpu().numpy()*255).astype('uint8')
  (ROOT/'sv3d').mkdir();angles=np.rad2deg(e.payloads['neighbor'][5].numpy());sheet=Image.new('RGB',(7*300,3*330),(20,30,45))
  for i,(im,angle) in enumerate(zip(views,angles)):
   Image.fromarray(im).save(ROOT/'sv3d'/f'{i:02}_{angle:.0f}.png');p=Image.fromarray(im).resize((300,300));sheet.paste(p,((i%7)*300,(i//7)*330));ImageDraw.Draw(sheet).text(((i%7)*300+8,(i//7)*330+303),f'view {i} | relative {angle:.0f} deg',fill='yellow')
  sheet.save(ROOT/'all21_views.jpg',quality=96);signal.alarm(0)
  result=dict(reproduced_video_exact=True,video_checks=checks,decoded_views=len(views),view_angles_deg=angles.tolist(),selected_query_indices=e.payloads['neighbor'][6].tolist(),seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,human_verdict=None)
  dump(ROOT/'replay_validation.json',result);state.update(state='complete',result=result);dump(ROOT/'state.json',state);print('R17_COMPLETE',result['seconds'],result['peak_allocated_gib'],flush=True)
 except Exception as exc:
  signal.alarm(0);state.update(state='failed_engineering',error=repr(exc));dump(ROOT/'state.json',state);raise

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[-1]]()
