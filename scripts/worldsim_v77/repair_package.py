"""固定30帧视频、全帧与放大对照；无需通过视觉检查才导出失败证据。"""
from repair_common import *
from PIL import Image,ImageDraw
import av,shutil
from fractions import Fraction

def encode(path,arrays):
 arrays=list(arrays);h,w=arrays[0].shape[:2]
 with av.open(str(path),'w',options={'movflags':'+faststart'}) as out:
  s=out.add_stream('libx264',rate=10);s.width=w;s.height=h;s.pix_fmt='yuv420p';s.thread_count=4;s.options={'crf':'18','preset':'fast','g':'10','bf':'0'}
  for i,a in enumerate(arrays):
   fr=av.VideoFrame.from_ndarray(a,format='rgb24');fr.pts=i;fr.time_base=Fraction(1,10)
   for p in s.encode(fr):out.mux(p)
  for p in s.encode():out.mux(p)
 with av.open(str(path)) as inp:
  frames=list(inp.decode(video=0));assert len(frames)==30 and abs(frames[-1].time-2.9)<1e-6
 return {'name':path.name,'frames':len(frames),'fps':10,'seconds':3,'width':w,'height':h,'decoded':True}
def marked(a,text,mask=None):
 im=Image.fromarray(a);d=ImageDraw.Draw(im);d.rectangle((0,0,1024,26),fill='#101923');d.text((8,7),text,fill='white')
 if mask is not None:
  y,x=np.where(mask)
  if len(y):d.rectangle((int(x.min())-2,int(y.min())-2,int(x.max())+2,int(y.max())+2),outline='yellow',width=2)
 return np.array(im)
def main():
 assert read(ROOT/'drive_state.json')['state']=='complete';review=ROOT/'review';review.mkdir(exist_ok=True);summary=[]
 guard=read(ROOT/'guard_summary.json') if (ROOT/'guard_summary.json').exists() else None
 for s in read(ROOT/'registration.json')['scenes']:
  out=ROOT/s['name'];dest=review/s['name'];dest.mkdir(exist_ok=True);parts={k:[] for k in ['original','precise','evidence_first','mask','evidence_map','zoom']};old=[];union=np.zeros((576,1024),bool)
  for i in range(30):union |= cv2.imread(str(out/'mask'/f'{i:05}.png'),0)>0
  y,x=np.where(union);rect=[max(0,int(x.min())-80),max(28,int(y.min())-60),min(1024,int(x.max())+100),min(576,int(y.max())+80)]
  for i,f in enumerate(s['source_frames']):
   rgb=np.array(Image.open(out/'rgb'/f'{i:05}.png'));a=np.array(Image.open(out/'precise'/f'{i:05}.png'));b=np.array(Image.open(out/'evidence_first'/f'{i:05}.png'))
   m=cv2.imread(str(out/'mask'/f'{i:05}.png'),0)>0;e=cv2.imread(str(out/'evidence_mask'/f'{i:05}.png'),0)>0
   label=f"{s['name']} actor {s['actor']} CAM{s['camera']} source f{f:03} t={f/10:.1f}s"
   parts['original'].append(marked(rgb,label,m));parts['precise'].append(marked(a,label+' | precise mask'));parts['evidence_first'].append(marked(b,label+' | evidence + residual'))
   mo=rgb.copy();mo[m]=(.5*mo[m]+.5*np.array([0,230,140])).astype('uint8');parts['mask'].append(marked(mo,label+' | green: deletion silhouette'))
   eo=rgb.copy();eo[m&~e]=(.4*eo[m&~e]+.6*np.array([195,60,200])).astype('uint8');eo[e]=[0,255,130];parts['evidence_map'].append(marked(eo,label+' | green: evidence / purple: residual'))
   x0,y0,x1,y1=rect;pieces=[]
   for im,title in [(rgb,'Original'),(a,'Precise mask'),(b,'Evidence + residual')]:
    crop=Image.fromarray(im[y0:y1,x0:x1]).resize((480,270));canvas=Image.new('RGB',(480,300),'#101923');canvas.paste(crop,(0,30));ImageDraw.Draw(canvas).text((10,8),f'{title} | source f{f}',fill='white');pieces.append(np.array(canvas))
   parts['zoom'].append(np.concatenate(pieces,axis=1))
   if s['name']!='official_000':old.append(marked(np.array(Image.open(FULL/s['name']/f"cam{s['camera']}/background/{f:05}.png")),label+' | prior full run'))
  if old:parts['legacy']=old
  if guard:
   for arm in ['precise','evidence_first']:parts['guard_'+arm]=[np.array(Image.open(out/'guard'/f'{arm}_{i:05}.jpg')) for i in range(30)]
  vids=[encode(dest/(k+'.mp4'),v) for k,v in parts.items()]
  # 10个等间隔时刻，不按生成质量挑选。
  picks=np.rint(np.linspace(0,29,10)).astype(int);sheet=Image.new('RGB',(1440,300*10))
  for j,i in enumerate(picks):sheet.paste(Image.fromarray(parts['zoom'][i]),(0,j*300))
  sheet.save(dest/'contact.jpg',quality=94)
  for i in [0,7,15,29]:Image.fromarray(parts['zoom'][i]).save(dest/f'zoom_{i:02}.jpg',quality=96)
  for p in ['selection.jpg','mask_comparison.jpg','mask_stats.json','evidence_stats.json','donor_index.json']:shutil.copy2(out/p,dest/p)
  if guard:shutil.copy2(out/'guard/summary.json',dest/'guard_summary.json')
  summary.append({'scene':s['name'],'actor':s['actor'],'camera':s['camera'],'source_frames':[s['source_frames'][0],s['source_frames'][-1]],'zoom_rect':rect,'videos':vids});print('packaged',s['name'],len(vids),flush=True)
 for p in ['registration.json','mask_summary.json','evidence_summary.json','drive_state.json','validation.json','guard_summary.json']:
  if (ROOT/p).exists():shutil.copy2(ROOT/p,review/p)
 dump(review/'summary.json',{'scenes':summary,'total_videos':sum(len(s['videos']) for s in summary),'total_decoded_frames':sum(sum(v['frames'] for v in s['videos']) for s in summary),'human_verdict':None})
if __name__=='__main__':main()
