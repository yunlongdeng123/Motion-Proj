"""同模型/seed/窗口对照：精确实例洞；真实证据+残余洞。"""
import fcntl,os,signal,time,shutil
os.environ.setdefault('DRIVEEDITOR_SEQUENTIAL_CFG','1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:128')
from repair_common import *
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
DE=pathlib.Path('/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor')
os.chdir(DE);sys.path.insert(0,str(DE))
from interactive_gui import GradioShow,load_model,set_seed
class Engine(GradioShow):
 def __init__(self):
  self.out_size=(576,1024);self.out_size_3d=(576,576);self.num_frames=10;self.num_frames_3d=21;self.device='cuda'
  self.to_tensor=transforms.ToTensor();self.transform_img=transforms.Compose([self.to_tensor,transforms.Lambda(lambda x:x*2-1)])
  self.transform_mask=transforms.Compose([transforms.Resize([72,128],interpolation=InterpolationMode.NEAREST),transforms.Lambda(lambda x:x*2-1)])
  self.previous_segment_last_frame=None;self.im_result=[];self.model=load_model('configs/sample.yaml','cuda',num_steps=25,num_frames=10,verbose=True)
 def get_deletion(self):
  cond=[]
  for im,m in zip(self.im,self.masks):
   c=self.to_tensor(im.copy())*2-1;c[:,m]=0;cond.append(c)
  mask=self.transform_mask(torch.from_numpy(np.stack(self.masks).astype('float32'))[:,None])
  return (torch.stack(cond),torch.ones(1,3,576,576),mask,torch.zeros_like(mask),torch.zeros(21,1),torch.zeros(21,1),torch.zeros(21,1),[{} for _ in cond],torch.ones(3,224,224),torch.ones(10,6,576,1024)*-1,torch.zeros(10),torch.zeros(10))
def main():
 lock=open(ROOT/'drive.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (ROOT/'evidence_summary.json').exists();assert not (ROOT/'drive_state.json').exists()
 reg=read(ROOT/'registration.json');torch.set_num_threads(4);state={'state':'loading','completed':[],'reused':[],'human_verdict':None};dump(ROOT/'drive_state.json',state);started=time.monotonic()
 def timeout(*_):raise TimeoutError('单窗180秒预算')
 signal.signal(signal.SIGALRM,timeout)
 try:
  e=Engine()
  for s in reg['scenes']:
   out=ROOT/s['name'];er=read(out/'evidence_stats.json')
   for arm,imgfolder,maskfolder in [('precise','rgb','mask'),('evidence_first','evidence','residual')]:
    dest=out/arm;dest.mkdir(exist_ok=False)
    if arm=='evidence_first' and sum(r['evidence_pixels'] for r in er)==0:
     for p in (out/'precise').glob('*.png'):shutil.copy2(p,dest/p.name)
     state['reused'].append({'scene':s['name'],'arm':arm,'reason':'零真实证据覆盖，输入与精确mask组完全相同，禁止重复推理'});dump(ROOT/'drive_state.json',state);continue
    for folder in ['native','windows']:(dest/folder).mkdir()
    prev=None;written=set()
    for start in range(0,29,9):
     assert time.monotonic()-started<2400
     ids=[min(start+j,29) for j in range(10)];valid=min(10,30-start)
     e.im=[np.array(Image.open(out/imgfolder/f'{i:05}.png').convert('RGB')) for i in ids];e.masks=[cv2.imread(str(out/maskfolder/f'{i:05}.png'),0)>0 for i in ids]
     e.previous_segment_last_frame=prev;e.im_result=[];set_seed(42);torch.cuda.reset_peak_memory_stats();t=time.monotonic();signal.alarm(180)
     try:e.predict(1,False,'Deletion')
     finally:signal.alarm(0)
     wd=dest/'windows'/f'{start:05}';wd.mkdir();assert len(e.im_result)==10
     for j,(i,raw,im,m) in enumerate(zip(ids,e.im_result,e.im,e.masks)):
      assert raw.dtype==np.uint8 and raw.shape==(576,1024,3)
      Image.fromarray(raw).save(wd/f'{j:02}.png');comp=np.where(m[...,None],raw,im)
      if j<valid and i not in written:Image.fromarray(comp).save(dest/f'{i:05}.png');Image.fromarray(raw).save(dest/'native'/f'{i:05}.png');written.add(i)
     prev=np.array(Image.open(dest/f'{ids[valid-1]:05}.png'))
     row={'scene':s['name'],'arm':arm,'start':start,'source_frames':[s['source_frames'][i] for i in ids],'seconds':time.monotonic()-t,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'previous_condition':e.used_previous_segment_condition};state['completed'].append(row);state['state']='running';dump(ROOT/'drive_state.json',state);print(row,flush=True)
    assert len(written)==30
  state.update(state='complete',elapsed_s=time.monotonic()-started);dump(ROOT/'drive_state.json',state)
 except Exception as ex:state.update(state='failed',error=repr(ex));dump(ROOT/'drive_state.json',state);raise
if __name__=='__main__':main()
