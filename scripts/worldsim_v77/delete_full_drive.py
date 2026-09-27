"""一次加载DriveEditor，按同帧重叠的10帧窗口完成六相机背景。"""
import fcntl,json,os,pathlib,signal,sys,time
import numpy as np,torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from delete_full_common import ROOT,DE,dump

reg=json.loads((ROOT/'registration.json').read_text());torch.set_num_threads(4)
lock=(ROOT/'drive.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (ROOT/'drive_state.json').exists(),'先检查已运行状态，不能覆盖'
os.chdir(DE);sys.path.insert(0,str(DE))
from interactive_gui import GradioShow,load_model,set_seed

class Engine(GradioShow):
    def __init__(self):
        self.out_size=(576,1024);self.out_size_3d=(576,576);self.num_frames=10;self.num_frames_3d=21;self.device='cuda'
        self.to_tensor=transforms.ToTensor();self.transform_img=transforms.Compose([self.to_tensor,transforms.Lambda(lambda x:x*2-1)])
        self.transform_mask=transforms.Compose([transforms.Resize([72,128],interpolation=InterpolationMode.NEAREST),transforms.Lambda(lambda x:x*2-1)])
        self.previous_segment_last_frame=None;self.im_result=[]
        self.model=load_model('configs/sample.yaml','cuda',num_steps=25,num_frames=10,verbose=True)
    def get_deletion(self):
        cond=[]
        for im,m in zip(self.im,self.masks):
            c=self.to_tensor(im.copy())*2-1;c[:,m]=0;cond.append(c)
        mask=self.transform_mask(torch.from_numpy(np.stack(self.masks).astype('float32'))[:,None])
        return (torch.stack(cond),torch.ones(1,3,576,576),mask,torch.zeros_like(mask),torch.zeros(21,1),torch.zeros(21,1),torch.zeros(21,1),[{} for _ in cond],torch.ones(3,224,224),torch.ones(10,6,576,1024)*-1,torch.zeros(10),torch.zeros(10))

state={'state':'running','pid':os.getpid(),'completed_windows':[],'total_windows':len(reg['windows']),'training_steps':0,'human_verdict':None}
dump(ROOT/'drive_state.json',state);start_all=time.monotonic()
def timeout_signal(*args):raise TimeoutError('单窗180秒预算到')
signal.signal(signal.SIGALRM,timeout_signal)
try:
    show=Engine();print('DRIVEEDITOR_FULL_READY',flush=True)
    for scene in reg['scenes']:
        name=scene['name']
        for stream in scene['streams']:
            c=stream['camera'];base=ROOT/name/f'cam{c}'
            for folder in ['background','native']:(base/folder).mkdir(exist_ok=False)
            written=set();prev=None;prev_frame=None
            for w in stream['windows']:
                assert time.monotonic()-start_all<3600
                ids=w['source_frames'];state['current']={'scene':name,'camera':c,'start':w['start']};dump(ROOT/'drive_state.json',state)
                show.im=[np.array(Image.open(base/'rgb'/f'{f:05}.png').convert('RGB')) for f in ids]
                show.masks=[np.array(Image.open(base/'mask'/f'{f:05}.png'))>0 for f in ids]
                show.previous_segment_last_frame=prev if prev_frame==ids[0] else None
                show.im_result=[];set_seed(42);torch.cuda.reset_peak_memory_stats();t=time.monotonic()
                signal.alarm(180)
                try:show.predict(1,False,'Deletion')
                finally:signal.alarm(0)
                folder=base/'windows'/f'{w["start"]:05}';folder.mkdir(parents=True,exist_ok=False)
                assert len(show.im_result)==10
                frame_rows=[]
                for i,(f,raw,im,m) in enumerate(zip(ids,show.im_result,show.im,show.masks)):
                    assert raw.shape==(576,1024,3) and raw.dtype==np.uint8
                    comp=np.where(m[...,None],raw,im)
                    Image.fromarray(raw).save(folder/f'native_{i:02}.png',compress_level=2)
                    if i<w['valid_count'] and f not in written:
                        Image.fromarray(raw).save(base/'native'/f'{f:05}.png',compress_level=2)
                        Image.fromarray(comp).save(base/'background'/f'{f:05}.png',compress_level=2);written.add(f)
                    frame_rows.append({'source_frame':f,'input_index':i,'valid_output':i<w['valid_count'],'mask_area':int(m.sum()),'composite_outside_mask_max':int(np.abs(comp.astype('int16')-im.astype('int16'))[~m].max()) if (~m).any() else 0})
                end=ids[w['valid_count']-1]
                prev=np.array(Image.open(base/'background'/f'{end:05}.png').convert('RGB'));prev_frame=end
                record={'scene':name,'camera':c,'start':w['start'],'previous_condition':show.used_previous_segment_condition,'elapsed_s':time.monotonic()-t,'peak_gib':torch.cuda.max_memory_allocated()/2**30,'frames':frame_rows}
                dump(folder/'result.json',record);state['completed_windows'].append({k:record[k] for k in ['scene','camera','start','previous_condition','elapsed_s','peak_gib']});dump(ROOT/'drive_state.json',state)
                print(json.dumps(state['completed_windows'][-1]),flush=True);torch.cuda.empty_cache()
            for f in range(scene['count']):
                if f not in written:
                    assert not np.array(Image.open(base/'mask'/f'{f:05}.png')).any()
                    for part in ['background','native']:
                        import shutil
                        shutil.copy2(base/'rgb'/f'{f:05}.png',base/part/f'{f:05}.png')
            assert len(list((base/'background').glob('*.png')))==scene['count']
    state['state']='complete';state['elapsed_s']=time.monotonic()-start_all;dump(ROOT/'drive_state.json',state)
except Exception as exc:
    state['state']='failed';state['error']=repr(exc);dump(ROOT/'drive_state.json',state);raise
