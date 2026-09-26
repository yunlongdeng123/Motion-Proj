"""固定两scene短窗与mask，串行比较ProPainter和DriveEditor；不训练。"""
import argparse, json, os, pathlib, sys, time

ROOT = pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DRIVEEDITOR-COMPARE-20260927/r1')
SRC = pathlib.Path('/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor')
OLD = pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
PREFLIGHT = pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DRIVEEDITOR-ASSESS-20260926/r1')
SCENES = {'scene_0230': ('22', 2, 0), 'scene_0255': ('25', 3, 15)}

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def prepare():
    import cv2, numpy as np
    from PIL import Image
    from scipy.ndimage import binary_dilation
    if (ROOT/'registration.json').exists():
        raise RuntimeError('输入已登记，不覆盖；检查当前run后继续后续阶段')
    ROOT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for scene,(actor,cam,start) in SCENES.items():
        base=ROOT/scene
        for part in ['rgb','mask_a','mask_b','target_sam']:
            (base/part).mkdir(parents=True,exist_ok=True)
        frame_rows=[]
        for idx,f in enumerate(range(start,start+10)):
            source=pathlib.Path('/root/autodl-tmp/data/v76_vadgs')/scene/'images'/f'{f:03}_{cam}.jpg'
            im=cv2.resize(cv2.imread(str(source)),(1024,576),interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(base/'rgb'/f'{idx:05}.png'),im)
            sam=np.array(Image.open(OLD/scene/'masks'/f'cam{cam}'/f'{f:05}.png').convert('L'))>0
            # 原基线4像素膨胀先在原688x384网格完成，再映射公共尺寸。
            a=np.array(Image.fromarray(binary_dilation(sam,iterations=4).astype('uint8')*255).resize((1024,576),Image.Resampling.NEAREST))
            b=np.array(Image.open(PREFLIGHT/scene/f'mask_{f:03}.png').convert('L'))
            s=cv2.resize(sam.astype('uint8'),(1024,576),interpolation=cv2.INTER_NEAREST)*255
            assert b.shape==(576,1024) and np.isin(b,[0,255]).all()
            for part,arr in [('mask_a',a),('mask_b',b),('target_sam',s)]:
                cv2.imwrite(str(base/part/f'{idx:05}.png'),arr)
            frame_rows.append({'index':idx,'source_frame':f,'source':str(source),'mask_a_fraction':float((a>0).mean()),'mask_b_fraction':float((b>0).mean()),'b_covers_sam':float((b[s>0]>0).mean())})
        rows.append({'scene':scene,'actor_id':actor,'camera':cam,'frames':frame_rows})
    dump(ROOT/'registration.json',{'task_id':'WS-V77-DRIVEEDITOR-COMPARE-20260927','run_id':'r1','source_commit':'de2a2d42','question':'分离删除mask扩大与生成模型对残影/邻车保持的影响','input_roles':'真实RGB为合法输入；GT框仅定义冻结Bmask，SAM2定义A；没有隐藏背景GT。两scene为已曝光开发例，不是泛化测试。','conditions':{'A':'ProPainter + 原SAM2/4px膨胀后映射','B':'ProPainter + 预检物化官方扩大mask','C':'训练后DriveEditor + 与B逐像素相同mask'},'size':[1024,576],'frames_per_scene':10,'fps':10,'seed':42,'steps':25,'decoding_t':1,'sequential_cfg':True,'mask_dilation_runtime':0,'training_steps':0,'resource':'单RTX3090，模型串行，单阶段最多1800秒，CPU4线程','scenes':rows,'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending','human_verdict':None})
    print('PREPARED',flush=True)

def paint(scene,variant):
    import subprocess
    out=ROOT/scene/variant
    if (out/'result.json').exists():
        print('ALREADY_COMPLETE',out);return
    if out.exists(): raise RuntimeError(f'保留未完成输出，不能覆盖: {out}')
    out.mkdir()
    source=pathlib.Path('/root/autodl-tmp/third_party/worldsim_v77/ProPainter-hfspace')
    cmd=[sys.executable,str(source/'inference_propainter.py'),'--video',str(ROOT/scene/'rgb'),'--mask',str(ROOT/scene/('mask_a' if variant=='A' else 'mask_b')),'--output',str(out),'--width','1024','--height','576','--mask_dilation','0','--save_fps','10','--save_frames','--fp16','--subvideo_length','10','--neighbor_length','10','--ref_stride','10']
    t=time.monotonic()
    subprocess.run(cmd,cwd=source,check=True,timeout=900)
    assert len(list((out/'rgb/frames').glob('*.png')))==10
    dump(out/'result.json',{'status':'complete','model':'ProPainter','scene':scene,'variant':variant,'command':cmd,'elapsed_s':time.monotonic()-t,'frames':10,'training_steps':0,'human_verdict':None})

def make_adapter():
    import numpy as np, torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    sys.path.insert(0,str(SRC))
    from interactive_gui import GradioShow, load_model
    class FrozenMaskDeletion(GradioShow):
        def __init__(self,scene,load=True):
            self.out_size=(576,1024);self.out_size_3d=(576,576)
            self.num_frames=10;self.num_frames_3d=21;self.device='cuda'
            self.to_tensor=transforms.ToTensor()
            self.transform_mask=transforms.Compose([transforms.Resize([72,128],interpolation=InterpolationMode.NEAREST),transforms.Lambda(lambda x:x*2.-1.)])
            self.previous_segment_last_frame=None;self.used_previous_segment_condition=False
            self.im=[];self.masks=[];self.im_result=[]
            from PIL import Image
            for idx in range(10):
                self.im.append(np.array(Image.open(ROOT/scene/'rgb'/f'{idx:05}.png').convert('RGB')))
                self.masks.append(np.array(Image.open(ROOT/scene/'mask_b'/f'{idx:05}.png'))>0)
            if load:
                self.model=load_model('configs/sample.yaml',self.device,num_steps=25,num_frames=10,verbose=True)
        def get_deletion(self):
            cond=[];masks=[]
            for im,m in zip(self.im,self.masks):
                c=self.to_tensor(np.copy(im))*2.-1.
                c[:,m]=0.
                cond.append(c);masks.append(torch.from_numpy(m.astype('float32')).unsqueeze(0))
            mask=self.transform_mask(torch.stack(masks))
            return (torch.stack(cond),torch.ones(1,3,576,576),mask,torch.zeros_like(mask),torch.zeros(21,1),torch.zeros(21,1),torch.zeros(21,1),[{} for _ in cond],torch.ones(3,224,224),torch.ones(10,6,576,1024)*-1.,torch.zeros(10),torch.zeros(10))
    return FrozenMaskDeletion

def validate():
    import numpy as np, torch
    Adapter=make_adapter()
    import interactive_gui as gui
    # 与官方删除分支逐张量比较；只把随机框函数替换为已冻结矩形。
    class DummyBox:
        def corners(self):return np.ones((3,8))
    original=gui.scale_bbox
    checked=0
    try:
        for scene in SCENES:
            show=Adapter(scene,load=False)
            show.data={'data':[{'box':DummyBox()} for _ in range(10)]}
            show.camera_intrinsic=np.eye(3);show.ratio=.64
            boxes=[]
            for m in show.masks:
                yy,xx=np.where(m);x0,x1=xx.min(),xx.max()+1;y0,y1=yy.min(),yy.max()+1
                assert m.sum()==(x1-x0)*(y1-y0)
                boxes.append(np.round(np.array([x0,y0,x1,y1])/.64).astype('int32'))
            iterator=iter(boxes)
            gui.scale_bbox=lambda *a,**k:next(iterator)
            expected=gui.GradioShow.get_deletion(show);actual=show.get_deletion()
            for a,b in zip(actual,expected):
                assert torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b
            checked+=10
    finally:gui.scale_bbox=original
    dump(ROOT/'adapter_validation.json',{'frames':checked,'check':'12返回项与官方冻结矩形deletion完全一致','model_forwards':0,'status':'passed'})
    print('ADAPTER_VALIDATED',checked,flush=True)

def drive(scene):
    import numpy as np, torch
    from PIL import Image
    out=ROOT/scene/'C'
    if (out/'result.json').exists():print('ALREADY_COMPLETE',out);return
    if out.exists():raise RuntimeError(f'保留未完成输出，不能覆盖: {out}')
    out.mkdir();(out/'native').mkdir();(out/'composite').mkdir()
    Adapter=make_adapter()
    from interactive_gui import set_seed
    os.chdir(SRC);torch.set_num_threads(4)
    t=time.monotonic();show=Adapter(scene)
    torch.cuda.reset_peak_memory_stats();set_seed(42)
    show.predict(1,False,'Deletion')
    rows=[]
    for idx,(raw,im,m) in enumerate(zip(show.im_result,show.im,show.masks)):
        assert raw.shape==im.shape==(576,1024,3) and raw.dtype==np.uint8
        Image.fromarray(raw).save(out/'native'/f'{idx:05}.png')
        composite=np.where(m[...,None],raw,im)
        Image.fromarray(composite).save(out/'composite'/f'{idx:05}.png')
        diff=np.abs(raw.astype('int16')-im.astype('int16'))
        rows.append({'index':idx,'native_outside_mask_mae':float(diff[~m].mean()),'composite_outside_mask_max':int(np.abs(composite.astype('int16')-im.astype('int16'))[~m].max())})
    assert len(rows)==10
    dump(out/'result.json',{'status':'complete','model':'DriveEditor official trained checkpoint','adapter':'固定预检mask，避免GUI demo依赖；复用现有sequential CFG与单帧解码','scene':scene,'frames':rows,'elapsed_s':time.monotonic()-t,'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'steps':25,'seed':42,'decoding_t':1,'previous_segment_condition':False,'training_steps':0,'human_verdict':None})
    print('DRIVEEDITOR_DONE',scene,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','paint','validate','drive']);p.add_argument('--scene',choices=list(SCENES));p.add_argument('--variant',choices=['A','B']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='paint':paint(a.scene,a.variant)
    elif a.action=='validate':validate()
    else:drive(a.scene)
