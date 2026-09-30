"""无权重CPU数据适配：完全沿用DriveEditor blank分支的字段/通道，不传完整X。"""
from pathlib import Path
import argparse,json,sys,inspect,ast,gc
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def deletion_batch(y,x,hole,seed=42,fps_id=10,index=0):
    """Y只进入监督jpg；条件仅由X和H构造。B/C标签不加入网络输入。"""
    if y.dtype!=np.uint8 or x.shape!=y.shape or x.dtype!=np.uint8 or hole.dtype!=bool or hole.shape!=x.shape[:3]:
        raise ValueError('需要同尺寸uint8 THWC Y/X以及bool THW H')
    if np.any((x!=y).any(-1)&~hole):raise ValueError('合成影响超出H；不能作为当前完全遮挡合同')
    t,h,w,_=y.shape
    if h%8 or w%8:raise ValueError('图像尺寸须可被8整除')
    gen=torch.Generator(device='cpu').manual_seed(seed)
    tensor=lambda a:torch.from_numpy(a.copy()).permute(0,3,1,2).float()/127.5-1
    target=tensor(y); condition=tensor(x);condition.masked_fill_(torch.from_numpy(hole[:,None]),0)
    noise=lambda shape:torch.randn(shape,generator=gen)
    aug=torch.exp(-3+.5*noise(())).float();obj=torch.ones(3,224,224);ref3d=torch.ones(1,3,576,576)
    mask=F.interpolate(torch.from_numpy(hole[:,None].astype(np.float32)),size=(h//8,w//8),mode='nearest')*2-1
    return {'jpg':target,'jpg_3d':noise((21,3,576,576)),
        'cond_frames':condition+aug*noise(condition.shape),'cond_frames_eval':condition+.02*noise(condition.shape),
        'cond_frames_without_noise':[condition[0],obj],
        'cond_frames_3d':ref3d+aug*noise(ref3d.shape),'cond_frames_3d_eval':ref3d+.02*noise(ref3d.shape),
        'cond_frames_without_noise_3d':ref3d,'depth':torch.full((t,6,h,w),-1.),'mask_concat':mask,
        'mask_fuse':torch.zeros_like(mask),'fps_id':torch.tensor(fps_id),'motion_bucket_id':torch.tensor(127),
        'cond_aug':aug,'cond_aug_3d':aug.repeat(21).unsqueeze(-1),'polars_rad_3d':torch.zeros(21,1),
        'azimuths_rad_3d':torch.zeros(21,1),'indices_3d':torch.zeros(t),'image_only_indicator':torch.zeros(t),
        'num_video_frames':t,'image_only_indicator_3d':torch.zeros(21),'num_video_frames_3d':21,
        'obj_pos':[{} for _ in range(t)],'valid_mask':torch.zeros(t),'obj_ratio':torch.zeros(t),'index':index}

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def verify(root,official,case):
    torch.set_num_threads(2);sys.path.insert(0,str(official))
    # 无卡实例仅2GiB内存；执行原文件中未修改的方法AST，避开包级PL/网络导入。
    tree=ast.parse((official/'sgm/data/nus.py').read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NuScenesDataset')
    method=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='get_blank')
    rand=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rand_log_normal')
    namespace={'torch':torch,'np':np};exec(compile(ast.Module(body=[rand,method],type_ignores=[]),str(official/'sgm/data/nus.py'),'exec'),namespace)
    obj=type('LoaderFields',(),{})();obj.num_frames=2;obj.num_frames_3d=21;obj.out_size=(64,112);obj.fps_id=10
    # 该控制RGB已是目标尺寸，不发生Resize；mask用官方相同nearest缩至1/8。
    obj.transform_img=lambda a:torch.from_numpy(a.copy()).permute(2,0,1).float()/255*2-1
    obj.transform_mask=lambda a:F.interpolate(a,size=(8,14),mode='nearest')*2-1
    rng=np.random.default_rng(42);y=rng.integers(0,256,(2,64,112,3),dtype=np.uint8);x=y.copy();h=np.zeros((2,64,112),bool);h[:,16:48,32:80]=True;x[h]=rng.integers(0,256,(int(h.sum()),3),dtype=np.uint8)
    pos=[np.array([32,16,80,48]) for _ in range(2)]
    obj.video_data=[{'im':list(y),'pos':pos}];torch.manual_seed(42);blank=namespace['get_blank'](obj,0)
    obj.video_data=[{'im':list(x),'pos':pos}];torch.manual_seed(42);blank_x=namespace['get_blank'](obj,0)
    keys=['cond_frames','cond_frames_eval','cond_frames_without_noise','mask_concat','depth','cond_frames_3d','cond_frames_without_noise_3d']
    assert all(equal(blank[k],blank_x[k]) for k in keys)
    assert not equal(blank['jpg'],blank_x['jpg']) # 直接拿X当官方im会错误地监督重建A。
    del blank_x;gc.collect()
    adapted=deletion_batch(y,x,h)
    assert set(adapted)==set(blank)
    def shape(a):return [shape(v) for v in a] if isinstance(a,list) else (list(a.shape) if isinstance(a,torch.Tensor) else type(a).__name__)
    assert all(shape(adapted[k])==shape(blank[k]) for k in blank)
    assert torch.allclose(adapted['jpg'],blank['jpg'],atol=1e-7) and torch.allclose(adapted['cond_frames_without_noise'][0],blank['cond_frames_without_noise'][0],atol=1e-7)
    assert torch.equal(adapted['mask_concat'],blank['mask_concat'])
    key_count=len(blank);del blank;gc.collect()
    changed=x.copy();changed[h]=255-changed[h];alt=deletion_batch(y,changed,h)
    assert all(equal(adapted[k],alt[k]) for k in adapted)
    del alt;gc.collect()
    y2=y.copy();y2[h]=255-y2[h];alt_y=deletion_batch(y2,x,h)
    assert all(equal(adapted[k],alt_y[k]) for k in adapted if k!='jpg') and not equal(adapted['jpg'],alt_y['jpg'])
    del alt_y;gc.collect()
    x2=x.copy();y3=y.copy();x2[:,0,0]=0;y3[:,0,0]=0;vis=deletion_batch(y3,x2,h)
    assert not equal(adapted['cond_frames_without_noise'],vis['cond_frames_without_noise'])
    del adapted,vis;gc.collect()
    # 当前真实产物，10帧；检查模型洞严格覆盖全部合成影响。
    def images(role):return np.stack([np.asarray(Image.open(p).convert('RGB')) for p in sorted((case/role).glob('*.png'))[:10]])
    ry=images('Y');rx=images('X');rh=np.stack([np.asarray(Image.open(p))>0 for p in sorted((case/'model_hole').glob('*.png'))[:10]])
    # 全尺寸逐帧查条件，避免同时构造两套大型dummy-SV3D视频。
    assert not np.any((rx!=ry).any(-1)&~rh)
    for i in range(len(ry)):
        cx=rx[i].astype(np.float32)/127.5-1;cy=ry[i].astype(np.float32)/127.5-1;cx[rh[i]]=0;cy[rh[i]]=0
        assert np.array_equal(cx,cy)
    result={'official_repository':str(official),'official_get_blank_executed_cpu':True,'official_method_AST_unmodified':True,'official_key_count':key_count,
        'shape_schema_matches':True,'official_masked_condition_X_equals_Y':True,
        'feeding_X_as_official_im_changes_training_target_wrongly':True,
        'changing_hidden_X_changes_any_batch_field':False,'changing_hidden_Y_changes_only_jpg':True,
        'visible_context_change_reaches_condition':True,'actual_case':str(case),'actual_frames':len(ry),
        'actual_masked_X_equals_masked_Y':True,'network_channels_unchanged':9,'added_protected_channels':0,
        'full_synthetic_X_seen_as_condition':False,'gpu_forwards':0,'training_steps':0,
        'source_locations':{'get_blank':method.lineno},
        'limits':'执行官方get_blank未修改AST及适配器CPU张量合同；控制RGB预先同尺寸、mask同nearest。全尺寸10帧只核对像素条件；未导入完整训练包、未加载VAE/UNet或前向/反向，不认证GPU显存/效果。'}
    (root/'driveeditor_contract_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--official',type=Path,required=True);p.add_argument('--case',type=Path,required=True);a=p.parse_args();verify(a.root,a.official,a.case)
