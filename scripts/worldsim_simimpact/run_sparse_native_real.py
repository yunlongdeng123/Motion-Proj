"""原生六相机 SparseDrive 前向；该进程不加载未来轨迹或 actor 真值。"""
import copy,json,os,random,sys,time,shutil
from pathlib import Path
import numpy as np
import torch

R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-SPARSE-NATIVE-01/20260915-r1')
B=Path('/root/autodl-tmp/external/worldsim_simimpact/SparseDrive')
sys.path.insert(0,str(B));os.chdir(B)
from mmcv import Config
from mmcv.parallel import collate,scatter
from mmcv.runner import wrap_fp16_model
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose
import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import NuScenes3DDataset

src=R/'real_inputs_r2';reg=json.loads((src/'registration.json').read_text());inputs=json.loads((src/'policy_inputs.json').read_text())
assert reg['prepared'] and len(inputs)==reg['expected_forward_count']
out=R/'real_outputs_r1';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'source_snapshot.py')
torch.set_num_threads(4);random.seed(reg['seed']);np.random.seed(reg['seed']);torch.manual_seed(reg['seed'])
checkpoint=torch.load(R/'assets/sparsedrive_stage2.pth',map_location='cpu');state=checkpoint['state_dict']
if all(k.startswith('module.') for k in state):state={k[7:]:v for k,v in state.items()}
cfg=Config.fromfile(str(B/'projects/configs/sparsedrive_small_stage2.py'))
anchors=out/'checkpoint_anchors';anchors.mkdir()
for head,key in [('det_head','anchor'),('map_head','anchor')]:
    values=state[f'head.{head}.instance_bank.{key}'].numpy();cfg.model.head[head].instance_bank.anchor=values
for kind in ['motion','plan']:
    p=anchors/f'{kind}_anchor.npy';np.save(p,state[f'head.motion_plan_head.{kind}_anchor'].numpy());cfg.model.head.motion_plan_head[kind+'_anchor']=str(p)
# 完整任务 checkpoint 已有 backbone 权重，无需再加载通用 ImageNet 初始化。
cfg.model.img_backbone.pretrained=None
model=build_detector(cfg.model);loaded=model.load_state_dict(state,strict=True);model.cuda().eval();wrap_fp16_model(model)
manifest={'task_id':reg['task_id'],'source_inputs':str(src),'started_unix':time.time(),'checkpoint_bytes':(R/'assets/sparsedrive_stage2.pth').stat().st_size,
 'strict_load':str(loaded),'state_tensors':len(state),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
 'architecture':'Official config, model, FP16 wrapper, deformable op, FlashAttention, temporal memory, planning rescore and decoder.',
 'initialization':'Kmeans initialization arrays recovered from the corresponding official checkpoint parameters. Every model parameter strictly loaded; no regenerated anchors or training.',
 'ego_status_contract':'CAN field is supplied by the official test Collect pipeline, but source audit finds it read only as a training-loss target; inference caches predicted ego status. Do not claim a measured CAN input effect.',
 'completed':False,'human_verdict':None}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));del checkpoint,state
pipeline=Compose(cfg.test_pipeline)
dataset_stub=type('AugmentationConfig',(),{'data_aug_conf':cfg.data_aug_conf,'test_mode':True})()
aug=NuScenes3DDataset.get_augmentation(dataset_stub)
rows=[]
for source in inputs:
    datum={k:copy.deepcopy(source[k]) for k in ['timestamp','img_filename','lidar2img','cam_intrinsic','lidar2global','ego_status','gt_ego_fut_cmd']}
    for k in ['lidar2img','cam_intrinsic']:datum[k]=[np.array(v,dtype=np.float64) for v in datum[k]]
    for k in ['lidar2global','ego_status','gt_ego_fut_cmd']:datum[k]=np.array(datum[k],dtype=np.float64 if k=='lidar2global' else np.float32)
    datum['aug_config']=copy.deepcopy(aug);batch=scatter(collate([pipeline(datum)],samples_per_gpu=1),[0])[0]
    begin=time.time()
    with torch.no_grad():result=model(**batch)[0]['img_bbox']
    torch.cuda.synchronize();duration=time.time()-begin
    arrays={k:v.detach().cpu().numpy() for k,v in result.items() if isinstance(v,torch.Tensor)}
    assert arrays['final_planning'].shape==(6,2) and np.isfinite(arrays['final_planning']).all()
    path=out/f"frame_{source['original_index']:02d}.npz";np.savez_compressed(path,**arrays)
    row={'original_index':source['original_index'],'sample_token':source['sample_token'],'evaluated':source['original_index'] in reg['evaluation_original_indices'],'seconds':duration,'prediction':arrays['final_planning'].tolist(),'output_file':str(path)}
    rows.append(row);(out/'forward_rows.json').write_text(json.dumps(rows,indent=2));print('FORWARD',row['original_index'],'seconds',duration,flush=True)
manifest.update(completed=True,completed_unix=time.time(),actual_forward_count=len(rows),evaluated_count=sum(r['evaluated'] for r in rows),peak_GPU_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('REAL_FORWARD_COMPLETED',flush=True)
