"""CPU：原 Full5 B 输入/数学，新目录，增加可靠身份绑定与诊断保存节点。"""
from pathlib import Path
import copy,hashlib,json,shutil
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=B/'scene008_full_dynamic_edit_20261107_r1'
BASE=B/'r15_gn11_stage1_v41_20261023/arms/B/run30k'
ENGINE=B/'r15_baseline_owner5_20261018/training_source/scene0008_089_137'
API=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def points(p):
 v=PlyData.read(str(p))['vertex'].data
 return np.column_stack([v[k] for k in ['x','y','z']]).astype(np.float64)
def main():
 cfg=json.loads((BASE/'effective_config.json').read_text());original=copy.deepcopy(cfg);src=Path(cfg['source_path'])
 out=ROOT/'inputs/scene0008_089_137';out.mkdir(parents=True,exist_ok=False)
 for p in sorted(src.iterdir()):
  q=out/p.name
  if p.name=='points3d.ply' or p.name.startswith('points3d_dynamic_') or p.name=='transform_matrix.txt':shutil.copy2(p,q)
  else:q.symlink_to(p,target_is_directory=p.is_dir())
 mapping=[];aliases=[5,26,18,6,37];clouds=[]
 for i,actor in enumerate(aliases):
  p=out/f'points3d_dynamic_{i}.ply';xyz=points(p);clouds.append(xyz)
  mapping.append(dict(actor_id=actor,source_actor_index=i,ply=str(p),rows=len(xyz),sha256=sha(p)))
 assert [x['rows'] for x in mapping]==[21358,15193,2444,11625,98452]
 assert sorted(out.glob('points3d_dynamic_*.ply'))==[Path(x['ply']) for x in mapping]
 cloud=np.concatenate(clouds);nearest,_=cKDTree(cloud).query(points(out/'points3d.ply'),k=1,eps=0,distance_upper_bound=.1,workers=1)
 static=int((nearest>=.1).sum());assert len(cloud)==149072
 cfg.update(source_path=str(out),model_path=str(ROOT/'formal5k'),start_checkpoint=None,
  test_iterations=[],save_iterations=[5000,10000],checkpoint_iterations=[5000,10000],
  test_interval=1000000000,paper_eval_train_each_checkpoint=False,
  dynamic_actor_ids=[5,6,18,26,37],delete_actor_ids=[])
 cfg['followup_scene_spec']['source_path']=str(out);cfg['followup_scene_spec']['actor_ply_paths']=[x['ply'] for x in mapping]
 diff={k:dict(baseline=original.get(k),current=cfg.get(k)) for k in set(original)|set(cfg) if original.get(k)!=cfg.get(k)}
 allowed={'source_path','model_path','start_checkpoint','followup_scene_spec','test_iterations','save_iterations','checkpoint_iterations','test_interval','paper_eval_train_each_checkpoint','dynamic_actor_ids','delete_actor_ids'}
 assert set(diff)<=allowed
 assert cfg['hybrid_lambda_static_rgb']==1. and cfg['hybrid_lambda_dynamic_opacity']==100.
 assert cfg['iterations']==cfg['position_lr_max_steps']==30000 and cfg['seed']==0
 write(ROOT/'config.json',cfg);write(ROOT/'baseline_config.json',original)
 identity=dict(dynamic_actor_ids=[5,6,18,26,37],source_ply_order=aliases,source_actor_rows=mapping,
  expected_initial_dynamic_rows=len(cloud),expected_initial_static_rows=static,
  initialization='Original native Full5B PLY rows/order, colors/time/KNN and strict0.1m static exclusion unchanged',
  source_scene=str(src),actor_ids_are_not_positions_in_public_list=True)
 write(ROOT/'identity_contract.json',identity)
 write(ROOT/'arm_contract.json',dict(formal_stop=5000,next_stop_if_requested=10000,learning_rate_horizon=30000,
  expected_initial_static_rows=static,expected_initial_dynamic_rows=149072,api_package_root=str(API),
  method_changes={},exact_config_differences=diff,baseline=str(BASE),baseline_config_sha256=sha(BASE/'effective_config.json'),
  additional_engineering='Fresh approximately540s smoke plus2 resumed updates; exact count from runtime receipts',
  pilot_frames=[105,130,137],automatic_full49=False,shutdown_allowed=False,gpu_reset_allowed=False))
 write(ROOT/'preparation_receipt.json',dict(status='PASS_FULL5_B_INPUT_AND_MATH_COPIED_IDENTITY_METADATA_ONLY',
  identity=identity,config_diff=diff,old_scene_point_sha256=sha(src/'points3d.ply'),new_scene_point_sha256=sha(out/'points3d.ply'),
  all_dynamic_ply_bytes_equal=all(sha(src/Path(x['ply']).name)==x['sha256'] for x in mapping),GPU_used=False,updates=0))
 print(json.dumps(dict(status='READY_FULL5_SMOKE_INPUT',root=str(ROOT),static=static,dynamic=149072)))
if __name__=='__main__':main()
