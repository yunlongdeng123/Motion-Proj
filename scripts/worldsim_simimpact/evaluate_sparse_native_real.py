"""使用作者原生规划指标评价保存预测，绝不把它冒称反馈闭环。"""
import argparse,json,sys,time,shutil
from pathlib import Path
import numpy as np
import torch
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-SPARSE-NATIVE-01/20260915-r1');B=Path('/root/autodl-tmp/external/worldsim_simimpact/SparseDrive');sys.path.insert(0,str(B))
from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import PlanningMetric
ap=argparse.ArgumentParser();ap.add_argument('--reference-dir',type=Path,default=R/'evaluation_r1');ap.add_argument('--outputs-dir',type=Path,default=R/'real_outputs_r1');args=ap.parse_args()
E=args.reference_dir;O=args.outputs_dir;manifest=json.loads((O/'manifest.json').read_text());assert manifest['completed']
all_rows=json.loads((O/'forward_rows.json').read_text());assert len(all_rows)==manifest['actual_forward_count']
conditions=list(dict.fromkeys(r.get('condition','real') for r in all_rows))
if len(conditions)>1:
    # 每条件交由同一原生评价入口运行，避免聚合时丢失条件维度。
    import subprocess
    by_condition={}
    for condition in conditions:
        sub=O/condition;sub.mkdir(exist_ok=False);selected=[r for r in all_rows if r['condition']==condition]
        (sub/'forward_rows.json').write_text(json.dumps(selected,indent=2));(sub/'manifest.json').write_text(json.dumps({**manifest,'actual_forward_count':len(selected)},indent=2))
        outref=E/condition;outref.mkdir(exist_ok=False)
        for name in ['registration.json','references.json']:shutil.copy2(E/name,outref/name)
        subprocess.run([sys.executable,__file__,'--reference-dir',str(outref),'--outputs-dir',str(sub)],check=True)
        by_condition[condition]=json.loads((outref/'real_results.json').read_text())
    baseline=by_condition['real_common'];original=by_condition['real_original'];render=by_condition['rendered_common']
    summary={'task_id':'WS-SIM-SPARSE-NATIVE-SENSOR-01','actual_forward_count':len(all_rows),'conditions':by_condition,'render_minus_common_mean_L2_m':render['mean_6waypoint_L2_m']-baseline['mean_6waypoint_L2_m'],'render_minus_common_FDE_m':render['mean_FDE_3s_m']-baseline['mean_FDE_3s_m'],'crop_minus_original_mean_L2_m':baseline['mean_6waypoint_L2_m']-original['mean_6waypoint_L2_m'],'scope':'Three native3s replay conditions with independent model memory. Not feedback, equal fitting information, or geometry causality.','human_verdict':None,'goal_status':'active'}
    (E/'paired_results.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='conditions'},indent=2));sys.exit(0)
refs=json.loads((E/'references.json').read_text());preds={r['original_index']:r for r in json.loads((O/'forward_rows.json').read_text())};metric=PlanningMetric();rows=[]
for ref in refs:
    i=ref['original_index'];p=torch.tensor(preds[i]['prediction'],dtype=torch.float32);g=torch.tensor(ref['future_lidar_origin_xyz'],dtype=torch.float32)[:,:2]
    boxes=[torch.tensor(b,dtype=torch.float32).reshape(1,-1,7) for b in ref['future_boxes_lidar_lwh']]
    raw_pred_collision=metric.evaluate_single_coll(p.clone(),boxes);gt_collision=metric.evaluate_single_coll(g.clone(),boxes)
    metric.update(p[None].clone(),g[None].clone(),torch.ones(1,6,2),boxes)
    error=torch.linalg.norm(p-g,dim=-1).numpy();prefix=np.cumsum(error)/np.arange(1,7)
    row={'original_index':i,'sample_token':ref['sample_token'],'prediction_lidar_xy':p.tolist(),'ground_truth_lidar_xy':g.tolist(),'endpoint_L2_m':error.tolist(),'prefix_L2_m':prefix.tolist(),'native_L2_avg_1_2_3s_m':float(prefix[[1,3,5]].mean()),'mean_6waypoint_L2_m':float(error.mean()),'FDE_3s_m':float(error[-1]),'pred_box_overlap_raw':raw_pred_collision.tolist(),'recorded_box_overlap':gt_collision.tolist(),'native_new_overlap_flags':(raw_pred_collision&~gt_collision).tolist()};rows.append(row)
assert len(rows)==8
native={k:v.tolist() for k,v in metric.compute().items()};prefix={k:(np.cumsum(v)/np.arange(1,7)).tolist() for k,v in native.items()}
report={'task_id':'WS-SIM-SPARSE-NATIVE-01','created_unix':time.time(),'model':'Official SparseDrive-S stage2 nuScenes','new_forward_count':manifest['actual_forward_count'],'warmup_forward_count':2,'evaluated_starts':len(rows),'native_raw_per_horizon':native,'native_prefix_metrics':prefix,'native_avg_1_2_3s':{k:float(np.array(v)[[1,3,5]].mean()) for k,v in prefix.items()},'mean_6waypoint_L2_m':float(np.mean([r['mean_6waypoint_L2_m'] for r in rows])),'mean_FDE_3s_m':float(np.mean([r['FDE_3s_m'] for r in rows])),'raw_pred_overlap_starts':sum(any(r['pred_box_overlap_raw']) for r in rows),'recorded_overlap_starts':sum(any(r['recorded_box_overlap']) for r in rows),'native_new_overlap_starts':sum(any(r['native_new_overlap_flags']) for r in rows),'rows':rows,'scope':'Native3s open-loop prediction only. No PDM vehicle execution or reconstructed-input evaluation; scene belongs to nuScenes train. GT route hint explicitly supplied.','human_verdict':None,'goal_status':'active'}
(E/'real_results.json').write_text(json.dumps(report,indent=2));shutil.copy2(__file__,E/'evaluator_source_snapshot.py');print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)
