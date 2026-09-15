"""按已测下游事件整理完整分母；不读取几何误差用于排序。"""
import json
from pathlib import Path
W=Path(__file__).resolve().parent;D=W/'evidence_milestone3';O=W.parents[1]/'outputs/Simulation_Impact_Research'
allrows=json.loads((D/'pdm_simulation_summary.json').read_text());base={r['scene']:r for r in allrows if r['condition']=='real'}
rows=[]
for r in allrows:
 if r.get('protocol')!='build_scale' or r['condition']!='full':continue
 b=base[r['scene']];q={k:r[k] for k in ['scene','method','variant']}
 q.update(endpoint_change_m=r['final_position_change_vs_real_m'],ADE_delta_vs_real_m=r['ADE_vs_recorded_ego_m']-b['ADE_vs_recorded_ego_m'],
          FDE_delta_vs_real_m=r['FDE_vs_recorded_ego_m']-b['FDE_vs_recorded_ego_m'],
          min_acceleration_delta_mps2=r['minimum_longitudinal_acceleration_mps2']-b['minimum_longitudinal_acceleration_mps2'],
          final_speed_mps=r['final_speed_mps'],real_final_speed_mps=b['final_speed_mps'],real_contact=b['actor_overlap_any'])
 q['events']={'new_annotated_contact':r['actor_overlap_any'] and not b['actor_overlap_any'],
              'endpoint_change_gt_1m':q['endpoint_change_m']>1.,
              'additional_braking_gt_2mps2':q['min_acceleration_delta_mps2']<-2.,
              'new_stop':q['final_speed_mps']<.5 and q['real_final_speed_mps']>2.}
 q['measured_replay_ADE_worsened']=q['ADE_delta_vs_real_m']>0
 q['claim_status']='Discovery signal only; coverage, baseline robustness, geometry-local recovery and independent confirmation required'
 rows.append(q)
rows.sort(key=lambda r:(not r['events']['new_annotated_contact'],not r['events']['new_stop'],not r['events']['additional_braking_gt_2mps2'],not r['events']['endpoint_change_gt_1m'],-r['ADE_delta_vs_real_m']))
out={'task':'WS-SIM-DOWNSTREAM-MINING-01','source':'Completed frozen six-log interaction cohort; exposed discovery, no new inference in ranking',
 'scope':'All 48 full reconstructed scans after ordinary BUILD scale. Endpoint changes are differences, not automatically harms. Min-gap/TTC/detection not invented when absent from this aggregate.',
 'event_counts':{e:sum(r['events'][e] for r in rows) for e in rows[0]['events']},
 'event_scene_counts':{e:len({r['scene'] for r in rows if r['events'][e]}) for e in rows[0]['events']},'rows':rows}
(O/'DOWNSTREAM_EVENT_RANKING.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
for r in rows:
 if any(r['events'].values()):print(r['scene'],r['method'],r['variant'],r['events'],round(r['ADE_delta_vs_real_m'],5))
