"""Summarize actual shared-training geometry, native supervision and support by epoch."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
records=[json.loads(line) for line in (args.run/'train.jsonl').read_text().splitlines() if line.strip()]
epochs=defaultdict(list)
for row in records: epochs[row['epoch']].append(row)
def distribution(values):
    values=np.asarray([value for value in values if value is not None],float)
    return {'count':len(values),'mean':float(values.mean()) if len(values) else None,
            'median':float(np.median(values)) if len(values) else None,
            'p95':float(np.quantile(values,.95)) if len(values) else None}
rows=[]
for epoch,values in sorted(epochs.items()):
    native=[row for row in values if row.get('native_observed_points',0)>0]
    views=[row for row in values if row['views']>0]
    count=sum(row['native_observed_points'] for row in native)
    rows.append({'epoch':epoch,'actor_presentations':len(values),'optimizer_updates':sum(row.get('optimizer_step',True) for row in values),
        'unique_actors':len({row['owner'] for row in values}),'actors_with_views':len(views),
        'predicted_support_fallback_with_views':sum(row.get('lidar_fallback',False) for row in views),
        'no_camera_pose_presentations':sum(row.get('fallback_reason')=='no_actor_camera_pose' for row in values),
        'native_supervised_presentations':len(native),'native_measurements':count,
        'native_huber_measurement_weighted_m':sum(row['native_sensor_huber_m']*row['native_observed_points'] for row in native)/count if count else None,
        'native_huber_supervised_actor_distribution_m':distribution([row['native_sensor_huber_m'] for row in native]),
        'metrics':{metric:distribution([row.get(metric) for row in values]) for metric in
            ['loss','coverage_m','free_intrusion_m','free_objective','gradient_norm_before_clip','native_candidates','query_count','step_s']},
        'gradient_groups':{group:distribution([row.get('group_gradient_norms_before_clip',{}).get(group) for row in values])
                           for group in ['native_dpt','query_decoder']}})
result={'run':str(args.run),'run_status':json.loads((args.run/'status.json').read_text()),'epochs':rows,
    'total_presentations_in_saved_history':len(records),'optimizer_updates_in_saved_history':sum(row.get('optimizer_step',True) for row in records),
    'boundary':'training-only process evidence; target/ray samples change by epoch, no paired heldout inference or generalization claim',
    'native_aggregation':'both supervised Actor-equal and native-measurement-weighted Huber; unobserved Actors not counted as zero native error',
    'resume_boundary':'saved train history already contains restored complete epochs; original discarded partial epoch stays in parent run and is not counted twice'}
resume=args.run/'resume.json'
if resume.is_file():
    result['resume']=json.loads(resume.read_text())
    parent=Path(result['resume']['parent_run'])
    interruption=parent/'interruption.json'
    if interruption.is_file():
        result['parent_interruption']=json.loads(interruption.read_text())
        result['actual_presentations_including_discarded_parent']=len(records)+result['parent_interruption']['unsaved_presentations']
summary=args.run/'summary.json'
if summary.is_file():
    completed=json.loads(summary.read_text())
    result['completed_run_resources']={key:completed.get(key) for key in
        ['wall_s','peak_gpu_gib','peak_rss_gib','trainable_parameters','native_project_max_change']}
    if 'parent_interruption' in result:
        result['combined_observed_wall_lower_bound_s']=completed['wall_s']+result['parent_interruption']['last_state']['elapsed_s']
        result['resource_boundary']='parent last observed elapsed + recovery wall; unknown parent exit delay and idle recovery gap excluded; peak memories are not additive'
args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'run_status':result['run_status'],'presentations':len(records),'first_epoch':rows[0],'last_epoch':rows[-1]}))
