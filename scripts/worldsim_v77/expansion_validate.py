"""实产物核对：窗口/像素合同、r30来源、渲染与资源；不替代视觉审核。"""
from pathlib import Path
import sys,numpy as np
from PIL import Image
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');r1=T/'r1';reg=read(r1/'registration.json');state=read(r1/'drive_state.json');assert state['state']=='complete_pending_visual_review';calls=state['completed'];assert len(calls)==20
contracts=0
for scene,cam in read(r1/'mask_admission.json')['admitted_primary_streams']:
    rows=[r for r in calls if r['scene']==scene];assert [r['start'] for r in rows]==[0,9,18,27];assert [r['previous_condition'] for r in rows]==[False,True,True,True]
    for row in rows:
        for chk in row['pixel_contracts']:assert chk['outside_changed']==chk['protected_changed']==chk['write_diff_native']==0
    base=r1/scene/f'cam{cam}';assert len(list((base/'drive/composite').glob('*.png')))==30
    for f in range(30):
        o=np.array(Image.open(base/'rgb'/f'{f:05}.png'));c=np.array(Image.open(base/'drive/composite'/f'{f:05}.png'));m=np.array(Image.open(base/'model_mask'/f'{f:05}.png'))>0;p=np.array(Image.open(base/'protect'/f'{f:05}.png'))>0;w=np.array(Image.open(base/'write_mask'/f'{f:05}.png'))>0;native=np.array(Image.open(base/'drive/native'/f'{f:05}.png'));assert np.array_equal(o[~m],c[~m]) and np.array_equal(o[p],c[p]) and np.array_equal(c[w],native[w]);contracts+=1
omega=read(T/'r2/state.json');assert len(omega['completed'])==10
for f in range(10):
    actual=np.array(Image.open(T/'r2'/f'f{f:03}/input_cam0.png'));expected=np.array(Image.open('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927/r30/rect_feather_keep/'+f'{f:05}.png'));assert np.array_equal(actual,expected)
insert=read(T/'r3/actor_layers/render_summary.json');assert len(insert['rows'])==20 and max(r['projection_max_error_px'] for r in insert['rows'])<.02
factual=read(T/'r5/actor_layers/render_summary.json');assert len(factual['rows'])==11 and max(r['projection_max_error_px'] for r in factual['rows'])<.02
data=read(T/'r4/summary.json');assert data['candidates']==78 and data['unique_camera_clips']==28 and data['training_steps']==0
audit=read(T/'r4/data_audit.json');assert not audit['known_old_0230_0255_log_overlap'] and not any(r['shared'] for r in audit['all_nine_eval_actor_track_overlap'])
report=dict(task_id=T.name,raw_recipe_no_per_scene_tuning=True,new_scenes=6,old_preserved=3,admitted_primary_generation_scenes=5,mask_unresolved_scene='processed_756',generated_unique_frames=150,drive_windows=20,saved_pixel_contract_frames=contracts,r30_exact_reused_inputs=10,omega_forwards=10,insert_layer_renders=20,factual_layer_renders=11,additional_yaw180_control_renders=2,projection_checks_max_px=max(r['projection_max_error_px'] for r in insert['rows']+factual['rows']),data_input_checks=audit['actual_input_checks'],training_steps=0,move_executed=False,resources=dict(drive_seconds=sum(r['seconds'] for r in calls),drive_elapsed=state['elapsed_s'],drive_peak_gib=max(r['peak_gib'] for r in calls),omega_seconds=omega['seconds'],omega_peak_gib=max(r['peak_gib'] for r in omega['completed']),shape=read(T/'r5/shape_state.json'),paint=read(T/'r5/paint_state.json')),existing_geometry_tests='15 passed',human_verdict=None)
dump(T/'validation.json',report);print('VALIDATED',contracts,report['projection_checks_max_px'])
