"""按用户要求停止退化试验，完整保留产物并恢复上一轮默认配置。"""
import shutil,subprocess,datetime
from repair_common import *
state=read(ROOT/'drive_state.json');shutil.copy2(ROOT/'drive_state.json',ROOT/'drive_state_before_user_stop.json')
rows=[]
for s in read(ROOT/'registration.json')['scenes']:
 counts={}
 for arm in ['precise','evidence_first']:
  ids=sorted(int(p.stem) for p in (ROOT/s['name']/arm).glob('*.png'));assert ids==list(range(len(ids)));counts[arm]=len(ids)
 rows.append({'scene':s['name'],'source_frames':s['source_frames'],'counts':counts,'comparison_count':min(counts.values())})
state.update(state='stopped_regression',reason='用户明确要求效果差则记failure并回退；已有三scene视觉反例，停止后续相同条件推理。',completed_windows=len(state['completed']),actual_outputs=rows,stopped_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=sum(x['seconds'] for x in state['completed']))
dump(ROOT/'drive_state.json',state);dump(ROOT/'comparison_plan.json',{'scenes':rows,'note':'只比较实际存在的共同帧，不补齐、不插帧，不把停止当模型完成。'})
baseline=read(FULL/'registration.json')
config={'active_version':'v77-delete-full','active_task':'WS-V77-DELETE-FULL-20260927','active_run':'r1','asset_root':str(FULL),'review_local':'outputs/v77-delete-full/index.html','source_commit':'1a2d5189','entrypoints':{'build':'scripts/worldsim_v77/delete_full_drive.py + delete_full_omega.py','query':'scripts/worldsim_v77/delete_full_query.py'},'scenes':[{'name':s['name'],'actor':s['actor'],'count':s['count'],'yaw':s['yaw']} for s in baseline['scenes']],'driveeditor':baseline['driveeditor'],'model_family':'DriveEditor trained deletion; frozen VGGT-Omega; existing GLB','experimental_precise_mask_route_enabled':False,'experimental_run_retained':str(ROOT),'rollback_reason':'用户要求：新效果更差则记录failure并回退。细轮廓直接作为DriveEditor输入产生白车/车形灰白残留，停止该默认路线。','known_limits_retained':['0230 CAM5原版仍可能再生车辆','0255原版后段模糊/变形','Ω碎裂和GLB错误遮挡尚未解决'],'automatic_model_jobs':False,'human_verdict':None}
dest=REPO/'configs/worldsim_v77/delete_pipeline_current.json';dest.parent.mkdir(parents=True,exist_ok=True)
if dest.exists():shutil.copy2(dest,ROOT/'repo_backups/delete_pipeline_current.before.json')
dump(dest,config);dump(ROOT/'rollback.json',config)
diff=subprocess.check_output(['git','diff','1a2d5189','--','scripts/worldsim_v77/delete_full_common.py','scripts/worldsim_v77/delete_full_drive.py','scripts/worldsim_v77/delete_full_omega.py','scripts/worldsim_v77/delete_full_query.py'],cwd=REPO,text=True)
assert not diff,'旧默认实现被改过，须核对后回退'
dump(ROOT/'rollback_validation.json',{'baseline_entrypoints_unchanged_from':'1a2d5189','baseline_registration_exists':True,'baseline_assets_preserved':True,'default_config':str(dest),'precise_route_enabled':False,'actual_experiment_frames':rows})
print(rows)
