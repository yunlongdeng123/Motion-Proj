"""Close the monitor and seal already-validated deliverables; no extra GPU work."""
from pathlib import Path
import json,datetime,hashlib,shutil,os,tomllib,ast
BASE=Path(__file__).resolve().parents[2];WORK=BASE/'work/shadow_sequence_v2';OUT=BASE/'outputs/shadow_sequence_v2'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
validation=read(OUT/'evidence/final_validation.json');assert validation['evidence_status']=='PASS'
config=tomllib.loads((Path(os.environ['USERPROFILE'])/'.codex/automations/b-c-15/automation.toml').read_text(encoding='utf-8'))
assert config['status']=='PAUSED'
process=read(OUT/'evidence/completion_process_check.json');assert not any(x['alive'] for x in process['jobs'])
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
receipt={'completed_utc':now,'experiment_work_complete':True,'overall_problem_fully_solved':False,
 'actual_gradient_updates':18000,'selected_training':'unselected_clean_v1','selected_evaluation':'final49_unselected_v1',
 'automation_id':'b-c-15','automation_status':config['status'],'owned_GPU_workers_all_exited':True,
 'GPU_left_powered_on':True,'no_subagents_used':True,'original_checkpoint_not_overwritten':True,
 'full49_diagnostic_complete':True,'final_full49_video_streams':13,'additional_fixed_raw101_edit_streams':2,
 'remaining':['outside-authority raw106–108 edge shard','large-yaw asset geometry defects','full vehicle asset purity uncertified','no independent clean/shadow or paired edit GT']}
write(OUT/'evidence/completion.json',receipt)
state=read(WORK/'state.json');state.update(complete=True,phase='complete',completed_training_updates=18000,
 experiment_work_complete=True,overall_problem_fully_solved=False,monitor_status='PAUSED',completed_utc=now,final_report=str(OUT/'report.html'))
write(WORK/'state.json',state)
(WORK/'NOTES.md').write_text('Completed. Do not restart experiments. User forbids subagents.\n\nSelected unselected_clean_v1 / final49_unselected_v1, actual cumulative18000 gradient updates. All owned GPU workers exited; automation b-c-15 PAUSED; GPU remains on. Original checkpoint preserved. Final report outputs/shadow_sequence_v2/report.html. Full problem partially solved: boundary shard raw106–108 outside authority; large-yaw original vehicle asset fails; asset purity and new views not certified. Full evidence and all49 videos validated. Read selected.json, completion_scope.json and evidence/completion.json for details.\n',encoding='utf-8')
for p in WORK.glob('*.py'):
 ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));shutil.copy2(p,OUT/'scripts'/p.name)
write(OUT/'manifest.json',{'created_utc':now,'files':{p.relative_to(OUT).as_posix():{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}})
all_files=list(OUT.rglob('*'));print(json.dumps({'completion':receipt,'video_files':len(list(OUT.rglob('*.mp4'))),'delivery_files':sum(p.is_file() for p in all_files)},ensure_ascii=False))
