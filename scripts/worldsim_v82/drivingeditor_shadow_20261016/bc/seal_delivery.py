import datetime,hashlib,json,re,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'outputs/shadow_bc_experiments' if HERE.name!='scripts' else HERE.parent
if HERE.name!='scripts':
    for name in ['seal_delivery.py','validate_results.py','finalize_report.py']:
        shutil.copy2(HERE/name,OUT/'scripts'/name)
time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
monitor={'automation_id':'b-c-15','status':'PAUSED','cadence_minutes_while_active':15,'reason':'All bounded experiments and final validation completed; owned workers exited.',
    'completed_client_time':time,'GPU_power_state_changed_by_agent':False,'new_training_queued':False,
    'recommended_variant':'deferred_R4','rejected_variants':['union_R4','receiver_only_rollback']}
(OUT/'evidence/monitor_completion.json').write_text(json.dumps(monitor,ensure_ascii=False,indent=2),encoding='utf-8')
report=OUT/'report.html';text=report.read_text(encoding='utf-8')
text=text.replace('已建立当前对话每15分钟检查：自动核验真实status、owned PID、短日志、图像与回执；状态无实质变化保持安静，完成或失败时反馈。该检查依赖本机与桌面应用保持运行；GPU实验本身由AutoDL独立进程执行。',
    '本轮已安排15分钟进度检查，并结合实际日志和图像做了自主核验。全部训练进程退出、报告验收完成后，该检查已暂停；没有排更多训练，也没有执行关GPU操作。')
text=text.replace('</main></html>','<p><a href="evidence/monitor_completion.json">本轮进度检查完成记录</a></p></main></html>')
report.write_text(text,encoding='utf-8')
refs=[x for x in re.findall(r'(?:src|href)="([^"]+)"',text) if not x.startswith(('http:','https:','#'))]
assert all((OUT/x).is_file() for x in refs)
validation=json.loads((OUT/'evidence/final_validation.json').read_text());validation['HTML_reference_count']=len(refs);validation['monitor_status']='PAUSED_AFTER_COMPLETION'
(OUT/'evidence/final_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
files=[]
for path in sorted(OUT.rglob('*')):
    if path.is_file() and path.name!='manifest.json':files.append({'path':path.relative_to(OUT).as_posix(),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(OUT/'manifest.json').write_text(json.dumps({'sealed_client_time':time,'files':files,'validation':validation},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'files':len(files),'html_references':len(refs),'recommended':'deferred_R4','quality':validation['quality_status'],'monitor':'PAUSED'},ensure_ascii=False))
