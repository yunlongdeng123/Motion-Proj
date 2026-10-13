from pathlib import Path
import json,datetime,shutil,zipfile
W=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-12/w');O=W/'outputs';R=W/'work/drivingeditor_resume/repo_active';T=W/'work/drivingeditor_resume/heartbeat_20261013_1056'
A=R/'docs/autoresearch/worldsim_v82/drivingeditor_followup_20261013/r1/final_followup_20261013'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
shutdown=load(O/'DrivingEditor_shutdown_receipt.json');paused=load(O/'DrivingEditor_automation_paused.json')
assert shutdown['signal_sent'] is True and shutdown['independent_platform_power_verified'] is False and paused['status']=='PAUSED'
for p in [O/'DrivingEditor_shutdown_receipt.json',O/'DrivingEditor_automation_paused.json',T/'saved_and_pushed_prerequisites.json',T/'prepare_remote_closeout.py']:
    shutil.copy2('\\\\?\\'+str(p.resolve()),'\\\\?\\'+str((A/p.name).resolve()))
p=R/'docs/v82/DRIVINGEDITOR_FOLLOWUP_20261013.json';d=load(p)
d['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
d['last_updated_singapore']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
d['state']='six_30k_runs_complete_reports_saved_pushed_shutdown_requested_heartbeat_paused'
d['automation']['status']='PAUSED'
d['final_closeout'].update(power_signal_sent=True,independent_platform_power_verified=False,
    shutdown_receipt='outputs/DrivingEditor_shutdown_receipt.json',heartbeat_status='PAUSED',original_remote_assets_preserved=True)
d['shutdown_contract']['request']=shutdown
d['shutdown_contract']['chosen_backend']='2026-10-13 12:03:48 SGT，fresh核对supervisord PID857/cmdline和零owned/GPU/无关任务后，实际日志+SIGTERM后端成功发送；跳过Trash删除，不执行wrapper。'
d['shutdown_contract']['verification']='远端os.kill成功返回并flush实际JSON acknowledgement到本地；SSH随后关闭。一次限时读取亦关闭，独立平台电源状态不可读；仅确认已发出关机请求。'
write(p,d)
s=R/'docs/RESEARCH_STATUS.md';text=s.read_text(encoding='utf-8')
text=text.replace('最后保存、推送与已授权关机收口中，当前尚未发出电源信号。','真实报告与离线包已保存，结果已提交推送；2026-10-13 12:03:48 SGT已发出已授权AutoDL关机请求，定时任务PAUSED。远端SIGTERM acknowledgement已捕获，独立平台电源状态不可读，不能由SSH关闭认证断电。')
text=text.replace('当前心跳待收口后暂停，其他旧自动化保持暂停。','当前`drivingeditor-30k`心跳已PAUSED，其他旧自动化保持暂停，不自动开机或重启研究。')
text=text.replace('平台不可读时仅报告“已发出关机请求”。','平台不可读时仅报告“已发出关机请求”。实际请求、fresh检查与自动化暂停凭据见最终证据目录及本聊天`outputs/DrivingEditor_shutdown_receipt.json`；远端canonical最后保存于关机前power_pending，本地/Git此终态为权威后续收口记录。')
s.write_text(text,encoding='utf-8')
dest=O/'DrivingEditor_final_reports.zip';names=['DrivingEditor_shutdown_receipt.json','DrivingEditor_automation_paused.json']
with zipfile.ZipFile(dest,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    assert all(n not in z.namelist() for n in names)
    for n in names:z.write(O/n,n)
with zipfile.ZipFile(dest) as z:
    for n in names:assert json.loads(z.read(n))==load(O/n)
    assert len(z.namelist())==len(set(z.namelist()))
    count=len(z.namelist())
receipt=load(O/'DrivingEditor_offline_package_receipt.json')
receipt.update(bytes=dest.stat().st_size,files=count,appended_closeout_entries=names,appended_entries_read_verified=True,
    original_report_entries_unchanged=True,independent_platform_power_verified=False)
write(O/'DrivingEditor_offline_package_receipt.json',receipt)
shutil.copy2('\\\\?\\'+str((O/'DrivingEditor_offline_package_receipt.json').resolve()),'\\\\?\\'+str((A/'DrivingEditor_offline_package_receipt.json').resolve()))
shutil.copy2('\\\\?\\'+str(Path(__file__).resolve()),'\\\\?\\'+str((A/'record_power_closeout.py').resolve()))
print(json.dumps({'state':d['state'],'signal_sent':True,'independent_platform_power_verified':False,'ZIP_files':count,'ZIP_bytes':dest.stat().st_size}))
