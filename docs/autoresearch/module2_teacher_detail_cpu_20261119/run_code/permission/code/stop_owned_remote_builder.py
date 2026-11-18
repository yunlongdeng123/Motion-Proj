"""仅停止本任务已核实的CPU builder，保留partial，不碰GPU/其他进程。"""
from pathlib import Path
import json,os,signal,time
Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
target=str(Q/'permission/code/build_average_plan.py');output=str(Q/'permission/avg_plan_gh31_r1')
matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=p.joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 values=[v.decode(errors='replace')for v in args]
 if len(values)>=4 and values[1:4]==[target,'--output',output]:matches.append((int(p.name),values))
if len(matches)!=1:raise RuntimeError('expected exactly one owned CPU builder: '+str(matches))
pid,args=matches[0];stat=Path('/proc',str(pid),'stat').read_text().split();hz=os.sysconf('SC_CLK_TCK')
progress=json.loads((Path(output)/'progress.json').read_text())
receipt=dict(stopped_pid=pid,verified_exact_argv=args,only_owned_CPU_builder=True,GPU_or_other_process_touched=False,
 reason='local identical 64row probe and samequadrature sparse compile passed; offload to localCPU',
 last_flushed_progress=progress,CPU_seconds=(float(stat[13])+float(stat[14]))/hz,
 wall_seconds=float(Path('/proc/uptime').read_text().split()[0])-float(stat[21])/hz,
 partial_files_preserved=True,partial_is_not_completed_plan=True,native_calls=0,optimizer_updates=0)
os.kill(pid,signal.SIGTERM)
(Path(output)/'offload_stop.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

