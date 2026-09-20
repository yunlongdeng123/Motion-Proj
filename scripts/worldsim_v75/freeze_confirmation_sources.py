"""只按本地完整性与既有文档曝光记录冻结有限新来源，不读模型结果。"""
import json
import subprocess
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path('/root/autodl-tmp/data/av2/sensor/val')
P=Path('/root/autodl-tmp/motion_proj')
OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-CONFIRM-SOURCES-01/20260920-r1')
assert not OUT.exists()
complete=[d for d in sorted(ROOT.iterdir()) if d.is_dir() and (d/'annotations.feather').exists()
          and len(list((d/'sensors/cameras/ring_front_center').glob('*.jpg')))>=200]
rows=[];selected=[]
for directory in complete:
    # 只保存是否在文档中被提及及文件名；不读取以前的模型结果。
    found=subprocess.run(['rg','-l','-F',directory.name,str(P/'docs')],capture_output=True,text=True)
    assert found.returncode in [0,1]
    refs=[str(Path(x).relative_to(P)) for x in found.stdout.splitlines()]
    row={'log_id':directory.name,'prior_document_references':refs,'eligible_unmentioned':not refs}
    rows.append(row)
    if not refs:selected.append(directory.name)
    if len(selected)==4:break
OUT.mkdir(parents=True)
result={'task_id':'WS-V75-CONFIRM-SOURCES-01','run_id':'20260920-r1','frozen_utc':datetime.now(timezone.utc).isoformat(),
        'status':'frozen' if len(selected)==4 else 'insufficient_unmentioned_sources',
        'logs':selected,'exposure_audit':rows,'local_complete_log_count':len(complete),
        'selection':'first four lexicographic complete local AV2 val logs not mentioned in existing project docs',
        'boundary':'absence from project docs is not proof of no prior exposure; training overlap unknown; not a final independent test',
        'screen':'unchanged source criteria: start first RGB+0.5s, full projected visibility through2s, >=48x32px, depth5..60m, >=6 causal central target LiDAR points; first eligible target UUID, no replacement',
        'measurement':'inspect all five real frames before model; fixed detector plus the frozen RAFT observer, unchanged thresholds; real trajectory support must cover the window',
        'stop':'exactly these four logs; no replacement, threshold changes, start changes or expansion if none pass',
        'model_outputs_read':False,'model_calls':0,'human_verdict':None,'failure_ledger_delta':'none'}
(OUT/'protocol.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'logs':selected,'audited':len(rows),'local_complete_logs':len(complete)}))
