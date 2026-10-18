"""CPU fixture for report completion claims; never writes model receipts/media."""
from pathlib import Path
import json
from build_direct_report import report_phase,runtime_io_phase
import datetime
W=Path(__file__).parent
q=dict(state='PARALLEL_A_B_FULL49_COMPLETE_AWAITING_REPORT_GIT');a=dict(state='complete',step=30000);b=dict(status='complete',iteration=30000)
ar=dict(frames=49,parameter_updates=0,records=[dict(frame_id=f'0008{f:03d}',actor=i) for f in range(89,138) for i in [6,37,5,26,18]])
br=dict(frames=49,parameter_updates=0,records=[dict(frame_id=f'0008{f:03d}') for f in range(89,138)])
ready=['matched5','allvehicles'];cases=[]
def check(name,queue,actor,native,left,right,models,want):
 got=report_phase(queue,actor,native,left,right,models);assert got['complete']==want,(name,got);cases.append(dict(case=name,result=got['code'],complete=got['complete']))
check('active training',q,dict(state='running',step=18200),b,ar,br,ready,False)
check('both trained but no final receipt',q,a,b,{},br,ready,False)
check('missing final identity frame',q,a,b,dict(ar,records=ar['records'][:-1]),br,ready,False)
check('wrong B frame despite49 rows',q,a,b,ar,dict(br,records=br['records'][:-1]+[dict(frame_id='0008089')]),ready,False)
check('final media absent',q,a,b,ar,br,['gn11five'],False)
check('coordinator not complete',dict(state='SERIAL_ZERO_UPDATE_FULL49_EVALUATIONS'),a,b,ar,br,ready,False)
check('all real-contract fields present',q,a,b,ar,br,ready,True)
check('failed controller never claims complete',dict(state='FAILED_PARALLEL_CAMPAIGN_PRESERVED_NO_RESTART'),a,b,ar,br,ready,False)
actual=json.loads((W/'direct_plan.json').read_text(encoding='utf-8'))['report']['phase']
assert actual['code'] in ['attention','storage_wait','training','evaluation_delivery','coordinator_pending','results_ready']
assert actual['complete']==(actual['code']=='results_ready')
now=datetime.datetime.now(datetime.timezone.utc)
h=dict(status='BLOCKED_HOST_STORAGE_WRITE_WAIT',at=now.isoformat(),A_reported_status=dict(step=20400))
base=dict(code='training',complete=False)
assert runtime_io_phase(base,h,dict(step=20400,state='running'),now)['code']=='storage_wait'
assert runtime_io_phase(base,h,dict(step=20500,state='running'),now)==base
assert runtime_io_phase(dict(code='results_ready',complete=True),h,a,now)['complete']
r=dict(status='PASSED_REPORT_COMPLETION_GATE_CPU_FIXTURE',cases=cases,storage_wait_cases=3,actual_current_phase=actual,synthetic_fixture_only=True,actual_final_models_validated=False,GPU_used=False,training_updates=0)
(W/'report_phase_validation.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(status=r['status'],cases=len(cases),actual_phase=actual['code'],actual_final_models_validated=False)))
