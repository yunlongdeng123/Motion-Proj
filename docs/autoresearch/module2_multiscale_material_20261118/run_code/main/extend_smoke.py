from pathlib import Path
import json,time,shutil
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
p=P/'lifecycle_contract.json';d=json.loads(p.read_text())
shutil.copy2(p,p.with_suffix('.json.before_real10min'))
d['runtime_measurement']=dict(source='root_native_field_probe_r1/receipt.json',
 actual_shared_native_forward_adjoint_seconds=[2.298695635050535,1.8828298971056938],
 physical1400_core_seconds=2927.0678725093603,interpretation='core49min plus evaluation/persistence is approximately1hour; require real10min smoke')
d['real10min_smoke']=dict(authorized=True,min_active_experiment_seconds=600,max_engineering_updates=360,
 lifecycle='exerciseA/B/C, actual checkpoint/evaluation/full model+optimizer+RNG/sampler restore and next-update/stage entry',
 no_idle_padding=True,counts='initial12-call/update engineering allowance and all duplicate next-update witnesses separately counted; no formal updates consumed',
 initialization='fresh frozen field for smoke; never reuse smoke learned values to initialize formal',
 formal_budget_unchanged=1400,method_budgets_each=800)
d['duplicate_probes']=dict(root_probe='root_native_field_probe_r1',root_native_calls=2,root_optimizer_updates=0,
 agent_probe='shared_gradient_r1 if completed; actual calls recorded independently',
 reason='coordination messages crossed native launch boundaries; preserve both, do not count as independent method evidence')
d['engineering_duration_extension_UTC']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
lease=json.loads((P/'gpu_lease.json').read_text());assert lease['owner']=='road_asset'
lease['scope']='registered about10min shared-field lifecycle smoke, <=360 engineering updates plus initial allowance; formal remains locked'
(P/'gpu_lease.json').write_text(json.dumps(lease,indent=2)+'\n')
print('Real10min lifecycle authorized; formal1400 unchanged and locked')
