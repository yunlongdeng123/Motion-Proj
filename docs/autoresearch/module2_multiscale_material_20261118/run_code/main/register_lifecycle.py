from pathlib import Path
import json

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
p=P/'lifecycle_contract.json'
if p.exists():raise FileExistsError('保留登记')
c=dict(schema='module2_shared_material_native_lifecycle.v1',task_id='MODULE2-MULTISCALE-MATERIAL-20261118',
 status='CONDITIONAL_ENGINEERING_ONLY',formal_launch_authorized=False,
 prerequisites=['GH15 actual-field and chart qualification passed; preserve failed GH7/affine evidence',
   'real source1 H mask and readonly fallback/eligible scope frozen before smoke',
   'execution_contract final sources/optimizer/sampler/regularizer definitions bound'],
 first_actions='zero-update current exact-center/filteredA0 five-view native baselines; actual shared L/H grid gradient probes through current GS raster',
 updates=dict(initial_engineering_limit=12,formal_updates=0,
   allocation='A/B/C stage and independent exact-restored next-update witnesses; actual per-stage calls counted even duplicates; keep all smoke output separate'),
 lifecycle=['save/evaluate/checkpoint','complete field/optimizers/sampler/RNG restoration','next step verify including stageB/phaseC entry',
   'serialize shared field and reconstruct/requery from geometry/chart before native/public reload, never ci-only persistence',
   'record time,peak memory,native forward/backward counts,finite losses/gradients,L/H update scope and frozen others'],
 gate='five-frame alpha equals parent, K77266 joint floatRGB0, no illegal independent color parameter; no formal if mechanical failure',
 runtime='estimate physical1400 formal updates + evaluations from actual timings; if>=1hour, first separately counted real about10min lifecycle smoke before formal; do not change formal steps/lr/method',
 GPU_keep_on=True,production_promoted=False,human_verdict=None)
p.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n')
print(str(p))
