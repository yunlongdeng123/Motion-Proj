import json, datetime
from pathlib import Path

here = Path(__file__).resolve().parent
path = here / 'task_manifest.json'
m = json.loads(path.read_text(encoding='utf-8-sig'))
m.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         gpu_owner='vgpu_component_diagnostic_solmax',
         gpu_owner_scope='isolated read-only inference; no training until root reviews continuation harness',
         gpu_status_path='component_diagnostic/supervisor_status.json',
         other_scene_queue='paused_by_user',
         known_findings={
             '30k_reset': 'Train scalars recover after late opacity reset; fixed-camera continuation verification pending.',
             'branch_separation': '20k static road already noisy; dynamic branch reconstructs road.',
             'road_coverage': 'Most audited car-footprint ground samples have other-frame observations supported by real LiDAR; not a dense counterfactual GT.',
             'depth_projection': 'Last-writer collision suspected; exact CPU scene impact pending.'})
path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
pause = {'status':'paused_by_user','updated_utc':m['updated_utc'],
         'reason':'User explicitly requested only vGPU paper-version bug debugging.',
         'gpu_training_authorized':False,'retain_all_existing_files':True,
         'resume_only_after_new_user_scope_or_completion_of_vgpu_debug':True}
(here/'other_scene_queue_pause.json').write_text(json.dumps(pause,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Scope and GPU ownership updated; wrote other-scene pause marker')
