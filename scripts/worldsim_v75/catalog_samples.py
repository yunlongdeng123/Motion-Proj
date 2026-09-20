"""按官方UUID冻结3个开发/3个保留样例，下载开发输入；不接触保留视频。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from huggingface_hub import hf_hub_download

RUN=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-COHORT-01/20260920-r1')
DEST=Path('/root/autodl-tmp/models/worldsim_v75/omni-dreams-samples')

def main():
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');a=p.parse_args()
    manifest=RUN/'samples_cohort_manifest.json'
    if manifest.exists():
        plan=json.loads(manifest.read_text())
    else:
        inventory=json.loads((RUN/'official_samples_inventory.json').read_text())[0]
        groups={}
        for row in inventory['files']:
            parts=row['path'].split('/')
            if len(parts)==4 and parts[:2]==['data','single_view'] and row['bytes'] is not None:
                groups.setdefault(parts[2],[]).append(row)
        chosen=sorted(groups)[:6]
        assert len(chosen)==6
        selected=[{'scene_uuid':uid,'role':'development' if i<3 else 'heldout_reserved',
                   'files':groups[uid]} for i,uid in enumerate(chosen)]
        plan={'task_id':'WS-V75-COHORT-01','run_id':'20260920-r1','repo_id':inventory['repo'],
              'revision':inventory['revision'],'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'available_scene_uuids':len(groups),'selection':'UUID lexicographic first3 development, next3 reserved; metadata only',
              'selected':selected,'model_outputs_viewed_for_selection':False,'heldout_video_viewed':False,
              'missing_policy':'记录排除和缺输入，不按模型表现替换场景',
              'scope':'原生RGB/hdmap视频基线；本仓库没有对应3D地图/actor轨迹，不能冒充完整场景输入',
              'human_verdict':None}
        manifest.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    items=[f for row in plan['selected'] if row['role']=='development' for f in row['files']]
    print(json.dumps({'frozen_scenes':len(plan['selected']),'development_bytes':sum(f['bytes'] for f in items)}),flush=True)
    if not a.download:
        return
    start=time.monotonic();done=[];failures=[]
    def download(item):
        path=hf_hub_download(plan['repo_id'],item['path'],repo_type='dataset',revision=plan['revision'],local_dir=str(DEST))
        assert Path(path).stat().st_size==item['bytes']
        return item
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(download,item):item for item in items}
        for future in as_completed(pending):
            try:
                done.append(future.result())
            except Exception as exc:
                failures.append({'path':pending[future]['path'],'error_type':type(exc).__name__,
                                 'http_status':getattr(getattr(exc,'response',None),'status_code',None)})
            result={'status':'downloading','completed_files':len(done),'total_files':len(items),
                    'completed_bytes':sum(x['bytes'] for x in done),'total_bytes':sum(x['bytes'] for x in items),
                    'elapsed_s':time.monotonic()-start,'failures':failures,
                    'completion_basis':'HF atomic completed return and repository byte count; not preallocation'}
            (RUN/'samples_download_result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
    result['status']='complete' if not failures else 'failed_stopped'
    (RUN/'samples_download_result.json').write_text(json.dumps(result,indent=2)+'\n')
    if failures:
        raise SystemExit(1)

if __name__=='__main__':
    main()
