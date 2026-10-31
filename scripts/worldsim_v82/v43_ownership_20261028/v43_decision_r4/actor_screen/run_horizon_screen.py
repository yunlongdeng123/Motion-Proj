"""两路CPU actor扫描，跟随已完成原片段；每场完成立即交付清单。"""
import concurrent.futures as futures
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/codex_tmp/v43_r4_actor_screen_20261031')
DEST=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/v43_r4_actor_screen_20261031/horizonforge')
RAW=Path('/root/autodl-tmp/datasets/waymo_v1_4_3/individual_files/validation')

def run(row):
    number=row['scene_number'];source=RAW/row['tfrecord_name'];deadline=time.monotonic()+1800
    while not source.exists() or source.stat().st_size!=row['full_raw_bytes']:
        if time.monotonic()>deadline:return dict(scene_number=number,status='raw_unavailable_before_deadline')
        time.sleep(3)
    output=DEST/('scene_'+number)
    with (ROOT/('screen_'+number+'.log')).open('w') as stream:
        result=subprocess.run([sys.executable,str(ROOT/'scan_waymo_actors.py'),'--raw',str(source),'--context',row['context_id'],'--scene-number',number,'--output',str(output)],stdout=stream,stderr=stream)
    answer=dict(scene_number=number,exit_code=result.returncode,output=str(output))
    if (output/'summary.json').exists():answer['summary']=json.loads((output/'summary.json').read_text())
    print(json.dumps(answer,ensure_ascii=False),flush=True)
    return answer

if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'horizon_source_registry.json').read_text(encoding='utf-8-sig'))['rows']
    rows.sort(key=lambda r:(not (RAW/r['tfrecord_name']).exists(),r['scene_number']))
    with futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,rows))
    (DEST/'batch_summary.json').write_text(json.dumps(dict(schema='horizonforge_all11_actor_screen_batch_v1',results=results),ensure_ascii=False,indent=2)+'\n')
