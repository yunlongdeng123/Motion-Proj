"""有界输入队列；没有电源操作，分割完成后仅准备新原生输入。"""
from pathlib import Path
import json,subprocess,os,time
ROOT=Path(__file__).resolve().parent.parent
PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
CPU='/root/autodl-tmp/envs/drivingeditor/bin/python'
def run(name,cmd,env=None):
    with (ROOT/(name+'.log')).open('x') as f:
        p=subprocess.Popen(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,env=env)
        (ROOT/'gpu_input_status.json').write_text(json.dumps(dict(stage=name,pid=p.pid,automatic_shutdown=False,unix=time.time()),indent=2))
        if p.wait():raise RuntimeError(name+' failed')
def main():
    assert json.loads((ROOT/'pilot_qa.json').read_text())['status']=='PASS_AGENT_INPUT_QA'
    raws=json.loads((ROOT/'inputs/receipt.json').read_text())['build_raws'];s=ROOT/'source'
    run('sam_all', ['/root/autodl-tmp/envs/worldsim-v77-sam2/bin/python',s/'mask_pilot.py','--bank',ROOT/'inputs','--output',ROOT/'masks_all_candidates_r1','--raws',*raws,'--point-prompts','--run-gpu'])
    run('components_all',[CPU,s/'filter_components.py','--candidates',ROOT/'masks_all_candidates_r1','--output',ROOT/'masks_all_candidates_anchored_r2'])
    run('resolve_all',[CPU,s/'resolve_masks.py','--bank',ROOT/'inputs','--candidates',ROOT/'masks_all_candidates_anchored_r2','--output',ROOT/'masks_all_resolved_r2'])
    env={**os.environ,'PYTHONPATH':'/root/autodl-tmp/envs/dggt_segformer/overlay:/root/autodl-tmp/external/dggt_segformer'}
    run('semantics_all',[PY,s/'export_semantics.py','--bank',ROOT/'inputs','--output',ROOT/'semantics'],env)
    old=Path('/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val029_v44_20261108_r1')
    run('prepare_native',[CPU,s/'prepare_native.py','--root','/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val065_native10k_20261108_r1','--prep',ROOT,'--engine-source',old/'engine','--config-source',old/'config.json','--api-source',old/'api'])
    (ROOT/'gpu_input_status.json').write_text(json.dumps(dict(stage='NATIVE_INPUTS_READY_PENDING_SMOKE',automatic_shutdown=False,unix=time.time()),indent=2))
if __name__=='__main__':main()
