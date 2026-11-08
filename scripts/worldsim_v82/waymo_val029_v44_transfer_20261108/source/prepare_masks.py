from pathlib import Path
import json,subprocess,sys,time
ROOT=Path('/root/autodl-tmp/codex_tmp/waymo_val029_cpu_20261108')
BANK=ROOT/'inputs'
def run(name,command):
    (ROOT/'mask_preparation_status.json').write_text(json.dumps(dict(stage=name,started=time.time(),command=command)))
    with (ROOT/(name+'.log')).open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
def main():
    build=json.loads((BANK/'receipt.json').read_text())['build_raws']
    run('sam2_all_build',['/root/autodl-tmp/envs/worldsim-v77-sam2/bin/python',str(ROOT/'source/mask_pilot.py'),'--bank',str(BANK),'--output',str(ROOT/'masks_all_candidates_r2'),'--raws',*map(str,build),'--point-prompts','--run-gpu'])
    run('resolve_all_build',['/root/autodl-tmp/envs/dggt_data_cpu/bin/python',str(ROOT/'source/resolve_masks.py'),'--bank',str(BANK),'--candidates',str(ROOT/'masks_all_candidates_r2'),'--output',str(ROOT/'masks_all_resolved_r2')])
    run('sky_all_build',['/root/autodl-tmp/envs/dggt_segformer/bin/python','/root/autodl-tmp/drivingeditor_repro/scripts/extract_sky.py','--scene',str(BANK),'--config','/root/autodl-tmp/external/dggt_segformer/local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py','--checkpoint','/root/autodl-tmp/drivingeditor_repro/weights/segformer.b5.1024x1024.city.160k.pth','--device','cuda:0'])
    (ROOT/'mask_preparation_status.json').write_text(json.dumps(dict(stage='COMPLETE_MASKS_AND_SKY',time=time.time())))
if __name__=='__main__':main()
