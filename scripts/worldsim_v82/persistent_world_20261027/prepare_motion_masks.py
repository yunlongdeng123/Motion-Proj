"""冻结既有DEVA/SAM预处理，三个新运动开发路段串行生成真实mask。

不会把自动instance ID当最终动态归属；此后仍须目标关联与资产QA。
"""
import os
import json
import hashlib
import subprocess
import time
import traceback
from pathlib import Path
from PIL import Image
from resource_control import require_start,stop_requested
from gpu_slot import exclusive

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
BASE=ROOT.parent.parent
SEG=BASE/'campaign/next_scene30k_20261012/segmentation_repository'
PYTHON='/root/autodl-tmp/envs/drivingeditor/bin/python'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for value in iter(lambda:stream.read(8*1024**2),b''):h.update(value)
    return h.hexdigest()


def main():
    state=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='preflight',completed=[])
    def save():
        p=ROOT/'motion_masks_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)
    save()
    try:
        for scene in sorted((ROOT/'motion_scene_inputs_v1').iterdir()):
            if not scene.is_dir():continue
            require_start();sensor=json.loads((scene/'sensor_preparation.json').read_text())
            images=sorted((scene/'image/image_0').glob('*.png'));assert len(images)==49
            output=scene/'mask/mask_0'
            assert not output.exists(),'保留已有分割现场，不覆盖/重复启动'
            command=[PYTHON,str(SEG/'demo/demo_automatic.py'),'--model',str(BASE/'weights/DEVA-propagation.pth'),
                     '--SAM_CHECKPOINT_PATH',str(BASE/'weights/sam_vit_h_4b8939.pth'),'--SAM_ENCODER_VERSION','vit_h',
                     '--sam_variant','original','--chunk_size','4','--img_path',str(scene/'image/image_0'),
                     '--temporal_setting','semionline','--size','480','--output',str(output),'--use_short_id',
                     '--suppress_small_objects','--SAM_PRED_IOU_THRESHOLD','0.7']
            for p in [BASE/'weights/DEVA-propagation.pth',BASE/'weights/sam_vit_h_4b8939.pth']:assert p.is_file()
            assert (BASE/'weights/sam_vit_h_4b8939.pth').stat().st_size==2564550879
            env=dict(os.environ,TORCH_HOME=str(BASE/'cache/torch'),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
            state.update(phase='segmentation',scene=scene.name,command=command);save()
            started=time.time()
            with exclusive():
                require_start()
                with (ROOT/('motion_mask_'+scene.name+'.log')).open('a') as log:
                    process=subprocess.Popen(command,cwd=str(SEG),env=env,stdout=log,stderr=subprocess.STDOUT)
                    state['child']=dict(pid=process.pid,start_ticks=Path('/proc/%d/stat'%process.pid).read_text().split()[21]);save()
                    while process.poll() is None:
                        if stop_requested():
                            process.terminate();process.wait(timeout=30);state.update(phase='resource_stop_partial_outputs_preserved');save();return
                        time.sleep(5)
                    assert process.returncode==0,'分割错误，保留输出和日志后工程修复'
            masks=sorted((output/'Annotations').glob('*.png'));assert [p.stem for p in masks]==[p.stem for p in images]
            mask_sizes=[]
            for a,b in zip(images,masks):
                with Image.open(a) as image,Image.open(b) as mask:
                    assert image.size==mask.size;mask_sizes.append(list(mask.size))
            receipt=dict(status='automatic_instance_masks_prepared',scene=scene.name,frames=49,
                         input_frames=[p.stem for p in images],command=command,seconds=time.time()-started,
                         precision='existing no-AMP recipe',execution_source_sha256=digest(SEG/'demo/demo_automatic.py'),
                         sam_generator_sha256=digest(SEG/'deva/ext/SAM/automatic_mask_generator.py'),
                         mask_sizes=mask_sizes,target_track=sensor['target_track'],
                         semantic_instance_binding_verified=False,asset_quality_accepted=False,
                         new_reconstruction_training_updates=0,quality_accepted=False,human_verdict=None)
            (output/'preparation_receipt.json').write_text(json.dumps(receipt,indent=2));state.pop('child',None)
            state['completed'].append(scene.name);save()
        state.update(phase='automatic_masks_complete_target_association_pending');save()
    except Exception:
        state.update(phase='engineering_repair_needed',error=traceback.format_exc());save();raise


if __name__=='__main__':main()
