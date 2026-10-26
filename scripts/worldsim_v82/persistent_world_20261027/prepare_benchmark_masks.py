"""Feed additional admitted input windows to the unchanged DEVA preprocessor.

Wait behind the first scene's immediate diagnostic round, never stop a healthy
training. Semantic contacts still need agent review before new native training.
"""
import json,os,subprocess,time,traceback
from pathlib import Path
from PIL import Image
from prepare_motion_masks import ROOT,BASE,SEG,PYTHON,digest
from native_scene_controller import dependency_result
from gpu_slot import exclusive
from resource_control import require_start,stop_requested

FIRST='14705303724557273004_3105_000_3125_000'
DEST=ROOT/'benchmark_mask_preparation_v1'

def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)

def main():
    require_start();DEST.mkdir(exist_ok=False)
    state=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_first_scene_diagnostics',child=None,completed=[],errors=[],native_training_started=[],new_training_updates=0)
    def update(**kw):state.update(kw);write(DEST/'status.json',state)
    update()
    while dependency_result(ROOT/'motion_suite_execution_v1'/FIRST/'status.json') is None:require_start();time.sleep(5)
    processed=set()
    while True:
        require_start();candidates=[]
        for summary in (ROOT/'benchmark_cases_v1/scenes').glob('*/summary.json'):
            d=json.loads(summary.read_text());context=d['context'];source=ROOT/'motion_scene_inputs_v1'/context
            if context in processed or (source/'mask/mask_0/preparation_receipt.json').exists():continue
            screen=json.loads((summary.parent/'input_quality.json').read_text());sensor=json.loads((source/'sensor_preparation.json').read_text())
            if screen['raw_input_status']!='raw_checks_passed' or not sensor.get('target_stats') or screen['target']['presence']!=49:continue
            candidates.append((d['split']=='heldout_reserved',context,source,sensor))
        candidates.sort()
        if not candidates:
            expansion=json.loads((ROOT/'benchmark_expansion_v1/status.json').read_text())
            if expansion['phase'] in ('inputs_and_cases_materialized','engineering_errors_preserved'):
                update(phase='available_masks_complete_target_review_and_training_pending');return
            update(phase='waiting_more_CPU_windows',child=None);time.sleep(10);continue
        _,context,source,sensor=candidates[0];processed.add(context);output=source/'mask/mask_0'
        try:
            assert not output.exists(),'Preserve partial mask attempt; engineering repair must use a new directory'
            command=[PYTHON,str(SEG/'demo/demo_automatic.py'),'--model',str(BASE/'weights/DEVA-propagation.pth'),'--SAM_CHECKPOINT_PATH',str(BASE/'weights/sam_vit_h_4b8939.pth'),'--SAM_ENCODER_VERSION','vit_h','--sam_variant','original','--chunk_size','4','--img_path',str(source/'image/image_0'),'--temporal_setting','semionline','--size','480','--output',str(output),'--use_short_id','--suppress_small_objects','--SAM_PRED_IOU_THRESHOLD','0.7']
            env=dict(os.environ,TORCH_HOME=str(BASE/'cache/torch'),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
            update(phase='waiting_GPU_for_segmentation',context=context,child=None)
            with exclusive():
                require_start();started=time.monotonic()
                with (DEST/(context+'.log')).open('x') as log:p=subprocess.Popen(command,cwd=str(SEG),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                update(phase='segmentation',child=dict(pid=p.pid,start_ticks=Path('/proc/%s/stat'%p.pid).read_text().split()[21]))
                while p.poll() is None:
                    if stop_requested():p.terminate();p.wait(timeout=45);update(phase='resource_stop_partial_masks_preserved');return
                    time.sleep(3)
                assert p.returncode==0,'Real segmentation failed; preserve output and log'
            frames=sorted((source/'image/image_0').glob('*.png'));masks=sorted((output/'Annotations').glob('*.png'));assert len(masks)==49 and [p.stem for p in frames]==[p.stem for p in masks]
            for rgb,mask in zip(frames,masks):
                with Image.open(rgb) as a,Image.open(mask) as b:assert a.size==b.size
            receipt=dict(status='automatic_instance_masks_prepared',scene=context,frames=49,input_frames=[p.stem for p in frames],seconds=time.monotonic()-started,command=command,execution_source_sha256=digest(SEG/'demo/demo_automatic.py'),sam_generator_sha256=digest(SEG/'deva/ext/SAM/automatic_mask_generator.py'),semantic_instance_binding_verified=False,new_reconstruction_training_updates=0,quality_accepted=False,human_verdict=None)
            write(output/'preparation_receipt.json',receipt)
            subprocess.run([PYTHON,str(ROOT/'code/prepare_native_scene.py'),'propose','--context',context],check=True)
            state['completed'].append(dict(context=context,stage='real_masks_and_target_contacts_ready',next='agent_visually_review_49_frames_then_build_actor_and_admit_native_smoke30K',reserved_output_is_not_a_method_score=sensor['split']=='heldout_reserved'))
            update(phase='target_visual_review_pending',child=None)
        except Exception:
            state['errors'].append(dict(context=context,error=traceback.format_exc()));update(phase='engineering_error_preserved',child=None)

if __name__=='__main__':main()
