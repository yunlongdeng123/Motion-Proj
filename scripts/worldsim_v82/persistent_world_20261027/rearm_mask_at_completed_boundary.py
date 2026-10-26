"""Finish the healthy GPU child once, then re-arm its parent's diagnostic gate.

The old parent had passed the first diagnostic's failed-attempt release. Keep
the GPU child running; prevent another long segmentation from overtaking the
repaired diagnostic. A stopped parent retains the child's actual exit status.
"""
import datetime,json,os,signal,subprocess,time
from pathlib import Path
from PIL import Image
from prepare_motion_masks import ROOT,SEG,PYTHON,digest
from native_scene_controller import same_process,ticks,write
from resource_control import stop_requested

def main():
    dest=ROOT/'mask_rearm_completed_boundary_v1';dest.mkdir(exist_ok=False)
    status_path=ROOT/'benchmark_mask_preparation_v1/status.json';state=json.loads(status_path.read_text());child=state['child']
    assert state['pid']==46606 and state['start_ticks']=='903469325' and state['phase']=='segmentation'
    assert same_process(state['pid'],state['start_ticks']) and same_process(child['pid'],child['start_ticks'])
    assert Path('/proc/%s/task/%s/children'%(state['pid'],state['pid'])).read_text().split()==[str(child['pid'])]
    receipt=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),start_ticks=ticks(os.getpid()),old=state,healthy_GPU_child_restarted=False,reason='repair first diagnostic before starting another segmentation',phase='wait_healthy_mask_child_boundary')
    write(dest/'receipt.json',receipt);os.kill(state['pid'],signal.SIGSTOP)
    for _ in range(100):
        if Path('/proc/%s/stat'%state['pid']).read_text().split()[2]=='T':break
        time.sleep(.01)
    assert Path('/proc/%s/stat'%state['pid']).read_text().split()[2]=='T'
    resource_stop=False
    while True:
        fields=Path('/proc/%s/stat'%child['pid']).read_text().split();assert fields[21]==child['start_ticks']
        if fields[2]=='Z':break
        if stop_requested() and not resource_stop:
            os.kill(child['pid'],signal.SIGTERM);resource_stop=True
        time.sleep(5)
    exit_status=int(fields[51]);receipt.update(actual_child_exit_status=exit_status,resource_stop=resource_stop)
    # Child has finished; the stopped parent cannot launch another GPU task.
    os.kill(state['pid'],signal.SIGTERM);os.kill(state['pid'],signal.SIGCONT)
    for _ in range(100):
        if not same_process(state['pid'],state['start_ticks']):break
        time.sleep(.05)
    assert not same_process(state['pid'],state['start_ticks'])
    state['child']=None
    if resource_stop:
        state['phase']='resource_stop_partial_masks_preserved';write(status_path,state);receipt['phase']='resource_stop';write(dest/'receipt.json',receipt);return
    assert exit_status==0,'GPU mask failed; preserve its actual exit and partial output for repair'
    scene=ROOT/'motion_scene_inputs_v1'/state['context'];output=scene/'mask/mask_0'
    images=sorted((scene/'image/image_0').glob('*.png'));masks=sorted((output/'Annotations').glob('*.png'));assert len(masks)==49 and [x.stem for x in images]==[x.stem for x in masks]
    for a,b in zip(images,masks):
        with Image.open(a) as aa,Image.open(b) as bb:assert aa.size==bb.size
    write(output/'preparation_receipt.json',dict(status='automatic_instance_masks_prepared',scene=scene.name,frames=49,input_frames=[x.stem for x in images],actual_exit_status=exit_status,execution_source_sha256=digest(SEG/'demo/demo_automatic.py'),sam_generator_sha256=digest(SEG/'deva/ext/SAM/automatic_mask_generator.py'),semantic_instance_binding_verified=False,new_reconstruction_training_updates=0,healthy_child_finished_without_restart=True,boundary_receipt=str(dest/'receipt.json'),quality_accepted=False,human_verdict=None))
    subprocess.run([PYTHON,str(ROOT/'code/prepare_native_scene.py'),'propose','--context',scene.name],check=True)
    state['completed'].append(dict(context=scene.name,stage='real_masks_and_target_contacts_ready',next='agent_visual_review_then_build_and_native_smoke30K'))
    state['phase']='paused_after_completed_mask_for_first_diagnostic_recovery';write(status_path,state)
    with (dest/'new_controller.log').open('x') as log:p=subprocess.Popen([PYTHON,str(ROOT/'code/prepare_benchmark_masks.py'),'--resume-boundary'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(phase='healthy_child_completed_parent_gate_rearmed',new=dict(pid=p.pid,start_ticks=ticks(p.pid)));write(dest/'receipt.json',receipt);write(ROOT/'benchmark_mask_rearmed_dispatch.json',receipt)

if __name__=='__main__':main()
