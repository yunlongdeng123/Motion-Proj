"""Finish the current mask child once; gate later masks behind dev diagnostics."""
import datetime,json,os,signal,subprocess,time
from pathlib import Path
from PIL import Image
from prepare_motion_masks import ROOT,PYTHON,SEG,digest
from native_scene_controller import same_process,ticks,write
from resource_control import stop_requested

def main():
    dest=ROOT/'mask_development_priority_boundary_v1';dest.mkdir(exist_ok=False)
    path=ROOT/'benchmark_mask_preparation_v1/status.json';state=json.loads(path.read_text())
    assert state['pid']==53383 and state['start_ticks']=='903655710' and same_process(state['pid'],state['start_ticks'])
    assert state['phase'] in ['segmentation','waiting_GPU_for_segmentation']
    os.kill(state['pid'],signal.SIGSTOP)
    for _ in range(200):
        if Path(f'/proc/{state["pid"]}/stat').read_text().split()[2]=='T':break
        time.sleep(.01)
    state=json.loads(path.read_text());child=state.get('child')
    receipt=dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),old=state,pid=os.getpid(),start_ticks=ticks(os.getpid()),healthy_GPU_child_restarted=False,training_updates_lost=0,phase='waiting_current_mask_child_boundary')
    write(dest/'receipt.json',receipt)
    try:
        if child:
            assert Path(f'/proc/{state["pid"]}/task/{state["pid"]}/children').read_text().split()==[str(child['pid'])]
            while True:
                fields=Path(f'/proc/{child["pid"]}/stat').read_text().split();assert fields[21]==child['start_ticks']
                if fields[2]=='Z':break
                if stop_requested():os.kill(child['pid'],signal.SIGTERM)
                time.sleep(3)
            exit_status=int(fields[51]);receipt['actual_child_exit_status']=exit_status
        else:
            assert state['phase']=='waiting_GPU_for_segmentation'
            assert not Path(f'/proc/{state["pid"]}/task/{state["pid"]}/children').read_text().split()
            exit_status=None
        os.kill(state['pid'],signal.SIGTERM);os.kill(state['pid'],signal.SIGCONT)
        for _ in range(200):
            if not same_process(state['pid'],state['start_ticks']):break
            time.sleep(.05)
        assert not same_process(state['pid'],state['start_ticks'])
        state['child']=None
        if stop_requested():
            state['phase']='resource_stop_partial_masks_preserved';write(path,state);return
        if child:
            assert exit_status==0,'Preserve failed mask output; do not claim complete'
            scene=ROOT/'motion_scene_inputs_v1'/state['context'];output=scene/'mask/mask_0'
            frames=sorted((scene/'image/image_0').glob('*.png'));masks=sorted((output/'Annotations').glob('*.png'));assert len(masks)==49 and [p.stem for p in frames]==[p.stem for p in masks]
            for a,b in zip(frames,masks):
                with Image.open(a) as aa,Image.open(b) as bb:assert aa.size==bb.size
            write(output/'preparation_receipt.json',dict(status='automatic_instance_masks_prepared',scene=scene.name,frames=49,actual_exit_status=0,input_frames=[x.stem for x in frames],healthy_child_finished_without_restart=True,semantic_instance_binding_verified=False,execution_source_sha256=digest(SEG/'demo/demo_automatic.py'),sam_generator_sha256=digest(SEG/'deva/ext/SAM/automatic_mask_generator.py'),quality_accepted=False,human_verdict=None))
            subprocess.run([PYTHON,str(ROOT/'code/prepare_native_scene.py'),'propose','--context',scene.name],check=True)
            state['completed'].append(dict(context=scene.name,stage='real_masks_and_target_contacts_ready',next='agent_input_review_only_until_method_freeze_if_heldout',reserved_output_is_not_a_method_score=True))
        state['phase']='paused_after_completed_mask_for_first_diagnostic_recovery';write(path,state)
        cmd=[PYTHON,str(ROOT/'code/prepare_benchmark_masks.py'),'--resume-boundary']
        for context in ['14705303724557273004_3105_000_3125_000','4295449061847708198_3769_000_3789_000']:cmd+=['--after-benchmark',context]
        with (dest/'new_controller.log').open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt.update(phase='completed_child_then_development_diagnostics_gate',new=dict(pid=p.pid,start_ticks=ticks(p.pid)),new_command=cmd);write(dest/'receipt.json',receipt);write(ROOT/'benchmark_mask_development_priority_dispatch.json',receipt)
    except Exception:
        if same_process(state['pid'],state['start_ticks']):os.kill(state['pid'],signal.SIGCONT)
        raise

if __name__=='__main__':main()
