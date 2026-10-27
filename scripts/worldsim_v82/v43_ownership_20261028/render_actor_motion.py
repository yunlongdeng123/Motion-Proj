"""Same physical object requests, rebuilt asset, unchanged static/env and backend."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from train_actor import Actor,render,NativeMotion,torch,cpu,prepare_rasterization,helpers
from segmentation import ROOT,OLD,write
from gpu_slot import exclusive
from v43_resource_control import require_start
def main():
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);a=p.parse_args();ctx=a.context;dest=ROOT/'actor_motion_v1'/ctx;dest.mkdir(parents=True,exist_ok=False)
    status=dict(pid=__import__('os').getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_GPU',context=ctx,frames=0,training_updates=0)
    def update(**kw):status.update(kw);write(dest/'status.json',status)
    update()
    with exclusive(),torch.no_grad():
        require_start();torch.set_num_threads(1);started=time.monotonic();dataset=ROOT/'datasets'/ctx;frames=json.loads((dataset/'manifest.json').read_text())['frames'];actor=Actor(np.load(dataset/'seeds.npz'));payload=torch.load(str(ROOT/'actor_campaign_v1'/ctx/'formal10k/checkpoint010000.pth'),map_location='cuda');actor.load_state_dict(payload['actor']);del payload
        n=NativeMotion(OLD/'native_scene_training_v1'/ctx/'formal30k',30000);cases=['no_edit','delete_target','translate_lateral_1m','turn_left15'];records=[];max_delete_difference=0.
        def save(p,t):p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray((np.clip(cpu(t).transpose(1,2,0),0,1)*255).round().astype('uint8')).save(p)
        for case in cases:
            spec=json.loads((OLD/'motion_edit_suite_v1'/ctx/(case+'.json')).read_text())
            for row in spec['frames']:
                require_start();i=row['camera_index'];f=frames[i];baseline=render(n,actor,f,i);raw=f['raw'];body=np.load(dataset/('%03d.npz'%raw))['body'];original_domain=binary_dilation(body,iterations=2)|(cpu(baseline['extra'])[0]>.05)
                if case=='delete_target':
                    result=n.render(row,dict(family='delete'));same=n.render(row,dict(family='delete'));max_delete_difference=max(max_delete_difference,float((same['rgb']-result['rgb']).abs().max()))
                else:result=baseline if case=='no_edit' else render(n,actor,f,i,edit=dict(rotation=row['target_rotation_delta_world'],center=row['target_requested_center_world']))
                mask=np.zeros((320,480),bool) if case=='no_edit' else original_domain&~(cpu(result['extra'])[0]>.05)
                out=dest/case;save(out/'native_rgb'/('%03d.png'%raw),result['rgb']);(out/'mask').mkdir(exist_ok=True);Image.fromarray(mask.astype('uint8')*255).save(out/'mask'/('%03d.png'%raw))
                records.append(dict(case=case,raw=raw,camera_index=i,model_time=f['model_time'],generation_pixels=int(mask.sum()),requested_rotation_world=row['target_rotation_delta_world'],requested_center_world=row['target_requested_center_world']));update(phase='rendering',frames=len(records),last_case=case,last_raw=raw)
            print('V43_MOTION',ctx,case,flush=True)
        assert n.unchanged();write(dest/'summary.json',dict(context=ctx,cases=cases,frames_per_case=31,records=records,checkpoint=str(ROOT/'actor_campaign_v1'/ctx/'formal10k/checkpoint010000.pth'),source_checkpoint_unchanged=True,static_and_environment_unchanged=True,delete_replay_max=max_delete_difference,backend_executed_here=False,new_training_updates=0,physical_requests_reused_from_v42=True,not_new_formal_benchmark_cases=True,quality_accepted=False,human_verdict=None));update(phase='native_complete',wall_seconds=time.monotonic()-started,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
if __name__=='__main__':main()
