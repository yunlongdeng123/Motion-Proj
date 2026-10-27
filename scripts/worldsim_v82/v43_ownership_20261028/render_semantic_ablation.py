"""PREPARED, NOT RUN. Exact RGB removal requires the native CUDA rasterizer.

No default invocation or queue registration. User requested stopping before GPU.
The explicit flag is an execution gate, not evidence of user approval by itself.
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from semantic_ablation_rules import classify_rows,removal_mask,MODES

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--candidates',required=True,type=Path);p.add_argument('--enable-gpu',action='store_true');a=p.parse_args()
    if not a.enable_gpu:raise SystemExit('GPU not enabled. Current user instruction: stop and report before GPU work.')
    # Delay every CUDA-aware import until the explicit execution boundary.
    from train_actor import Actor,NativeMotion,prepare_rasterization,helpers,torch,ROOT,OLD
    from v43_resource_control import require_start
    from gpu_slot import exclusive
    from diagnose_first_badcases import save
    require_start(a.context)
    ledger=json.loads(a.candidates.read_text());assert ledger['context']==a.context
    a.output.mkdir(parents=True,exist_ok=False)
    def status(**kw):
        d=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),training_updates=0,**kw)
        (a.output/'status.json').write_text(json.dumps(d,indent=2))
    status(phase='waiting_GPU',frames=0)
    with exclusive(),torch.no_grad():
        torch.set_num_threads(1);ds=ROOT/'datasets'/a.context;frames=json.loads((ds/'manifest.json').read_text())['frames'];actor=Actor(np.load(ds/'seeds.npz'))
        actor.load_state_dict(torch.load(str(ROOT/'actor_campaign_v1'/a.context/'formal10k/checkpoint010000.pth'),map_location='cuda')['actor'])
        original={k:v.clone() for k,v in actor.state_dict().items()};n=NativeMotion(OLD/'native_scene_training_v1'/a.context/'formal30k',30000);ns=len(n.static._xyz)
        requests=json.loads((OLD/'motion_edit_suite_v1'/a.context/'no_edit.json').read_text())['frames'];records=[]
        status(phase='rendering',frames=0)
        for row in requests:
            require_start(a.context)
            i=row['camera_index'];f=frames[i];raw=f['raw'];cam=n.camera(row,{})
            st=prepare_rasterization(cam,n.static,n.args);st['means2D']=torch.zeros_like(st['means3D']);dy=actor.kwargs(n,cam,f)
            # Use matching no-edit Actor10K evidence, never original dynamic row IDs.
            observed=np.load(ROOT/'boundary_semantics_v2'/a.context/('actor10k_%03d.npz'%raw))
            st['features']=st['features'][:,:2];dy['features']=dy['features'][:,:2];kw=helpers.merge_rows(st,dy)
            env=n.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            for radius in (0,5):
                labels=classify_rows(observed,len(kw['means3D']),ns,ledger['ids'],radius,.2)
                for mode in MODES+('delete_actor',):
                    remove=removal_mask(labels,mode) if mode!='delete_actor' else np.arange(len(labels))>=ns
                    query={**kw,'opacities':kw['opacities'].clone()};query['opacities'][torch.from_numpy(remove).to(query['opacities'].device)]=0
                    result=helpers.raster(cam,n.args,n.static.active_sh_degree,query,n.bg);rgb=result['rgb']+(1-result['alpha'])*env
                    save(a.output/(str(radius)+'px')/mode/('%03d.png'%raw),rgb)
                    np.savez_compressed(a.output/(str(radius)+'px')/mode/('%03d_maps.npz'%raw),alpha=result['alpha'][0].cpu().numpy().astype('float16'))
                    records.append(dict(raw=raw,radius=radius,mode=mode,suppressed_rows=int(remove.sum()),suppressed_static_rows=int(remove[:ns].sum()),rgb_exact_native=True))
            status(phase='rendering',frames=len(records)//10,last_raw=raw);print('ABLATION',raw,len(records),flush=True)
        assert n.unchanged() and all(torch.equal(v,original[k]) for k,v in actor.state_dict().items())
        (a.output/'summary.json').write_text(json.dumps(dict(records=records,all_vehicle_classes=list(range(14,20)),all_unknown_including_unobserved_rows=True,scope='global semantic-removal upper bound, not target-only deletion',checkpoint_unchanged=True,training_updates=0,rgb_ghost_fully_removed=None,human_verdict=None,quality_accepted=False),indent=2))
        status(phase='complete',frames=len(requests),images=len(records),cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())

if __name__=='__main__':main()
