"""Frozen GPU audit: true Gaussian support and alpha*T in each supervision domain."""
import sys,time
from identity_common import *

def main():
    sys.path.insert(0,str(TASK/'infra'))
    from priority import require_gpu_slot
    slot,admission=require_gpu_slot(str(ROOT.parent/'gpu_slot.lock'))
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(ROOT/'shadow_rebuild_20261017_v3/implementation_r15_train_v3/shadow_rebuild_v3'))
    from shadow_rebuild.config import Config,LossWeights
    from shadow_rebuild.data import Dataset
    from shadow_rebuild.field import GaussianField
    from shadow_rebuild.train import seed_from_state
    from shadow_rebuild.raster import cuda_raster
    from render_identity_matrix import save_png
    out=TASK/'tristate/projection'
    if out.exists():raise FileExistsError(out)
    out.mkdir();start=time.monotonic()
    ck=R15/'paused_checkpoint_004000_20261017/checkpoint.pt';before=stat(ck)
    saved=torch.load(ck,map_location='cpu',weights_only=False)
    raw=saved['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate()
    data=Dataset(R15/'data','cuda',True);records=[]
    palette=torch.tensor([[56,165,240],[38,230,140],[255,182,39]],dtype=torch.float32,device='cuda')/255
    try:
        for stage in ['init','4k']:
            field=GaussianField(data.seed if stage=='init' else seed_from_state(saved['state']),cfg).cuda()
            if stage=='4k':
                state={k[6:]:v for k,v in saved['state'].items() if k.startswith('field.')}
                field.load_state_dict(state)
            field.requires_grad_(False)
            with torch.no_grad():
                for i in range(len(data.rows)):
                    frame=data.frame(i);fid=frame.frame_id;folder=out/stage/fid;folder.mkdir(parents=True)
                    g=field.evaluate(frame.poses,frame.valid,frame.camera.center)
                    rawstate=field.identity_evidence_status[g['rows']]
                    display=torch.where(rawstate==2,0,torch.where(rawstate==1,1,2))
                    one=torch.nn.functional.one_hot(display,3).float()
                    owners=(g['owner'][:,None]==torch.tensor(ACTORS,device='cuda')[None]).float()
                    full=cuda_raster(frame.camera,g['xyz'],g['cov'],g['opacity'],torch.cat((one,owners),-1),10)
                    color=full[...,:3]@palette
                    save_png(folder/'scene_state.png',color)
                    # Individual state alpha images keep other classes/owners as occluders.
                    for c,label in enumerate(['0','1','U']):save_png(folder/('scene_'+label+'.png'),full[...,c])
                    for j,actor in enumerate(ACTORS):
                        take=g['owner']==actor
                        isolated=cuda_raster(frame.camera,g['xyz'][take],g['cov'][take],g['opacity'][take],one[take],10)
                        save_png(folder/('a%02d_state.png'%actor),isolated@palette)
                        alpha=isolated.sum(-1);visible=full[...,3+j]
                        save_png(folder/('a%02d_alpha.png'%actor),alpha)
                        save_png(folder/('a%02d_visible.png'%actor),visible)
                        valid=frame.mask_valid[...,actor]&frame.mask_known[actor]
                        pos=frame.masks[...,actor]&valid
                        domains=[valid&~pos,pos,~valid]
                        v=[float(visible[m].sum()) for m in domains];s=[float(alpha[m].sum()) for m in domains]
                        records.append(dict(stage=stage,frame_id=fid,actor=actor,
                            visible_alpha_mass_by_pixel_domain_0_1_U=v,
                            isolated_alpha_mass_by_pixel_domain_0_1_U=s,
                            target_rows=int(take.sum()),target_raw_status_counts={str(k):int(((rawstate==k)&take).sum()) for k in [0,1,2]},
                            target_no_Gaussians=int(take.sum())==0))
                    if i%7==0:print(json.dumps(dict(stage=stage,frames=i+1)),flush=True)
            del field,g,full,isolated
            torch.cuda.empty_cache()
        assert before==stat(ck)
        write(out/'projection_receipt.json',dict(status='COMPLETE_REVIEW_PENDING',checkpoint=before,
            parameter_updates=0,shadows_evaluated=False,admission=admission,seconds=time.monotonic()-start,
            records=records,frames=49,stages=['init','4k'],
            semantics='Evidence labels rendered with actual covariance/opacity and original alpha*T occlusion; no recolouring of pixel masks.',
            limitations=['An isolated pixel outside a visible body mask may be occluded valid geometry.',
                'A static evidence label means static class, not definitely negative for a specific actor.',
                'Initial renders use original initial opacity and scales; not normalization to 4K alpha.'],human_verdict=None))
    except BaseException as e:
        write(out/'failure.json',dict(error=repr(e),assets_preserved=True));raise
    finally:slot.close()

if __name__=='__main__':main()
