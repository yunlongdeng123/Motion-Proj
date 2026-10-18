"""Reuse the already validated full49 hard-owner renderer with explicit new paths."""
from pathlib import Path
W=Path(__file__).parent
s=(W.parent/'identity_completion/bundle/render_identity_matrix.py').read_text(encoding='utf-8')
s=s[s.index('def run_r15('):s.index('\ndef main():')]
s=s.replace("    sys.path.insert(0,str(ROOT/'shadow_rebuild_20261017_v3/implementation_r15_train_v3/shadow_rebuild_v3'))\n",'')
s=s.replace('from packed_shadow import install,factors_for_geometry','from shadow_rebuild.factor_shadow import factors_for_geometry')
s=s.replace("ck=R15/'paused_checkpoint_004000_20261017/checkpoint.pt'","ck=Q/'train/checkpoint_004000.pt'")
s=s.replace("    install(model,dtype=torch.float64,block_size=256,pair_batch=256,recompute=False)\n",'')
s=s.replace("    assert cfg.sun_samples==5", "    assert cfg.sun_samples==5 and cfg.shadow_integrator=='factor_fp64' and saved['global_step']==4000")
s=s.replace("'validated FP64 factor adapter; original weights unchanged'","'integrated validated FP64 factor kernel; freshly trained repair4K'")
s=s.replace("                    isolated_points=int(take.sum()),isolated_render_mask_multiplied=False))", """                    isolated_points=int(take.sum()),isolated_render_mask_multiplied=False,
                    known_body_pixels=int((frame.masks[...,actor]&frame.mask_valid[...,actor]).sum()),
                    known_negative_pixels=int((~frame.masks[...,actor]&frame.mask_valid[...,actor]).sum()),
                    visible_alpha_in_body=float(p['visible_labels'][...,actor][frame.masks[...,actor]&frame.mask_valid[...,actor]].sum()),
                    visible_alpha_in_known_negative=float(p['visible_labels'][...,actor][~frame.masks[...,actor]&frame.mask_valid[...,actor]].sum()),
                    rgb_mae_in_body=float((full_on-frame.rgb).abs()[frame.masks[...,actor]].mean()) if frame.masks[...,actor].any() else None))""")
header='''"""Frozen repaired4K true-owner 49-frame independent/delete evaluation."""
from pathlib import Path
import sys,time,json
Q=Path(__file__).resolve().parent
sys.path.insert(0,str(Q/'implementation'))
TASK=Q.parent.parent.parent/'identity_completion_20261018'
sys.path.insert(0,str(TASK/'code'))
from identity_common import *
from render_identity_matrix import save_png
'''
tail='''
def main():
    sys.path.insert(0,str(TASK/'infra'))
    from priority import require_gpu_slot
    slot,admission=require_gpu_slot(str(ROOT.parent/'gpu_slot.lock'))
    import torch
    torch.set_num_threads(2);out=Q/'renders_repair4k';out.mkdir(exist_ok=False)
    try:
        start=time.monotonic();r=run_r15(None,out)
        r.update(status='COMPLETE_REVIEW_PENDING',model='r15_repair4k',frames=49,parameter_updates=0,
            checkpoint_file_unchanged=True,seconds=time.monotonic()-start,admission=admission,
            human_verdict=None,paired_edited_GT_available=False)
        write(out/'render_receipt.json',r)
        saved=torch.load(Q/'train/checkpoint_004000.pt',map_location='cpu',weights_only=False)
        old=torch.load(R15/'paused_checkpoint_004000_20261017/checkpoint.pt',map_location='cpu',weights_only=False)
        from shadow_rebuild.data import Dataset
        data=Dataset(R15/'data','cpu',True)
        half=torch.tensor(read(R15/'box_half_sizes.json')['half_sizes_m']);stats={}
        for label,state in [('original4k',old['state']),('repair4k',saved['state'])]:
            records=[]
            for a in ACTORS:
                take=state['field.owner']==a;x=state['field.xyz'][take];status=state['field.identity_evidence_status'][take]
                seed=data.seed['xyz'][data.seed['owner']==a]
                records.append(dict(actor=a,initial_points=len(seed),points=len(x),growth=len(x)/len(seed),
                    outside_exact_box=int((x.abs()>half[a]).any(-1).sum()),
                    outside_padded_box=int((x.abs()>half[a]+.1).any(-1).sum()),
                    raw_status_counts={str(k):int((status==k).sum()) for k in [0,1,2]},
                    maximum_sigma_quantile=torch.quantile(state['field.log_scales'][take].exp().max(-1).values,torch.tensor([.5,.95,.99,1.])).tolist()))
            stats[label]=records
        write(Q/'geometry_comparison.json',stats)
        del saved,old,data
        import gc
        gc.collect();torch.cuda.empty_cache()
        from render_branches import execute
        write(TASK/'branch_queue_status.json',dict(state='RUNNING',phase='repair4k_branches'))
        execute('repair4k',already_locked=True)
    except BaseException as e:
        write(out/'failure.json',dict(error=repr(e),assets_preserved=True));raise
    finally:slot.close()

if __name__=='__main__':
    import subprocess
    for name in ['old5','r15']:
        write(TASK/'branch_queue_status.json',dict(state='RUNNING',phase=name+'_branches'))
        subprocess.run([sys.executable,str(TASK/'code/render_branches.py'),'--model',name],check=True)
    write(TASK/'branch_queue_status.json',dict(state='RUNNING',phase='repair4k_identity_matrix'))
    main()
    write(TASK/'branch_queue_status.json',dict(state='ALL_BRANCHES_COMPLETE_AWAITING_PLAYER_DELIVERY',models=['old5','r15','repair4k']))
'''
(W/'evaluate_repair.py').write_text(header+s+tail,encoding='utf-8')
