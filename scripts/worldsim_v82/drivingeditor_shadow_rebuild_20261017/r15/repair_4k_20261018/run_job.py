"""Finite R15 repaired experiment. Requires the existing shared GPU slot."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
Q=Path(__file__).resolve().parent
R=Q.parent.parent;D=Q.parent;ROOT=R.parent
TASK=ROOT/'identity_completion_20261018'
sys.path.insert(0,str(Q/'implementation'));sys.path.insert(0,str(TASK/'infra'))

def main():
    p=argparse.ArgumentParser();p.add_argument('job',choices=['preflight','smoke','train']);a=p.parse_args()
    import torch
    from shadow_rebuild.config import Config
    from shadow_rebuild.data import Dataset
    from shadow_rebuild.train import run,preflight,write,seed_from_state
    torch.set_num_threads(2);cfg=Config.load(Q/'config.json');data=Dataset(D/'data','cpu',True)
    receipt=preflight(data,cfg)
    if a.job=='preflight':
        write(Q/'input_preflight.json',receipt);print(json.dumps(receipt));return
    from priority import require_gpu_slot
    slot,admission=require_gpu_slot(str(ROOT.parent/'gpu_slot.lock'))
    try:
        if a.job=='train':
            if json.loads((Q/'smoke_receipt.json').read_text())['status']!='PASSED':raise ValueError('GPU smoke not passed')
            if (Q/'train').exists():raise FileExistsError('No implicit resume or overwrite')
        data.device='cuda';start=time.monotonic()
        output=Q/('smoke' if a.job=='smoke' else 'train')
        model,result=run(data,cfg,output,'cuda','cuda',max_updates=2 if a.job=='smoke' else 4000)
        if a.job=='smoke':
            from shadow_rebuild.factor_shadow import transmittance
            from shadow_rebuild.math3d import disk_directions
            from shadow_rebuild.repair_geometry import geometry_receipt
            from shadow_rebuild.evidence_audit import run_evidence_audit
            from shadow_rebuild.model import SceneModel
            frame=data.frame(data.train_indices[0])
            with torch.no_grad():
                g,p=model.buffers(frame,len(data.actor_ids),'cuda',isolated=False)
                rows=(p['road_alpha'].reshape(-1)>1e-5).nonzero().flatten()[::512][:256]
                pos=p['road_position'].reshape(-1,3)[rows];normal=p['road_normal'].reshape(-1,3)[rows]
                origins=pos+normal*cfg.ray_bias_m;dirs=model.light.direction().expand_as(origins)
                values=[transmittance(origins,dirs,g.shadow_factors,block_size=b,recompute=False) for b in [128,256,512]]
                error=max(float((v-values[0]).abs().max()) for v in values[1:]) if len(rows) else 0.
                assert len(rows)>0 and error<=1e-6,(len(rows),error)
            del g,p,values
            saved=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
            restored=SceneModel(seed_from_state(saved['state']),cfg)
            restored.load_state_dict(saved['state'],strict=True)
            for key,value in model.state_dict().items():assert torch.equal(value.cpu(),restored.state_dict()[key]),key
            del restored,saved
            # Exercise the REAL complete current-evidence path without adopting
            # the smoke model, its optimizer, or any of its decisions for training.
            audit=run_evidence_audit(model,data,model.optimizer('geometry'),2,Q/'smoke_audit')
            geo=geometry_receipt(model.field)
            assert all(v['outside_allowed_envelope']==0 and v['drift_violations']==0 for v in geo)
            result=dict(status='PASSED',training_smoke_updates=2,actual_source='same raw R15 seed and original masks',
                real_frame_shadow_chunk_max_abs=error,real_evidence_audit_frames=audit['frames'],strict_checkpoint_restore=True,
                geometry=geo,seconds=time.monotonic()-start,admission=admission,smoke_not_adopted=True)
        else:
            progress=json.loads((output/'progress.json').read_text());assert progress['global_step']==4000
            audits=[json.loads((output/'ownership'/f'{s:06d}'/'receipt.json').read_text()) for s in [1500,2500,3500]]
            assert all(v['status']=='COMPLETE' for v in audits)
            result.update(status='TRAIN_4K_COMPLETE_REQUIRES_EVALUATION',seconds=time.monotonic()-start,
                admission=admission,actual_owner_changes=[v['actual_owner_changes'] for v in audits])
        write(Q/(a.job+'_receipt.json'),result);print(json.dumps(result),flush=True)
    finally:slot.close()

if __name__=='__main__':main()
