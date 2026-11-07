"""CPU复核固定400步参数/Adam/导出，以及保存float的全图geometry不变量。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random

os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    formal=ROOT/'E_formal400_r1';returned=json.loads((formal/'returned.json').read_text())
    assert returned['status']=='COMPLETE_FIXED_400_PAIRED_COLOUR_RESIDUAL_BAKE_PENDING_GATES'
    assert returned['actual_optimizer_updates']==returned['final_step']==400
    assert returned['actual_native_forwards']==808 and returned['actual_native_backwards']==800
    assert returned['original_Gaussian_SH_geometry_opacity_identity_updates']==0
    expected=[];generator=random.Random(20261107)
    for epoch in range(100):
        raws=[94,105,128,137];generator.shuffle(raws);expected.extend(raws)
    checks=[];final=None
    for step in [100,200,400]:
        path=formal/f'state{step}.pth';state=torch.load(str(path),map_location='cpu',weights_only=False)
        d=state['delta'];assert d.shape==(34376,3) and d.dtype==torch.float32 and torch.isfinite(d).all() and float(d.abs().max())<=1.
        assert state['step']==step and len(state['history'])==step
        assert [x['raw'] for x in state['history']]==expected[:step]
        opt=state['optimizer'];assert len(opt['param_groups'])==1 and len(opt['state'])==1
        group=opt['param_groups'][0];assert group['params']==[0] and group['lr']==.01 and group['betas']==(.9,.999) and group['eps']==1e-8
        moments=opt['state'][0];assert int(moments['step'])==step
        for key in ['exp_avg','exp_avg_sq']:
            assert moments[key].shape==d.shape and torch.isfinite(moments[key]).all()
        assert (moments['exp_avg_sq']>=0).all()
        checks.append(dict(step=step,path=str(path),sha256=sha(path),delta_max_abs=float(d.abs().max()),
            saturated_scalar_count=int((d.abs()==1).sum()),delta_L1_sum=float(d.abs().double().sum()),
            optimizer_only_private_delta=True,sampler_history_exact_registered_seed=True))
        if step==400:final=state
    material_path=formal/'evaluation_material/appearance_manifest.json'
    material=json.loads(material_path.read_text());assert material['paired_optimizer_steps']==400 and material['parameter_bounds']==[-1.,1.]
    assert material['model_binding']==final['config']['model_binding']
    assert material['fit_contract_sha256']==returned['fit_contract_sha256']
    payload=material_path.parent/material['material_file'];assert sha(payload)==material['material_sha256']
    qual_path=ROOT/'E4_appearance_qualification/qualification_manifest.json';qual=json.loads(qual_path.read_text())
    assert material['qualification_manifest_sha256']==sha(qual_path)
    with np.load(payload,allow_pickle=False) as z,np.load(qual_path.parent/qual['selection_file'],allow_pickle=False) as q:
        assert np.array_equal(z['static_row_ids'],q['static_safe_row_ids']) and np.array_equal(z['road_row_ids'],q['road_safe_row_ids'])
        delta=np.concatenate((z['static_rgb_delta'],z['road_rgb_delta']))
        assert delta.dtype==np.float32 and np.array_equal(delta,final['delta'].numpy())
        assert np.max(np.abs(delta))<=1
    fields=['alpha','depth_sum','depth_m','group_alpha_T','added_owner_rgb_T'];frames=[]
    for raw in [94,105,128,137]:
        for state in ['keep','delete']:
            out=formal/f'evaluation/frames/{raw:03d}/{state}_float.npz';base=ROOT/f'baseline4/frames/{raw:03d}/{state}_float.npz'
            with np.load(out,allow_pickle=False) as z,np.load(base,allow_pickle=False) as b:
                assert z['rgb'].shape==(3,320,480) and np.isfinite(z['rgb']).all()
                for key in fields:assert np.array_equal(z[key],b[key]),(raw,state,key)
                frames.append(dict(raw=raw,state=state,all_global_geometry_and_semantic_RGB_exact=True,
                    RGB_min=float(z['rgb'].min()),RGB_max=float(z['rgb'].max()),float_sha256=sha(out)))
    after=json.loads((ROOT/'contract/E_target_guard_after_formal.json').read_text())
    before=json.loads((ROOT/'contract/E_target_guard_before_formal.json').read_text())
    assert before['files']==after['files']
    assert not torch.cuda.is_initialized()
    report=dict(status='PASS_INDEPENDENT_CPU_E400_STATE_EXPORT_AND_GEOMETRY_INVARIANTS',
        checkpoint_checks=checks,export_manifest=str(material_path),export_manifest_sha256=sha(material_path),
        export_delta_exact_checkpoint400=True,export_rows_exact_qualification=True,delta_bounds=[-1,1],
        same_fixed_delta_KD=True,frames=frames,exact_fields=fields,
        all_image_geometry_weights_and_semantic_RGB_exact=True,target_sources_bytes_mtime_unchanged=True,
        actual_formal_optimizer_updates=400,actual_formal_native_forwards=808,actual_formal_native_backwards=800,
        original_Gaussian_model_updates=0,new_GPU_forwards_for_this_audit=0,cuda_initialized=False,
        quality_gate_or_visual_acceptance_claim=False,
        limitations=['Original registered photometric/seam/vehicle protection and root visual verdict are separate.',
            'This audit validates saved state/export/floats and does not add training or49-frame evaluation.'])
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],checkpoints=[x['step'] for x in checks],delta_max_abs=checks[-1]['delta_max_abs'],export_exact=True,frames=8,geometry_exact=True)))


if __name__=='__main__':main()
