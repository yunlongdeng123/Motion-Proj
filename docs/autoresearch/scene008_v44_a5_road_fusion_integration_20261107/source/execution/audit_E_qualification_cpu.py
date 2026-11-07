"""从保存逐point质量独立汇总E资格/覆盖；CPU只读，无新光栅。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch

from scripts.worldsim_v82.drivingeditor_dynamic_edit.baseline_road_appearance import read_qualification

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
E=ROOT/'scene008_v44_a5_road_fusion_integration_20261107_r1/E4_appearance_qualification'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4194304),b''):h.update(data)
    return h.hexdigest()


def tensor_hash(values):
    h=hashlib.sha256()
    for k,v in sorted(values.items()):
        a=v.detach().cpu().contiguous().numpy()
        h.update(k.encode());h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    started=time.time();mp=E/'qualification_manifest.json';manifest=json.loads(mp.read_text())
    binding=json.loads((E/'model_binding_before.json').read_text())
    q,safe,npz=read_qualification(mp,binding)
    receipt=json.loads((E/'receipt.json').read_text());protocol=json.loads((E/'qualification_protocol.json').read_text())
    assert q['model_binding']==binding and receipt['manifest_sha256']==sha(mp)
    assert receipt['source_files_and_model_binding_unchanged']
    assert receipt['joint_forwards']==8 and receipt['dummy_feature_backwards']==8
    ns=q['branch_rows']['native_static'];nr=q['branch_rows']['oldroad'];n=ns+nr
    indices=np.r_[safe['native_static'],ns+safe['oldroad']];mask=np.zeros(n,bool);mask[indices]=True
    accumulation=np.zeros((n,2),np.float64);coverage=[];hashes={};maxerror=0.
    records={(r['raw'],r['state']):r for r in receipt['rows']}
    cov_original={(r['raw'],r['state']):r for r in receipt['coverage']}
    for raw in q['calibration_raw_ids']:
        for state in q['states']:
            p=E/'point_mass'/f'{raw:03d}_{state}.npz';hashes[str(p)]=sha(p)
            with np.load(p,allow_pickle=False) as z:
                mass=z['mass'].copy();assert int(z['static_row_count'])==ns
            assert mass.shape==(n,6) and mass.dtype==np.float32 and np.isfinite(mass).all() and np.all(mass>=0)
            accumulation+=mass[:,:2].astype(np.float64)
            rows=[];record=records[(raw,state)]
            for column,name in enumerate(protocol['probe_domains']):
                selected=float(mass[mask,column].astype(np.float64).sum())
                total=float(mass[:,column].astype(np.float64).sum())
                static=float(mass[:ns,column][mask[:ns]].astype(np.float64).sum())
                road=float(mass[ns:,column][mask[ns:]].astype(np.float64).sum())
                old=cov_original[(raw,state)]['domains'][column]
                for key,value in [('safe_mass',selected),('all_candidate_branch_mass',total),('safe_static_mass',static),('safe_road_mass',road)]:
                    # Float64逐column/axis0归约顺序可差数个ULP；资格predicate仍逐值相等。
                    assert np.isclose(value,old[key],rtol=1e-13,atol=1e-8),(raw,state,name,key,value,old[key])
                    maxerror=max(maxerror,abs(value-old[key]))
                rows.append(dict(domain=name,pixels=old['pixels'],safe_mass=selected,
                    safe_static_mass=static,safe_road_mass=road,total_branch_mass=total,
                    baseline_joint_alpha_mass=old['joint_alpha_mass'],
                    safe_fraction_of_joint_alpha_mass=selected/old['joint_alpha_mass'] if old['joint_alpha_mass'] else None))
            assert rows[1]['safe_mass']==0.
            assert max(record['mass_sum_relative_error'])<=1e-5
            assert all(v==0 for v in record['baseline_parity_max_abs'].values())
            coverage.append(dict(raw=raw,state=state,domains=rows))
    with np.load(npz,allow_pickle=False) as z:
        assert np.array_equal(accumulation[:ns,0],z['static_inside_mass'])
        assert np.array_equal(accumulation[:ns,1],z['static_outside_mass'])
        assert np.array_equal(accumulation[ns:,0],z['road_inside_mass'])
        assert np.array_equal(accumulation[ns:,1],z['road_outside_mass'])
    # 直接CPU mmap真实旧checkpoint，比对资格时实际已加载模型的全部核心参数指纹。
    parent=ROOT/'scene008_blue_global_rgb_probe_20261105_r1/global_10k/chkpnt10000.pth'
    payload,iteration=torch.load(str(parent),map_location='cpu',weights_only=False,mmap=True)
    keys=['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_t','_scaling_t','_velocity']
    actual_static_hash=tensor_hash({k:payload[i] for i,k in enumerate(keys,1)})
    assert actual_static_hash==binding['native_static_tensor_sha256']
    repair_path=ROOT/'scene008_blue_global_road1k_20261105_r1/formal/checkpoint001000.pt'
    repair=torch.load(str(repair_path),map_location='cpu',weights_only=False,mmap=True)['repair']
    actual_repair_hash=tensor_hash(repair);assert actual_repair_hash==binding['oldroad_repair_state_sha256']
    contract=ROOT/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json'
    assert sha(contract)==binding['v44_contract_sha256']
    sem_path=ROOT/'scene008_v44_full5_integration_20261107_r1/semantic_actor5_calibration49/semantic_actor5_manifest.json'
    sem=json.loads(sem_path.read_text());assert sha(sem_path)==binding['semantic_manifest_sha256']
    assert sem['selection_sha256']==binding['semantic_selection_sha256']
    assert sem['gaussian_files_sha256']==binding['full5_gaussian_files_sha256']
    baseline_hashes=q['baseline_float_input_sha256']
    for path,value in baseline_hashes.items():assert sha(path)==value
    assert not torch.cuda.is_initialized()
    primary=[dict(raw=r['raw'],state=r['state'],**r['domains'][5]) for r in coverage if r['raw']==128]
    result=dict(status='PASS_INDEPENDENT_CPU_E_QUALIFICATION_AND_COVERAGE',manifest=str(mp),manifest_sha256=sha(mp),
        safe_rows={k:len(v) for k,v in safe.items()},point_mass_hashes=hashes,
        all8_mass_aggregate_exact_saved_selection=True,strict_predicate_recomputed=True,
        safe_guard_mass_zero_in_all8_measured_states=True,coverage_recompute_max_absolute_error=maxerror,
        coverage=coverage,primary128_68px=primary,native_static_checkpoint_core_tensor_hash=actual_static_hash,
        oldroad_checkpoint_repair_state_hash=actual_repair_hash,
        source_model_binding_and_frozen_baseline_inputs_verified=True,
        oldroad_final_GPU_fine_colour_hash_role='preserved runtime before/after fingerprint; not independently recomputed with CPU GEMM',
        calibration_cost_only=dict(joint_forwards=8,dummy_backwards=8),new_GPU_forwards_for_this_audit=0,
        fit_steps=0,model_updates=0,cuda_initialized=False,
        limits=['Four calibrated BUILD views only, not independent view generalization.',
            'Zero guard contribution at fixed geometry makes row RGB changes invisible in those measured guard pixels.',
            'Mass coverage is representational leverage, not a guarantee of fitting quality or seam improvement.'],
        seconds=time.time()-started)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','safe_rows','primary128_68px','seconds']},indent=2))


if __name__=='__main__':main()
