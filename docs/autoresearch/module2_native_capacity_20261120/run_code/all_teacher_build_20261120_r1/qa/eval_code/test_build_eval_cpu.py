from pathlib import Path
import numpy as np
import pytest
import eval_build_endpoints as b

def split_fixture(shape=(32,36)):
    score=np.ones(shape,bool);seen=score.copy();seen[:,shape[1]//2:]=False;held=score&~seen
    return dict(A_seen=seen,A_held=held,A_seen_context=seen.copy(),A_held_context=held.copy(),A_unmapped=np.zeros(shape,bool),
        K=np.zeros(shape,bool),fixed_alpha=np.ones(shape,np.float32))

def metric_path():
    p=Path(__file__).resolve().parents[3]/'road_quality_bench_20261120/metrics/road_detail_metrics.py'
    return p if p.exists()else b.B.parent/'quality_bench_20261120_r1/metrics/road_detail_metrics.py'

def test_union_precedes_filtering_and_restores_old_artificial_boundary_core():
    s=split_fixture();score,ctx=b.union_source(s,'A')
    old=b.E.metric_module(metric_path())
    union_of_old=old._strict_core(s['A_seen_context'],12)|old._strict_core(s['A_held_context'],12)
    rebuilt=old._strict_core(ctx,12)&score
    assert not union_of_old.any()and rebuilt.sum()==96

def test_protected_or_unmapped_labels_are_rejected_not_reclassified_fit():
    s=split_fixture();s['K'][5,5]=True
    with pytest.raises(ValueError):b.union_source(s,'A')
    s=split_fixture();s['A_unmapped'][5,5]=True
    with pytest.raises(ValueError):b.union_source(s,'A')

def test_build_report_has_only_fit_metadata_and_new_union_core():
    s=split_fixture();shape=s['K'].shape;T=np.full((3,*shape),.4,np.float32)
    p=dict(road_rgb=T,road_alpha=s['fixed_alpha']);prior=dict(pixels=576,mean_support_mass={'known':1.},original_source_weight_sum={'1':576.},pixels_with_any_seed=0,pixels_with_any_teacher_H_permission=5)
    receipt=dict(F0={'seen':prior,'held':prior},seen_blocks=2,held_blocks=1)
    r=b.build_region(b.E.metric_module(metric_path()),p,T,s,'A',receipt)
    assert r['data_role']==b.LABEL and r['scales'][-1]['core']['pixels']==96
    assert r['interpretation_status'].startswith('training_fit_only')and r['former_held_now_training']
    rows=b.flat_rows(r,'F0',86,'A','test.json')
    assert len(rows)==3 and all(x['split']==b.LABEL and x['blind_test']is False for x in rows)

def test_initial_ci_reuse_requires_bytes_exact_not_tolerance(tmp_path):
    x=np.ones((5,3),np.float32)*.4;ids=np.arange(5);mask=np.ones(5,bool)
    old=tmp_path/'old.npz';new=tmp_path/'new.npz'
    np.savez(old,ci0=x,active_world_rows=ids,eligible_active=mask);np.savez(new,ci0=x.copy(),active_world_rows=ids,eligible_active=mask)
    assert b.verify_initial_ci(new,old)['F0_native_packet_reuse_allowed']
    y=x.copy();y[0,0]=np.nextafter(y[0,0],np.float32(1.))
    np.savez(new,ci0=y,active_world_rows=ids,eligible_active=mask)
    with pytest.raises(ValueError):b.verify_initial_ci(new,old)

def test_build_endpoint_counters_use_frozen_budget_and_lastaccepted():
    proto=dict(arms={'G_BUILD_FIELD':dict(total_steps=1200),'G_BUILD_NUMERIC':dict(maxFG=2000)})
    assert b.validate_build_endpoint(dict(arm='G_BUILD_FIELD',completed=True,extra=dict(logical_steps=1200)),'G_BUILD_FIELD',proto)['steps']==1200
    n=dict(arm='G_BUILD_NUMERIC',completed=True,extra=dict(FGcount=2000,accepted_steps=1900,export_point='last accepted NEW_X'))
    assert b.validate_build_endpoint(n,'G_BUILD_NUMERIC',proto)['FGcount']==2000
    n['extra']['export_point']='best trial'
    with pytest.raises(ValueError):b.validate_build_endpoint(n,'G_BUILD_NUMERIC',proto)
