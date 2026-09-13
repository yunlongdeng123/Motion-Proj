import numpy as np
from motion_proj.worldsim_v81.geometry import transform,apply,project,reference_filter,depth_metrics,interaction_interval,remove_boxes

def test_camera_z_is_not_ego_ray_distance():
    T=transform({'rotation':[1,0,0,0],'translation':[2,0,0]});K=np.array([[100,0,50],[0,100,40],[0,0,1]])
    xyz=np.array([[3.,4.,10.]]);uv,z,c=project(apply(xyz,T),T,K)
    np.testing.assert_allclose(uv,[[80,80]]);np.testing.assert_allclose(z,[10])
    assert np.linalg.norm(apply(xyz,T)[0])!=z[0]

def test_reference_requires_disjoint_scan_support_and_rejects_edge():
    uv=np.array([[10,10],[10,10],[20,20],[20,20],[30,30],[30,30]])
    z=np.array([10,10.1,10,20,10,10.1]);src=np.array([0,1,0,1,0,0])
    np.testing.assert_array_equal(reference_filter(uv,z,src,100,100),[0,1])

def test_missing_predictions_keep_denominator():
    m=depth_metrics([10,np.nan,0],[10,10,10]);assert m['support']==3 and m['valid']==1 and m['coverage']==1/3
    assert depth_metrics([np.nan],[10])['absrel'] is None

def test_scene_bootstrap_does_not_count_pixels_as_logs():
    rows=[{'log':'one','cohort':c,'error':1} for c in ['C00','C10','C01','C11'] for _ in range(1000)]
    assert interaction_interval(rows)['status']=='INSUFFICIENT_MATCHED_LOGS'
    rows=[{'log':str(log),'cohort':c,'error':v} for log in range(4) for c,v in zip(['C00','C10','C01','C11'],[1,2,3,8])]
    r=interaction_interval(rows);assert r['interaction']==4 and r['ci95']==[4,4]

def test_nuscenes_box_width_length_axes():
    b={'rotation':[1,0,0,0],'translation':[0,0,0],'size':[2,6,2]}
    points=np.array([[2,0,0],[0,2,0],[4,0,0]])
    np.testing.assert_array_equal(remove_boxes(points,[b],0),points[1:])

def test_view_interventions_keep_target_and_temporal_camera_order():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('v81_infer',Path(__file__).resolve().parents[1]/'scripts/run_worldsim_v81_inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    select_views,pixel_affine=module.select_views,module.pixel_affine
    cameras=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']
    def frame(sample,time):return [{'camera':c,'sample_token':sample,'timestamp_us':time} for c in cameras]
    m={'views':frame('now',1000000),'context_views':frame('before',500000)+frame('after',1500000)}
    for variant in ['sparse2','sparse3']:assert select_views(m,variant,'CAM_BACK_LEFT')[0]['camera']=='CAM_BACK_LEFT'
    result=select_views(m,'temporal18','CAM_BACK_LEFT')
    assert len(result)==18
    assert [v['camera'] for v in result[:6]]==[v['camera'] for v in result[6:12]]==[v['camera'] for v in result[12:]]
    size,A=pixel_affine([1600,900],'dvgt');assert size==[512,288]
    np.testing.assert_allclose(np.array(A)@[-.5,-.5,1],[-.5,-.5,1])

def test_matching_cannot_rescue_missing_cells_with_unmatched_or_confounded_rows():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('v81_eval',Path(__file__).resolve().parents[1]/'scripts/evaluate_worldsim_v81.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[{'roi_id':str(i),'coverage':1.,'error':1.,'visual_screen':'UNREVIEWED'} for i in range(4)]
    blocks=[{'cases':dict(zip(['C00','C10','C01','C11'],map(str,range(4))))}]
    assert module.matched_rows(rows,[])==[]
    assert len(module.matched_rows(rows,blocks))==4
    assert module.matched_rows(rows[:-1],blocks)==[]
    rows[0]['visual_screen']='CONFOUND_EXCLUDE_MAIN'
    assert module.matched_rows(rows,blocks)==[]
