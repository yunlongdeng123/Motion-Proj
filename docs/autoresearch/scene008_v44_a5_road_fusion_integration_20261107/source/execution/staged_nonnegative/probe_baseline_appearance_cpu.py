"""CPU小反例：固定branch-row residual不改geometry/SH视角差异或原tensor。"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
import scripts.worldsim_v82.drivingeditor_dynamic_edit.baseline_road_appearance as api


def main():
    global api
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--api-path');args=parser.parse_args()
    if args.api_path:
        spec=importlib.util.spec_from_file_location('_isolated_staged_appearance',args.api_path)
        api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    # 只缩小当前隔离CPU进程的branch fixture；生产源文件/常量未改。
    production_counts=api.BRANCH_ROWS.copy();api.BRANCH_ROWS=dict(native_static=2,oldroad=2)
    with tempfile.TemporaryDirectory(prefix='baseline_appearance_cpu_') as tmp:
        folder=Path(tmp);npz=folder/'safe.npz'
        np.savez_compressed(npz,static_safe_row_ids=np.array([1],np.int64),
            static_inside_mass=np.array([0,1],np.float64),static_outside_mass=np.zeros(2,np.float64),
            road_safe_row_ids=np.array([0],np.int64),road_inside_mass=np.array([1,0],np.float64),
            road_outside_mass=np.array([0,1],np.float64))
        q=dict(schema=api.QUALIFICATION_SCHEMA,branch_rows=api.BRANCH_ROWS,
            calibration_raw_ids=api.CALIBRATION_RAW_IDS,states=['keep','delete'],
            calibration_scene='latest_semantic_without_added_road_geometry',model_binding={'fixture':'frozen'},
            threshold={'inside_strictly_greater_than':0.,'outside_equal':0.},model_updates=0,colour_updates=0,
            selection_file=npz.name,selection_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
            safe_rows={'native_static':1,'oldroad':1})
        qp=folder/'qualification.json';qp.write_text(json.dumps(q))
        _,safe,_=api.read_qualification(qp,q['model_binding'])
        try:api.read_qualification(qp,{'wrong':'parent'})
        except ValueError:pass
        else:raise AssertionError('mismatched parent was accepted')
        mp=api.write_appearance_material(qp,{k:np.zeros((len(v),3),np.float32) for k,v in safe.items()},folder/'zero_export')
        assert json.loads(mp.read_text())['geometry_updates']==0
        originals={}
        def make_scene(raw,delete):
            # 删除a37会改变oldroad的全局offset；branch-local row仍相同。
            groups=[5,5,2]+([] if delete else [3])+[4,4,6]
            n=len(groups);f=torch.zeros(n,24)
            for row,column in enumerate(groups):f[row,column]=1
            rows=dict(means3D=torch.arange(n*3,dtype=torch.float32).reshape(n,3),
                means2D=torch.zeros(n,3),shs=None,colors_precomp=torch.full((n,3),.3+raw*.001),
                opacities=torch.full((n,1),.8),scales=torch.ones(n,3),rotations=torch.ones(n,4),
                cov3D_precomp=None,mask=torch.ones(n,dtype=torch.bool),features=f)
            originals[(raw,delete)]=rows
            return dict(rows=rows,meta={})
        runtime=api.BaselineRoadAppearanceV44Integration.__new__(api.BaselineRoadAppearanceV44Integration)
        runtime.semantic=SimpleNamespace(torch=torch,scene=make_scene)
        runtime.assert_unchanged=lambda full=False:True
        runtime.safe=safe;runtime.material={'material_sha256':'CPUfixture'};runtime._qualification_path=qp
        runtime._ids={k:torch.as_tensor(v) for k,v in safe.items()}
        runtime._delta=dict(native_static=torch.tensor([[.1,.2,.3]]),oldroad=torch.tensor([[-.02,.03,.04]]))
        selected={}
        for raw in (94,105):
            for delete in (False,True):
                scene=runtime.scene(raw,delete=delete);old=originals[(raw,delete)];rows=scene['rows'];f=rows['features']
                for key in old:
                    if key!='colors_precomp' and old[key] is not None:assert rows[key] is old[key],key
                expected=torch.zeros_like(old['colors_precomp'])
                for branch,column in [('native_static',5),('oldroad',4)]:
                    branch_rows=torch.nonzero(f[:,column]==1).flatten();index=branch_rows[runtime._ids[branch]]
                    expected[index]=runtime._delta[branch]
                    selected[(raw,delete,branch)]=rows['colors_precomp'][index].clone()
                assert torch.allclose(rows['colors_precomp']-old['colors_precomp'],expected,atol=3e-8,rtol=0)
                assert torch.equal(old['colors_precomp'],torch.full_like(old['colors_precomp'],.3+raw*.001))
        for branch in safe:
            for delete in (False,True):
                difference=selected[(105,delete,branch)]-selected[(94,delete,branch)]
                assert torch.allclose(difference,torch.full_like(difference,.011),atol=5e-8,rtol=0)
        for k in runtime._delta:runtime._delta[k].zero_()
        rows=runtime.scene(94,delete=False)['rows']
        assert torch.equal(rows['colors_precomp'],originals[(94,False)]['colors_precomp'])
        runtime._delta['native_static'][:]=torch.tensor([[-1.,2.,0.]])
        projected=runtime.scene(94,delete=False)['rows']['colors_precomp'][1]
        assert float(projected[0])==0 and float(projected[1])>1
        assert not torch.cuda.is_initialized()
    module_path=Path(api.__file__)
    result=dict(status='PASS_REAL_PYTHON39_CPU_BASELINE_APPEARANCE_FIXTURE',
        api_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),production_branch_counts=production_counts,
        isolated_fixture_branch_counts=api.BRANCH_ROWS,
        full_predicate_and_wrong_binding_rejection=True,zero_delta_RGB_exact=True,
        original_geometry_opacity_mask_features_tensor_identity=True,original_colour_tensor_unchanged=True,
        selected_branch_local_rows_correct_after_KD_global_offset_change=True,
        camera_dependent_base_colour_difference_preserved_when_projection_inactive=True,
        selected_negative_RGB_projected_to_zero=True,HDR_upper_values_preserved=True,
        no_new_geometry=True,optimizer_updates=0,GPU_forwards=0,cuda_initialized=False,
        limits='Small branch fixture in isolated CPU process; no assertion of real CUDA image quality/qualification.')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
