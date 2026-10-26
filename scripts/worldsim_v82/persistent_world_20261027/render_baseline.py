"""阶段一：真实高斯状态→编辑请求→原生渲染；始终保留生成前输出。"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp')
from editor_common import *
from scipy.ndimage import binary_dilation, binary_erosion
from world_state import WorldState


def dump(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--frames',type=int,default=49)
    a=parser.parse_args();dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    n=Native();ckpath=OUT/'edit_smoke_preserve/full199/checkpoint1000_from0.pth'
    ck=torch.load(str(ckpath),weights_only=False)
    road=Road(FULL/'full199_initialization.npz')
    with torch.no_grad():road.dc.copy_(ck['dc']);road.opacity.copy_(ck['opacity'])
    multiplier=torch.ones(len(n.static.get_xyz),device='cuda').index_copy(0,ck['ids'],ck['gate'].sigmoid())
    state=WorldState(dest/'world.sqlite','waymo_scene0008_development')
    state.bind({str(k):list(v) for k,v in n.before.items()})
    state.revision('native',None,dict(route='native',editing=False))
    state.revision('background_preserve_199','native',dict(
        checkpoint=str(ckpath),initialization=str(FULL/'full199_initialization.npz'),
        xyz_scale_rotation_fixed=True,optimized_views=[102,129,137],
        note='已有开发原型；颜色拟合不能代替几何验证'))
    init=dict(np.load(FULL/'full199_initialization.npz'))
    source_matrix=init['source_valid'] if 'source_valid' in init else None
    rows=[]
    for i,xyz in enumerate(init['world_xyz']):
        eligible=bool(init['eligible'][i]);supported=bool(init['surface_supported'][i])
        status='observed_candidate' if eligible else 'supported_uncertain' if supported else 'unknown'
        values=np.asarray(source_matrix[:,i]).ravel() if source_matrix is not None else np.array([])
        src=init['source_raw'][values.astype(bool)].tolist() if len(values) else []
        count=len(src)
        rows.append((i,xyz.tolist(),status,src,count,'background_preserve_199' if eligible else None))
    state.surfaces(rows)
    cases=['native','delete_all_unrepaired','delete_all_repaired','repeat_delete_reloaded','repaired_with_native_actors']
    for case in cases:
        for sub in ['native_rgb','mask','float']:(dest/case/sub).mkdir(parents=True,exist_ok=True)
    started=time.time();metrics=[];repeat_max=0.
    frames=list(range(89,89+a.frames))
    with torch.no_grad():
        for raw in frames:
            camera=n.camera(raw)
            native=n.render(camera,dynamic=True)
            static=n.render(camera)
            edited=n.render(camera,road,static_multiplier=multiplier)
            restored=n.render(camera,road,static_multiplier=multiplier)
            with_actors=n.render(camera,road,dynamic=True,static_multiplier=multiplier)
            difference=float((edited['rgb']-restored['rgb']).abs().max());repeat_max=max(repeat_max,difference)
            labels=np.load(f'/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023_dense_r7/frames/0008{raw:03d}.npz')['actor_labels']
            # 所有动态移除，保留此目标定义；不能将其冒称只删除a37/a6。
            target=labels>0
            repair_domain=(labels==4)|(labels==5)
            generation=binary_dilation(target,iterations=2)
            protected=~generation
            per=dict(raw=raw,optimized_view=raw in [102,129,137],target=int(target.sum()),repair_target=int(repair_domain.sum()),
                repeat_max_difference=difference,
                outside_change_MAE=float(cpu((edited['rgb']-static['rgb']).abs()).mean(0)[protected].mean()),
                full_with_edit_state_change_MAE=float((with_actors['rgb']-native['rgb']).abs().mean()),
                no_edit_native_route='master without edit state')
            for case,package in zip(cases,[native,static,edited,restored,with_actors]):
                path=dest/case
                save(path/'native_rgb'/f'{raw:03d}.png',package['rgb'])
                Image.fromarray((generation*255).astype('uint8')).save(path/'mask'/f'{raw:03d}.png')
                np.savez_compressed(path/'float'/f'{raw:03d}.npz',rgb=cpu(package['rgb']),alpha=cpu(package['alpha']),depth_sum=cpu(package['depth_sum']))
            metrics.append(per)
            if raw%10==0:print('RENDER',raw,round(time.time()-started,2),flush=True)
    assert n.unchanged()
    # 独立重载checkpoint及数据库后复渲染，检查版本重用；并不等于新算法一致性收益。
    state.close();state=WorldState(dest/'world.sqlite','waymo_scene0008_development')
    ck2=torch.load(str(ckpath),weights_only=False)
    with torch.no_grad():
        road.dc.copy_(ck2['dc']);road.opacity.copy_(ck2['opacity'])
        multiplier2=torch.ones(len(n.static.get_xyz),device='cuda').index_copy(0,ck2['ids'],ck2['gate'].sigmoid())
        camera=n.camera(102);actual=n.render(camera,road,static_multiplier=multiplier2)
        old=np.load(dest/'delete_all_repaired/float/102.npz')['rgb']
        reload_diff=float(np.abs(cpu(actual['rgb'])-old).max())
        native_again=n.render(camera,dynamic=True)
        native_old=np.load(dest/'native/float/102.npz')['rgb']
        master_diff=float(np.abs(cpu(native_again['rgb'])-native_old).max())
    for case in cases:
        revision='native' if case in ['native','delete_all_unrepaired'] else 'background_preserve_199'
        state.record_request(case,revision,dict(operation=case,frames=frames,scope='development_same_sequence'),dest/case)
    summary=dict(stage='phase1_native_baseline',complete=True,frames=frames,cases=cases,metrics=metrics,
        reload_max_rgb_difference=reload_diff,native_roundtrip_max_rgb_difference=master_diff,
        repeat_same_state_max_rgb_difference=repeat_max,persistence=state.summary(),
        selected_checkpoint=str(ckpath),master_files_unchanged=n.unchanged(),
        limitations=['仅单场景开发集','背景修复只优化三个视图','当前删全部动态，单身份编辑另行接入','世界几何未验收','重载复现是工程合同，不是记忆算法收益'],
        seconds=time.time()-started,cuda_peak_allocated=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved=torch.cuda.max_memory_reserved(),new_training_updates=0,
        quality_accepted=False,human_verdict=None)
    state.close();dump(dest/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='metrics'}),flush=True)


if __name__=='__main__':main()
