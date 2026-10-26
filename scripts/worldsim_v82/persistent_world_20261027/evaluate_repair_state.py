"""同49帧比较两个固定10K状态：颜色、深度统计、未知分母与保护域。"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp')
from editor_common import *
import surface_moments


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    for sub in ['native_rgb','mask','float']:(dest/sub).mkdir(exist_ok=True)
    ck=torch.load(a.checkpoint,weights_only=False);config=ck['config'];n=Native();assert ck['master']==n.before
    check=surface_moments.install(n);road=Road(Path(config['initialization']))
    with torch.no_grad():road.dc.copy_(ck['dc']);road.opacity.copy_(ck['opacity'])
    mult=torch.ones(len(n.static.get_xyz),device='cuda').index_copy(0,ck['ids'],ck['gate'].sigmoid())
    rows=[]
    with torch.no_grad():
        for raw in range(89,138):
            camera=n.camera(raw);out=n.render(camera,road,static_multiplier=mult);base=n.render(camera)
            d=dict(np.load(Path(config['reference_root'])/f'{raw}.npz'));valid=d['valid']&np.isfinite(d['depth'])&(d['depth']>0)
            rgb=cpu(out['rgb']).transpose(1,2,0);error=np.abs(np.clip(rgb,0,1)-np.clip(d['rgb'],0,1)).mean(2)
            depth=cpu(out['depth_sum']/out['alpha'].clamp_min(1e-6))[0]/config['meter_to_model']
            geometry=np.abs(depth-d['depth']);outside=~d['target']
            rows.append(dict(raw=raw,optimization_view=raw in config['train_views'],labeled_pixels=int(valid.sum()),target=int(d['target'].sum()),unknown=int((d['target']&~valid).sum()),
                clamped_MAE=float(error[valid].mean()),mean_expected_depth_difference_m=float(geometry[valid].mean()),
                depth_fraction_below03m=float((geometry[valid]<.3).mean()),
                outside_change_MAE=float(cpu((out['rgb']-base['rgb']).abs()).mean(0)[outside].mean()),
                road_mean_contribution_diagnostic=float(cpu(out['contribution'])[0,valid].mean())))
            save(dest/'native_rgb'/f'{raw:03d}.png',out['rgb'])
            # 固定生成区域，避免把路由掩码变化混进上游几何对照。
            source=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027/phase1_native/delete_all_evidence_routed/mask')/f'{raw:03d}.png'
            Image.open(source).save(dest/'mask'/source.name)
            np.savez_compressed(dest/'float'/f'{raw:03d}.npz',rgb=cpu(out['rgb']),alpha=cpu(out['alpha']),depth_sum=cpu(out['depth_sum']),depth2_sum=cpu(out['depth2_sum']))
    assert n.unchanged()
    groups={}
    for name,items in [('train_views',[r for r in rows if r['optimization_view']]),('other46_views',[r for r in rows if not r['optimization_view']])]:
        count=sum(r['labeled_pixels'] for r in items)
        groups[name]=dict(reference_pixels=count,frames=len(items),**{metric:sum(r[metric]*r['labeled_pixels'] for r in items)/count
            for metric in ['clamped_MAE','mean_expected_depth_difference_m','depth_fraction_below03m','road_mean_contribution_diagnostic']})
    summary=dict(status='complete',checkpoint=a.checkpoint,config=config,rows=rows,groups=groups,
        extra_moment_rgb_check=check,final_step=ck['step'],master_files_unchanged=True,
        reference_role='同片段来源一致性诊断，非独立三维GT；期望深度非首交面',
        generation_mask='固定第一轮证据路由mask；同后端对照',quality_accepted=False,human_verdict=None)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(groups),flush=True)


if __name__=='__main__':
    from gpu_slot import exclusive
    with exclusive():main()
