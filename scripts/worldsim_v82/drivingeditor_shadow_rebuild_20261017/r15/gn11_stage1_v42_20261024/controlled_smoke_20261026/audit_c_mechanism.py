"""Zero-update component interventions on the saved full199 editing checkpoint.

These interventions identify dependence of this fitted scene, not causal effects
of retraining ablations. Surface-depth agreement is internal consistency only.
"""
from editor_common import *
from scipy.ndimage import binary_dilation, binary_erosion


def render_depth_moments(n, camera, road, mult, scale):
    kw = prepare_rasterization(camera, n.static, n.args)
    kw['means2D'] = torch.zeros_like(kw['means3D'])
    kw['opacities'] = kw['opacities'] * mult[:, None]
    rr = road_kwargs(camera, road, n.args, kw['shs'].shape[1] if kw['shs'] is not None else 1,
                     precomputed_color=kw['colors_precomp'] is not None)
    z = rr['features'][:, :1] / scale
    kw['features'] = torch.cat((kw['features'], torch.zeros((len(kw['means3D']), 3), device='cuda')), 1)
    rr['features'] = torch.cat((rr['features'], torch.ones_like(z), z, z*z), 1)
    out = raster(camera, n.args, n.static.active_sh_degree, merge_rows(kw, rr), n.bg)
    weight, first, second = out['extra']
    mean = first / weight.clamp_min(1e-8)
    std = (second / weight.clamp_min(1e-8) - mean*mean).clamp_min(0).sqrt()
    return cpu(weight), cpu(mean), cpu(std)


def main():
    dest = OUT/'c_mechanism'; dest.mkdir(exist_ok=True)
    n = Native()
    path = OUT/'edit_visibility/full199/checkpoint1000_from0.pth'
    ck = torch.load(path, weights_only=False)
    init = Road(FULL/'full199_initialization.npz')
    trained = Road(FULL/'full199_initialization.npz')
    initial49 = Road(FULL/'window49_initialization.npz')
    trained.dc.data.copy_(ck['dc']); trained.opacity.data.copy_(ck['opacity'])
    mult = torch.ones(len(n.static.get_xyz), device='cuda').index_copy(0, ck['ids'], ck['gate'].sigmoid())
    old = torch.load(OUT/'edit_smoke_preserve/full199/checkpoint1000_from0.pth', weights_only=False)
    oldroad = Road(FULL/'full199_initialization.npz')
    oldroad.dc.data.copy_(old['dc']); oldroad.opacity.data.copy_(old['opacity'])
    oldmult = torch.ones_like(mult).index_copy(0, old['ids'], old['gate'].sigmoid())
    coloronly = Road(FULL/'full199_initialization.npz'); coloronly.dc.data.copy_(ck['dc'])
    opacityonly = Road(FULL/'full199_initialization.npz'); opacityonly.opacity.data.copy_(ck['opacity'])
    variants = [('B_static', None, None), ('initial_road_only_change', init, None),
                ('trained_road_no_gate', trained, None), ('gate_no_road', None, mult),
                ('gate_initial_road', init, mult), ('C_full', trained, mult),
                ('same_gate_49_initial', initial49, mult), ('RGB_only_previous', oldroad, oldmult),
                ('gate_trained_color_initial_opacity', coloronly, mult),
                ('gate_initial_color_trained_opacity', opacityonly, mult)]
    rows = []; depth_rows = []; master_difference = 0.0
    scale = read('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json')['meter_to_model']
    with torch.no_grad():
        for raw in range(89,138):
            c = n.camera(raw)
            labels = np.load(C.parent/'r15_gn11_stage1_v41_20261023_dense_r7'/ 'frames'/f'0008{raw:03d}.npz')['actor_labels']
            target = (labels==4)|(labels==5); outside = ~binary_dilation(target, iterations=2)
            base = n.render(c); original = cpu(base['rgb']); native_before = n.render(c,dynamic=True)['rgb'].clone()
            d = dict(np.load(OUT/'crossview'/f'{raw}.npz')) if raw in [102,129,137] else None
            for name,road,gate in variants:
                out = base if name=='B_static' else n.render(c,road,static_multiplier=gate)
                rgb=cpu(out['rgb']); con=cpu(out['contribution']).squeeze(); change=np.abs(rgb-original).mean(0)
                row=dict(raw=raw,variant=name,outside_pixels=int(outside.sum()),outside_MAE=float(change[outside].mean()),
                         outside_p99=float(np.quantile(change[outside],.99)),outside_changed_gt005=int((change[outside]>.05).sum()),actors={})
                for actor,k in [('37',5),('6',4)]:
                    core=binary_erosion(labels==k,iterations=5)
                    row['actors'][actor]=dict(core=int(core.sum()),visible=int((core&(con>=.05)).sum()),
                                             contribution_sum=float(con[core].sum()))
                if d is not None:
                    valid=d['valid']; error=np.abs(rgb-d['rgb'].transpose(2,0,1)).mean(0)
                    base_error=np.abs(original-d['rgb'].transpose(2,0,1)).mean(0)
                    row.update(valid=int(valid.sum()),valid_MAE=float(error[valid].mean()),valid_MAE_clamped=float(np.abs(np.clip(rgb,0,1)-d['rgb'].transpose(2,0,1)).mean(0)[valid].mean()),
                               better_fraction=float((error[valid]<base_error[valid]-1e-4).mean()),worse_fraction=float((error[valid]>base_error[valid]+1e-4).mean()))
                if raw==102:
                    row['critical']={f'{x},{y}':dict(road_contribution=float(con[y,x]),rgb=rgb[:,y,x].tolist(),
                                                  color_MAE=float(np.abs(rgb[:,y,x]-d['rgb'][y,x]).mean())) for x,y in [(255,240),(256,219)]}
                if raw in [102,129,133,137]:save(dest/f'{name}_{raw}.png',out['rgb'])
                if name=='C_full':
                    save(dest/f'C_full_{raw}.png',out['rgb'])
                    if raw in [102,129,137]:
                        w,z,std=render_depth_moments(n,c,road,mult,scale);valid=d['valid']&(w>=.05)
                        dr=dict(raw=raw,valid=int(valid.sum()),mean_abs_to_initial_surface=float(np.abs(z[valid]-d['depth'][valid]).mean()),
                                p90_abs_to_initial_surface=float(np.quantile(np.abs(z[valid]-d['depth'][valid]),.9)))
                        if raw==102:
                            dr['critical']={f'{x},{y}':dict(road_weight=float(w[y,x]),road_weighted_z_m=float(z[y,x]),road_z_std_m=float(std[y,x]),
                                                          reference_surface_z_m=float(d['depth'][y,x])) for x,y in [(255,240),(256,219)]}
                        depth_rows.append(dr)
                rows.append(row)
            native_after=n.render(c,dynamic=True)['rgb'];master_difference=max(master_difference,float((native_after-native_before).abs().max()))
            if raw%10==2:print('raw',raw,'done',flush=True)
        # Save direct initial/trained road images for the key frame.
        for name,road in [('road_initial',init),('road_trained',trained)]:save(dest/f'{name}_102.png',render_road(n.camera(102),road,n.args)['rgb'])
        same_geometry=all(torch.equal(getattr(init,key),getattr(trained,key)) for key in ['xyz','scaling','rotation'])
        values=cpu(mult);gate_summary=dict(total_rows=len(values),selected_rows=len(ck['ids']),below_05=int((values<.5).sum()),below_01=int((values<.1).sum()),below_001=int((values<.01).sum()),quantiles=np.quantile(values,[0,.01,.05,.25,.5,.9,1]).tolist())
        summary={}
        for name,_,_ in variants:
            items=[r for r in rows if r['variant']==name];supervised=[r for r in items if 'valid_MAE' in r]
            summary[name]=dict(reference_pixels=sum(r['valid'] for r in supervised),reference_MAE=sum(r['valid_MAE']*r['valid'] for r in supervised)/sum(r['valid'] for r in supervised),
                               outside_MAE=sum(r['outside_MAE']*r['outside_pixels'] for r in items)/sum(r['outside_pixels'] for r in items),
                               worst_outside=max(items,key=lambda r:r['outside_MAE']),actors={actor:{key:sum(r['actors'][actor][key] for r in items) for key in ['core','visible','contribution_sum']} for actor in ['37','6']})
        assert n.unchanged() and same_geometry and master_difference==0
        result=dict(checkpoint=str(path),source_variant='edit_visibility/full199',updates_this_audit=0,
                    qualification_rescue_used=False,freeinpaint_used=False,geometry_unchanged=same_geometry,
                    native_repeat_maxabs=master_difference,master_files_unchanged=n.unchanged(),gate=gate_summary,
                    limitations=['Fixed-state interventions are not retrained ablations.', 'Three color-reference views were used in optimization.',
                                 'Other 46 are same-sequence checks, not new-scene generalization.', 'Depth comparison reuses initialization surface: internal consistency, not independent geometry truth.',
                                 'Saved checkpoint needs B, engine, calibration and full199 initialization.', 'Road visibility does not certify geometry or hidden appearance.'],
                    rows=rows,summary=summary,depth_consistency=depth_rows,quality_accepted=False,human_verdict=None)
        (dest/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(summary={k:{s:v for s,v in value.items() if s not in ['worst_outside','actors']} for k,value in summary.items()},gate=gate_summary,depth=depth_rows)),flush=True)


if __name__=='__main__':main()
