"""补导出同一深度排序的分支 αT；原始场景和结果只读。"""
from pathlib import Path
import sys, time, json, tarfile, hashlib
import numpy as np
from PIL import Image

BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene065_closedloop_20261116_r1')
sys.path.insert(0, str(BASE/'code'))
import coverage_handoff
import closedloop as t

STOPS = [(0., '#0b1424'), (.25, '#285ca5'), (.5, '#2a9d9b'), (.75, '#ecb84e'), (1., '#fff2bd')]
COLORS = dict(gap='#ff4d67', thin='#ffba49', occluded='#39c6e6')

def color(s):
    return np.array([int(s[i:i+2],16) for i in (1,3,5)],np.float32)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(t.R/'scripts/worldsim_v82'))
    from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
    run=t.C; out=run/'contribution_diagnostics_20261116'
    out.mkdir(exist_ok=False); delivery=out/'diagnostic_delivery'; delivery.mkdir()
    (out/'float').mkdir()
    scene, manifest, receipt=load_persistent_handoff(run/'persistent/scene.json',run/'persistent/proof.json')
    cfg=t.read(BASE.parent/'scene065_material_loop_20261114_r1/config.json')
    existing=t.read(run/'evaluation/evaluation.json')
    eval_frames={int(r['raw']):r for r in existing['frames']}
    rows=[];checks=[];start=time.time()
    for raw in range(199):
        with torch.no_grad():
            pack,values,camera=t.render_parts(scene,raw)
        static=pack['extra'][20].cpu().numpy()
        dynamic=pack['extra'][21].cpu().numpy()
        road=pack['extra'][3].cpu().numpy()
        total=pack['alpha'].cpu().numpy().squeeze()
        assert np.isfinite(static).all() and static.min()>=-1e-6 and static.max()<=1+1e-5
        branch_error=float(np.max(abs(static+dynamic+road-total)))
        assert branch_error<1e-5,branch_error
        with np.load(run/'rendered'/f'{raw:03d}'/'arrays.npz') as z:
            rgb=z['rgb'].transpose(1,2,0).copy(); only=z['road_only_alpha'].squeeze().copy()
            rgb_error=float(np.max(abs(pack['rgb'].cpu().numpy()-z['rgb'])))
            road_error=float(np.max(abs(road-z['road_alpha'].squeeze())))
        assert rgb_error<1e-5 and road_error<1e-5,(raw,rgb_error,road_error)
        with np.load(Path(cfg['road_views'])/f'{raw:03d}.npz') as z:
            domain=z['target']&z['inside']&~z['retained_reveal_guard']&~z['retained_box_guard']
        with np.load(run/'domains'/f'{raw:03d}.npz') as z:
            domain &= ~z['K']
        masks=dict(gap=domain&(only<.05),thin=domain&(only>=.05)&(only<=.9),
                   occluded=domain&(only>.9)&(road<.5))
        n=int(domain.sum()); gap=int(masks['gap'].sum()); support=int((domain&(only>.9)).sum())
        prior=eval_frames[raw]['stages']['full']['road_support']
        assert (n,gap,support)==(prior['domain_pixels'],prior['lt0p05'],prior['gt0p9'])
        overlay=np.clip(rgb*255,0,255)
        for label,mask in masks.items():
            overlay[mask]=.32*overlay[mask]+.68*color(COLORS[label])
        # 单像素轮廓只界定评价域，不扩大缺口，也不进入原始渲染。
        interior=domain.copy()
        interior[1:]&=domain[:-1]; interior[:-1]&=domain[1:]
        interior[:,1:]&=domain[:,:-1]; interior[:,:-1]&=domain[:,1:]
        edge=domain&~interior&~(masks['gap']|masks['thin']|masks['occluded'])
        overlay[edge]=.45*overlay[edge]+.55*np.array([225,235,245])
        x=np.clip(static,0,1); heat=np.stack([np.interp(x,[v[0] for v in STOPS],[color(v[1])[c] for v in STOPS]) for c in range(3)],-1)
        assets={}
        for name,array in [('road_gap',overlay),('static_alphaT',heat)]:
            dest=delivery/name/f'{raw:03d}.png';dest.parent.mkdir(exist_ok=True)
            Image.fromarray(np.uint8(np.clip(array,0,255)+.5)).save(dest,compress_level=6)
            assets[name]=dict(path=str(dest.relative_to(delivery)),sha256=sha(dest))
        np.savez_compressed(out/'float'/f'{raw:03d}.npz',static_alphaT=static,dynamic_alphaT=dynamic,
                            road_alphaT=road,road_only_alpha=only,domain=domain)
        stats=dict(domain_pixels=n,gap_pixels=gap,thin_pixels=int(masks['thin'].sum()),occluded_pixels=int(masks['occluded'].sum()),
                   supported_pixels=support,gap_fraction=gap/n if n else None,
                   static_alphaT_domain_mean=float(static[domain].mean()) if n else None)
        rows.append(dict(raw=raw,**assets,stats=stats))
        checks.append(dict(raw=raw,rgb_error=rgb_error,road_error=road_error,alphaT_sum_error=branch_error,evaluation_counts_exact=True))
        del pack,values
        if raw%25==0:print(json.dumps(dict(raw=raw,seconds=time.time()-start,stats=stats)),flush=True)
    scene.assert_persistent_unchanged()
    legend=dict(colors=COLORS,static_alphaT=dict(stops=STOPS,min=0,max=1,definition='sum_static(alpha_i * T_i), same_depth_order'),
                domain='road_views.target & inside & ~retained_reveal_guard & ~retained_box_guard & ~K',
                gap='domain & road_only_alpha < 0.05',thin='domain & 0.05 <= road_only_alpha <= 0.9',
                occluded='domain & road_only_alpha > 0.9 & shared_road_alphaT < 0.5',
                outside_domain='not_evaluated; zero_contribution_is_not_a_gap',
                warning='low_visible_road_contribution_is_not_proof_of_bad_static_ownership')
    result=dict(schema='scene065_contribution_diagnostics.v1',frame_count=199,preview_wh=[480,320],frames=rows,legend=legend,
                complete=True,source_run=str(run),loader=receipt,training_updates=0,source_render_unchanged=True,
                checks=checks,seconds=time.time()-start,GPU='keep_on',
                summary=dict(domain_pixel_frames=sum(r['stats']['domain_pixels'] for r in rows),gap_pixel_frames=sum(r['stats']['gap_pixels'] for r in rows)))
    t.write(delivery/'manifest.json',result)
    t.write(out/'completed.json',result)
    archive=out/'diagnostic_delivery.tar'
    with tarfile.open(archive,'w') as tar:tar.add(delivery,arcname='diagnostic_delivery')
    t.write(out/'delivery_ready.json',dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size))
    print('COMPLETE',result['summary'],result['seconds'],archive.stat().st_size,flush=True)

if __name__=='__main__':main()
