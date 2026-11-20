"""Bind completed BUILD public assets; does not run a renderer or training."""
from pathlib import Path
import argparse,json
import numpy as np
import eval_build_endpoints as V
E=V.E

def bind(root):
    root=Path(root);public=root/'public1920/export_r1';mp=public/'packets_manifest.json';rp=public/'receipt.json';vp=public/'visibility_manifest_r1.json'
    protocol=root/'protocol.json';cfg=E.read(protocol);manifest=E.read(mp);receipt=E.read(rp);vis=E.read(vp)
    if receipt.get('complete')is not True or receipt.get('optimizer_updates')!=0:raise ValueError('BUILD actual public export incomplete')
    if receipt.get('all_K_exact_zero')is not True:
        subs=receipt.get('subphase_receipts',[])
        if len(subs)!=2:raise ValueError('BUILD public K proof missing')
        for s in subs:
            if E.sha(s['path'])!=s['SHA256']or s.get('all_K_exact_zero')is not True or E.read(s['path']).get('all_K_exact_zero')is not True:raise ValueError('BUILD subphase public K proof differs')
    initial=V.verify_initial_ci(root/'common/initial_ci.npz',V.B/'formal_r1/common_initial_ci.npz')
    rows={(x['arm'],int(x['raw'])):x for x in manifest['rows']};vrows={(x['arm'],int(x['raw'])):x for x in vis['rows']}
    arms=['F0','G_BUILD_FIELD','G_BUILD_NUMERIC'];frames=[86,110,134,179,198]
    if set(rows)!={(a,f)for a in arms for f in frames}:raise ValueError('BUILD public15packet set incomplete')
    old={(x['arm'],int(x['raw'])):x for x in E.read(V.B/'public1920/export_r1/packets_manifest.json')['rows']}
    plan=E.read(Path(__file__).parent/'plan_template.json');plan.update(protocol=str(protocol),protocol_sha256=E.sha(protocol),packets=[])
    source={str(x):E.sha(x)for x in [protocol,mp,rp,vp,root/'common/initial_ci.npz']};checks=[]
    endpoints={'G_BUILD_FIELD':root/'field/formal_r1/receipt.json','G_BUILD_NUMERIC':root/'numerical/formal_r1/receipt.json'}
    permission=E.load(root/'common/initial_ci.npz',['active_world_rows','eligible_active'])
    readonly=permission['active_world_rows'][~permission['eligible_active']]
    original=E.load(V.Q.parent/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded/road.npz')
    immutable={}
    for arm in arms:
        asset=Path(rows[arm,86]['asset_dir']);road=E.load(asset/'road.npz')
        nonRGB={k:(x.shape==road[k].shape and x.dtype==road[k].dtype and x.tobytes()==road[k].tobytes())for k,x in original.items()if k!='reference_srgb'}
        rgb=bool(np.array_equal(original['reference_srgb'][readonly],road['reference_srgb'][readonly]))
        if not all(nonRGB.values())or not rgb:raise ValueError('BUILD immutable geometry/readonly RGB differs')
        immutable[arm]=dict(nonRGB_exact=nonRGB,readonly_RGB_exact=rgb,asset_road_SHA256=E.sha(asset/'road.npz'))
    for arm in arms:
        if arm!='F0':
            p=endpoints[arm];ep=E.read(p);V.validate_build_endpoint(ep,arm,cfg);source[str(p)]=E.sha(p)
        for raw in frames:
            r=rows[arm,raw]
            if r['resolution']!=[1920,1280]or E.sha(r['path'])!=r['SHA256']:raise ValueError('BUILD native packet binding differs')
            if r.get('authority')and E.sha(r['authority'])!=r['authority_SHA256']:raise ValueError('BUILD authority changed')
            item=dict(arm=arm,raw=raw,path=r['path'],baseline=old['F0',raw]['path'])
            if arm=='F0':
                if r['SHA256']!=old['F0',raw]['SHA256']:raise ValueError('F0 reuse packet differs from original actualF0')
            else:item['endpoint_receipt']=str(endpoints[arm])
            v=vrows[arm,raw]
            if v['available']:
                if raw not in[86,110]or not v['alpha_exact']or not v['K_exact']or E.sha(v['path'])!=v['SHA256']:raise ValueError('BUILD visibility source mismatch')
                item.update(visibility_sidecar=v['path'],frozen_visibility_reuse_confirmed=True)
                # Road readonly is qualified by unchanged scope/rowcolors/geometry,
                # completed public freeze proof and exact same initialci authority.
                basis=V.B.parent/f'quality_bench_20261120_r1/highres/native_r1/{raw:03d}/road_basis.npz'
                item.update(readonly_road_basis=str(basis),readonly_basis_reuse_confirmed=True);source[str(basis)]=E.sha(basis)
            elif raw in[86,110]:raise ValueError('BUILD train visibility unexpectedly unavailable')
            plan['packets'].append(item)
    for arm,p in endpoints.items():
        ep=E.read(p)
        for key in ['source_freeze','source_binding']:
            if ep.get(key):
                sp=Path(ep[key]);source[str(sp)]=E.sha(sp);s=E.read(sp)
                for q,h in s.get('binding',{}).items():
                    actual=E.sha(q)
                    if actual!=h:raise ValueError('BUILD frozen source changed: '+q)
                    checks.append(dict(path=q,SHA256=h,passed=True))
    plan['source_binding_receipt']=str(root/'qa/evaluation_binding_r1.json')
    pp=root/'qa/evaluation_plan_r1.json';bp=root/'qa/evaluation_binding_r1.json'
    if pp.exists()or bp.exists():raise FileExistsError('preserve BUILD evaluation plan')
    E.write(pp,plan);E.write(bp,dict(schema='BUILD-evaluation-source-binding.v1',passed=True,initial_ci=initial,immutable_assets=immutable,source_bindings=source,source_checks=checks,
        plan_SHA256=E.sha(pp),data_role=V.LABEL,old_QUERY_results_modified=False,GPU_executed=False,native_calls=0,optimizer_updates=0))
    print(json.dumps(dict(plan=str(pp),packets=15,initial_ci_exact=True)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=V.BUILD);a=p.parse_args();bind(a.root)
