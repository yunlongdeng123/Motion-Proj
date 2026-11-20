"""All-label BUILD public export; all old experiment outputs stay immutable.

Old seen and held mapped labels are both BUILD_fit supervision in this branch.
No held-generalization or representation-ceiling claim is made here.
"""
from pathlib import Path
import argparse,gc,json,sys,time,traceback
import numpy as np

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=C/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119';B=G/'native_capacity_20261120_r1'
sys.path.insert(0,str(B/'public1920/code'))
from export_public_endpoints import read,write,file_sha,validate_asset
FRAMES=(86,110,134,179,198);F0ROOT=B/'public1920/export_r1'


def inspect(asset,authority,completion,arm,field_authority=None):
    if arm not in('G_BUILD_FIELD','G_BUILD_NUMERIC'):raise ValueError('registered BUILD arm required')
    asset=Path(asset);authority=Path(authority);completion=Path(completion);result=read(completion)
    if result.get('complete')is not True and result.get('completed')is not True:raise RuntimeError('numeric work not completed')
    for p in(asset/'road.npz',asset/'provenance.npz',asset/'persistent/scene.json',asset/'persistent/proof.json',authority):
        if not p.exists():raise FileNotFoundError(p)
    base=read(F0ROOT/'receipt.json')
    if not base['complete']or not base['all_K_exact_zero']:raise RuntimeError('F0 baseline not qualified')
    if arm=='G_BUILD_FIELD'and(field_authority is None or not Path(field_authority).exists()):raise FileNotFoundError('FIELD field.pt authority required')
    return dict(schema='all_teacher_BUILD_public1920_plan.v1',arm=arm,asset_dir=str(asset),authority=str(authority),
        authority_SHA256=file_sha(authority),completion=str(completion),completion_SHA256=file_sha(completion),
        F0_manifest=str(F0ROOT/'packets_manifest.json'),F0_manifest_SHA256=file_sha(F0ROOT/'packets_manifest.json'),
        frames=list(FRAMES),nativeF_expected=20,nativeF_cap=40,two_arms_total_nativeF_cap=80,optimizer_updates=0,
        field_authority=str(field_authority)if field_authority else None,field_authority_SHA256=file_sha(field_authority)if field_authority else None,
        training_namespace=str(B/'all_teacher_build_20261120_r1'),training_role='BUILD_fit_all_qualified',held_generalization_claim=False,
        renderer='actual ordinary GS; shared-field-derived ci or independent bounded ci, never pixel shader')


def execute(plan,out,lease,owner,authorization,*,release_lease=True):
    out=Path(out);auth=read(authorization);ls=read(lease)
    if auth.get('GPU_authorized')is not True or auth.get('optimizer_updates')!=0 or auth.get('max_native_forwards')!=80:
        raise RuntimeError('explicit two-arm80F/0update authorization required')
    if ls.get('active')is not True or ls.get('owner')!=owner or ls.get('gpu_authorized')is not True:raise RuntimeError('new caller-owned lease required before CUDA')
    if out.exists():raise FileExistsError('preserve old numeric public output')
    out.mkdir(parents=True);write(out/'plan.json',plan);ctx=None;public=None;ledger=[];rows=[];checks=[];visibility_rows=[];start=time.monotonic();arm=plan['arm']
    try:
        import torch
        torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
        sys.path[:0]=[str(B/'code'),str(B/'gs_oracle/code')]
        from native_capacity_inputs import make_context,ASSET
        from gs_color_oracle import BoundedGSColors
        ctx=make_context(lease,owner,frames=FRAMES)
        from road_capacity_native import NativeRoadContext
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
        asset=Path(plan['asset_dir']);frozen=validate_asset(ASSET,asset)
        if file_sha(plan['authority'])!=plan['authority_SHA256']or file_sha(plan['completion'])!=plan['completion_SHA256']:raise RuntimeError('numeric authority/completion changed')
        oracle=BoundedGSColors.load(plan['authority'],device='cuda')
        with np.load(Q/'permission/final_r1/teacher_evidence.npz',allow_pickle=False)as z:active=z['active_world_rows'];eligible=z['common_write_active']
        if not np.array_equal(active,ctx.active_world_rows)or not np.array_equal(oracle.layout.active_world_rows,active)or not np.array_equal(oracle.layout.eligible_active,eligible):raise RuntimeError('numeric scope changed')
        with torch.no_grad():colors=oracle().detach()
        readonly=torch.as_tensor(~eligible,device='cuda')
        if not torch.equal(colors[readonly],ctx.base_rgb[readonly]):raise RuntimeError('readonly numeric rows changed')
        with np.load(asset/'road.npz',allow_pickle=False)as z:stored=z['reference_srgb'][active]
        ci_error=float(np.max(np.abs(colors.cpu().numpy()-stored)))
        if ci_error>1e-6:raise RuntimeError('numeric authority and effective saved ci differ')
        scene,m,loader=load_persistent_handoff(asset/'persistent/scene.json',asset/'persistent/proof.json')
        data=dict(m.data);data['render_wh']=[1920,1280];headers=SceneManifest(data,asset/'persistent').camera_headers();cameras={}
        parent_data=read(ASSET/'persistent/scene.json');parent_data['render_wh']=[1920,1280];ph=SceneManifest(parent_data,ASSET/'persistent').camera_headers()
        for raw in FRAMES:
            if not np.array_equal(headers[raw]['W'],ph[raw]['W'])or any(headers[raw][k]!=ph[raw][k]for k in('width','height','fx','fy','cx','cy','timestamp')):raise RuntimeError('numeric native camera differs')
        def camera(raw):
            if raw not in cameras:cameras[raw]=make_native_camera(headers[raw])
            return cameras[raw]
        scene.camera_provider=camera;public=NativeRoadContext(scene,m,loader,asset,FRAMES,lease,owner)
        def call(label,raw,fn):
            if len(ledger)>=40:raise RuntimeError('40F numeric public cap')
            ctx.require_lease();t=time.monotonic();v=fn();torch.cuda.synchronize();ledger.append(dict(label=label,raw=raw,seconds=time.monotonic()-t));return v
        folder=out/'rendered'/arm;folder.mkdir(parents=True)
        base_rows={r['raw']:r for r in read(F0ROOT/'packets_manifest.json')['rows']if r['arm']=='F0'}
        old_visibility=read(F0ROOT/'visibility_manifest_r1.json');visibility={r['raw']:r for r in old_visibility['rows']if r['arm']=='F0'}
        with torch.no_grad():
            for raw in FRAMES:
                er=call('authority_road',raw,lambda:ctx.render_road(raw,colors));ej=call('authority_joint',raw,lambda:ctx.render_joint(raw,colors))
                road=call('public_road',raw,lambda:public.render_road(raw));joint=call('public_joint',raw,lambda:public.render_joint(raw))
                check=dict(arm=arm,raw=raw,authority_to_asset_ci_max=ci_error)
                for view,actual,expected in(('road',road,er),('joint',joint,ej)):
                    check[view+'_RGB_max']=float((actual['rgb']-expected['rgb']).abs().max());check[view+'_alpha_max']=float((actual['alpha']-expected['alpha']).abs().max())
                    check[view+'_RGB8_max']=int((torch.round(actual['rgb']*255)-torch.round(expected['rgb']*255)).abs().max())
                    if check[view+'_RGB_max']>1e-6 or check[view+'_alpha_max']!=0:raise RuntimeError('numeric public/source mismatch')
                baseline=Path(base_rows[raw]['path'])
                if file_sha(baseline)!=base_rows[raw]['SHA256']:raise RuntimeError('F0 baseline changed')
                with np.load(baseline,allow_pickle=False)as f:
                    K=public.K_masks[raw]
                    if not np.array_equal(K,f['K']):raise RuntimeError('numeric K mask changed')
                    check['F0_road_alpha_exact']=np.array_equal(road['alpha'].squeeze().cpu().numpy(),f['road_alpha'])
                    check['F0_joint_alpha_exact']=np.array_equal(joint['alpha'].squeeze().cpu().numpy(),f['joint_alpha'])
                    check['joint_K_RGB_max']=float(np.max(np.abs(joint['rgb'].cpu().numpy()-f['joint_rgb'])[:,K]));check['K_pixels']=int(K.sum())
                if not(check['F0_road_alpha_exact']and check['F0_joint_alpha_exact']):raise RuntimeError('numeric coverage differs from F0')
                path=folder/f'{raw:03d}_native.npz';packet=dict(road_rgb=road['rgb'].cpu().numpy(),road_alpha=road['alpha'].squeeze().cpu().numpy(),
                    joint_rgb=joint['rgb'].cpu().numpy(),joint_alpha=joint['alpha'].squeeze().cpu().numpy(),K=K);np.savez_compressed(path,**packet)
                rows.append(dict(arm=arm,raw=raw,path=str(path),keys=list(packet),resolution=[1920,1280],RGB_layout='CHW',alpha_layout='HW',SHA256=file_sha(path),
                    authority=plan['authority'],authority_SHA256=plan['authority_SHA256'],asset_dir=str(asset),renderer='actual_public_ordinary_GS',
                    visibility_sidecar=None,teacher_GT_available=raw in(86,110),
                    teacher_role='real A and frozen pseudo C/S; not hidden-road truth'if raw in(86,110)else'diagnostic only; no teacher',
                    evaluation_role='BUILD_fit_all_qualified'if raw in(86,110)else'diagnostic_only',held_generalization_claim=False,
                    field_authority=plan['field_authority'],field_authority_SHA256=plan['field_authority_SHA256']))
                vr=dict(visibility[raw]);vr.update(arm=arm,F0_alpha_exact=True,nonRGB_geometry_ledger_frozen=True)
                visibility_rows.append(vr)
                checks.append(check);print('EXPORTED',arm,raw,flush=True)
        public.scene.assert_persistent_unchanged();public.assert_frozen();ctx.assert_frozen()
        receipt=dict(schema='all_teacher_BUILD_actual_public1920.v1',complete=True,arm=arm,optimizer_updates=0,native_forward_calls=len(ledger),native_ledger=ledger,
            checks=checks,all_K_exact_zero=all(x['joint_K_RGB_max']==0 for x in checks),frozen_asset=frozen,
            source_SHA256={str(Path(__file__)):file_sha(__file__),str(authorization):file_sha(authorization),plan['authority']:plan['authority_SHA256'],
                plan['completion']:plan['completion_SHA256'],plan['F0_manifest']:plan['F0_manifest_SHA256'],str(F0ROOT/'visibility_manifest_r1.json'):file_sha(F0ROOT/'visibility_manifest_r1.json')},
            visibility_source='same original scope/camera/geometry/opa/order and F0 alpha exact; no new native feature pass',
            seconds=time.monotonic()-start,GPU_keep_on=True,production_assets_modified=False,promotion=False,human_verdict=None,
            training_role='BUILD_fit_all_qualified',old_held_now_build_fit=True,held_generalization_claim=False,
            field_authority=plan['field_authority'],field_authority_SHA256=plan['field_authority_SHA256'],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(out/'packets_manifest.json',dict(schema='native_capacity_public_packets.v1',rows=rows))
        write(out/'visibility_manifest_r1.json',dict(schema='native_capacity_visibility_sidecars.v1',complete=True,CPU_only=True,new_native_calls=0,rows=visibility_rows,
            source=str(F0ROOT/'visibility_manifest_r1.json'),source_SHA256=file_sha(F0ROOT/'visibility_manifest_r1.json')))
        write(out/'receipt.json',receipt)
    except Exception:
        write(out/'failed.json',dict(error=traceback.format_exc(),native_ledger=ledger,rows=rows,seconds=time.monotonic()-start));raise
    finally:
        if public is not None:public.close()
        if ctx is not None:ctx.close()
        state=read(lease)
        if release_lease and state.get('owner')==owner:state.update(active=False,owner=None,ended_at=time.time());write(lease,state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--asset',required=True);p.add_argument('--authority',required=True);p.add_argument('--completion',required=True);p.add_argument('--out',required=True)
    p.add_argument('--arm',required=True,choices=['G_BUILD_FIELD','G_BUILD_NUMERIC']);p.add_argument('--field-authority')
    p.add_argument('--execute',action='store_true');p.add_argument('--lease');p.add_argument('--owner');p.add_argument('--authorization');a=p.parse_args()
    plan=inspect(a.asset,a.authority,a.completion,a.arm,a.field_authority)
    if not a.execute:print(json.dumps(plan,indent=2));return
    if not all((a.lease,a.owner,a.authorization)):p.error('new lease/owner/authorization required')
    execute(plan,a.out,a.lease,a.owner,a.authorization)
if __name__=='__main__':main()
