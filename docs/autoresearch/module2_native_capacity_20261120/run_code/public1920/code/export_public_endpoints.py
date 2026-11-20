"""Independent actual-public1920 exporter. Defaults to CPU inspection only.

Only F0 and the four registered ordinary-GS endpoints are accepted. No pixel
shader, optimization, source modification, lease creation or power operation.
"""
from pathlib import Path
import argparse,copy,gc,hashlib,json,sys,time,traceback
import numpy as np
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=C/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119';B=G/'native_capacity_20261120_r1'
FRAMES=(86,110,134,179,198);ARMS=('F0','G_RGB','G_DETAIL','G_ORACLE','IDEAL_ORACLE')

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for a in iter(lambda:f.read(1024*1024),b''):h.update(a)
    return h.hexdigest()
def array_sha(x):
    a=np.ascontiguousarray(x);return hashlib.sha256(str((a.shape,a.dtype)).encode()+a.tobytes()).hexdigest()


def validate_asset(parent,asset):
    """All non-RGB road fields and original prior ledger bytes remain frozen."""
    parent=Path(parent);asset=Path(asset)
    with np.load(parent/'road.npz',allow_pickle=False)as p,np.load(asset/'road.npz',allow_pickle=False)as a:
        if set(p.files)!=set(a.files):raise ValueError('road field set changed')
        hashes={}
        for key in p.files:
            if key=='reference_srgb':continue
            hp=array_sha(p[key]);ha=array_sha(a[key]);hashes[key]=hp
            if hp!=ha:raise ValueError('non-RGB road field changed: '+key)
    changed_effective=[];ledger_hashes={}
    with np.load(parent/'provenance.npz',allow_pickle=False)as p,np.load(asset/'provenance.npz',allow_pickle=False)as a:
        for key in p.files:
            if key not in a.files:raise ValueError('original ledger field missing: '+key)
            if key in('effective_row_rgb','asset_mode'):
                if array_sha(p[key])!=array_sha(a[key]):changed_effective.append(key)
                continue
            ledger_hashes[key]=array_sha(p[key])
            if ledger_hashes[key]!=array_sha(a[key]):raise ValueError('original prior ledger changed: '+key)
    return dict(nonRGB_road_SHA256=hashes,all_original_prior_ledger_bytes_exact=True,
        immutable_original_ledger_SHA256=ledger_hashes,allowed_effective_metadata_changes=changed_effective)


def inspect(formal):
    formal=Path(formal);rows=[]
    for arm in ARMS[1:]:
        p=formal/arm/'endpoint.json'
        if not p.exists():raise FileNotFoundError('formal endpoint not yet ready: '+str(p))
        e=read(p)
        if e.get('completed')is not True or e['arm']!=arm:raise ValueError('endpoint incomplete or identity changed')
        for key in('asset_dir','authority','source_bindings','common_initial_ci'):
            if not Path(e[key]).exists():raise FileNotFoundError(e[key])
        rows.append(dict(arm=arm,endpoint=str(p),endpoint_SHA256=file_sha(p),asset_dir=e['asset_dir'],authority=e['authority'],
            authority_SHA256=file_sha(e['authority']),source_bindings=e['source_bindings'],source_bindings_SHA256=file_sha(e['source_bindings'])))
    for key in('common_initial_ci.npz','common_initial_field.pt'):
        if not(formal/key).exists():raise FileNotFoundError(formal/key)
    return dict(schema='native_capacity_public1920_plan.v1',formal_root=str(formal),frames=list(FRAMES),arms=list(ARMS),endpoints=rows,
        F0_ci=str(formal/'common_initial_ci.npz'),F0_field=str(formal/'common_initial_field.pt'),
        renderer='one RGB per physical Gaussian, actual public GS raster, never pixel-world shader',
        maximum_native_forwards=100,optimizer_updates=0,expected_native_forwards_if_all_controller_caches_ready=82,
        output_packet_keys=['road_rgb','road_alpha','joint_rgb','joint_alpha','K'],RGB_layout='CHW encoded-sRGB; road is black-background premultiplied P',
        teacher_policy='external original highres targets; never multiply teacher preview by rendered alpha')


def execute(plan,out,lease,owner,authorization):
    out=Path(out)
    if out.exists():raise FileExistsError('preserve prior export directory: '+str(out))
    auth=read(authorization)
    if auth.get('GPU_authorized')is not True or auth.get('optimizer_updates')!=0 or int(auth.get('max_native_forwards',0))<100:
        raise RuntimeError('explicit zero-update100-forward public export authorization required')
    lease_state=read(lease)
    if lease_state.get('active')is not True or lease_state.get('owner')!=owner or lease_state.get('gpu_authorized')is not True:
        raise RuntimeError('new caller-owned active GPU lease required before importing CUDA')
    out.mkdir(parents=True);write(out/'plan.json',plan);ctx=None;started=time.monotonic();ledger=[];manifest=[];checks=[];frozen_assets={}
    try:
        import torch
        torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
        sys.path[:0]=[str(B/'code'),str(B/'shared_field/code'),str(B/'gs_oracle/code')]
        from native_capacity_inputs import make_context,field_and_reader,ASSET
        from gs_color_oracle import BoundedGSColors
        ctx=make_context(lease,owner,frames=FRAMES);initial_model,reader,_=field_and_reader(ctx.base_rgb,'cuda')
        from road_capacity_native import NativeRoadContext
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
        formal=Path(plan['formal_root']);source_bindings={str(Path(__file__)):file_sha(__file__),str(authorization):file_sha(authorization)}
        for e in plan['endpoints']:
            if file_sha(e['endpoint'])!=e['endpoint_SHA256']or file_sha(e['authority'])!=e['authority_SHA256']:raise RuntimeError('endpoint source changed after inspect')
            source_bindings[e['endpoint']]=e['endpoint_SHA256'];source_bindings[e['authority']]=e['authority_SHA256']
        def call(label,arm,raw,fn):
            if len(ledger)>=100:raise RuntimeError('100-forward export cap')
            ctx.require_lease();t=time.monotonic();v=fn();torch.cuda.synchronize()
            ledger.append(dict(label=label,arm=arm,raw=raw,seconds=time.monotonic()-t));return v
        scope=np.zeros(len(ctx.road['active']),bool);scope[ctx.active_world_rows]=reader.eligible
        readonly=torch.as_tensor(~reader.eligible,device='cuda');parent_joint={}
        with torch.no_grad():
            # Reuse already-computed public parent scene KRGB only if available
            # in the cache? F0 itself is a scope-safe parent for all endpoints,
            # because K color contribution of every eligible Gaussian is zero.
            F0,_=type(initial_model).load(plan['F0_field'],device='cuda');ci0=reader.colors(F0).detach()
            f0_asset=ctx.save_asset(ci0,scope,out/'F0_asset',training_record={'role':'derived_common_F0_export_only','authority':plan['F0_field'],'ordinary_GS_primary':True})
        for arm in ARMS:
            if arm=='F0':asset=f0_asset;authority=plan['F0_field'];color=ci0;ep=None
            else:
                e=next(e for e in plan['endpoints']if e['arm']==arm);ep=read(e['endpoint']);asset=Path(e['asset_dir']);authority=e['authority']
                if arm in('G_RGB','G_DETAIL'):
                    field,_=type(initial_model).load(authority,device='cuda')
                    with torch.no_grad():color=reader.colors(field).detach()
                else:
                    oracle=BoundedGSColors.load(authority,device='cuda')
                    if not np.array_equal(oracle.layout.active_world_rows,ctx.active_world_rows)or not np.array_equal(oracle.layout.eligible_active,reader.eligible):raise RuntimeError('oracle scope differs')
                    with torch.no_grad():color=oracle().detach()
            frozen_assets[arm]=validate_asset(ASSET,asset)
            with np.load(asset/'road.npz',allow_pickle=False)as z:stored=z['reference_srgb'][ctx.active_world_rows].copy()
            ci_error=float(np.max(abs(color.cpu().numpy()-stored)))
            if ci_error>1e-6 or not torch.equal(color[readonly],ctx.base_rgb[readonly]):raise RuntimeError('authority/asset ci or readonly changed')
            scene,m,loader=load_persistent_handoff(asset/'persistent/scene.json',asset/'persistent/proof.json')
            data=dict(m.data);data['render_wh']=[1920,1280];headers=SceneManifest(data,asset/'persistent').camera_headers();cameras={}
            def camera(raw):
                if raw not in cameras:cameras[raw]=make_native_camera(headers[raw])
                return cameras[raw]
            scene.camera_provider=camera;public=NativeRoadContext(scene,m,loader,asset,FRAMES,lease,owner)
            folder=out/'rendered'/arm;folder.mkdir(parents=True)
            with torch.no_grad():
                for raw in FRAMES:
                    cache=formal/('common_initial'if arm=='F0'else arm+'/rendered')/f'{raw:03d}_native.npz'
                    cached={}
                    if cache.exists():
                        with np.load(cache,allow_pickle=False)as z:cached={k:z[k].copy()for k in('road_rgb','road_alpha','joint_rgb','joint_alpha')if k in z.files}
                    expected={};cacheused={}
                    for view in('road','joint'):
                        if view+'_rgb'in cached and view+'_alpha'in cached:
                            expected[view]={'rgb':torch.as_tensor(cached[view+'_rgb'],device='cuda'),'alpha':torch.as_tensor(cached[view+'_alpha'],device='cuda')};cacheused[view]=True
                        else:
                            expected[view]=call('source_authority_'+view,arm,raw,lambda view=view:ctx.render_road(raw,color)if view=='road'else ctx.render_joint(raw,color));cacheused[view]=False
                    road=call('public_road',arm,raw,lambda:public.render_road(raw));joint=call('public_joint',arm,raw,lambda:public.render_joint(raw))
                    K=public.K_masks[raw]
                    if not np.array_equal(K,ctx.K_masks[raw]):raise RuntimeError('public native K source changed')
                    row=dict(arm=arm,raw=raw,authority_to_asset_ci_max=ci_error,expected_cache=cacheused,cache=str(cache)if cached else None)
                    for view,v in(('road',road),('joint',joint)):
                        ref=expected[view];rgb=float((v['rgb']-ref['rgb']).abs().max());alpha=float((v['alpha'].squeeze()-ref['alpha'].squeeze()).abs().max())
                        row[view+'_RGB_max']=rgb;row[view+'_alpha_max']=alpha;row[view+'_RGB8_max']=int((torch.round(v['rgb']*255)-torch.round(ref['rgb']*255)).abs().max())
                        if rgb>1e-6 or alpha!=0:raise RuntimeError('public/source mismatch: '+str(row))
                    if arm=='F0':parent_joint[raw]=joint['rgb'].detach().clone()
                    row['K_pixels']=int(K.sum());row['joint_K_RGB_max']=float((joint['rgb']-parent_joint[raw]).abs().amax(0)[torch.as_tensor(K,device='cuda')].max())
                    packet=dict(road_rgb=road['rgb'].cpu().numpy(),road_alpha=road['alpha'].squeeze().cpu().numpy(),
                        joint_rgb=joint['rgb'].cpu().numpy(),joint_alpha=joint['alpha'].squeeze().cpu().numpy(),K=K)
                    path=folder/f'{raw:03d}_native.npz';np.savez_compressed(path,**packet)
                    manifest.append(dict(arm=arm,raw=raw,path=str(path),keys=list(packet),resolution=[1920,1280],RGB_layout='CHW',alpha_layout='HW',
                        SHA256=file_sha(path),asset_dir=str(asset),authority=str(authority),authority_SHA256=file_sha(authority),
                        renderer='actual_public_ordinary_GS',visibility_sidecar=None,visibility_note='no new contribution-feature pass; same-sort sidecar may be bound independently, otherwise N/A',
                        teacher_GT_available=raw in(86,110),teacher_role='frozen pseudo C/S and real A, not hiddenGT'if raw in(86,110)else'diagnostic only, no teacher'))
                    checks.append(row);print('EXPORTED',arm,raw,flush=True)
                    del road,joint,expected,cached;gc.collect()
            scene.assert_persistent_unchanged();public.close();del public,scene,color;gc.collect();torch.cuda.empty_cache()
        ctx.assert_frozen()
        receipt=dict(schema='native_capacity_actual_public1920_export.v1',complete=True,optimizer_updates=0,native_forward_calls=len(ledger),
            native_ledger=ledger,checks=checks,all_K_exact_zero=all(r['joint_K_RGB_max']==0 for r in checks),
            source_bindings=source_bindings,frozen_assets=frozen_assets,frames=list(FRAMES),arms=list(ARMS),seconds=time.monotonic()-started,
            GPU_keep_on=True,production_assets_modified=False,promotion=False,human_verdict=None,
            K_failure_policy='reported on complete unmodified fixed mask, no screen copy or scope shrink; root decides acceptance',
            frozen_K_qualification=str(B/'smoke_r1/K_qualification/receipt.json'),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(out/'packets_manifest.json',dict(schema='native_capacity_public_packets.v1',rows=manifest));write(out/'receipt.json',receipt)
    except Exception:
        write(out/'failed.json',dict(error=traceback.format_exc(),native_ledger=ledger,manifest_rows=manifest,seconds=time.monotonic()-started));raise
    finally:
        if ctx is not None:ctx.close()
        lease_state=read(lease)
        if lease_state.get('owner')==owner:
            lease_state.update(active=False,owner=None,ended_at=time.time());write(lease,lease_state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--formal-root',required=True);p.add_argument('--out',required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--lease-file');p.add_argument('--owner');p.add_argument('--authorization');a=p.parse_args();plan=inspect(a.formal_root)
    if not a.execute:print(json.dumps(plan,ensure_ascii=False,indent=2));return
    if not all((a.lease_file,a.owner,a.authorization)):p.error('explicit new lease/owner/authorization required')
    execute(plan,a.out,a.lease_file,a.owner,a.authorization)
if __name__=='__main__':main()
