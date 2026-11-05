"""只补缺失的49帧原生RGB；不拟合、不改资产、相机、renderer或背景参数。"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib,inspect,json,shutil,sys,tarfile,time
from pathlib import Path
import numpy as np
from PIL import Image

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ART=CAM/'scene008_a6_insert_artifacts_20261105'
CONTROLS=CAM/'scene008_a6_asset_controls_20261105'
OLD=CAM/'scene008_a6_appearance_formal_20261105/output_adapter/projective_surface/formal_observed/formal/complete_a6.npz'
NEW=ART/'surface_fit/combined_v1/observed_fit/formal/complete_a6.npz'
POST=ART/'surface_fit/masked_depth_v1/posterior.npz'
DEST=ART/'frame_compare/native49_v1'
LEGACY=ART/'scale/surface_lod_v1/evaluation49'
SO_SHA='0eaff4d7a9b8ef6e711627f4c19824bf427c4387f7c6453fcb769a2c919c920e'
EXPECTED={str(OLD):'70826fc7c9c011ee073f7ffea4f22501a57f39ffa2cd0d52cd5ebd9c37b14e53',
    str(NEW):'0b1f69e13d0e54328734325c8f118bc27dc7b75080869f2d52f55038afc0b20d',
    str(POST):'b98a0d932730fbf1462e555eb7d5eb931030b51fe68848990f073b9752879e06'}
REUSE=(105,106,113,130)
METHODS=('old_formal','c2_matte','surface_lod')
ALIASES=dict(old_formal='old_native',c2_matte='final_native',surface_lod='surface_LOD')
STATES=('keep','delete')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def png(p,a):
    if a.dtype!=np.uint8:a=np.uint8(np.clip(a,0,1)*255+.5)
    assert a.shape==(320,480,3)
    Image.fromarray(a).save(p)
def descriptor(path,raw,state,method,origin,asset=None):
    with Image.open(path) as im:assert im.size==(480,320) and im.mode=='RGB';im.load()
    row=dict(path=path.relative_to(DEST).as_posix(),sha256=sha(path),width=480,height=320,raw=raw,state=state,method=method,origin=origin)
    if asset:row['asset_sha256']=EXPECTED[str(asset)]
    return row

def crop_from_observation(obs):
    # 原有标注框投影足迹包含amodal范围；不拿某个方法alpha挑窗口。
    m=(obs['box_near_m'][0]>0)|obs['visible_reviewed'][0].astype(bool)
    y,x=np.where(m)
    if not len(x):return [0,0,480,320],dict(source='Empty original a6 reference footprint; full viewport fallback',padding_px=16)
    crop=[max(0,int(x.min())-16),max(0,int(y.min())-16),min(480,int(x.max())+17),min(320,int(y.max())+17)]
    return crop,dict(source='Original OBS a6 padded-box ray footprint union original reviewed a6 silhouette',
        padding_px=16,reference_pixels=int(m.sum()),predicted_alpha_used=False,display_only=True,
        same_crop_for_all_methods_states_and_reference=True,shadow_judgment_uses_full_street=True)

def main():
    DEST.mkdir(parents=True,exist_ok=False);start=time.time()
    for p,h in EXPECTED.items():assert sha(p)==h,p
    old_receipt=read(LEGACY/'receipt.json');legacy_lod=read(ART/'scale/surface_lod_v1/receipt.json')
    proof=read(ART/'scale/receipt.json')['binding']['files']
    for p,h in old_receipt['source_files_sha256'].items():assert sha(p)==h,p
    for p,h in proof.items():assert sha(p)==h,p
    for p,h in legacy_lod['sources'].items():assert sha(p)==h,p
    archive=ART/'scale/surface_lod_v1/evaluation49_delivery_20261105.tgz'
    assert sha(archive)=='e0a175ecf6bcb7342a03dfec5903e9f60347d0cc58fbf0be2226e47a6ef4ae65'
    original_png_hashes={}
    with tarfile.open(archive) as t:
        for raw in REUSE:
            for state in STATES:
                for method in METHODS:
                    name=f'raw{raw}/{ALIASES[method]}_{state}480.png'
                    h=digest(t.extractfile('evaluation49/'+name).read());assert sha(LEGACY/name)==h
                    original_png_hashes[name]=h
    sys.path.insert(0,str(CONTROLS/'source'));sys.path.insert(0,str(ART/'source/scale'))
    import torch
    import scene_runtime,sh_asset,editor_common,diff_gaussian_rasterization
    from diff_gaussian_rasterization import _C
    from scene_runtime import Runtime,raster
    from sh_asset import SHAsset
    from surface_lod_asset import SurfaceLODAsset
    # 与另一独立a5任务共存；仅限制本进程allocator，不改变渲染数值。
    torch.cuda.set_per_process_memory_fraction(.40)
    renderer_files={str(Path(p).resolve()):sha(p) for p in [scene_runtime.__file__,sh_asset.__file__,editor_common.__file__,
        inspect.getsourcefile(raster),diff_gaussian_rasterization.__file__,_C.__file__]}
    assert renderer_files[str(Path(_C.__file__).resolve())]==SO_SHA
    renderer_id=digest(json.dumps(renderer_files,sort_keys=True,separators=(',',':')).encode())
    rt=Runtime();assert json.loads(json.dumps(rt.meta))==old_receipt['runtime']
    old=SHAsset.from_geometry(OLD).cuda();new=SHAsset.from_geometry(NEW).cuda();lod=SurfaceLODAsset(new,POST)
    assert lod.persistent_SF_hash==old_receipt['persistent_SF_hash']
    assets=dict(old_formal=old,c2_matte=new,surface_lod=lod)
    columns=[dict(id='old_formal',label='旧 formal',asset=str(OLD),asset_sha256=EXPECTED[str(OLD)]),
        dict(id='c2_matte',label='新 C2 + matte',asset=str(NEW),asset_sha256=EXPECTED[str(NEW)]),
        dict(id='surface_lod',label='同 S/F SurfaceLOD',asset=str(NEW),asset_sha256=EXPECTED[str(NEW)],
            posterior_sha256=EXPECTED[str(POST)],persistent_SF_hash=lod.persistent_SF_hash,
            wrapper_sha256=sha(ART/'source/scale/surface_lod_asset.py'))]
    frames=[];native_calls=0;conservation=0.
    with torch.no_grad():
        # 113先落盘，方便UI实现；最终manifest仍按原始raw顺序输出。
        for raw in [113]+[r for r in range(89,138) if r!=113]:
            d=rt.raw(raw);assert d['cam'].image_width==480 and d['cam'].image_height==320
            outdir=DEST/'frames'/f'{raw:03d}';outdir.mkdir(parents=True)
            crop,crop_provenance=crop_from_observation(d['obs'])
            record=dict(raw=raw,crop_xyxy=crop,crop_provenance=crop_provenance,
                observation_sha256=sha(CONTROLS.parent/'scene008_instances_20261105_r1/observations/frames'/f'{raw:03d}.npz'),
                camera=dict(width=480,height=320,fx=float(d['cam'].fx),fy=float(d['cam'].fy),cx=float(d['cam'].cx),cy=float(d['cam'].cy),
                    FoVx=float(d['cam'].FoVx),FoVy=float(d['cam'].FoVy),timestamp=float(d['cam'].timestamp),
                    world_view=d['cam'].world_view_transform.cpu().numpy().tolist(),full_projection=d['cam'].full_proj_transform.cpu().numpy().tolist()),keep={},delete={})
            real=np.asarray(d['obs']['rgb']);png(outdir/'reference.png',real)
            record['reference']=descriptor(outdir/'reference.png',raw,'observed_a37_present','original_RGB','Original fixed OBS rgb')
            record['reference']['label']='真实480 RGB：a37仍在，非delete GT'
            for state in STATES:
                folder=outdir/state;folder.mkdir()
                for method,asset in assets.items():
                    p=folder/(method+'.png')
                    if raw in REUSE:
                        source=LEGACY/f'raw{raw}'/(ALIASES[method]+'_'+state+'480.png')
                        shutil.copy2(source,p)
                        assert sha(p)==original_png_hashes[f'raw{raw}/{source.name}']
                        row=descriptor(p,raw,state,method,'Byte-exact prior fixed49 render',OLD if method=='old_formal' else NEW)
                        row['reused_source']=str(source);row['legacy_receipt_sha256']=sha(LEGACY/'receipt.json')
                    else:
                        result=rt.render_scene(raw,asset,delete=state=='delete');native_calls+=1
                        assert result['meta']['native_a6_rows_input']==0
                        if state=='delete':assert result['meta']['native_a37_packed_rows']==0
                        conservation=max(conservation,float((result['component_alpha_T'].sum(0)-result['alpha'][0]).abs().max()))
                        image=result['rgb'].cpu().numpy().transpose(1,2,0);assert np.isfinite(image).all();png(p,image)
                        row=descriptor(p,raw,state,method,'New actual native480 render',OLD if method=='old_formal' else NEW)
                        row['native_meta']=result['meta'];del result,image
                    row['renderer_id']=renderer_id;record[state][method]=row
            record['lod_sampling']=lod.sampling(d['pose'],d['cam'])[1]
            write(outdir/'metadata.json',record);frames.append(record)
            if raw==113:
                write(DEST/'schema_preview.json',dict(status='PARTIAL_UI_SCHEMA_ONLY',raw_ids=list(range(89,138)),states=list(STATES),
                    columns=columns,frames=[record],viewport=dict(width=480,height=320),default_raw=113,default_state='delete'))
            write(DEST/'progress.json',dict(completed_raws=sorted(x['raw'] for x in frames),native_raster_forwards=native_calls))
            print(json.dumps(dict(stage='native49_RGB_capture',raw=raw,completed=len(frames),reused=raw in REUSE)),flush=True)
            del d;rt._data.clear();torch.cuda.empty_cache()
    assert native_calls==270 and len(frames)==49 and conservation<2e-5
    assert rt.modelunchanged() and old.assert_geometry_unchanged() and lod.assert_persistent_unchanged()
    for p,h in EXPECTED.items():assert sha(p)==h
    for p,h in renderer_files.items():assert sha(p)==h
    manifest=dict(schema='scene008.native49.same_camera_RGB.v1',status='COMPLETE',raw_ids=list(range(89,138)),states=list(STATES),
        columns=columns,frames=sorted(frames,key=lambda x:x['raw']),viewport=dict(width=480,height=320),default_raw=113,default_state='delete',
        images=294,references=49,reference_semantics='Original camera RGB with a37 present; never delete-state ground truth',
        renderer=dict(id=renderer_id,files_sha256=renderer_files,native_neg_fov=bool(rt.pipe.neg_fov),
            background_parameters=rt.meta,native_kernel_or_camera_or_environment_updated=False),
        source_sha256=sha(__file__),asset_files_sha256=EXPECTED,
        reuse_provenance=dict(raws=list(REUSE),prior_receipt=str(LEGACY/'receipt.json'),prior_receipt_sha256=sha(LEGACY/'receipt.json'),
            prior_archive_sha256=sha(archive),original_PNG_bytes_verified_against_prior_archive=True,
            SO_runtime_loader_and_assets_match_prior_recorded_hashes=True,frozen_runtime_meta_exact=True),
        no_interpolation_or_2D_frame_synthesis=True,no_neural_colour_or_geometry_fit_updates=True,
        current_surface_geometry_is_research_candidate=True,HELD_previous_development_role_preserved=True)
    write(DEST/'manifest.json',manifest)
    receipt=dict(status='PASS_FIXED49_RGB_CAPTURE',frames=49,methods=list(METHODS),states=list(STATES),PNG_rendered_or_reused=294,reference_PNGs=49,
        newly_rendered_raws=[r for r in range(89,138) if r not in REUSE],new_native_raster_forwards=native_calls,reused_PNGs=24,
        no_asset_parameters_changed=True,no_training_or_colour_fit_updates=True,manifest_sha256=sha(DEST/'manifest.json'),
        group_conservation_maxabs=conservation,torch_allocator_cap_fraction=.40,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        model_and_renderer_unchanged=True,other_jobs_or_services_stopped=False,shutdown_performed=False,
        seconds=time.time()-start,source_sha256=sha(__file__))
    write(DEST/'receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
