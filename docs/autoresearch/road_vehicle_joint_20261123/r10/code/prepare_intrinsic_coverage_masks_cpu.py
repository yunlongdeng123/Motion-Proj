"""Fixed local-only intrinsic-alpha coverage probe; original evaluation band is preserved."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from PIL import Image

SCHEMA='R10.intrinsic_local_coverage_edit_masks.v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def bank(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def checked(ref):assert sha(ref['path'])==ref['sha256'];return Path(ref['path'])
def validate_intrinsic(receipt,manifest,asset_path):
    assert receipt['complete'] and not receipt.get('error')
    assert receipt['actual_component']=='local_only' and receipt['intrinsic_alpha_from_actual_native_output'] is True
    assert receipt['joint_local_alphaT_not_used'] is True and receipt['target_free_loader'] is True and receipt['training_targets_read'] is False
    assert receipt['asset_manifest']['sha256']==sha(asset_path)
    assert receipt['nativeF']==len(manifest['frames']) and receipt['nativeB']==0 and receipt['updates']==0

def prepare_or_validate(a,cfg,manifest):
    asset_path=a.asset/'manifest.json';intrinsic_path=a.intrinsic/'receipt.json';intrinsic=read(intrinsic_path)
    validate_intrinsic(intrinsic,manifest,asset_path)
    surface_path=a.surface_review/'receipt.json';surface=read(surface_path)
    assert surface['complete'] and surface['geometry_sha256']==manifest['geometry']['npz']['sha256'] and surface['final_atlas_sha256']==manifest['source']['field']['sha256']
    cam_path=a.intrinsic/'camera_timestamp_audit.json';assert cam_path.is_file()
    rp=a.masks/'receipt.json'
    if not a.masks.exists():
        assert not getattr(a,'execute',False),'Prepare/freeze CPU masks before GPU measurement'
        a.masks.mkdir(parents=True);rows=[]
        for raw in cfg['frames']:
            source=next(r for r in intrinsic['rows'] if r['raw']==raw)
            ip=checked(source['render']);alpha=bank(ip)['alpha'];assert alpha.dtype==np.float32 and alpha.shape==(768,1152)
            assert np.isfinite(alpha).all() and alpha.min()>=0 and alpha.max()<=1
            if 'intrinsic_alpha' in source:
                ap=checked(source['intrinsic_alpha']);assert np.array_equal(np.load(ap,allow_pickle=False),alpha)
            ep=Path(cfg['fixed_eval'])/f'{raw:03d}/fixed_eval.npz';mp=Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz'
            ev=bank(ep);m=bank(mp);band=ev['full_original_band'].astype(bool);K=ev['K_protected'].astype(bool)
            assert np.array_equal(K,m['protected'])
            if raw in [86,110]:car=m['actor_visible'].astype(bool);kind='opened_strict_real_V'
            else:car=m['actor_visible_proxy'].astype(bool);kind='actor_visible_proxy_no_heldout_V_read'
            positive=alpha>0;covered=alpha>=.95;inside=(band|positive)&~K&~car;replacement=inside&covered
            yy,xx=np.nonzero(replacement);assert len(xx)
            pick=np.argmin((xx-(replacement.shape[1]-1)/2)**2+(yy-(replacement.shape[0]-1)/2)**2)
            sentinel=np.zeros_like(replacement);sentinel[yy[pick],xx[pick]]=True
            out=a.masks/f'{raw:03d}';out.mkdir()
            np.savez_compressed(out/'scope_masks.npz',full_image=np.ones_like(inside),outside_allowed_edit=~inside,actor_protected=car,K_protected=K,
                inside_allowed_edit=inside,full_original_band=band,original_band_eval_domain=band&~K&~car,
                intrinsic_alpha_positive=positive,intrinsic_coverage_ge95=covered,outside_replacement=~replacement,
                inside_replacement=replacement,single_pixel_sentinel=sentinel,original_band_without_intrinsic_coverage=band&~covered)
            preview=np.zeros((*inside.shape,3),np.uint8);preview[band]=[75,75,90];preview[inside]=[50,125,160];preview[replacement]=[30,185,100];preview[car]=[70,100,230];preview[K]=[225,65,90]
            Image.fromarray(preview).save(out/'intrinsic_coverage_edit_domain.png')
            sp=a.surface_review/f'{raw:03d}/surface_review.npz'
            rows.append(dict(raw=raw,car_mask_kind=kind,masks=pin(out/'scope_masks.npz'),fixed_eval=pin(ep),rectified_masks=pin(mp),direct_surface_projection=pin(sp),
                intrinsic_alpha_npz=pin(ip),full_original_band_pixels=int(band.sum()),inside_allowed_edit_pixels=int(inside.sum()),outside_allowed_edit_pixels=int((~inside).sum()),
                actor_protected_pixels=int(car.sum()),K_pixels=int(K.sum()),inside_replacement_pixels=int(replacement.sum()),outside_replacement_pixels=int((~replacement).sum()),
                intrinsic_positive_pixels=int(positive.sum()),intrinsic_ge95_pixels=int(covered.sum()),original_band_without_intrinsic_coverage_pixels=int((band&~covered).sum()),
                sentinel_xy=[int(xx[pick]),int(yy[pick])],sentinel_selection='nearest image centre insideReplacement, row-major tie break; no RGB selection',
                exact_inside_outside_disjoint_full_partition=True,exact_replacement_partition=True,original_evaluation_band_unchanged=True))
        write(rp,dict(schema=SCHEMA,complete=True,CPU_only=True,parent_asset_manifest=pin(asset_path),measurement_contract=pin(a.measurement_contract),
            fixed_eval_receipt=pin(Path(cfg['fixed_eval'])/'receipt.json'),direct_surface_receipt=pin(surface_path),engineering_contract=pin(a.engineering_contract),
            intrinsic_coverage_receipt=pin(intrinsic_path),intrinsic_camera_timestamp_audit=pin(cam_path),frames=cfg['frames'],rows=rows,
            alpha_semantics='local_only_intrinsic_alpha',joint_local_alphaT_used=False,intrinsic_alpha_positive_threshold=0.0,replacement_alpha_threshold=.95,
            definition='allowed = (original full band union intrinsic alpha>0) minus K minus actual strict V/proxy',
            replacement_definition='allowed intersect intrinsic alpha>=.95; current local standalone physical opacity coverage, not joint alphaT or colour qualification',
            original_full_band_and_unknown_evaluation_denominators_preserved=True,source_material_from_immutable_asset_pin=True,
            source_colour_used_for_masks=False,current_Gaussian_alpha_used_for_masks=True,alpha_source_is_local_only_intrinsic=True,
            broad_canonical_alpha_not_used_as_car_mask=True,heldout_real_V_not_read=True,GPU_forward=0,GPU_backward=0,updates=0))
    r=read(rp);assert r['schema']==SCHEMA and r['complete'] and r['parent_asset_manifest']['sha256']==sha(asset_path)
    assert r['measurement_contract']['sha256']==sha(a.measurement_contract) and r['engineering_contract']['sha256']==sha(a.engineering_contract)
    assert r['intrinsic_coverage_receipt']['sha256']==sha(intrinsic_path) and r['direct_surface_receipt']['sha256']==sha(surface_path)
    assert r['frames']==cfg['frames'] and r['intrinsic_alpha_positive_threshold']==0.0 and r['replacement_alpha_threshold']==.95
    for row in r['rows']:
        for k in ['masks','fixed_eval','rectified_masks','direct_surface_projection','intrinsic_alpha_npz']:checked(row[k])
        z=bank(row['masks']['path']);ev=bank(row['fixed_eval']['path']);m=bank(row['rectified_masks']['path']);alpha=bank(row['intrinsic_alpha_npz']['path'])['alpha']
        car=m['actor_visible'] if row['car_mask_kind']=='opened_strict_real_V' else m['actor_visible_proxy']
        expected=(ev['full_original_band']|(alpha>0))&~m['protected']&~car
        assert np.array_equal(z['full_original_band'],ev['full_original_band'])
        assert np.array_equal(z['inside_allowed_edit'],expected) and np.array_equal(z['outside_allowed_edit'],~expected)
        assert np.array_equal(z['inside_replacement'],expected&(alpha>=.95)) and np.array_equal(z['outside_replacement'],~z['inside_replacement'])
        assert np.array_equal(z['K_protected'],m['protected']) and np.array_equal(z['actor_protected'],car)
        assert z['full_image'].all() and z['single_pixel_sentinel'].sum()==1 and not(z['single_pixel_sentinel']&~z['inside_replacement']).any()
    return r

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['asset','measurement-contract','intrinsic','surface-review','engineering-contract','masks']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();cfg=read(a.measurement_contract);m=read(a.asset/'manifest.json');r=prepare_or_validate(a,cfg,m)
    print(json.dumps(dict(complete=True,mask_receipt=pin(a.masks/'receipt.json'),counts={r['raw']:[r['inside_allowed_edit_pixels'],r['inside_replacement_pixels']] for r in r['rows']},GPU_calls=0)))
if __name__=='__main__':main()
