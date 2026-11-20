"""CPU-only same-scene visibility binding; never regenerate native weights."""
from pathlib import Path
import json,sys
import numpy as np
from export_public_endpoints import read,write,file_sha,validate_asset

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=C/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119';B=G/'native_capacity_20261120_r1'
H=G/'quality_bench_20261120_r1/highres';PARENT=C/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'

def main():
    out=B/'public1920/export_r1';manifest=read(out/'packets_manifest.json');export=read(out/'receipt.json')
    if not export['complete']:raise RuntimeError('public export incomplete')
    old=read(H/'native_completed.json');scope=G/'native_qualification_r2/basis/110/row_scope.npz'
    permission=Q/'permission/final_r1/teacher_evidence.npz'
    if file_sha(PARENT/'persistent/scene.json')!=old['original_manifest_SHA256']or file_sha(PARENT/'persistent/proof.json')!=old['proof_SHA256']or file_sha(scope)!=old['scope_SHA256']:
        raise RuntimeError('old same-sort geometry/source/scope pins no longer match')
    with np.load(scope,allow_pickle=False)as a,np.load(permission,allow_pickle=False)as b:
        if not np.array_equal(a['active_world_rows'],b['active_world_rows'])or not np.array_equal(a['eligible_active'],b['common_write_active']):raise RuntimeError('old alphaT scope and current formal field scope differ')
    sys.path.insert(0,'/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82')
    from drivingeditor_dynamic_edit.v44_manifest import SceneManifest
    data=read(PARENT/'persistent/scene.json');data['render_wh']=[1920,1280];headers=SceneManifest(data,PARENT/'persistent').camera_headers()
    for raw in(86,110):
        a=old['camera'][str(raw)];b=headers[raw]
        if not np.array_equal(np.asarray(a['world_view_model']),b['W'])or any(a[k]!=b[k]for k in('width','height','fx','fy','cx','cy','timestamp')):raise RuntimeError('camera/order source differs')
    rows=[]
    for row in manifest['rows']:
        arm,raw=row['arm'],row['raw'];packet=Path(row['path'])
        if raw not in(86,110):
            rows.append(dict(arm=arm,raw=raw,available=False,reason='no fully bound whole-road same-sort alphaT sidecar here; source-only eligible-K proof remains separate'))
            continue
        side=H/f'native_r1/{raw:03d}/joint_basis.npz'
        with np.load(packet,allow_pickle=False)as p,np.load(side,allow_pickle=False)as s:
            if not np.array_equal(p['joint_alpha'],s['alpha'])or not np.array_equal(p['K'],s['K']):raise RuntimeError('visibility alpha/K identity mismatch')
            if p['joint_alpha'].shape!=(1280,1920):raise RuntimeError('visibility resolution differs')
        # Export already byte-audited every endpoint; reference its full hashes.
        if not export['frozen_assets'][arm]['all_original_prior_ledger_bytes_exact']:raise RuntimeError('original ledger changed')
        rows.append(dict(arm=arm,raw=raw,available=True,path=str(side),SHA256=file_sha(side),
            keys={'joint_eligible_road_alphaT':'eligible_weight','joint_whole_road_alphaT':'entire_road_weight'},
            alpha_exact=True,K_exact=True,RGB_components_reused=False,
            qualification='same original manifest/proof, fixed perGaussian geometry/opacity/activation/order, exact same1920camera/scope and alpha/K; colors do not enter alphaT'))
    target=out/'visibility_manifest_r1.json'
    if target.exists():raise FileExistsError('preserve prior visibility binding')
    write(target,dict(schema='native_capacity_visibility_sidecars.v1',complete=True,CPU_only=True,new_native_calls=0,
        public_manifest=str(out/'packets_manifest.json'),public_manifest_SHA256=file_sha(out/'packets_manifest.json'),
        source_highres_receipt=str(H/'native_completed.json'),source_highres_receipt_SHA256=file_sha(H/'native_completed.json'),
        original_manifest_SHA256=old['original_manifest_SHA256'],proof_SHA256=old['proof_SHA256'],scope_SHA256=old['scope_SHA256'],
        formal_permission=str(permission),formal_permission_SHA256=file_sha(permission),formal_scope_equal_old_scope=True,
        rows=rows,source_file=str(Path(__file__)),source_SHA256=file_sha(__file__)))
    print(str(target))
if __name__=='__main__':main()
