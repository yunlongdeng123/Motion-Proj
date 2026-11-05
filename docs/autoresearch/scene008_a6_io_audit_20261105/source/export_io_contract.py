"""Export the unchanged M2 with explicit geometry/material I/O provenance.

RGB-loss support is not a certificate of point-level semantic truth. All
unknown material remains labelled inferred. This is an interface correction,
not a new texture model or a quality improvement.
"""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD=CAM/'scene008_a6_asset_controls_20261105'
RUN=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(OLD/'source'))
from sh_asset import SHAsset
from scene_runtime import Runtime,raster,tagged

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

@torch.no_grad()
def main():
    dest=RUN/'contract_asset';dest.mkdir(exist_ok=False)
    source=OLD/'m2/complete_a6.npz';arrays=dict(np.load(source));before=sha(source)
    audit=dict(np.load(RUN/'audit/appearance_provenance.npz'))
    arrays.pop('provenance_hash')
    arrays['rgb_loss_support_state']=audit['category']
    arrays['rgb_loss_support_mass']=audit['support_mass'].astype(np.float32)
    arrays['rgb_loss_support_frames']=audit['support_views'].astype(np.int16)
    arrays['rgb_loss_design_nnz']=audit['design_nnz'].astype(np.int32)
    arrays['rgb_default_fallback_prior']=audit['global_fallback_prior']
    meta=json.loads(str(arrays['metadata_json']))
    contract=dict(schema='a6.geometry_material_io.v1',geometry_model='Frozen PVD car_completion_3199',
       geometry_model_input='200 XYZ coordinates; no RGB, no image, no material conditioning',
       geometry_model_output='2048 XYZ coordinates: 200 fixed observations +1848 generated',
       renderer_adapter_output='Object-local surface Gaussians; PCA normals/scales, opacity, thickness are downstream conventions',
       material_model='BUILD40 real-visible RGB SH0 fit; no learned hidden-appearance model',
       rgb_loss_support_codes={'0':'No RGB loss column support; material stays fallback, NOT recovered texture',
          '1':'Weak/limited RGB loss support; appearance remains inferred',
          '2':'At least2 frame and1 alpha*T pixel-mass support; NOT point-semantic or hidden-surface ground truth'},
       material_source='Reliable native appearance reference only inside true observed BUILD pixels, then real RGB fit; unseen global median',
       source_M2_sha256=before,source_design_sha256=sha(OLD/'m2/design.npz'),
       source_inputs_sha256=sha(OLD/'m2/inputs.npz'),unknown_colour_inferred=True,
       front_orientation_status='Unverified adapter convention; sensitivity probe does not establish orientation truth',
       asset_status='Unchanged M2 research reference with explicit I/O contract; not a newly improved or accepted reconstruction')
    meta['explicit_geometry_material_io']=contract;arrays['metadata_json']=np.asarray(json.dumps(meta,ensure_ascii=False,sort_keys=True))
    asset=SHAsset(arrays,meta);export=asset.export(dest/'m2_with_io_contract.npz')
    fresh=SHAsset.from_geometry(dest/'m2_with_io_contract.npz');base=SHAsset.from_geometry(source)
    assert fresh.geometry_hash==base.geometry_hash and torch.equal(fresh.coefficients,base.coefficients)
    assert torch.equal(fresh.colour_inferred,base.colour_inferred)
    for key in ('xyz','normal','sigma','opacity','thickness'):assert torch.equal(getattr(fresh,key),getattr(base,key))
    assert np.array_equal(fresh.rgb_loss_support_state.numpy(),audit['category'])
    assert bool(fresh.colour_inferred[fresh.rgb_loss_support_state==0].all())
    rt=Runtime();fresh=fresh.cuda();base=base.cuda();checks=[]
    for raw in (105,113,130):
        d=rt.raw(raw)
        def solo(a):return raster(d['cam'],rt.pipe,3,tagged(a.rows(d['pose'],d['cam'],rt.pipe,16),'asset'),rt.n.bg.new_zeros(3))
        a,b=solo(fresh),solo(base)
        assert all(torch.equal(a[key],b[key]) for key in ('rgb','alpha','depth_sum'))
        for delete in (False,True):
            a=rt.render_scene(raw,fresh,delete=delete);b=rt.render_scene(raw,base,delete=delete)
            assert all(torch.equal(a[key],b[key]) for key in ('rgb','alpha','depth_sum','component_alpha_T'))
            assert a['meta']['native_a6_rows_input']==0
        checks.append(dict(raw=raw,solo_keep_delete_render_bitwise_equal=True));rt._data.clear()
    assert rt.modelunchanged() and fresh.assert_geometry_unchanged() and sha(source)==before
    write(dest/'io_contract.json',contract)
    write(dest/'receipt.json',dict(status='PASS_INTERFACE_ONLY',export=export,geometry_bitwise_unchanged=True,
        RGB_coefficients_bitwise_unchanged=True,inferred_flags_unchanged=True,three_view_replay=checks,
        new_training_steps=0,source_bytes_unchanged=True,scene_adoption=False))

if __name__=='__main__':main()
