"""Real-GPU complete asset wiring/save/reload/next-stage smoke; no fitting."""
import json, time
from pathlib import Path
import numpy as np
import torch
from full_pvd import FixedAsset, RUN
from scene_runtime import Runtime, tagged, raster
from linear_weights import design

@torch.no_grad()
def main():
    start=time.time(); out=RUN/'smoke';out.mkdir(exist_ok=True)
    runtime=Runtime(fine_road=True)
    asset=FixedAsset.from_geometry(RUN/'geometry.npz').cuda()
    rng=np.random.RandomState(20261105)
    asset.set_colours(rng.uniform(.05,.95,(2048,3)).astype(np.float32))
    record=dict(geometry_hash=asset.geometry_hash, runtime=runtime.meta, frames=[])
    for raw in (105,113,130):
        d=runtime.raw(raw)
        rows=asset.rows(d['pose'],d['cam'],runtime.pipe,1,precomputed=True)
        # Include visible area plus surrounding projected support, not a lower-body mask.
        reference=raster(d['cam'],runtime.pipe,0,tagged(rows,'asset'),runtime.n.bg.new_zeros(3))
        mask=(reference['alpha'][0]>.01)|d['a6_visible'].bool()
        linear=design(d['cam'],runtime.pipe,rows,mask)
        native=runtime.native_reference(raw)
        before=runtime.render_scene(raw,baseline=True)
        replacement=runtime.render_scene(raw,asset)
        deleted=runtime.render_scene(raw,asset,delete=True)
        assert replacement['meta']['native_a6_rows_input']==deleted['meta']['native_a6_rows_input']==0
        assert replacement['meta']['native_a37_packed_rows']>0
        assert deleted['meta']['native_a37_packed_rows']==0
        assert float((linear['raster_rgb']-reference['rgb']).abs().max())<5e-6
        saved=asset.export(out/f'{raw}_restore.npz')
        restored=FixedAsset.from_geometry(out/f'{raw}_restore.npz').cuda()
        again=runtime.render_scene(raw,restored,delete=True)
        error=float((again['rgb']-deleted['rgb']).abs().max());assert error==0
        record['frames'].append(dict(raw=raw,shape=list(d['rgb'].shape),visible=int(d['a6_visible'].sum()),
             a6_depth=int(d['a6_depth_valid'].sum()),selected=int(mask.sum()),nnz=linear['matrix'].nnz,
             alpha_maxabs=linear['alpha_maxabs'],colour_maxabs=linear['colour_maxabs'],
             reload_rgb_maxabs=error,replacement=replacement['meta'],delete=deleted['meta']))
        print(json.dumps(record['frames'][-1]),flush=True)
        del reference,linear,native,before,replacement,deleted,restored,again
        runtime._data.clear();torch.cuda.empty_cache()
    assert runtime.modelunchanged() and asset.assert_geometry_unchanged()
    record.update(status='PASS',seconds=time.time()-start,appearance_updates=0,
                  peak_vram_bytes=torch.cuda.max_memory_allocated(),restored_entry='same full scene renderer')
    (out/'receipt.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(dict(status='PASS',seconds=record['seconds'])),flush=True)

if __name__=='__main__':main()
