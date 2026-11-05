"""Fixed M2 material-only ray operators; no RGB or geometry changes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz,save_npz,vstack
from scipy.ndimage import binary_erosion
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign');OLD=CAM/'scene008_a6_asset_controls_20261105'
RUN=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(OLD/'source'))
from sh_asset import SHAsset
from scene_runtime import Runtime
from linear_weights import design
HELD=(89,95,101,105,113,121,125,130,137)
BUILD=tuple(raw for raw in range(89,138) if raw not in HELD)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

@torch.no_grad()
def main():
    dest=RUN/'data';dest.mkdir(exist_ok=False);start=time.time()
    # Geometry seed contains uniform .5, not the already-fitted M2 material.
    asset=SHAsset.from_geometry(OLD/'geometry_dense.npz').cuda();assert bool((asset.rgb==.5).all())
    rt=Runtime();oldA=load_npz(OLD/'m2/design.npz')
    with np.load(OLD/'m2/inputs.npz') as old_inputs:old_target=old_inputs['target'].copy()
    counts={r['raw']:r['fit'] for r in json.loads((OLD/'m2/collection.json').read_text())['frames']}
    offsets={};position=0
    for raw in BUILD:offsets[raw]=(position,position+counts[raw]);position+=counts[raw]
    records=[];matrices=[];targets=[];allpixels=[];rowraw=[];position=0
    for raw in range(89,138):
        d=rt.raw(raw);mask=binary_erosion(d['a6_visible'].cpu().numpy(),iterations=1)
        pixels=np.flatnonzero(mask.ravel())
        if raw in offsets:
            low,high=offsets[raw];A=oldA[low:high].tocsr();assert len(pixels)==len(range(low,high))
            target=d['rgb'].flatten(1)[:,pixels].T.cpu().numpy()
            assert np.array_equal(target,old_target[low:high])
            # Original M2 normalized rows were checked by native alpha*T extraction.
            source='Original exact M2 BUILD operator'
        else:
            lin=design(d['cam'],rt.pipe,asset.rows(d['pose'],d['cam'],rt.pipe,1,precomputed=True),torch.as_tensor(mask,device='cuda'))
            assert np.array_equal(pixels,lin['pixels'])
            A=lin['matrix'].multiply(1/np.maximum(lin['alpha'][:,None],1e-5)).tocsr()
            target=d['rgb'].flatten(1)[:,pixels].T.cpu().numpy();source='Fixed geometry HELD operator, original observed mask'
        A.eliminate_zeros();matrices.append(A);targets.append(target);allpixels.append(pixels);rowraw.append(np.full(len(pixels),raw,np.int16))
        records.append(dict(raw=raw,split='BUILD' if raw in BUILD else 'HELD',start=position,end=position+len(pixels),rows=len(pixels),source=source));position+=len(pixels)
        print(json.dumps(dict(stage='collect',raw=raw,rows=len(pixels))),flush=True);rt._data.clear();torch.cuda.empty_cache()
    full=vstack(matrices,format='csr');save_npz(dest/'all_design.npz',full)
    np.savez_compressed(dest/'sealed_targets.npz',target=np.concatenate(targets),pixels=np.concatenate(allpixels),raw=np.concatenate(rowraw),
        BUILD=np.asarray(BUILD),HELD=np.asarray(HELD))
    assert rt.modelunchanged() and asset.assert_geometry_unchanged()
    write(dest/'collection.json',dict(status='PASS',frames=records,rows=position,columns=full.shape[1],nnz=full.nnz,
        design_sha256=sha(dest/'all_design.npz'),target_sha256=sha(dest/'sealed_targets.npz'),
        geometry_sha256=sha(OLD/'geometry_dense.npz'),geometry_hash=asset.geometry_hash,
        source_BUILD_design_sha256=sha(OLD/'m2/design.npz'),seconds=time.time()-start,
        operator_type='Normalized isolated RGB only; no alpha statistics are inferred from this operator.',
        no_material_or_geometry_updates=True))

if __name__=='__main__':main()
