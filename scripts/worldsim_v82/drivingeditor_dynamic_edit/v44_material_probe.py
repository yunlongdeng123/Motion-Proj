"""Short real-CUDA material backend probe; never fits or changes source assets.

This file is CPU-authored. Running it deliberately calls the supplied CUDA
libraries. It checks one explicitly selected saved camera, sparse design,
Matrix/transpose, and the DC coefficient gradient against Torch autograd.
The production material contract remains the separate 1000-step FISTA entry.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time
import numpy as np
from PIL import Image
try:
    from .v44_manifest import SceneManifest, make_native_camera, _path, _pose
    from .v44_vehicle_material import (asset_class, backend_paths, observation_files,
        source_camera_header, source_mask, _match_file, sha_file, RIDGE)
    from .v44_linear_weights import design, native, load_raster_extension
    from .v44_deterministic_csr import Matrix, validated_csr
except ImportError:
    from v44_manifest import SceneManifest, make_native_camera, _path, _pose
    from v44_vehicle_material import (asset_class, backend_paths, observation_files,
        source_camera_header, source_mask, _match_file, sha_file, RIDGE)
    from v44_linear_weights import design, native, load_raster_extension
    from v44_deterministic_csr import Matrix, validated_csr


def select_frame(frames, key_text, json_key=False):
    key = json.loads(key_text) if json_key else key_text
    matches = [f for f in frames if (type(f.key) is type(key) and f.key == key)] if json_key else [f for f in frames if str(f.key) == key]
    if len(matches) != 1: raise ValueError('Probe camera key must identify one explicit manifest frame; use --frame-key-json for a typed key')
    return matches[0]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','bank','geometry','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--track-id',required=True); p.add_argument('--actor-id',type=int,required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--frame-key'); group.add_argument('--frame-key-json')
    p.add_argument('--samples',type=int,default=32)
    return p


def main(argv=None):
    options = parser().parse_args(argv); started = time.monotonic()
    if options.samples <= 0: raise ValueError('samples must be positive')
    manifest = SceneManifest.load(options.manifest)
    if manifest.actor_ids((options.track_id,)) != (options.actor_id,): raise ValueError('Track/actor identity differs from manifest')
    frame = select_frame(manifest.frames,options.frame_key_json if options.frame_key_json is not None else options.frame_key,
                         options.frame_key_json is not None)
    if manifest.world_to_model is None: raise ValueError('Probe requires explicit world_to_model')
    saved = manifest.camera_headers()[frame.key]; paths = backend_paths(manifest,True)
    observation = observation_files(SimpleNamespace(frames=(frame,)),options.bank)[frame.key]
    obs = json.loads(observation.read_text(encoding='utf-8'))
    source_name = obs['image_path'].replace('\\','/').rsplit('/',1)[-1]
    saved_name = frame.image_name.replace('\\','/').rsplit('/',1)[-1]
    if source_name != saved_name and Path(source_name).stem != Path(saved_name).stem: raise ValueError('Observation image differs from selected saved camera')
    if frame.source_time_s is not None:
        timestamp = obs.get('timestamp_s',None if obs.get('timestamp_micros') is None else obs['timestamp_micros']/1e6)
        if timestamp is None or not np.isfinite(timestamp) or abs(timestamp-frame.source_time_s) > 1e-6: raise ValueError('Observation clock differs from selected source binding')
    boxes = [box for box in obs['boxes'] if box['track'] == options.track_id]
    if len(boxes) != 1: raise ValueError('Probe needs one actual track observation in the selected camera')
    raw_pose = np.asarray(boxes[0]['world_from_actor'])
    if raw_pose.shape == (16,): raw_pose = raw_pose.reshape(4,4)
    pose = manifest.world_to_model @ _pose(raw_pose,'observed world_from_actor')
    config = json.loads((manifest.model/'effective_config.json').read_text(encoding='utf-8'))
    source_path = _path(manifest.model,config['source_path'])
    label_path = _match_file(frame.image_name,list((source_path/'mask/mask_0/Annotations').rglob('*.png')))
    support, coarse_count = source_mask(np.array(Image.open(label_path)),options.actor_id,
        (saved['width'],saved['height']),obs['camera_size'])
    if coarse_count < 4: raise ValueError('Selected observation has insufficient real vehicle support after inherited erosion')
    header = source_camera_header(saved,obs); image_path = _path(options.bank.resolve(),obs['image_path'])
    image = np.array(Image.open(image_path).convert('RGB'))
    if image.shape[:2] != (header['height'],header['width']): raise ValueError('Actual RGB does not match declared camera_size')
    sys.path.insert(0,str(manifest.engine))
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('This probe deliberately requires a real CUDA device')
    asset = asset_class().from_geometry(options.geometry).cuda(); asset.requires_grad_(False)
    before_geometry, before_provenance = asset.geometry_hash, asset.provenance_hash
    if bool((asset.coefficients[:,1:] != 0).any()): raise ValueError('DC-only probe needs zero initial higher SH coefficients')
    camera = make_native_camera(header); pipe = SimpleNamespace(**config)
    if Path(sys.modules[type(camera).__module__].__file__).resolve() != manifest.engine/'scene/cameras.py': raise ValueError('Native Camera came from another engine')
    raster = load_raster_extension(paths['raster_extension'])
    with torch.no_grad():
        rows = asset.rows(torch.as_tensor(pose,device=asset.xyz.device,dtype=torch.float32),camera,pipe,1,precomputed=True)
        result,_ = native(camera,pipe,0,rows,asset.xyz.new_zeros(3),raster)
        alpha = result[3][1].detach().cpu().numpy()
        candidates = np.flatnonzero(support & (alpha > .9))
        if not len(candidates): raise RuntimeError('Actual vehicle support has no opaque projected asset pixels')
        ids = candidates[np.linspace(0,len(candidates)-1,min(options.samples,len(candidates)),dtype=int)]
        selected = np.zeros(support.shape,bool); selected.flat[ids] = True
        lin = design(camera,pipe,rows,torch.tensor(selected,device=asset.xyz.device),
            raster_extension=raster,replay_library=paths['replay_library'])
    A = validated_csr(lin['matrix'].multiply(1/lin['alpha'][:,None])); A.eliminate_zeros()
    m,n = A.shape
    if not A.nnz: raise RuntimeError('Design has no actual Gaussian hits')
    target = torch.tensor(image.reshape(-1,3)[lin['pixels']].astype(np.float32)/255,device=asset.xyz.device)
    x = asset.rgb.detach().clone(); prior = x.clone()
    matrix, transpose = Matrix(A,paths['csr_library']), Matrix(A.T,paths['csr_library'])
    prediction = matrix(x); gradient = 2*transpose(prediction-target)/m+2*RIDGE/n*(x-prior)
    reference = torch.tensor(np.asarray(A@x.cpu().numpy()),device=x.device)
    spmm_error = float((prediction-reference).abs().max())
    # 仅有实际hit的列构成小矩阵；Autograd独立验证DC链式梯度，不训练或改asset。
    active = np.unique(A.indices); dense = torch.tensor(A[:,active].toarray(),device=x.device)
    coeff = asset.coefficients[torch.as_tensor(active,device=x.device),0].detach().clone().requires_grad_(True)
    dc = coeff*.28209479177387814+.5
    dense_loss = (dense@dc-target).square().sum()/m+RIDGE/n*(dc-prior[active]).square().sum()
    direct_gradient, = torch.autograd.grad(dense_loss,coeff)
    grad_error = float((direct_gradient-gradient[active]*.28209479177387814).abs().max())
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(gradient).all()) or not np.isfinite(spmm_error+grad_error): raise RuntimeError('Nonfinite material backend result')
    if spmm_error >= 5e-6 or grad_error >= 5e-6: raise RuntimeError('Material Matrix/DC gradient mismatch: '+str((spmm_error,grad_error)))
    asset.assert_geometry_unchanged()
    if (asset.geometry_hash,asset.provenance_hash) != (before_geometry,before_provenance): raise RuntimeError('Probe changed persistent asset state')
    torch.cuda.synchronize()
    out = options.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    receipt = dict(status='PASSED_MATERIAL_BACKEND_CAMERA_PROBE',frame_key=frame.key,image_name=frame.image_name,
        model_time=frame.model_time,source_time_s=frame.source_time_s,track_id=options.track_id,actor_id=options.actor_id,
        native_wh=obs['camera_size'],render_wh=[saved['width'],saved['height']],selected_pixels=m,
        available_supported_opaque_pixels=len(candidates),hits=lin['hits'],nnz=A.nnz,columns=n,
        alpha_maxabs=lin['alpha_maxabs'],colour_maxabs=lin['colour_maxabs'],spmm_maxabs=spmm_error,
        dc_gradient_maxabs=grad_error,dc_gradient_nonzero=int(torch.count_nonzero(gradient)),
        finite=True,geometry_unchanged=True,provenance_unchanged=True,
        geometry_hash=before_geometry,provenance_hash=before_provenance,
        production_fista_steps_executed=0,does_not_certify_1000_step_fit_or_all_frames=True,
        inputs={name:dict(path=str(path),sha256=sha_file(path)) for name,path in
            [('manifest',options.manifest),('geometry',options.geometry),('observation',observation),('labels',label_path),('rgb',image_path)]},
        kernels={name:dict(path=str(path),sha256=sha_file(path)) for name,path in paths.items()},
        seconds=time.monotonic()-started)
    (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,allow_nan=False),flush=True)


if __name__ == '__main__': main()
