"""Manifest-driven fixed-geometry RGB material fit; inherited 1000-step FISTA.

This is the inherited DC-RGB fit, not new SH3 coefficient learning. Higher SH
must initially be zero. The module imports CUDA/native Camera only at main(),
while metadata, camera-grid and solver CPU-reference contracts are testable.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.sparse import vstack, save_npz, load_npz
try:
    from .v44_manifest import SceneManifest, make_native_camera, _path, _pose, _positive_hw
    from .v44_deterministic_csr import Matrix, validated_csr, library_path
except ImportError:
    from v44_manifest import SceneManifest, make_native_camera, _path, _pose, _positive_hw
    from v44_deterministic_csr import Matrix, validated_csr, library_path

FISTA_STEPS, RIDGE = 1000, .001


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def asset_class():
    """Support both package -m and a directly invoked portable source file."""
    package = __package__
    if not package:
        directory = Path(__file__).resolve().parent
        sys.path.insert(0, str(directory.parent)); package = directory.name
    return importlib.import_module(package + '.v44_sh_asset').SHAsset


def source_camera_header(saved_header, observation):
    """Scale saved render intrinsics to the declared real source image grid."""
    native_w, native_h = _positive_hw(observation['camera_size'], 'camera_size width,height')
    render_w, render_h = _positive_hw((saved_header['width'], saved_header['height']), 'saved render width,height')
    result = deepcopy(saved_header); result.update(width=native_w, height=native_h)
    for key, scale in [('fx',native_w/render_w),('cx',native_w/render_w),
                       ('fy',native_h/render_h),('cy',native_h/render_h)]: result[key] *= scale
    return result


def source_mask(labels, actor_id, saved_wh, native_wh):
    if type(actor_id) is not int or actor_id <= 0: raise ValueError('actor_id must be an explicit positive integer')
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.dtype.kind not in 'iu': raise ValueError('Source labels must be a 2D integer identity map')
    render_w, render_h = _positive_hw(saved_wh, 'saved render width,height')
    native_w, native_h = _positive_hw(native_wh, 'camera_size width,height')
    coarse = np.array(Image.fromarray(np.uint8(labels == actor_id)).resize((render_w,render_h), Image.Resampling.NEAREST)) > 0
    selected = binary_erosion(coarse, iterations=1)  # 原协议：渲染网格侵蚀1像素，不截断大整数ID。
    native = np.array(Image.fromarray(np.uint8(selected)).resize((native_w,native_h), Image.Resampling.NEAREST)) > 0
    return native, int(selected.sum())


def _match_file(name, paths):
    name = str(name).replace('\\','/'); basename = name.rsplit('/',1)[-1]
    exact = [p for p in paths if p.name == basename]
    matches = exact or [p for p in paths if p.stem == basename or p.stem == Path(basename).stem]
    if len(matches) != 1: raise ValueError('Ambiguous/missing saved image-name mapping: ' + name)
    return matches[0]


def observation_files(manifest, bank):
    """Index actual image_path names; never parse a numeric suffix as frame ID."""
    bank = Path(bank).resolve(); indexed = []
    if any(frame.observation is None for frame in manifest.frames):
        for path in sorted((bank/'observations').glob('*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(data.get('image_path'),str):
                indexed.append((path,data['image_path'].replace('\\','/').rsplit('/',1)[-1]))
    result = {}
    for frame in manifest.frames:
        if frame.observation is not None:
            path = frame.observation
        else:
            name = frame.image_name.replace('\\','/').rsplit('/',1)[-1]
            matches = [p for p,n in indexed if n == name] or [p for p,n in indexed if Path(n).stem == name or Path(n).stem == Path(name).stem]
            if len(matches) != 1: raise ValueError('Frame needs an explicit or unique observation image_path mapping: ' + str(frame.key))
            path = matches[0]
        if not path.is_file(): raise FileNotFoundError(path)
        result[frame.key] = path
    return result


def validate_design(A, target, prior):
    A = validated_csr(A); A.eliminate_zeros()
    target, prior = np.asarray(target), np.asarray(prior)
    m,n = A.shape
    if (not m or not n or not A.nnz or (A.data < 0).any() or target.shape != (m,3) or prior.shape != (n,3)
            or target.dtype.kind not in 'iuf' or prior.dtype.kind not in 'iuf'
            or not np.isfinite(target).all() or not np.isfinite(prior).all()
            or (target < 0).any() or (target > 1).any() or (prior < 0).any() or (prior > 1).any()):
        raise ValueError('Design needs nonnegative measured weights, finite [0,1] RGB targets/prior, and matching nonempty topology')
    return A, target.astype(np.float32), prior.astype(np.float32)


def fit_rgb(A, target, prior, *, device='cuda', csr_library=None, checkpoint=None):
    """Exactly the inherited diagonal-majorized FISTA equations and 1000 steps."""
    A, target, prior = validate_design(A,target,prior); m,n = A.shape
    unknown = A.getnnz(axis=0) == 0; cpu = str(device) == 'cpu'
    matrix, transpose = Matrix(A,csr_library,device), Matrix(A.T,csr_library,device)
    upper = np.asarray(A.T @ np.asarray(A.sum(1)).ravel()).ravel()
    if cpu:
        copy = lambda x: x.copy(); clamp = lambda x: np.clip(x,0,1)
        D = np.asarray(2*(upper/m + RIDGE/n),np.float32)[:,None]
        double_sum = lambda x: np.square(x.astype(np.float64)).sum()
        finite = lambda x: bool(np.isfinite(x).all()); snapshot = lambda x: x.copy()
    else:
        import torch
        target = torch.tensor(target,device=device); prior = torch.tensor(prior,device=device)
        D = torch.tensor(2*(upper/m + RIDGE/n),device=device,dtype=torch.float32)[:,None]
        copy = lambda x: x.clone(); clamp = lambda x: x.clamp(0,1)
        double_sum = lambda x: x.double().square().sum()
        finite = lambda x: bool(torch.isfinite(x).all()); snapshot = lambda x: x.detach().cpu().numpy().copy()
    x = copy(prior); y = copy(x); t = 1.; restarts = 0; trace = []
    def objective(value): return float(double_sum(matrix(value)-target)/m + RIDGE*double_sum(value-prior)/n)
    loss = objective(x)
    for step in range(1,FISTA_STEPS+1):
        gradient = 2*transpose(matrix(y)-target)/m + 2*RIDGE/n*(y-prior)
        z = clamp(y-gradient/D); value = objective(z)
        if value > loss + 1e-9:
            t = 1.; gradient = 2*transpose(matrix(x)-target)/m + 2*RIDGE/n*(x-prior)
            z = clamp(x-gradient/D); value = objective(z); restarts += 1
        if not finite(z): raise RuntimeError('FISTA produced nonfinite RGB')
        tn = float((1+np.sqrt(1+4*t*t))/2); y = z+float((t-1)/tn)*(z-x); x = z; t = tn; loss = value
        if step % 100 == 0: trace.append(dict(step=step,objective=loss))
        if checkpoint is not None and step in (500,1000):
            checkpoint(dict(step=step,x=snapshot(x),y=snapshot(y),t=t,restarts=restarts,loss=loss))
    result = snapshot(x); original = snapshot(prior)
    if not np.array_equal(result[unknown],original[unknown]): raise RuntimeError('Unobserved geometry colors changed')
    return result, dict(steps=FISTA_STEPS,ridge=RIDGE,objective=loss,objective_trace=trace,
        solver_restarts=restarts,rows=m,columns=n,unknown_columns=int(unknown.sum()),backend=matrix.backend)


def backend_paths(manifest, need_design):
    config = manifest.data.get('material_backend',{}); result = {}
    for name,env in [('csr_library','V44_CSR_LIBRARY'),('replay_library','V44_REPLAY_LIBRARY'),('raster_extension','V44_RASTER_EXTENSION')]:
        value = config.get(name) or os.environ.get(env)
        if value: result[name] = _path(manifest.base,value)
    result['csr_library'] = library_path(result.get('csr_library'))
    if need_design:
        result.setdefault('replay_library',Path(__file__).parent/'native/libreplay_quantile.so')
        if 'raster_extension' not in result: raise ValueError('Fresh design requires explicit material_backend.raster_extension')
        for name in ('replay_library','raster_extension'):
            if not result[name].is_file(): raise FileNotFoundError(result[name])
    return result


def design_binding(manifest, track_id, actor_id, geometry_hash):
    """Cache reuse is bound to actual camera/clock artifacts, not just row count."""
    return dict(schema='v44.material_design.v1',track_id=track_id,actor_id=actor_id,geometry_hash=geometry_hash,
        camera_sha256=sha_file(manifest.model/'cameras.json'),
        clock_binding_sha256=sha_file(manifest.model/'actual_scene_binding.json'),
        world_to_model=None if manifest.world_to_model is None else manifest.world_to_model.tolist(),
        frames=[dict(key=f.key,image_name=f.image_name,model_time=f.model_time,source_time_s=f.source_time_s)
                for f in manifest.frames])


def collect_design(manifest, bank, track_id, actor_id, asset, pipe, source_path, paths):
    import torch
    try:
        from .v44_linear_weights import design, load_raster_extension
    except ImportError:
        from v44_linear_weights import design, load_raster_extension
    if manifest.world_to_model is None: raise ValueError('Source material design requires explicit world_to_model')
    headers = manifest.camera_headers(); observations = observation_files(manifest,bank)
    annotations = list((source_path/'mask/mask_0/Annotations').rglob('*.png'))
    raster = load_raster_extension(paths['raster_extension']); matrices,targets,records = [],[],[]
    for frame in manifest.frames:
        obs = json.loads(observations[frame.key].read_text(encoding='utf-8'))
        actual_name = obs['image_path'].replace('\\','/').rsplit('/',1)[-1]
        saved_name = frame.image_name.replace('\\','/').rsplit('/',1)[-1]
        if actual_name != saved_name and Path(actual_name).stem != Path(saved_name).stem:
            raise ValueError('Explicit observation does not belong to the saved camera image')
        if frame.source_time_s is not None:
            source_time = obs.get('timestamp_s', None if obs.get('timestamp_micros') is None else obs['timestamp_micros']/1e6)
            if source_time is None or not np.isfinite(source_time) or abs(source_time-frame.source_time_s) > 1e-6:
                raise ValueError('Observation acquisition time differs from the explicit frame binding')
        boxes = [b for b in obs['boxes'] if b['track'] == track_id]
        if not boxes: continue
        if len(boxes) != 1: raise ValueError('Observation repeats the same original track ID')
        header = source_camera_header(headers[frame.key],obs)
        labels = np.array(Image.open(_match_file(frame.image_name,annotations)))
        mask, coarse_pixels = source_mask(labels,actor_id,(headers[frame.key]['width'],headers[frame.key]['height']),obs['camera_size'])
        if coarse_pixels < 4: continue
        raw_pose = np.asarray(boxes[0]['world_from_actor'])
        if raw_pose.shape == (16,): raw_pose = raw_pose.reshape(4,4)
        pose = manifest.world_to_model @ _pose(raw_pose,'observed world_from_actor')
        camera = make_native_camera(header)
        camera_module = sys.modules[type(camera).__module__]
        if Path(camera_module.__file__).resolve() != manifest.engine/'scene/cameras.py':
            raise ValueError('Loaded Camera from a different engine')
        with torch.no_grad():
            rows = asset.rows(torch.as_tensor(pose,device=asset.xyz.device,dtype=torch.float32),camera,pipe,1,precomputed=True)
            lin = design(camera,pipe,rows,torch.tensor(mask,device=asset.xyz.device),
                         raster_extension=raster,replay_library=paths['replay_library'])
        accepted = lin['alpha'] > .9
        if not accepted.any(): continue
        A = lin['matrix'][accepted].multiply(1/np.maximum(lin['alpha'][accepted,None],1e-6)).tocsr(); A.eliminate_zeros()
        pixels = lin['pixels'][accepted]
        image_path = _path(Path(bank),obs['image_path'])
        image = np.array(Image.open(image_path).convert('RGB'))
        if image.shape[:2] != (header['height'],header['width']): raise ValueError('Source RGB differs from declared camera_size')
        rgb = image.reshape(-1,3)[pixels].astype(np.float32)/255
        matrices.append(A); targets.append(rgb)
        records.append(dict(frame_key=frame.key,image_name=frame.image_name,model_time=frame.model_time,
            observation=str(observations[frame.key]),rows=len(rgb),nnz=A.nnz,
            native_wh=obs['camera_size'],saved_render_wh=[headers[frame.key]['width'],headers[frame.key]['height']],
            alpha_maxabs=lin['alpha_maxabs'],colour_maxabs=lin['colour_maxabs']))
        print('DESIGN',frame.key,len(rgb),flush=True)
    if not matrices: raise ValueError('No accepted source-image material constraints for the registered track')
    return vstack(matrices,format='csr').astype(np.float32),np.concatenate(targets),records


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','bank','geometry','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--track-id',required=True); p.add_argument('--actor-id',type=int,required=True)
    p.add_argument('--design-source',type=Path)
    return p


def main(argv=None):
    options = parser().parse_args(argv); started = time.monotonic()
    manifest = SceneManifest.load(options.manifest)
    if manifest.actor_ids((options.track_id,)) != (options.actor_id,): raise ValueError('track-id/actor-id disagree with explicit manifest registry')
    headers = manifest.camera_headers()  # 先CPU验证真实模型相机/时间契约。
    config = json.loads((manifest.model/'effective_config.json').read_text(encoding='utf-8'))
    paths = backend_paths(manifest,options.design_source is None)
    source_path = _path(manifest.model,config['source_path'])
    sys.path.insert(0,str(manifest.engine))
    import torch
    SHAsset = asset_class(); C0 = .28209479177387814
    asset = SHAsset.from_geometry(options.geometry).cuda(); asset.requires_grad_(False)
    if bool((asset.coefficients[:,1:] != 0).any()): raise ValueError('Inherited DC-RGB FISTA requires zero initial higher SH coefficients')
    out = options.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    records = []; cache_binding_verified = options.design_source is None
    if options.design_source is not None:
        A = load_npz(options.design_source/'design.npz'); target = np.load(options.design_source/'target.npy',allow_pickle=False)
        metadata = options.design_source/'design_binding.json'
        if metadata.is_file():
            saved = json.loads(metadata.read_text(encoding='utf-8'))
            expected = design_binding(manifest,options.track_id,options.actor_id,asset.geometry_hash)
            if saved != expected: raise ValueError('Cached design geometry/identity/frame binding differs from this manifest')
            cache_binding_verified = True
    else:
        A,target,records = collect_design(manifest,options.bank,options.track_id,options.actor_id,asset,SimpleNamespace(**config),source_path,paths)
    prior = asset.rgb.detach().cpu().numpy().copy(); A,target,prior = validate_design(A,target,prior)
    save_npz(out/'design.npz',A); np.save(out/'target.npy',target)
    def checkpoint(state):
        torch.save(dict(state,x=torch.from_numpy(state['x']),y=torch.from_numpy(state['y'])),out/f'solver{state["step"]}.pt')
        print('FISTA',state['step'],state['loss'],flush=True)
    solution,receipt = fit_rgb(A,target,prior,csr_library=paths['csr_library'],checkpoint=checkpoint)
    unknown = A.getnnz(axis=0) == 0; known = torch.tensor(~unknown,device=asset.xyz.device)
    coefficients = asset.coefficients.clone()
    coefficients[known,0] = (torch.tensor(solution,device=asset.xyz.device)[known]-.5)/C0
    asset.set_coefficients(coefficients); asset.metadata['colour_fit'] = dict(steps=FISTA_STEPS,ridge=RIDGE,
        geometry_frozen=True,track_id=options.track_id,actor_id=options.actor_id,mode='inherited_DC_RGB_FISTA')
    artifact = asset.export(out/'complete_asset.npz')
    binding = design_binding(manifest,options.track_id,options.actor_id,asset.geometry_hash)
    if cache_binding_verified: (out/'design_binding.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2),encoding='utf-8')
    receipt.update(status='COMPLETED_FIXED_GEOMETRY_RGB_MATERIAL',frames=records,
        source_frame_binding_verified=cache_binding_verified,design_source=None if options.design_source is None else str(options.design_source.resolve()),
        manifest_sha256=sha_file(options.manifest),input_geometry_sha256=sha_file(options.geometry),
        design_sha256=sha_file(out/'design.npz'),target_sha256=sha_file(out/'target.npy'),
        kernels={k:dict(path=str(v),sha256=sha_file(v)) for k,v in paths.items()},
        track_id=options.track_id,actor_id=options.actor_id,artifact=artifact,seconds=time.monotonic()-started)
    (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('MATERIAL_COMPLETE',receipt['rows'],receipt['columns'],receipt['objective'],flush=True)


if __name__ == '__main__': main()
