"""Stream derived, hash-bound diagnostic assets to stdout. No remote output writes."""
import base64
import datetime as dt
import io
import json
import sys
import tarfile
import zlib
from pathlib import Path
import numpy as np
from PIL import Image
from cloud_sampling import Sampler
from backend import Adapter, CAMPAIGN, ARMS, MANIFEST_SHA, clean, digest


def packed(arrays):
    payload, records, offset = [], {}, 0
    for key, array in arrays.items():
        array = np.ascontiguousarray(array)
        raw = array.tobytes()
        records[key] = dict(dtype=array.dtype.str, shape=array.shape, offset=offset, bytes=len(raw))
        payload.append(raw); offset += len(raw)
    return dict(arrays=records, encoding='zlib+base64-little-endian',
                data=base64.b64encode(zlib.compress(b''.join(payload), 7)).decode())


def export():
    adapter = Adapter(); sampler=Sampler(adapter); source_files = {}
    tar = tarfile.open(fileobj=sys.stdout.buffer, mode='w|')
    def put(name, value):
        if not isinstance(value, bytes): value = json.dumps(clean(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        ti = tarfile.TarInfo(name); ti.size = len(value); tar.addfile(ti, io.BytesIO(value))
    def js(name, key, value):
        put(name, ('window.GSDATA=window.GSDATA||{};window.GSDATA['+json.dumps(key)+']='+json.dumps(clean(value), separators=(',', ':'), allow_nan=False)+';').encode())
    def source(path):
        path = Path(path); stat = path.stat(); record = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        rel = str(path.relative_to(CAMPAIGN)) if path.is_relative_to(CAMPAIGN) else str(path)
        if rel in adapter.manifest:
            expected = adapter.manifest[rel]
            if stat.st_size != expected['bytes'] or digest(path) != expected['sha256']: raise ValueError('source hash mismatch '+rel)
            record['sha256'] = expected['sha256']
        else: record['sha256'] = digest(path)
        source_files[str(path)] = record

    js('data/sampling_meta.js', 'samplingMeta', sampler.metadata)
    bundle = adapter.meta
    source(adapter.bundle); source(adapter.bundle.parent/bundle['units'])
    with np.load(adapter.bundle.parent/bundle['units']) as z:
        gates = ['in_view', 'surface_supported', 'mask_clear', 'road_support', 'measured_depth_match', 'boxes_clear', 'angle_ok', 'resolution_ok', 'quality_ok']
        bits = sum(z['gate_'+gate].astype(np.uint16)*(1 << i) for i, gate in enumerate(gates))
        road = {k: z[k].astype(np.float32) for k in ('model_xyz', 'world', 'normal_model', 'surface_distance_m')}
        # World position must retain centimeter precision despite a large global origin.
        road['world_origin'] = np.mean(z['world'], 0)
        road['world_local'] = (z['world']-road['world_origin']).astype(np.float32)
        del road['world']
        for k in ('source_count', 'prior_graph_component', 'prior_graph_context_hops'): road[k] = z[k].astype(np.int32)
        for k in ('provenance', 'surface_supported', 'is_query'): road[k] = z[k].astype(np.uint8)
        road['gates'] = bits
        road['source_uv'] = z['source_uv'].astype(np.float32)
        road['source_rgb'] = np.round(np.clip(z['source_rgb'], 0, 1)*255).astype(np.uint8)
        road['source_depth_m'] = (z['source_depth']/adapter.scale).astype(np.float16)
    js('data/road.js', 'road', packed(road))
    camera_records = []
    for raw in range(89, 138):
        cam = adapter.camera(raw)
        camera_records.append(dict(raw=raw, timestamp=cam['timestamp'], W=cam['W'], K=cam['K'],
            center=cam['center'], boxes=cam['meta']['selected_boxes_model']))
    js('data/scene.js', 'scene', dict(cameras=camera_records, world_to_model=adapter.A,
        model_to_world=adapter.invA, meter_to_model=adapter.scale, gates=gates, raw_start=89, raw_end=137,
        source='frozen V3 serial_r1 30K', road_cells=26088, supported_road=14034,
        gpu_used=False, optimizer_updates=0, native_manifest_sha256=MANIFEST_SHA,
        precision=dict(map_alpha='uint16 /65535', map_depth='uint16 centimeters, 65535 invalid; >=655.34 m clipped',
                       geometry='float32 model coordinates; sampled centers, not full novel-view rendering',
                       source_rgb='8bit display only; unquantized source remains remote')))
    summary = {}
    for arm in ARMS:
        print('export '+arm, file=sys.stderr, flush=True)
        cloud_pack = sampler.query(dict(arm=arm, raw=102, static_rate=2, dynamic_rate=2))
        cloud = cloud_pack['branches']; totals = {b['branch']: b['total'] for b in cloud}
        js(f'data/{arm}/cloud.js', arm+'/cloud', cloud_pack)
        # Actual optimized source weights/visits, separate from original geometry acceptance.
        optpath = CAMPAIGN/f'arms/{arm}/run30k/road_audit/iteration_030000/optimization.npz'
        source(optpath)
        with np.load(optpath) as z:
            opt = {k: z[k].astype(np.float32) for k in ('reliability', 'effective_rgb_weight', 'gradient_l2_cumulative')}
            opt['objective_visits'] = z['objective_visits'].astype(np.int32)
            opt['gradient_samples'] = z['gradient_samples'].astype(np.int32)
        js(f'data/{arm}/optimization.js', arm+'/optimization', packed(opt))
        summary[arm] = dict(totals=totals, displayed=sum(b['displayed'] for b in cloud), source_snapshot=30000)
        for raw in range(89, 138):
            folder = CAMPAIGN/f'final49/{arm}_r2/0008{raw:03d}'
            for filename in ('render_float.npz', 'world_unit_float.npz'): source(folder/filename)
            with np.load(folder/'render_float.npz') as z:
                maps = {}
                for k in ('static_alpha', 'joint_alpha', 'dynamic_alpha', 'road_alpha'):
                    maps[k] = np.round(np.clip(z[k][0], 0, 1)*65535).astype(np.uint16)
                for k, path in [('road_bg', 'road_in_background'), ('road_joint', 'road_in_joint')]:
                    maps[k] = np.round(np.clip(z[path][0], 0, 1)*65535).astype(np.uint16)
                for k in ('lidar_depth_m', 'static_depth_m', 'joint_depth_m', 'dynamic_depth_m'):
                    a = z[k][0]; valid = np.isfinite(a) & (a > 0)
                    maps[k] = np.where(valid, np.round(np.clip(a*100, 0, 65534)), 65535).astype(np.uint16)
                maps['body_mask'] = z['body_mask'][0].astype(np.uint8)
                maps['residual'] = np.round(np.clip(np.abs(z['joint']-z['gt']).mean(0), 0, 1)*65535).astype(np.uint16)
                # Stored rendered-depth normals are image-space proxies, never measured surface normals.
                if arm == 'full':
                    gt = np.round(np.clip(z['gt'].transpose(1, 2, 0), 0, 1)*255).astype(np.uint8)
                    buf = io.BytesIO(); Image.fromarray(gt).save(buf, format='WEBP', lossless=True)
                    put(f'assets/gt/{raw}.webp', buf.getvalue())
                js(f'data/{arm}/{raw}.js', f'{arm}/{raw}', packed(maps))
                # Sparse measured LiDAR point cloud: exact valid depth pixels at deterministic sample positions.
                valid = np.argwhere(np.isfinite(z['lidar_depth_m'][0]) & (z['lidar_depth_m'][0] > 0))
                valid = valid[np.linspace(0, len(valid)-1, min(6000, len(valid))).astype(int)]
                y, x = valid.T; depth = z['lidar_depth_m'][0, y, x]
                if arm == 'full':
                    cam = adapter.camera(raw); K = cam['K']
                    point = np.stack(((x-K[0, 2])/K[0, 0]*depth, (y-K[1, 2])/K[1, 1]*depth, depth), 1)*adapter.scale
                    invW = np.linalg.inv(cam['W']); point = point@invW[:3, :3].T+invW[:3, 3]
                    js(f'data/lidar/{raw}.js', f'lidar/{raw}', packed(dict(xyz=point.astype(np.float32))))
            with np.load(folder/'world_unit_float.npz') as z:
                units = {k: z[k].astype(np.float32) for k in ('raster_uv', 'road_only_alpha', 'road_alpha_in_background',
                    'road_alpha_in_joint', 'road_depth_model', 'source_depth_model', 'road', 'deleted', 'joint')}
                units['in_view'] = z['in_view'].astype(np.uint8)
                units['verified_exposure'] = z['verified_exposure'].astype(np.uint8)
                # Compact float16 is display data only; original is SHA bound above.
                for k in units:
                    if units[k].dtype.kind == 'f': units[k] = units[k].astype(np.float16)
                js(f'data/{arm}/units_{raw}.js', f'{arm}/units/{raw}', packed(units))
        # CPU reference, only a small fixed set; live CPU bridge can inspect other pixels.
        for raw in (89, 102, 131, 137):
            for x, y in ((120, 210), (255, 210), (220, 280)):
                result = adapter.ray(arm, raw, x, y)
                js(f'data/rays/{arm}_{raw}_{x}_{y}.js', f'ray/{arm}/{raw}/{x}/{y}', result)
        print('done '+arm, file=sys.stderr, flush=True)
    unchanged = all(Path(path).stat().st_size == record['bytes'] and Path(path).stat().st_mtime_ns == record['mtime_ns'] for path, record in source_files.items())
    if not unchanged: raise RuntimeError('protected input changed during CPU export')
    put('export_receipt.json', dict(status='CPU_DIAGNOSTIC_EXPORT_COMPLETE', created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        training_updates=0, optimizer_created=False, gpu_used=False, remote_assets_modified=False,
        source_files=source_files, protected_inputs_unchanged=unchanged, summary=summary,
        cuda_topk_status='PENDING_GPU', counterfactual_render_status='PENDING_GPU'))
    tar.close()


if __name__ == '__main__': export()
