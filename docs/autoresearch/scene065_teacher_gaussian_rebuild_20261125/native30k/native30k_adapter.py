"""Frozen, complete native 30k scene. No editing scene or replacement assets.

Construction loads CUDA models; importing this module is CPU only.
packet(raw) returns every original static/dynamic row, in checkpoint row order.
The 24-column feature ABI is the native V44Scene diagnostic ABI; columns22/23
identify native static/dynamic, and column8 tags original native owner21.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import sys
import time
import numpy as np

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def pinned(spec):
    path = Path(spec['path'])
    assert path.is_file() and sha(path) == spec['sha256'], path
    return path

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

class _Cameras(dict):
    def __init__(self, headers, maker):
        super().__init__(); self.headers = headers; self.maker = maker
    def __missing__(self, raw):
        raw = int(raw)
        camera = self.maker(self.headers[raw]); self[raw] = camera
        return camera

class Native30KAdapter:
    def __init__(self, contract, output=None):
        self.cfg = cfg = read(contract)
        assert cfg['schema'] == 'scene065.native30k.frozen_baseline.v1'
        assert cfg['iteration'] == 30000 and cfg['complete_unedited_native_30k'] is True
        assert cfg['retained_native_parameters_trainable'] is False
        assert cfg['row_providers'] == [] and cfg['static_gates'] is None
        assert cfg['semantic_ownership'] is None and cfg['actor_replacement_assets'] == []
        assert cfg['retire_native_rows'] == []
        source = cfg['native_sources']; model = Path(cfg['model'])
        for role, prefix in [('static', 'chkpnt'), ('dynamic', 'dynamic_chkpnt'), ('environment', 'env_light_chkpnt')]:
            assert pinned(source[role]).resolve() == (model / f'{prefix}30000.pth').resolve()
        for spec in cfg['source_metadata'].values():
            pinned(spec)
        for spec in cfg['native_engine_sources'].values():
            pinned(spec)
        for spec in cfg.get('rasterizer_sources', {}).values():
            pinned(spec)
        completed = read(cfg['source_metadata']['completion']['path'])
        assert completed['iteration'] == 30000 and completed['editing_run'] is False
        assert read(cfg['source_metadata']['last_complete_node']['path'])['iteration'] == 30000
        api = Path(cfg['repo']) / 'scripts/worldsim_v82'; sys.path.insert(0, str(api))
        from drivingeditor_dynamic_edit.scene_io import load_native_models, model_tensors
        from drivingeditor_dynamic_edit.ownership import actor_ids
        from drivingeditor_dynamic_edit.v44_manifest import SceneManifest, make_native_camera
        from drivingeditor_dynamic_edit.v44_scene import V44Scene
        import torch
        assert torch.cuda.is_available()
        self.torch = torch
        self.pipe, self.models, self.renderer = load_native_models(model, 30000, engine=cfg['engine'])
        self.static = self.models['static']; self.dynamic = self.models['dynamic']; self.environment = self.models['environment']
        for value in model_tensors(self.models).values():
            value.requires_grad_(False)
        assert all(getattr(m, 'optimizer', None) is None for m in self.models.values())
        self.frames = [int(x) for x in cfg['frames']]
        saved = read(model / 'cameras.json'); binding = read(model / 'actual_scene_binding.json')
        raw_id = lambda name: int(re.search(r'(\d{3})$', str(name)).group(1))
        image_names = {raw_id(x['img_name']): x['img_name'] for x in saved}
        clock = {raw_id(n): float(t) for n, t in zip(binding['frame_ids'], binding['timestamps'])}
        records = [dict(key=raw, image_name=image_names[raw], model_time=clock[raw]) for raw in self.frames]
        manifest = dict(schema='v44.scene.v1', model=str(model), engine=cfg['engine'], iteration=30000,
                        render_wh=cfg['render_wh'], frames=records, actors=[], ownership=None,
                        world_to_model=cfg['source_metadata']['world_to_model']['path'], surfaces=[], road=None, static_gates=None)
        self.manifest = SceneManifest(manifest)
        self.camera_headers = self.manifest.camera_headers()
        self.cameras = _Cameras(self.camera_headers, make_native_camera)
        self.transform = np.asarray(self.manifest.world_to_model, np.float64)
        self.meter_to_model = float(np.linalg.norm(self.transform[:3, :3], axis=0).mean())
        assert np.allclose(np.linalg.norm(self.transform[:3, :3], axis=0), self.meter_to_model, rtol=1e-6, atol=0)
        self.native_static_rows = len(self.static.get_xyz); self.native_dynamic_rows = len(self.dynamic.get_xyz)
        self.native_row_count = self.native_static_rows + self.native_dynamic_rows
        self.owner_static = torch.full((self.native_static_rows,), -1, dtype=torch.int64, device='cuda')
        self.owner_dynamic = actor_ids(self.dynamic).detach().clone()
        self.owner_ids = torch.cat([self.owner_static, self.owner_dynamic])
        self.global_row_ids = torch.arange(self.native_row_count, dtype=torch.int64, device='cuda')
        self.row_groups = torch.cat([torch.zeros_like(self.owner_static), torch.ones_like(self.owner_dynamic)])
        self.scene = V44Scene(self.static, self.dynamic, self.environment, self.pipe,
                              lambda raw: self.cameras[int(raw)],
                              source_owner_ids=dict(static=self.owner_static, dynamic=self.owner_dynamic),
                              semantic_owner_ids=None, row_providers=(), static_gates=None,
                              retire_static=None, retire_dynamic=None)
        assert not self.scene._providers and self.scene._gates is None
        assert not bool(self.scene._retire_s.any()) and not bool(self.scene._retire_d.any())
        self.frozen_assets = {k: v['sha256'] for k, v in source.items()}
        self.frozen_assets.update({k: v['sha256'] for k, v in cfg['source_metadata'].items()})
        self.calls = 0; self.backward_calls = 0
        self.native_near_model = float(cfg['native_near_model'])
        self.trainable = []; self.initial_parameters = []
        self._model_tensors = model_tensors
        self._signature = self._parameter_signature()
        self.assert_fixed()
        if output is not None:
            write(Path(output) / 'native30k_load_receipt.json', self.load_receipt())

    def _parameter_signature(self):
        return {k: (id(v), v.data_ptr(), v._version, tuple(v.shape), str(v.dtype), bool(v.requires_grad))
                for k, v in self._model_tensors(self.models).items()}

    def assert_fixed(self):
        self.scene.assert_unchanged()
        assert self._parameter_signature() == self._signature, 'Original native parameters changed'
        assert all(not v.requires_grad for v in self._model_tensors(self.models).values())
        assert not self.scene._providers and self.scene._gates is None
        assert not bool(self.scene._retire_s.any()) and not bool(self.scene._retire_d.any())

    def packet(self, raw):
        """Original full static then dynamic rows, precomputed SH colors, no deletion.

        Fields match the native merge_rows/raster interface. Parent code can
        select retained rows and merge newly constructed rows without altering
        any original parameter. All features have 24 columns, not RGB masks.
        """
        self.assert_fixed()
        rows, owners, _, _ = self.scene._rows(int(raw), (), (21,), True)
        assert len(rows['means3D']) == self.native_row_count
        assert self.torch.equal(owners, self.owner_ids)
        return rows

    def row_metadata(self, raw):
        return dict(global_row_ids=self.global_row_ids, group=self.row_groups, owner_ids=self.owner_ids,
                    native_static_rows=self.native_static_rows, native_dynamic_rows=self.native_dynamic_rows,
                    raw=int(raw), static_then_dynamic=True)

    def gids(self, raw):
        return self.global_row_ids

    def owners(self, raw, semantic=False):
        return self.load_ownership_metadata()['global_owner_ids'] if semantic else self.owner_ids

    def load_ownership_metadata(self):
        """Read the aligned labels only; never apply labels to rendering or masks.

        Native source ownership and measured semantic ownership are distinct.
        Native dynamic lineage supplies source owner IDs; the sidecar also
        labels static rows. Exact sidecar zeros are returned without relabeling.
        """
        if hasattr(self, '_ownership_metadata'):
            return self._ownership_metadata
        spec = self.cfg['ownership_metadata']; receipt = read(pinned(spec['receipt']))
        assert Path(receipt['model']).resolve() == Path(self.cfg['model']).resolve()
        assert receipt['iteration'] == 30000
        assert receipt['static_rows'] == self.native_static_rows and receipt['dynamic_rows'] == self.native_dynamic_rows
        assert receipt['gaussian_parameter_updates'] == 0 and receipt['edit_query_used'] is False
        with np.load(pinned(spec['sidecar']), allow_pickle=False) as z:
            arrays = {name: z[key].copy() for name, key in [('static', 'static_actor_ids'), ('dynamic', 'dynamic_actor_ids')]}
        assert arrays['static'].shape == (self.native_static_rows,) and arrays['dynamic'].shape == (self.native_dynamic_rows,)
        assert all(x.dtype.kind in 'iu' for x in arrays.values())
        tensors = {k: self.torch.as_tensor(v, dtype=self.torch.int64, device=self.owner_ids.device) for k, v in arrays.items()}
        self._ownership_metadata = dict(**tensors, global_owner_ids=self.torch.cat([tensors['static'], tensors['dynamic']]),
                                        static_offset=0, dynamic_offset=self.native_static_rows, source=spec,
                                        applied_to_native_render=False, exact_sidecar_zero_preserved=True)
        return self._ownership_metadata

    def native_visible_mask(self, raw, packet=None):
        """Keep original CUDA center-near eligibility if a parent changes units."""
        packet = self.packet(raw) if packet is None else packet
        view = self.cameras[int(raw)].world_view_transform
        depth = packet['means3D'] @ view[:3, 2] + view[3, 2]
        return packet['mask'] & (depth > self.native_near_model)

    def raster_packet(self, raw, packet):
        """One original-model-unit native raster; environment uses the same camera."""
        camera = self.cameras[int(raw)]
        degree = 0 if packet['shs'] is None else self.static.active_sh_degree
        out = self.scene._bindings().raster(camera, self.pipe, degree, packet, self.scene._background)
        rgb = out['rgb'] + (1 - out['alpha']) * self.environment(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
        self.calls += 1
        return dict(rgb=rgb, alpha=out['alpha'], depth_sum_m=out['depth_sum'] / self.meter_to_model,
                    extra=out['extra'], raw_rgb=out['rgb'])

    def render(self, raw):
        out = self.scene.render(int(raw), diagnostic_actor_ids=(21,), diagnostic_branch_alpha=True)
        self.calls += 1; self.assert_fixed()
        return dict(rgb=out['rgb'], alpha=out['alpha'], depth_sum_m=out['depth_sum'] / self.meter_to_model,
                    native_static_alpha=out['branch_alpha_T'][0], native_dynamic_alpha=out['branch_alpha_T'][1],
                    original_A20_alpha=out['owner_alpha_T'][0], raw_rgb=out['raw_rgb'], environment_contribution=out['env_map'])

    def load_receipt(self):
        return dict(schema='scene065.native30k.load.v1', iteration=30000,
                    original_unedited_source=True, source_model=self.cfg['model'], native_engine=self.cfg['engine'],
                    native_sources=self.cfg['native_sources'], native_static_rows=self.native_static_rows,
                    native_dynamic_rows=self.native_dynamic_rows, total_native_rows=self.native_row_count,
                    original_native_A20_owner21_rows=int((self.owner_dynamic == 21).sum()),
                    original_native_parameters_frozen=True, optimizer_created=False,
                    replacement_actor_loaded=False, external_road_loaded=False, semantic_ownership_loaded=False,
                    provider_count=0, retired_native_rows=0, render_wh=self.cfg['render_wh'], frames=self.frames,
                    meter_to_model=self.meter_to_model, world_to_model=self.transform.tolist())

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--raw', type=int, default=110); a = p.parse_args()
    assert not a.output.exists(), 'Keep earlier baseline evidence'
    a.output.mkdir(parents=True); start = time.perf_counter(); rt = Native30KAdapter(a.contract, a.output)
    with rt.torch.no_grad():
        arrays = {k: v.detach().cpu().numpy() for k, v in rt.render(a.raw).items()}
    assert all(np.isfinite(v).all() for v in arrays.values())
    np.savez_compressed(a.output / f'{a.raw:03d}.npz', **arrays)
    from PIL import Image
    Image.fromarray(np.rint(np.clip(arrays['rgb'].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)).save(a.output / f'{a.raw:03d}.png')
    rt.assert_fixed(); receipt = rt.load_receipt()
    receipt.update(schema='scene065.native30k.actual_baseline.v1', complete=True, raw=a.raw,
                   native_forward_calls=rt.calls, native_backward_calls=0, optimizer_updates=0,
                   baseline_has_no_editing_instructions=True, seconds=time.perf_counter() - start,
                   peak_GPU_bytes=int(rt.torch.cuda.max_memory_allocated()),
                   RGB_range=[float(arrays['rgb'].min()), float(arrays['rgb'].max())],
                   finished_utc=datetime.now(timezone.utc).isoformat())
    write(a.output / 'receipt.json', receipt); print(json.dumps(receipt, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
