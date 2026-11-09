"""Portable scene/camera contract. Frame and actor identifiers are data, not code."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from numbers import Real
import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    key: object
    image_name: str
    model_time: float
    source_time_s: float | None = None
    observation: Path | None = None


def _path(base, value):
    if not isinstance(value, (str, Path)) or (isinstance(value, str) and not value.strip()):
        raise ValueError('Manifest paths must be explicit nonempty strings or Paths')
    p = Path(value)
    return (p if p.is_absolute() else base / p).resolve()


def _finite_number(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(name + ' must be a real numeric scalar')
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(name + ' must be finite' + (' and positive' if positive else ''))
    return value


def _positive_hw(value, name):
    if (not isinstance(value, (list, tuple, np.ndarray))
            or (isinstance(value, np.ndarray) and value.ndim != 1) or len(value) != 2):
        raise ValueError(name + ' must contain exactly two positive integers')
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x <= 0 for x in value):
        raise ValueError(name + ' must contain positive integers without float truncation')
    return tuple(int(x) for x in value)


def _matrix(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise ValueError(name + ' must be a finite real array with shape ' + str(shape))
    return a.astype(np.float64, copy=True)


def _pose(value, name, similarity=False):
    a = _matrix(value, (4, 4), name); linear = a[:3, :3]
    scale = np.linalg.norm(linear, axis=0)
    expected = float(scale.mean()) if similarity else 1.
    if not np.isfinite(expected) or expected <= 0:
        raise ValueError(name + ' has invalid scale')
    rotation = linear / expected
    if (not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(scale / expected, 1., atol=2e-6, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or np.linalg.det(rotation) <= 0):
        raise ValueError(name + (' must be a positive similarity without shear/reflection' if similarity else ' must be SE3'))
    return a


def _frame_key(value):
    if type(value) not in (str, int) or (isinstance(value, str) and not value):
        raise ValueError('Frame keys must be explicit strings or integers, not booleans')
    return value


def _image_name(value):
    if not isinstance(value, str) or not value:
        raise ValueError('Camera image names must be nonempty strings')
    return value


class SceneManifest:
    """The saved model clock and real acquisition seconds remain distinct."""
    def __init__(self, data, base=Path('.')):
        self.data = dict(data); base = Path(base).resolve()
        if data.get('schema') != 'v44.scene.v1':
            raise ValueError('Expected v44.scene.v1')
        self.model = _path(base, data['model'])
        self.engine = _path(base, data['engine'])
        self.iteration = data['iteration']
        if type(self.iteration) is not int or self.iteration <= 0:
            raise ValueError('iteration must be a positive integer')
        self.frames = tuple(FrameRecord(
            _frame_key(row['key']), _image_name(row['image_name']), _finite_number(row['model_time'], 'model_time'),
            None if row.get('source_time_s') is None else _finite_number(row['source_time_s'], 'source_time_s'),
            None if row.get('observation') is None else _path(base, row['observation']))
            for row in data['frames'])
        keys = [f.key for f in self.frames]
        if (not keys or any(type(k) not in (str, int) for k in keys)
                or len(set(keys)) != len(keys)
                or any(not f.image_name or not np.isfinite(f.model_time) for f in self.frames)
                or any(f.source_time_s is not None and not np.isfinite(f.source_time_s) for f in self.frames)):
            raise ValueError('Frame keys/names/times must be explicit, unique and finite')
        if len({f.image_name for f in self.frames}) != len(self.frames):
            raise ValueError('Camera image names must be unique in one worker')
        actors = data['actors']
        self.track_to_actor = {r['track_id']: r['actor_id'] for r in actors}
        if (len(self.track_to_actor) != len(actors)
                or any(not isinstance(t, str) or not t for t in self.track_to_actor)
                or any(type(i) is not int or i <= 0 or i > np.iinfo(np.int64).max for i in self.track_to_actor.values())
                or len(set(self.track_to_actor.values())) != len(actors)):
            raise ValueError('Explicit track and positive actor IDs must form a bijection')
        self.actor_to_track = {v: k for k, v in self.track_to_actor.items()}
        self.ownership = None if data.get('ownership') is None else _path(base, data['ownership'])
        self.zero_semantic_policy = data.get('zero_semantic_policy', 'unknown')
        if self.zero_semantic_policy not in ('unknown', 'source_fallback'):
            raise ValueError('Unknown semantic-zero policy')
        transform = data.get('world_to_model')
        self.world_to_model = None
        if transform is not None:
            self.world_to_model = (np.loadtxt(_path(base, transform)) if isinstance(transform, (str, Path))
                                   else transform)
            self.world_to_model = _pose(self.world_to_model, 'world_to_model', similarity=True)
        self.surfaces = []
        for item in data.get('surfaces', []):
            item = dict(item)
            if item['track_id'] not in self.track_to_actor:
                raise ValueError('Surface track is not registered')
            for name in ('geometry', 'posterior', 'poses', 'retire_static_rows', 'retire_dynamic_rows'):
                if item.get(name) is not None: item[name] = _path(base, item[name])
            if item.get('geometry') is None or item.get('poses') is None:
                raise ValueError('Surface entries require geometry and explicit pose data paths')
            if item.get('posterior') is not None and 'reference_hw' not in item:
                raise ValueError('Surface LOD requires explicit reference_hw')
            if 'reference_hw' in item:
                item['reference_hw'] = _positive_hw(item['reference_hw'], 'reference_hw')
            for flag in ('retire_native_owner', 'retire_dynamic_source_owner', 'require_complete_coverage'):
                if flag in item and type(item[flag]) is not bool:
                    raise ValueError(flag + ' must be boolean')
            self.surfaces.append(item)
        self.road = None if data.get('road') is None else _path(base, data['road'])
        self.static_gates = None
        if data.get('static_gates') is not None:
            if not isinstance(data['static_gates'], dict) or set(data['static_gates']) != {'indices', 'checkpoint'}:
                raise ValueError('static_gates requires explicit indices and checkpoint paths')
            self.static_gates = {k: _path(base, v) for k, v in data['static_gates'].items()}
        self.base = base

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        return cls(json.loads(path.read_text(encoding='utf-8')), path.parent)

    def camera_headers(self):
        """Match explicit saved image names; no three-digit suffix or clip range."""
        rows = json.loads((self.model / 'cameras.json').read_text())
        saved = {_image_name(r['img_name']): r for r in rows}
        if len(saved) != len(rows): raise ValueError('Saved camera names are duplicated')
        binding = json.loads((self.model / 'actual_scene_binding.json').read_text())
        wh = _positive_hw(self.data.get('render_wh', binding.get('render_wh')), 'render_wh')
        names = [_image_name(n) for n in binding['frame_ids']]
        if len(names) != len(binding['timestamps']) or len(set(names)) != len(names):
            raise ValueError('Saved camera/time binding is ambiguous')
        times = {n: _finite_number(t, 'saved model timestamp') for n, t in zip(names, binding['timestamps'])}
        basename = lambda n: n.replace('\\', '/').rsplit('/', 1)[-1]
        headers = {}
        for index, frame in enumerate(self.frames):
            matches = ([frame.image_name] if frame.image_name in times else
                       [name for name in names if basename(name) == basename(frame.image_name)])
            if frame.image_name not in saved or len(matches) != 1:
                raise ValueError('Manifest frame is not in the reconstructed camera set')
            if abs(times[matches[0]] - frame.model_time) > 1e-9:
                raise ValueError('Manifest changes the trained model clock')
            row = saved[frame.image_name]; c2w = np.eye(4)
            c2w[:3, :3] = _matrix(row['rotation'], (3, 3), 'saved camera rotation')
            c2w[:3, 3] = _matrix(row['position'], (3,), 'saved camera position')
            c2w = _pose(c2w, 'saved camera pose')
            source_wh = _positive_hw((row['width'], row['height']), 'saved camera width/height')
            header = dict(uid=index, frame_key=frame.key, image_name=frame.image_name,
                          timestamp=frame.model_time, W=np.linalg.inv(c2w), width=wh[0], height=wh[1])
            for key, axis in [('fx', 0), ('fy', 1), ('cx', 0), ('cy', 1)]:
                header[key] = _finite_number(row[key], key, positive=key in ('fx', 'fy')) * wh[axis] / source_wh[axis]
            if not all(np.isfinite(header[k]) for k in ('fx', 'fy', 'cx', 'cy')) or min(header['fx'], header['fy']) <= 0:
                raise ValueError('Invalid camera intrinsics')
            headers[frame.key] = header
        return headers

    def actor_ids(self, tracks):
        if isinstance(tracks, (str, bytes)):
            raise ValueError('Requested tracks must be a sequence, not one string')
        requested = tuple(tracks)
        if len(set(requested)) != len(requested) or not set(requested) <= set(self.track_to_actor):
            raise ValueError('Requested tracks must be distinct registered identities')
        return tuple(self.track_to_actor[t] for t in requested)


def make_native_camera(header):
    import torch
    from scene.cameras import Camera
    w = header['W']
    camera = Camera(header['uid'], w[:3, :3].T.copy(), w[:3, 3].copy(),
                    cx=header['cx'], cy=header['cy'], fx=header['fx'], fy=header['fy'],
                    image=torch.zeros((3, header['height'], header['width'])),
                    image_name=header['image_name'], resolution=(header['width'], header['height']),
                    timestamp=header['timestamp'])
    if not np.allclose(camera.world_view_transform.detach().cpu().numpy().T, w, atol=1e-5, rtol=0):
        raise ValueError('Native camera conversion changed coordinates')
    return camera
