"""Registered-support handoff partitioned along the measured car-frame normal.

The layers are physical 3-D crop boxes, not per-pixel image gates or semantic
labels on complete Gaussian rows.  Existing ShellWindow conditional integrals
crop each eligible native hit's support; all parent/PVD geometry stays fixed.
Evaluation always uses Registered's original thin measurement_boxes.
"""
from copy import deepcopy
from itertools import product
from pathlib import Path
import sys

import numpy as np


RUN = Path(__file__).resolve().parent.parent
SOURCE = RUN.parent / 'scene008_a6_registered_handoff_20261105'
sys.path.insert(0, str(SOURCE / 'source'))
import registered

core = registered.core
fit_root = registered.fit_root
fit_joint = registered.fit_joint
run_coupled = registered.run_coupled
OBS = registered.OBS
BUILD, HELD = core.BUILD, core.HELD
read, write, PRIMARY = registered.read, registered.write, registered.PRIMARY
fixed_group_mask = registered.fixed_group_mask

# registered's imports patch these globals to its older run.  Apply our run
# afterwards so downstream construction/fitting cannot write into that run.
core.rescue.volume.RUN = RUN
core.rescue.volume.depth_order.takeover.calibrate.RUN = RUN
fit_root.RUN = RUN
fit_joint.RUN = RUN
run_coupled.RUN = RUN
fit_root.group_mask = fixed_group_mask
fit_joint.group_mask = fixed_group_mask
run_coupled.group_mask = fixed_group_mask

NORMAL_HALF_WIDTH_M = .12
BOX_PADDING_M = .28
LAYERS = ('near', 'same', 'far', 'full')


class LayeredAbstain(ValueError):
    """An unusable physical partition; callers can preserve its JSON audit."""
    status = 'ABSTAIN'

    def __init__(self, reason, details):
        self.details = deepcopy(details)
        self.details.update(status=self.status, reason=reason)
        super().__init__('ABSTAIN: ' + reason)

    def audit(self):
        return deepcopy(self.details)


def _physical_layers(ledger):
    """CPU-only geometry and BUILD camera-side preflight, before model setup."""
    center = np.asarray(ledger['center'], dtype=np.float64)
    basis = np.asarray(ledger['basis'], dtype=np.float64)
    thin = np.asarray(ledger['bounds'], dtype=np.float64)
    car_box = np.asarray(ledger['box'], dtype=np.float64)
    audit = {
        'status': 'ABSTAIN',
        'source_run': str(SOURCE), 'output_run': str(RUN),
        'normal_plane_m': 0., 'normal_split_m': [-NORMAL_HALF_WIDTH_M, NORMAL_HALF_WIDTH_M],
        'car_box_padding_m': BOX_PADDING_M,
        'build': [int(raw) for raw in BUILD],
        'held': [int(raw) for raw in HELD],
        'selection_used_heldout': False,
        'camera_records': [],
        'coordinates': '(xyz_local_m-center)@basis; column 2 is the fitted plane normal',
        'operator': 'Car-frame normal layers; exact conditional support crop on each registered native hit',
        'image_gating': 'No per-pixel image gate; no measurement-depth gating at edit time',
        'registration': 'Original persistent native-row eligibility is reused; linkage is not semantic purity',
        'evaluation': 'Original thin measurement_boxes and registered.fixed_group_mask are unchanged',
        'parent_and_pvd_geometry_changed': False,
    }
    shape_ok = (center.shape == (3,) and basis.shape == (3, 3)
                and thin.shape == (2, 3) and car_box.shape == (3,))
    if not shape_ok or not all(np.isfinite(value).all() for value in (center, basis, thin, car_box)):
        raise LayeredAbstain('invalid_group_geometry', audit)
    if (np.any(car_box <= 0) or np.any(thin[1] <= thin[0])
            or not np.allclose(basis.T @ basis, np.eye(3), atol=2e-5, rtol=0)
            or not np.isclose(np.linalg.det(basis), 1., atol=2e-5, rtol=0)):
        raise LayeredAbstain('invalid_rigid_group_frame_or_bounds', audit)
    audit.update(center_m=center.tolist(), basis=basis.tolist(),
                 measurement_box=thin.tolist(), measured_car_box_lwh_m=car_box.tolist(),
                 registered_eligible_count=int(np.count_nonzero(ledger['eligible'])))

    signs, cosines = [], []
    for raw in BUILD:
        try:
            with np.load(OBS / 'frames' / f'{raw:03d}.npz', allow_pickle=False) as frame:
                camera_from_local = frame['world_to_camera'] @ frame['world_from_local'][0]
            if camera_from_local.shape != (4, 4) or not np.isfinite(camera_from_local).all():
                raise ValueError('nonfinite or malformed camera transform')
            origin = np.linalg.inv(camera_from_local)[:3, 3]
        except (OSError, KeyError, ValueError, np.linalg.LinAlgError) as error:
            audit.update(failed_camera_raw=int(raw), camera_error=str(error))
            raise LayeredAbstain('cannot_establish_build_camera_side', audit) from error
        offset = origin - center
        local = offset @ basis
        distance = float(local[2])
        norm = float(np.linalg.norm(offset))
        cosine = distance / norm if norm > 0 else None
        sign = int(np.sign(distance))
        audit['camera_records'].append({
            'raw': int(raw), 'origin_object_local_m': origin.tolist(),
            'origin_group_m': local.tolist(), 'normal_distance_m': distance,
            'camera_sign': sign, 'normal_cosine': cosine,
        })
        signs.append(sign)
        cosines.append(cosine)
    if not signs or 0 in signs or len(set(signs)) != 1:
        audit['camera_signs'] = signs
        raise LayeredAbstain('build_cameras_do_not_share_one_nonzero_plane_side', audit)
    camera_sign = signs[0]
    audit.update(camera_sign=camera_sign,
                 normal_cosine_min=float(min(cosines)), normal_cosine_max=float(max(cosines)),
                 abs_normal_cosine_min=float(min(abs(cosine) for cosine in cosines)))

    # Full normal extent is determined only by the original measured car box,
    # padded on each face.  Never reuse the registered mode-quantile prism.
    corners = np.asarray(list(product((-1., 1.), repeat=3))) * (car_box / 2 + BOX_PADDING_M)
    projected = (corners - center) @ basis
    lower, upper = float(projected[:, 2].min()), float(projected[:, 2].max())
    negative_cut = float(np.clip(-NORMAL_HALF_WIDTH_M, lower, upper))
    positive_cut = float(np.clip(NORMAL_HALF_WIDTH_M, lower, upper))
    negative = [lower, negative_cut]
    same = [negative_cut, positive_cut]
    positive = [positive_cut, upper]
    ranges = {'near': positive if camera_sign > 0 else negative,
              'same': same,
              'far': negative if camera_sign > 0 else positive,
              'full': [lower, upper]}
    boxes = {}
    for layer, interval in ranges.items():
        box = thin.copy()
        box[:, 2] = interval
        boxes[layer] = box
        assert np.array_equal(box[:, :2], thin[:, :2])
    audit.update(
        status='PASS', reason='consistent_build_camera_side_and_fixed_physical_partition',
        camera_near_normal_direction='positive' if camera_sign > 0 else 'negative',
        padded_car_corners_object_local_m=corners.tolist(),
        padded_car_corners_group_m=projected.tolist(),
        normal_ranges={key: list(value) for key, value in ranges.items()},
        layer_boxes={key: value.tolist() for key, value in boxes.items()},
        split_clipped_to_full_range=(negative_cut != -NORMAL_HALF_WIDTH_M
                                    or positive_cut != NORMAL_HALF_WIDTH_M),
        partition='near+same+far=full; shared interval endpoints carry zero Gaussian mass',
    )
    return boxes, audit


class Layered(registered.Registered):
    """Registered eligibility with switchable near/same/far physical support."""

    def __init__(self, ledger, layer='near'):
        if layer not in LAYERS:
            raise ValueError(f'Unknown layer {layer!r}; expected one of {LAYERS}')
        boxes, self._layer_audit = _physical_layers(ledger)
        # Camera-side failures abstain before GPU/model initialization.  The
        # input ledger is never modified; Registered keeps its original mask.
        super().__init__(ledger)
        self.layer_boxes = {key: self.zero.new_tensor(value[None]) for key, value in boxes.items()}
        self.camera_sign = self._layer_audit['camera_sign']
        self.status = 'PASS'
        self.set_layer(layer)

    def set_layer(self, layer):
        """Select one [1,2,3] crop tensor; existing bundles must be rebuilt."""
        if layer not in self.layer_boxes:
            raise ValueError(f'Unknown layer {layer!r}; expected one of {LAYERS}')
        self.layer = layer
        self.boxes = self.layer_boxes[layer]
        # Current Coupled.bundle builds fresh windows.  Clear its reserved
        # cache as well so later cache use cannot retain a different crop.
        self.bundles.clear()
        return self

    def audit(self):
        """Return JSON-safe metadata only; no file writes or rendering."""
        result = deepcopy(self._layer_audit)
        result.update(layer=self.layer,
                      layer_box=result['layer_boxes'][self.layer],
                      normal_range=result['normal_ranges'][self.layer])
        return result
