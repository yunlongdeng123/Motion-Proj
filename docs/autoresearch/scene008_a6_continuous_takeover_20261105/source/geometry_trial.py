"""Audit persistent local lower-support planes on the continuous takeover."""
import hashlib

import numpy as np
from scipy.spatial import cKDTree
import torch

from takeover import EPS, RAWS, RESIDUAL, RUN, Experiment as Continuous, read, write


PLANES = ('vertical', 'vertical_clearance', 'footprint_xz')
CLEARANCE_M = .03
EXTRA_CLEARANCE_M = .05
FOOTPRINT_WEIGHT = .30
MAX_LIDAR_MASS = .10
WITNESS_VIEWS = (137, 97)
GROUP_EDGES = (0, 24, 72, 168, 360, 744, 1512, 3048)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(actual, expected):
    assert len(actual) == len(expected) == len(RAWS)
    max_delta = 0.
    for row, reference in zip(actual, expected):
        assert row['raw'] == reference['raw'] and not row['violations'], row
        assert row['metrics'].keys() == reference['metrics'].keys()
        for key, value in row['metrics'].items():
            old = reference['metrics'][key]
            delta = abs(value['sum'] - old['sum'])
            assert value['count'] == old['count'] and delta < .001, (row['raw'], key, delta)
            max_delta = max(max_delta, delta)
    return max_delta


class Experiment(Continuous):
    def __init__(self):
        super().__init__()
        self.geometry = None
        self.codes = {}

    def load_parent(self):
        self.geometry = None
        path = RUN / 'candidate.npz'
        self.parent_sha256 = digest(path)
        with np.load(path, allow_pickle=False) as data:
            assert np.array_equal(data['native_row_id'], self.tr.native_ids.cpu().numpy())
            weight = self.zero.new_tensor(data['strength'])
            residual = self.zero.new_tensor(data['residual'])
        assert torch.isfinite(weight).all() and torch.isfinite(residual).all()
        assert (weight >= 0).all() and (residual >= 0).all() and (weight + residual <= 1).all()
        self.parent_rows = read(RUN / 'black_search.json')['combined']
        self.parent_delta = replay(self.audit(weight, residual), self.parent_rows)
        return weight, residual

    def fit(self):
        tr = self.tr
        with np.load(RESIDUAL / 'shared_planes.npz', allow_pickle=False) as data:
            assert np.array_equal(data['native_row_id'], tr.native_ids.cpu().numpy())
            order = torch.as_tensor(data['order'], device=self.zero.device)
            prior_normal = self.zero.new_tensor(data['normal_local'])
            prior_cut = self.zero.new_tensor(data['cut_local_m'])
            target, known, outside, lidar = self.zero.new_tensor(data['masses'])
        order = order[~self.options[order] & (lidar[order] <= MAX_LIDAR_MASS)]
        assert len(order) > 0

        # Native centers express each row in the same canonical a6 frame as LiDAR.
        center = prior_normal.new_zeros((len(order), 3))
        count = self.zero.new_zeros(len(order))
        for raw in RAWS:
            rows = self.cache[raw][0]
            inverse_pose = torch.linalg.inv(tr.raw(raw)['pose'])
            active = rows['mask'][order]
            local = rows['means3D'][order] @ inverse_pose[:3, :3].T + inverse_pose[:3, 3]
            center += torch.where(active[:, None], local, 0.)
            count += active
        order, center, count = (field[count > 0] for field in (order, center, count))
        center /= count[:, None]
        assert len(order) > 0 and torch.isfinite(center).all()

        lower = float(tr.anchors[:, 2].min())
        z_cut = lower - CLEARANCE_M
        assert abs(z_cut - tr.zcut) < 1e-6
        vertical = torch.zeros_like(prior_normal)
        vertical[:, 2] = 1.
        vertical_cut = torch.full_like(prior_cut, z_cut)

        # Project the old signed plane onto XZ at the row's canonical y. Then
        # tilt it toward gravity and keep a buffer below nearby measured points.
        xz = prior_normal[order].clone()
        y = xz[:, 1].clone()
        xz[:, 1] = 0.
        size = xz.norm(dim=1).clamp_min(1e-6)
        xz /= size[:, None]
        xz_cut = (prior_cut[order] - y * center[:, 1]) / size
        mixed = (1. - FOOTPRINT_WEIGHT) * vertical[order] + FOOTPRINT_WEIGHT * xz
        mixed_cut = (1. - FOOTPRINT_WEIGHT) * z_cut + FOOTPRINT_WEIGHT * xz_cut
        size = mixed.norm(dim=1)
        mixed /= size[:, None]
        mixed_cut /= size
        _, neighbors = cKDTree(tr.anchors[:, :2]).query(center[:, :2].cpu().numpy(), k=32, workers=1)
        nearby = mixed.new_tensor(tr.anchors[neighbors])
        protected = (nearby * mixed[:, None]).sum(2).amin(1) - CLEARANCE_M
        tilted_normal = vertical.clone()
        tilted_cut = vertical_cut.clone()
        tilted_normal[order] = mixed
        tilted_cut[order] = torch.minimum(mixed_cut, protected)

        # Use only order information from the two tightest guards. The frozen
        # masks and thresholds still decide each 49-frame acceptance.
        pressure = self.zero.clone()
        for raw in WITNESS_VIEWS:
            rows = self.cache[raw][0]
            original_features = dict(rows, features=rows['features'][:, :2])
            mass = tr.row_weights(raw, original_features, tr.protection[raw]['outside'])
            pressure.index_add_(0, tr.lineage[raw], mass)
        ranking = target / (known + outside + 4 * lidar + 8 * pressure + .01)
        order = order[torch.argsort(ranking[order], descending=True, stable=True)]
        normals = torch.stack((vertical, vertical, tilted_normal))
        cuts = torch.stack((vertical_cut, vertical_cut - EXTRA_CLEARANCE_M, tilted_cut))
        assert torch.isfinite(normals[:, order]).all() and torch.isfinite(cuts[:, order]).all()
        self.design = dict(lidar_lower_m=lower, clearance_m=CLEARANCE_M,
                           additional_vertical_clearance_m=EXTRA_CLEARANCE_M,
                           footprint_weight=FOOTPRINT_WEIGHT, local_lidar_neighbors=32,
                           maximum_row_lidar_mass=MAX_LIDAR_MASS,
                           candidate_count=len(order), pressure_views=WITNESS_VIEWS)
        return order, normals, cuts

    def prepare_codes(self, specs):
        self.codes = {}
        for raw in RAWS:
            native = self.cache[raw][0]
            lineage = self.tr.lineage[raw]
            codes = []
            for variant, eligible, normals, cuts in specs:
                assert not (eligible & self.options).any()
                positions, parents, code = self.tr.camera_code(raw, native, lineage, eligible, normals, cuts)
                assert (native['features'][positions, 6:] == 0).all()
                codes.append((variant, positions, parents, code))
            self.codes[raw] = codes

    def actor(self, raw, weight, extra):
        rows = super().actor(raw, weight, extra)
        if self.geometry is None:
            return rows
        features = None
        for variant, positions, parents, code in self.codes[raw]:
            active = self.geometry[parents] == variant
            if not active.any():
                continue
            if features is None:
                features = rows['features'].clone()
            features[positions[active], 6:] = code[active]
        return rows if features is None else dict(rows, features=features)

    def audit_geometry(self, geometry, weight, extra, *, stop=False):
        previous = self.geometry
        self.geometry = geometry
        try:
            rows = self.audit(weight, extra, stop=stop)
        finally:
            self.geometry = previous
        for row in rows:
            old = self.parent_rows[row['raw'] - 89]['metrics']
            lower = row['metrics']['lower_pvd']['sum']
            isolated = old['isolated_lower_alpha']['sum']
            if isolated and old['lower_pvd']['sum'] - lower > .01 * isolated:
                row['violations'].append('lower_vs_continuous_parent')
        return rows

    def search_geometry(self, order, weight, extra):
        selected = self.zero.to(torch.int64)
        best_rows = self.parent_rows
        best = self.tr.fraction(best_rows)
        trace = []
        edges = (*[edge for edge in GROUP_EDGES if edge < len(order)], len(order))
        groups = [order[start:end] for start, end in zip(edges, edges[1:])]
        for number, group in enumerate((order, *groups)):
            for variant, label in enumerate(PLANES, 1):
                proposal = selected.clone()
                proposal[group] = variant
                rows = self.audit_geometry(proposal, weight, extra, stop=True)
                violations = [dict(raw=row['raw'], keys=row['violations'])
                              for row in rows if row['violations']]
                score = self.tr.fraction(rows) if len(rows) == len(RAWS) else None
                take = score is not None and not violations and score > best + EPS
                trace.append(dict(group=number, variant=label, rows=len(group),
                                  frames_read=len(rows), ratio=score, violations=violations, adopted=take))
                if take:
                    selected, best, best_rows = proposal, score, rows
            print('GEOMETRY_GROUP', number, len(group), best, flush=True)
        return selected, best_rows, trace

    def prepare_replay(self, selected, normal, cut):
        eligible = selected > 0
        self.prepare_codes([(variant, eligible & (selected == variant), normal, cut)
                            for variant in range(1, len(PLANES) + 1)
                            if bool((selected == variant).any())])
        self.geometry = selected

    def load_geometry_candidate(self):
        weight, residual = self.load_parent()
        result = read(RUN / 'geometry_black_search.json')
        assert result['parent_candidate_sha256'] == self.parent_sha256
        with np.load(RUN / 'geometry_candidate.npz', allow_pickle=False) as data:
            assert np.array_equal(data['native_row_id'], self.tr.native_ids.cpu().numpy())
            selected = torch.as_tensor(data['variant'], device=self.zero.device)
            assert ((selected >= 0) & (selected <= len(PLANES))).all()
            self.prepare_replay(selected, self.zero.new_tensor(data['normal_local']),
                                self.zero.new_tensor(data['cut_local_m']))
        replay(self.audit_geometry(selected, weight, residual), result['rows'])
        return weight, residual

    def run(self):
        weight, residual = self.load_parent()
        order, normals, cuts = self.fit()
        eligible = self.zero.bool()
        eligible[order] = True
        self.prepare_codes([(index + 1, eligible, normals[index], cuts[index])
                            for index in range(len(PLANES))])
        selected, best, trace = self.search_geometry(order, weight, residual)
        ids = torch.where(selected > 0)[0]
        normal = torch.zeros_like(normals[0])
        cut = torch.zeros_like(cuts[0])
        normal[ids] = normals[selected[ids] - 1, ids]
        cut[ids] = cuts[selected[ids] - 1, ids]
        self.prepare_replay(selected, normal, cut)
        final = self.audit_geometry(selected, weight, residual)
        maximum_delta = replay(final, best)
        score = self.tr.fraction(final)
        assert score >= self.tr.fraction(self.parent_rows) - EPS and self.tr.exp.unchanged()
        assert digest(RUN / 'candidate.npz') == self.parent_sha256
        np.savez_compressed(RUN / 'geometry_candidate.npz', native_row_id=self.tr.native_ids.cpu().numpy(),
                            variant=selected.cpu().numpy(), normal_local=normal.cpu().numpy(),
                            cut_local_m=cut.cpu().numpy())
        write(RUN / 'geometry_black_search.json', dict(frames=len(RAWS), training_updates=0,
              parent_candidate_sha256=self.parent_sha256, parent_replay_max_metric_sum_delta=self.parent_delta,
              compact_replay_max_metric_sum_delta=maximum_delta, parent_ratio=self.tr.fraction(self.parent_rows),
              ratio=score, selected={name:int((selected == index).sum()) for index, name in enumerate(PLANES, 1)},
              design=self.design, rows=final, trace=trace))
        print('GEOMETRY_FINAL', score, len(ids), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        Experiment().run()
