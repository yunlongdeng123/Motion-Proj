"""Verify and replay a6 takeover with the accepted road field in both edit states."""
from pathlib import Path
import csv
import shutil
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
import torch

from takeover import Experiment, RUN, RAWS, PILOTS, metric, read, save, write
sys.path.insert(0, str(RUN.parent / 'scene008_a6_residual_replay_20261105' / 'source'))
from report import image_grid, pooled

FUSION = RUN.parent / 'scene008_a6_local_fusion_20261105'
GUARDS = dict(road_frame=.00005, road_mean=.000002, a6=.001, a37=.0005, protected=.001,
              visible_a37_pvd=.0005, lower_frame=.01)


def mean(item):
    return item['sum'] / max(1, item['count'])


def average(items):
    return sum(m['sum'] for m in items) / max(1, sum(m['count'] for m in items))


def pool_street(rows):
    out = {}
    for state in ('keep', 'delete'):
        selected = [r[state] for r in rows]
        values = {key: average([r[key] for r in selected]) for key in selected[0]}
        values['before_fraction'] = values['lower_before'] / values['isolated']
        values['after_fraction'] = values['lower_after'] / values['isolated']
        out[state] = values
    return out


class Verify(Experiment):
    def comparison(self, raw, state, weight, extra):
        delete = state == 'delete'
        e = self.evidence(raw, delete)
        scene = self.assemble(raw, delete)
        self.colourize(scene, self.road_colour)
        old = self.render(raw, scene)
        new = self.street(raw, delete, weight, extra, scene)
        q = self.tr.evidence[raw]['mask']
        d = self.tr.evidence[raw]['lower_alpha']
        b = (old['extra'][2] > .15).cpu().numpy()
        seam = e['road'] & torch.as_tensor(binary_dilation(b, iterations=2) & ~binary_erosion(b, iterations=2),
                                          device='cuda')
        a37 = torch.zeros_like(e['a37']) if delete else e['a37']
        rgb_change = (new['rgb'] - old['rgb']).abs().mean(0)
        row = dict(isolated=metric(d, q), a6_delta=metric(rgb_change, e['a6']),
                   a37_delta=metric(rgb_change, a37), protected_delta=metric(rgb_change, e['protect']),
                   visible_a37_pvd=metric((new['extra'][3] - old['extra'][3]).clamp_min(0), a37))
        for name, pkg in (('before', old), ('after', new)):
            legacy = pkg['alpha'][0] - pkg['extra'].sum(0)
            assert torch.isfinite(legacy).all() and float(legacy.min()) > -4e-6
            assert not delete or not pkg['extra'][1].any()
            error = (pkg['rgb'] - e['target']).abs().mean(0)
            raw_error = (pkg['rgb'] - e['true_rgb']).abs().mean(0)
            for k, m in (('road', metric(error, e['road'])), ('seam', metric(error, seam)),
                         ('a6', metric(raw_error, e['a6'])), ('lower', metric(pkg['extra'][3], q))):
                row[f'{k}_{name}'] = m
        parent = self.previous_road['rows'][raw - 89]['states'][state]['world']
        for key, previous_key in (('road', 'road'), ('seam', 'seam'), ('a6', 'a6'), ('lower', 'reveal_pvd')):
            m, prior = row[f'{key}_before'], parent[previous_key]
            assert m['count'] == prior['count'] and abs(m['sum'] - prior['sum']) < .002, (raw, state, key)
        failed = [key for key in ('a6_delta', 'a37_delta', 'protected_delta', 'visible_a37_pvd')
                  if mean(row[key]) > GUARDS[key.removesuffix('_delta')]]
        if mean(row['road_after']) > mean(row['road_before']) + GUARDS['road_frame']:
            failed.append('road_frame')
        if row['isolated']['sum'] and (row['lower_before']['sum'] - row['lower_after']['sum']) / row['isolated']['sum'] > GUARDS['lower_frame']:
            failed.append('lower_frame')
        return row, failed, old, new

    def audit_joint(self, weight, extra):
        rows = []
        failed = []
        for raw in RAWS:
            row = dict(raw=raw)
            for state in ('keep', 'delete'):
                row[state], issues, _, _ = self.comparison(raw, state, weight, extra)
                if issues:
                    failed.append(dict(raw=raw, state=state, issues=issues))
            rows.append(row)
        stats = pool_street(rows)
        for state, s in stats.items():
            for key in ('road', 'seam'):
                if s[key + '_after'] > s[key + '_before'] + GUARDS['road_mean']:
                    failed.append(dict(state=state, issues=[key + '_mean']))
        d = stats['delete']
        if d['after_fraction'] <= d['before_fraction'] + 1e-7:
            failed.append(dict(state='delete', issues=['lower_pool']))
        return rows, stats, failed

    def render_outputs(self, weight, extra, expected):
        folder = RUN / 'render'
        folder.mkdir()
        for raw in RAWS:
            e = self.tr.evidence[raw]
            for name, w, a in (('before', self.zero, self.zero), ('after', weight, extra)):
                pkg = self.black(raw, w, a)
                deficit = (e['lower_alpha'] - pkg['extra'][1]).clamp_min(0) * e['mask']
                save(folder / f'{raw:03d}_{name}.png', pkg['rgb'])
                save(folder / f'{raw:03d}_{name}_pvd.png', pkg['extra'][:1].expand(3, -1, -1))
                save(folder / f'{raw:03d}_{name}_blocked.png', torch.stack((deficit, .2 * e['mask'], .08 * e['mask'])))
            for state in ('keep', 'delete'):
                row, failed, old, new = self.comparison(raw, state, weight, extra)
                assert row == expected[raw - 89][state] and not failed
                save(folder / f'{raw:03d}_{state}_before.png', old['rgb'])
                save(folder / f'{raw:03d}_{state}_after.png', new['rgb'])

    def run_verify(self):
        with np.load(RUN / 'candidate.npz', allow_pickle=False) as f:
            assert np.array_equal(f['native_row_id'], self.tr.native_ids.cpu().numpy())
            w, a = (self.zero.new_tensor(f[name]) for name in ('strength', 'residual'))
        source = read(RUN / 'black_search.json')
        trials = []
        winner = None
        for strength, residual in ((1., 1.), (1., 0.), (.75, .75), (.5, .5), (.25, .25)):
            trial_w, trial_a = w * strength, a * residual
            black = self.audit(trial_w, trial_a)
            score = self.tr.fraction(black)
            if strength == residual == 1.:
                assert abs(score - source['scores']['combined']) < 1e-7
            if any(r['violations'] for r in black):
                trials.append(dict(strength=strength, residual=residual, black=score, passes=False))
                continue
            rows, street, issues = self.audit_joint(trial_w, trial_a)
            tests = dict(strength=strength, residual=residual, black=score, street=street,
                         street_failures=issues, passes=not issues)
            trials.append(tests)
            print('VERIFY', strength, residual, score, street['delete']['after_fraction'], issues[:10], flush=True)
            if not issues and (winner is None or score > winner[0]['black']):
                winner = tests, trial_w, trial_a, black, rows
            if not issues and strength == residual == 1.:
                break
        if winner is None:
            write(RUN / 'verify.json', dict(status='NO_PASS', trials=trials, guards=GUARDS))
            return
        selection, weight, extra, black, rows = winner
        assert self.tr.exp.unchanged()
        self.render_outputs(weight, extra, rows)
        np.savez_compressed(RUN / 'accepted.npz', native_row_id=self.tr.native_ids.cpu().numpy(),
                            strength=weight.cpu().numpy(), residual=extra.cpu().numpy())
        write(RUN / 'verify.json', dict(status='PASS', frames=49, chosen=selection, trials=trials,
              black=black, street=rows, guards=GUARDS, training_updates=0, replay_98_images=True,
              native_and_pvd_assets_unchanged=True, shared_rows_across_edits=True))
        print('ACCEPTED', selection, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        Verify().run_verify()
