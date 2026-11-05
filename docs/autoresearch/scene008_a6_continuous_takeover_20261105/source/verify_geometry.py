"""Check LiDAR-guided support cuts separately and on the signed tradeoff."""
import argparse
import numpy as np
import torch

from geometry_trial import Experiment as Geometry
from tradeoff import Tradeoff
from takeover import RUN, RAWS, read, write, save
from verify import GUARDS


class Check(Geometry, Tradeoff):
    def before_black(self, raw):
        keep = self.geometry
        self.geometry = None
        try:
            return self.black(raw, self.zero, self.zero)
        finally:
            self.geometry = keep

    def images(self, weight, extra, expected, destination):
        destination.mkdir()
        for raw in RAWS:
            evidence = self.tr.evidence[raw]
            old, new = self.before_black(raw), self.black(raw, weight, extra)
            for name, pkg in (('before', old), ('after', new)):
                deficit = (evidence['lower_alpha'] - pkg['extra'][1]).clamp_min(0) * evidence['mask']
                save(destination / f'{raw:03d}_{name}.png', pkg['rgb'])
                save(destination / f'{raw:03d}_{name}_pvd.png', pkg['extra'][:1].expand(3, -1, -1))
                save(destination / f'{raw:03d}_{name}_blocked.png',
                     torch.stack((deficit, .2 * evidence['mask'], .08 * evidence['mask'])))
            for state in ('keep', 'delete'):
                row, failed, old, new = self.comparison(raw, state, weight, extra)
                assert row == expected[raw - 89][state] and not failed
                save(destination / f'{raw:03d}_{state}_before.png', old['rgb'])
                save(destination / f'{raw:03d}_{state}_after.png', new['rgb'])

    def run_check(self, combo):
        self.load_geometry_candidate()
        full_geometry = self.geometry.clone()
        with np.load(RUN / 'geometry_candidate.npz', allow_pickle=False) as f:
            normal, cut = f['normal_local'].copy(), f['cut_local_m'].copy()
        weight, extra = self.zero, self.zero
        prefix = 'geometry'
        minimum_black = self.previous_black['ratio']
        minimum_street = None
        if combo:
            controls = []
            geometry = read(RUN / 'geometry_verified.json')
            if geometry['status'] == 'PASS':
                controls.append(geometry['chosen'])
            if (RUN / 'verify.json').is_file():
                tradeoff = read(RUN / 'verify.json')['chosen']
                controls.append(tradeoff)
                with np.load(RUN / 'accepted.npz', allow_pickle=False) as f:
                    assert bool(f['signed_relative_to_previous'])
                    weight, extra = (self.zero.new_tensor(f[name]) for name in ('strength', 'residual'))
                expected_black = tradeoff['black']
            else:
                with np.load(RUN / 'tradeoff_candidates.npz', allow_pickle=False) as f:
                    assert np.array_equal(f['native_row_id'], self.tr.native_ids.cpu().numpy())
                    weight, extra = (self.zero.new_tensor(f[name][-1]) for name in ('strength', 'residual'))
                expected_black = read(RUN / 'tradeoff_search.json')['best_black']
            self.geometry = None
            b = self.audit(weight, extra)
            assert abs(self.tr.fraction(b) - expected_black) < 1e-7 and not any(r['violations'] for r in b)
            prefix = 'combined'
            if controls:
                minimum_black = max(c['black'] for c in controls)
                minimum_street = max(c['street']['delete']['after_fraction'] for c in controls)
        assert int((full_geometry > 0).sum()) > 0
        trace = []
        winner = None

        def check(name, selected, w, a):
            nonlocal winner
            self.geometry = selected
            black = self.audit(w, a)
            ratio = self.tr.fraction(black)
            failed = [dict(raw=r['raw'], keys=r['violations']) for r in black if r['violations']]
            rec = dict(arm=name, black=ratio, geometry_rows=int((selected > 0).sum()), black_failures=failed)
            trace.append(rec)
            if not failed and ratio > minimum_black + 1e-7:
                rows, stats, issues = self.audit_joint(w, a)
                if minimum_street is not None and stats['delete']['after_fraction'] < minimum_street - 1e-7:
                    issues.append(dict(state='delete', issues=['lower_vs_tradeoff']))
                rec.update(street=stats, street_failures=issues, passes=not issues)
                if not issues and (winner is None or ratio > winner[0]['black']):
                    winner = rec, selected, w, a, black, rows
            print('GEOMETRY_JOINT', prefix, name, ratio, rec.get('passes', False),
                  failed[:4], rec.get('street_failures', [])[:4], flush=True)

        if not combo:
            check('all_planes', full_geometry, weight, extra)
        else:
            check('all_planes', full_geometry, weight, extra)
            if winner is None:
                for f in (.9, .75, .5, .25):
                    check(f'planes_tradeoff_{f}', full_geometry, f * weight, f * extra)
                for variant in (1, 2, 3):
                    selected = torch.where(full_geometry == variant, full_geometry, 0)
                    if selected.any():
                        check(f'one_plane_{variant}', selected, weight, extra)
        out = RUN / f'{prefix}_verified.json'
        if winner is None:
            write(out, dict(status='NO_PASS', trials=trace))
            return
        choice, selected, w, a, black, street = winner
        self.geometry = selected
        self.images(w, a, street, RUN / f'{prefix}_render')
        np.savez_compressed(RUN / f'{prefix}_accepted.npz', native_row_id=self.tr.native_ids.cpu().numpy(),
                            strength=w.cpu().numpy(), residual=a.cpu().numpy(),
                            variant=selected.cpu().numpy(), normal_local=normal, cut_local_m=cut,
                            signed_relative_to_previous=True)
        assert self.tr.exp.unchanged()
        write(out, dict(status='PASS', frames=49, chosen=choice, trials=trace,
              black=black, street=street, guards=GUARDS, training_updates=0, replay_98_images=True,
              native_and_pvd_assets_unchanged=True, shared_rows_across_edits=True))
        print('GEOMETRY_JOINT_ACCEPTED', prefix, choice, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--combo', action='store_true')
    opt = parser.parse_args()
    torch.set_num_threads(1)
    with torch.no_grad():
        Check().run_check(opt.combo)
