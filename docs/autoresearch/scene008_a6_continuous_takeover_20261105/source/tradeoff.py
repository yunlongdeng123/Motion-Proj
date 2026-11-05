"""Find guard-aware swaps among persistent support components."""
import sys
import numpy as np
from scipy.optimize import linprog
import torch

from verify import Verify, GUARDS
from takeover import RUN, RAWS, RESIDUAL, read, write
from fuse_a6 import PARAMS
from replay_retirement import OTHER_ALPHA, OTHER_RGB

LIMITS = ('outside_rgb_delta', 'outside_alpha_delta', 'known_rgb', 'known_rgb_change',
          'measured_depth', 'known_alpha', 'lower_pvd')


class Tradeoff(Verify):
    include_accepted = True
    prefix = 'tradeoff'
    round_budget = 4

    def actor(self, raw, weight, extra):
        return super().actor(raw, self.selected.float() + weight, extra)

    def vector(self, rows):
        values = []
        bounds = []
        for r in rows:
            raw = r['raw']
            m = r['metrics']
            parent = self.previous_black['rows'][raw - 89]['metrics']
            earlier = self.tr.latest['outcomes'][self.tr.latest['chosen']]['rows'][raw - 89]['metrics']
            native = self.tr.previous['rows'][raw - 89]['variants']['native']
            mean = self.tr.mean
            target = max(mean(parent['lower_pvd']), mean(earlier['lower_pvd']))
            target -= .01 * mean(parent['isolated_lower_alpha'])
            values.extend(mean(m[k]) if k not in ('known_alpha', 'lower_pvd') else -mean(m[k]) for k in LIMITS)
            bounds.extend((OTHER_RGB, OTHER_ALPHA, mean(native['known_rgb']) + PARAMS['rgb_guard'],
                           PARAMS['rgb_guard'], mean(native['measured_depth']) + PARAMS['depth_guard_m'],
                           -PARAMS['evidence_guard_fraction'] * mean(native['known_alpha']), -target))
        return np.asarray(values), np.asarray(bounds)

    def groups(self):
        with np.load(RESIDUAL / 'shared_planes.npz', allow_pickle=False) as f:
            assert np.array_equal(f['native_row_id'], self.tr.native_ids.cpu().numpy())
            target, known, outside, lidar = self.zero.new_tensor(f['masses'])
            residual = torch.as_tensor(f['order'], device='cuda')
        pressure = []
        for raw in (137, 97, 112):
            # A center-sort attribution proxy is only used to stratify groups.
            # The solver's gain and all constraints come from actual renders.
            rows = self.tr.latest_rows[raw]
            marks = self.tr.row_weights(raw, rows, self.tr.protection[raw]['outside'])
            collapsed = torch.zeros_like(target)
            collapsed.index_add_(0, self.tr.lineage[raw], marks)
            pressure.append(collapsed / max(1, int(self.tr.protection[raw]['outside'].sum())))
        pressure = torch.stack(pressure)
        primary = pressure.argmax(0)
        ratio = target / (known + outside + 4 * lidar + .01)
        groups = []
        for name, ids, low, high, splits in (
                ('release', self.selected.nonzero().flatten(), -1., 0., 6),
                ('matched', (self.options & ~self.selected).nonzero().flatten(), 0., 1., 8),
                ('residual', residual[~self.options[residual]], 0., .2, 8)):
            for focus in range(3):
                matching = ids[primary[ids] == focus]
                ordered = matching[torch.argsort(ratio[matching], descending=True, stable=True)]
                for sub in torch.tensor_split(ordered, splits):
                    if len(sub):
                        groups.append((name, focus, sub, low, high))
        return groups

    def fields(self, groups, x):
        w, a = self.zero.clone(), self.zero.clone()
        for group, value in zip(groups, x):
            name, _, ids, _, _ = group
            if name == 'residual':
                a[ids] = float(value)
            else:
                w[ids] = float(value)
        return w, a

    def run_tradeoff(self):
        groups = self.groups()
        x = np.zeros(len(groups))
        base = self.audit(self.zero, self.zero)
        assert not any(r['violations'] for r in base)
        before = self.tr.fraction(base)
        assert abs(before - getattr(self, 'start_ratio', self.previous_black['ratio'])) < 1e-7
        state = base
        history = []
        finalists = []
        print('TRADEOFF_GROUPS', len(groups), flush=True)
        for outer in range(self.round_budget):
            vector, limits = self.vector(state)
            score = self.tr.fraction(state)
            jac, gain, bounds = [], [], []
            for index, (name, focus, ids, low, high) in enumerate(groups):
                amp = (.30 if name != 'residual' else .07) * (.65 ** outer)
                h = min(amp, high - x[index])
                if h < amp / 2:
                    h = -min(amp, x[index] - low)
                assert h != 0
                probe = x.copy()
                probe[index] += h
                rows = self.audit(*self.fields(groups, probe))
                v, _ = self.vector(rows)
                jac.append((v - vector) / h)
                gain.append((self.tr.fraction(rows) - score) / h)
                bounds.append((max(low - x[index], -amp), min(high - x[index], amp)))
            jac = np.stack(jac, axis=1)
            gain = np.asarray(gain)
            headroom = np.maximum(limits - vector, 0.)
            # Keep a small reserve when available, and preserve the actual
            # almost-saturated constraints in the line-search replay.
            margin = np.maximum(0., headroom - np.tile([2e-5, 2e-5, 0, 0, 0, 0, 0], len(RAWS)))
            fit = linprog(-gain, A_ub=jac, b_ub=margin, bounds=bounds, method='highs')
            assert fit.success, fit.message
            line = []
            for factor in (1., .75, .5, .25, .125, .0625):
                candidate = x + factor * fit.x
                fields = self.fields(groups, candidate)
                rows = self.audit(*fields)
                proposed = self.tr.fraction(rows)
                failed = [dict(raw=r['raw'], issues=r['violations']) for r in rows if r['violations']]
                line.append(dict(scale=factor, ratio=proposed, failures=failed))
                if not failed and proposed > score + 1e-7:
                    x, state = candidate, rows
                    finalists.append((outer, x.copy(), rows))
                    break
            history.append(dict(round=outer, score=score, linear_improvement=float(-fit.fun),
                                max_gain=float(gain.max()), line=line,
                                nonzero_groups=int(np.count_nonzero(x))))
            print('TRADEOFF_ROUND', outer, score, float(-fit.fun), line, flush=True)
        ledger = [self.fields(groups, p) for _, p, _ in finalists]
        if ledger:
            np.savez_compressed(RUN / f'{self.prefix}_candidates.npz',
                                native_row_id=self.tr.native_ids.cpu().numpy(),
                                rounds=np.asarray([outer for outer, _, _ in finalists]),
                                strength=np.stack([w.cpu().numpy() for w, _ in ledger]),
                                residual=np.stack([a.cpu().numpy() for _, a in ledger]))
        search = dict(history=history, groups=[
            dict(name=name, focus=focus, n=len(ids)) for name, focus, ids, _, _ in groups],
            starting_black=before, best_black=self.tr.fraction(state))
        write(RUN / f'{self.prefix}_search.json', search)
        if ledger:
            self.verify_tradeoff(search)

    def verify_tradeoff(self, search):
        with np.load(RUN / f'{self.prefix}_candidates.npz', allow_pickle=False) as f:
            assert np.array_equal(f['native_row_id'], self.tr.native_ids.cpu().numpy())
            finalists = [(int(i), self.zero.new_tensor(w), self.zero.new_tensor(a))
                         for i, w, a in zip(f['rounds'], f['strength'], f['residual'])]
        chosen = None
        trials = []
        for outer, w, a in reversed(finalists):
            black = self.audit(w, a)
            for factor in (1., .75, .5):
                trial_w, trial_a = factor * w, factor * a
                adjusted = self.audit(trial_w, trial_a) if factor != 1 else black
                if any(r['violations'] for r in adjusted):
                    continue
                rows, stats, issues = self.audit_joint(trial_w, trial_a)
                result = dict(round=outer, scale=factor, black=self.tr.fraction(adjusted), street=stats,
                              street_failures=issues, passes=not issues)
                trials.append(result)
                print('TRADEOFF_STREET', result['black'], stats['delete']['after_fraction'], issues[:6], flush=True)
                if not issues and (chosen is None or result['black'] > chosen[0]['black']):
                    chosen = result, trial_w, trial_a, adjusted, rows
                if not issues and factor == 1.:
                    break
            if chosen is not None and chosen[0]['round'] == outer and chosen[0]['scale'] == 1.:
                break
        write(RUN / 'tradeoff.json', dict(search, trials=trials))
        if chosen is None:
            return
        selection, w, a, black, rows = chosen
        self.render_outputs(w, a, rows)
        np.savez_compressed(RUN / 'accepted.npz', native_row_id=self.tr.native_ids.cpu().numpy(),
                            strength=w.cpu().numpy(), residual=a.cpu().numpy(),
                            signed_relative_to_previous=True)
        assert self.tr.exp.unchanged()
        write(RUN / 'verify.json', dict(status='PASS', frames=49, chosen=selection, trials=trials,
              black=black, street=rows, guards=GUARDS, training_updates=0, replay_98_images=True,
              native_and_pvd_assets_unchanged=True, shared_rows_across_edits=True))
        print('TRADEOFF_ACCEPTED', selection, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        trial = Tradeoff()
        if '--verify' in sys.argv:
            trial.verify_tradeoff(read(RUN / 'tradeoff_search.json'))
        else:
            trial.run_tradeoff()
