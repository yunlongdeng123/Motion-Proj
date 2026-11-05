"""Evaluate the order-only change in both edit states and save paired renders."""
from scipy.ndimage import binary_dilation, binary_erosion
import torch

from depth_order import Experiment, RUN, RAWS, MODES, read, write, metric, save, GUARDS, pool_street, mean


class Evaluate(Experiment):
    def street_pair(self, raw, state, before, after):
        delete = state == 'delete'
        e = self.evidence(raw, delete)
        q = self.tr.evidence[raw]['mask']
        den = self.tr.evidence[raw]['lower_alpha']
        b = (before['extra'][2] > .15).cpu().numpy()
        seam = e['road'] & torch.as_tensor(binary_dilation(b, iterations=2) & ~binary_erosion(b, iterations=2),
                                          device='cuda')
        a37 = torch.zeros_like(e['a37']) if delete else e['a37']
        diff = (after['rgb'] - before['rgb']).abs().mean(0)
        row = dict(isolated=metric(den, q), a6_delta=metric(diff, e['a6']),
                   a37_delta=metric(diff, a37), protected_delta=metric(diff, e['protect']),
                   visible_a37_pvd=metric((after['extra'][3] - before['extra'][3]).clamp_min(0), a37))
        for name, pkg in (('before', before), ('after', after)):
            residual = pkg['alpha'][0] - pkg['extra'].sum(0)
            assert torch.isfinite(residual).all() and float(residual.min()) > -4e-6
            assert not delete or not pkg['extra'][1].any()
            err = (pkg['rgb'] - e['target']).abs().mean(0)
            rgb = (pkg['rgb'] - e['true_rgb']).abs().mean(0)
            row[f'road_{name}'] = metric(err, e['road'])
            row[f'seam_{name}'] = metric(err, seam)
            row[f'a6_{name}'] = metric(rgb, e['a6'])
            row[f'lower_{name}'] = metric(pkg['extra'][3], q)
        reference = self.prior['street'][raw - 89][state]
        for k in ('road', 'seam', 'a6', 'lower'):
            old, got = reference[k + '_after'], row[k + '_before']
            assert old['count'] == got['count'] and abs(old['sum'] - got['sum']) < .002, (raw, state, k)
        keys = ('a6_delta', 'a37_delta', 'protected_delta', 'visible_a37_pvd')
        bad = [k for k in keys if mean(row[k]) > GUARDS[k.removesuffix('_delta')]]
        if mean(row['road_after']) > mean(row['road_before']) + GUARDS['road_frame']:
            bad.append('road_frame')
        if row['isolated']['sum'] and (row['lower_before']['sum'] - row['lower_after']['sum']) / row['isolated']['sum'] > GUARDS['lower_frame']:
            bad.append('lower_frame')
        return row, bad

    def evaluate_street(self, arms):
        stats = {arm: {'rows': [], 'issues': []} for arm in arms}
        parity = []
        for raw in RAWS:
            row = {arm: {'raw': raw} for arm in arms}
            for state in ('keep', 'delete'):
                trace = self.street_trace(raw, state == 'delete')
                base = trace.base()
                noop, _ = trace.replay(0, 0.)
                p = self.parity(base, noop)
                assert p['depth_sum'] < 5e-5 and max(p[k] for k in ('rgb', 'alpha', 'extra')) < 5e-6, (raw, state, p)
                parity.append(p)
                for arm in arms:
                    result, _ = trace.replay(*MODES[arm])
                    values, issues = self.street_pair(raw, state, base, result)
                    row[arm][state] = values
                    if issues:
                        stats[arm]['issues'].append(dict(raw=raw, state=state, issues=issues))
            for arm in arms:
                stats[arm]['rows'].append(row[arm])
        for arm, item in stats.items():
            item['pool'] = pool_street(item['rows'])
            for state, s in item['pool'].items():
                for key in ('road', 'seam'):
                    if s[key + '_after'] > s[key + '_before'] + GUARDS['road_mean']:
                        item['issues'].append(dict(state=state, issues=[key + '_mean']))
            s = item['pool']['delete']
            if s['after_fraction'] <= s['before_fraction'] + 1e-7:
                item['issues'].append(dict(state='delete', issues=['no_lower_gain']))
        return stats, {k: max(p[k] for p in parity) for k in parity[0]}

    def run_joint(self):
        black = read(RUN / 'black_order.json')
        summary = black['summary']
        arms = [k for k, v in summary.items() if not v['failures'] and v['ratio'] > black['baseline'] + 1e-7]
        if not arms:
            write(RUN / 'street_order.json', dict(status='NO_BLACK_PASS', selected=None, arms={}))
            return
        street, p = self.evaluate_street(arms)
        candidates = [arm for arm in arms if not street[arm]['issues']]
        winner = max(candidates, key=lambda arm: (summary[arm]['ratio'], street[arm]['pool']['delete']['after_fraction'])) if candidates else None
        write(RUN / 'street_order.json', dict(status='PASS' if winner else 'NO_STREET_PASS', selected=winner,
              arms=street, guards=GUARDS, center_noop_max=p, frames=49))
        print('STREET', {name:(item['pool']['delete']['after_fraction'], item['issues'][:8]) for name,item in street.items()},
              'SELECTED', winner, flush=True)
        if winner:
            self.render_winner(winner, black, street[winner])

    def render_winner(self, name, black, street):
        output = RUN / 'render'
        output.mkdir()
        for raw in RAWS:
            trace = self.black_trace(raw)
            base = trace.base()
            after, _ = trace.replay(*MODES[name])
            row = self.black_row(raw, after, self.tr.measure(raw, base))
            expected = black['rows'][raw - 89]['modes'][name]
            assert row['metrics'] == expected['metrics'] and not row['violations']
            e = self.tr.evidence[raw]
            for arm, pkg in (('before', base), ('after', after)):
                gap = (e['lower_alpha'] - pkg['extra'][1]).clamp_min(0) * e['mask']
                save(output / f'{raw:03d}_{arm}.png', pkg['rgb'])
                save(output / f'{raw:03d}_{arm}_pvd.png', pkg['extra'][:1].expand(3, -1, -1))
                save(output / f'{raw:03d}_{arm}_blocked.png', torch.stack((gap, .2*e['mask'], .08*e['mask'])))
            for state in ('keep', 'delete'):
                trace = self.street_trace(raw, state == 'delete')
                base = trace.base()
                after, _ = trace.replay(*MODES[name])
                row, issues = self.street_pair(raw, state, base, after)
                assert row == street['rows'][raw-89][state] and not issues
                save(output / f'{raw:03d}_{state}_before.png', base['rgb'])
                save(output / f'{raw:03d}_{state}_after.png', after['rgb'])
        assert self.tr.exp.unchanged()
        write(RUN / 'receipt.json', dict(status='PASS', mode=name, frames=49, street_states=['keep','delete'],
              replay_matches=True, unchanged_assets=True, training_updates=0))


if __name__ == '__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        Evaluate().run_joint()
