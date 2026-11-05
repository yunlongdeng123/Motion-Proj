"""Guarded per-lineage support adjustment from exact forward hit derivatives."""
import numpy as np
from scipy.optimize import linprog
import torch

from depth_order import RUN, RAWS, PILOTS, read, write, save, pool_street, mean, GUARDS
from joint_order import Evaluate
from tradeoff import Tradeoff, LIMITS
from adjoint import ray_adjoint

FIRST_CAP = .16
SECOND_CAP = .12
TOTAL_CAP = .25
ORDER = (137, 97, 112, 113, 105, 130, *[r for r in RAWS if r not in (137,97,112,113,105,130)])


def color(pkg):
    return pkg['rgb'].permute(1, 2, 0).reshape(-1, 3)


class Support(Evaluate):
    def __init__(self):
        super().__init__()
        self.gate = self.zero.clone()
        self.actor_lineage = {}
        for raw in RAWS:
            base = self.tr.lineage[raw]
            parents = self.cache[raw][2]
            append = parents[(self.weight + self.selected.float())[parents] > 0]
            self.actor_lineage[raw] = torch.cat((base, append))
        self.nrows = len(self.zero)

    def actor(self, raw, weight, extra):
        actor = super().actor(raw, weight, extra)
        assert len(actor['features']) == len(self.actor_lineage[raw])
        return dict(actor, opacities=actor['opacities'] * (1 - self.gate[self.actor_lineage[raw], None]))

    def pack_grad(self, raw, trace):
        scene = trace.base()
        tr = self.tr
        d = tr.raw(raw)
        protected = tr.protection[raw]
        masks = dict(known=d['obs'].flatten(), depth=d['depth_valid'].flatten(),
                     outside=protected['outside'].flatten(), target=tr.evidence[raw]['mask'].flatten())
        _, ids, alpha = trace.gather(0, 0.)
        lineage = self.actor_lineage[raw]
        h = ray_adjoint(trace, (ids, alpha, trace.offsets), len(lineage), lineage, masks)
        pix, owner = h['pixels'], h['ids']
        u = h['rgb']
        q = color(scene)
        current_a = scene['alpha'].flatten()
        current_d = scene['depth_sum'].flatten()
        raw_rgb = d['rgb'].permute(1, 2, 0).reshape(-1, 3)
        old = tr.raw38[raw]
        native = protected['untouched']
        g_rgb = lambda ref: (u * (q[pix] - ref[pix]).sign()).mean(dim=1)
        g_alpha = h['alpha'] * (current_a[pix] - old['alpha'].flatten()[pix]).sign()
        den = current_a[pix].clamp_min(1e-5)
        dz = (h['depth_sum'] / den - (current_d[pix] / den.square()) *
              h['alpha'] * (current_a[pix] > 1e-5)) / tr.scale
        depth_residual = (current_d[pix] / den / tr.scale - d['depth'].flatten()[pix]).sign()
        values = [(g_rgb(color(old)), 'outside'), (g_alpha, 'outside'),
                  (g_rgb(raw_rgb), 'known'), (g_rgb(color(native)), 'known'),
                  (dz * depth_residual, 'depth'), (-h['alpha'], 'known'), (-h['pvd'], 'target')]
        if raw in (97, 112, 113):
            # At the tight guards, bound RGB/alpha drift by the sum of
            # individual absolute responses instead of allowing cancellations.
            values[0] = (u.abs().mean(dim=1), 'outside')
            values[1] = (h['alpha'].abs(), 'outside')
        witness = None
        if raw == 137:
            counts = (trace.offsets[1:] - trace.offsets[:-1]).long()
            all_pixels = trace.pixels.long().repeat_interleave(counts, output_size=len(ids))
            held = (masks['outside'] | masks['known'] | masks['depth'])[all_pixels]
            protected_ids = ids[held & (ids < len(lineage))].long()
            witness = torch.zeros_like(self.zero, dtype=torch.bool)
            witness[lineage[protected_ids]] = True
        m = []
        for delta, key in values:
            mask = masks[key]
            take = mask[pix]
            v = delta.new_zeros(self.nrows)
            v.index_add_(0, owner[take], delta[take])
            m.append(v / max(1, int(mask.sum())))
        objective = -m[-1] * int(masks['target'].sum())
        metrics = tr.measure(raw, scene)
        assert not metrics['violations'], (raw, metrics['violations'])
        return metrics, torch.stack(m).cpu().numpy(), objective.cpu().numpy(), h['meta'], witness

    def linearization(self):
        prior = []
        matrix = []
        total = np.zeros(self.nrows)
        notes = []
        protected_rows = None
        for raw in RAWS:
            trace = self.black_trace(raw)
            row, grad, obj, note, witness = self.pack_grad(raw, trace)
            if witness is not None:
                protected_rows = witness.cpu().numpy()
            prior.append(row)
            matrix.append(grad)
            total += obj
            notes.append(dict(raw=raw, **note))
        x, limits = Tradeoff.vector(self, prior)
        lower_index = LIMITS.index('lower_pvd')
        for raw in RAWS:
            old = self.prior['black'][raw-89]['metrics']
            key = (raw-89)*len(LIMITS)+lower_index
            limits[key] = min(limits[key], -mean(old['lower_pvd']) + .01*mean(old['isolated_lower_alpha']))
        matrix = np.concatenate(matrix, axis=0)
        den = sum(r['metrics']['isolated_lower_alpha']['sum'] for r in prior)
        assert np.isfinite(matrix).all() and np.isfinite(total).all() and den > 0
        assert protected_rows is not None
        return prior, matrix, total / den, x, limits, notes, protected_rows

    def validate_black(self, gate, early=False):
        self.gate = gate
        rows = []
        for raw in ORDER:
            out = self.black(raw, self.weight, self.extra)
            m = self.tr.measure(raw, out)
            old = self.prior['black'][raw - 89]['metrics']
            if old['isolated_lower_alpha']['sum'] and (old['lower_pvd']['sum'] - m['metrics']['lower_pvd']['sum']) > .01*old['isolated_lower_alpha']['sum']:
                m['violations'].append('lower_vs_parent')
            rows.append(m)
            if early and m['violations']:
                break
        return sorted(rows, key=lambda r:r['raw'])

    def validate_street(self, gate):
        result = []
        issues = []
        for raw in RAWS:
            row = dict(raw=raw)
            for state in ('keep','delete'):
                delete = state == 'delete'
                self.gate = self.zero
                before = self.street(raw, delete, self.weight, self.extra)
                self.gate = gate
                after = self.street(raw, delete, self.weight, self.extra)
                values, bad = self.street_pair(raw, state, before, after)
                row[state] = values
                if bad:
                    issues.append(dict(raw=raw, state=state, issues=bad))
            result.append(row)
        pooled = pool_street(result)
        for state, m in pooled.items():
            for k in ('road','seam'):
                if m[k + '_after'] > m[k + '_before'] + GUARDS['road_mean']:
                    issues.append(dict(state=state, issues=[k + '_mean']))
        return result, pooled, issues

    def save_render(self, gate, expected_black, expected_street):
        folder = RUN / 'support_render'
        folder.mkdir()
        for raw in RAWS:
            e = self.tr.evidence[raw]
            for name, g in (('before', self.zero), ('after', gate)):
                self.gate = g
                black = self.black(raw, self.weight, self.extra)
                deficit = (e['lower_alpha'] - black['extra'][1]).clamp_min(0)*e['mask']
                save(folder/f'{raw:03d}_{name}.png', black['rgb'])
                save(folder/f'{raw:03d}_{name}_pvd.png', black['extra'][:1].expand(3,-1,-1))
                save(folder/f'{raw:03d}_{name}_blocked.png', torch.stack((deficit,.2*e['mask'],.08*e['mask'])))
                if name == 'after':
                    assert self.tr.measure(raw, black)['metrics'] == expected_black[raw-89]['metrics']
            for state in ('keep','delete'):
                self.gate = self.zero
                before = self.street(raw, state == 'delete', self.weight, self.extra)
                self.gate = gate
                after = self.street(raw, state == 'delete', self.weight, self.extra)
                row, issues = self.street_pair(raw, state, before, after)
                assert row == expected_street[raw-89][state] and not issues
                save(folder/f'{raw:03d}_{state}_before.png', before['rgb'])
                save(folder/f'{raw:03d}_{state}_after.png', after['rgb'])

    def run(self):
        baseline = self.validate_black(self.zero)
        assert baseline == self.prior['black']
        incumbent = self.zero.clone()
        best_black = self.tr.fraction(baseline)
        best_street = self.prior['chosen']['street']['delete']['after_fraction']
        accepted = None
        history = []
        for attempt, cap in enumerate((FIRST_CAP, SECOND_CAP)):
            self.gate = incumbent
            black, jac, objective, x, limits, meta, protected_rows = self.linearization()
            assert abs(self.tr.fraction(black) - best_black) < 1e-8
            use = np.flatnonzero((objective > 1e-9) & ~protected_rows)
            full = incumbent.cpu().numpy()[use]
            room = np.minimum(cap, (TOTAL_CAP - full)/(1-full))
            gap = np.maximum(limits-x, 0.)
            reserve = np.tile([1e-5, 2e-5, 0., 0., 0., 0., 0.], len(RAWS))
            b = np.maximum(gap-reserve, 0.)
            fit = linprog(-objective[use]*10000, A_ub=jac[:,use]*1000, b_ub=b*1000,
                          bounds=list(zip(np.zeros(len(use)), room)), method='highs')
            assert fit.success, fit.message
            step = self.zero.clone()
            step[use] = self.zero.new_tensor(fit.x)
            np.savez_compressed(RUN/f'proposal_{attempt}.npz', native_row_id=self.tr.native_ids.cpu().numpy(),
                                base=incumbent.cpu().numpy(), proposed_relative=step.cpu().numpy())
            trace = dict(round=attempt, baseline=best_black, proposed_gain=float(-fit.fun/10000),
                         eligible=len(use), frozen_raw137=int(protected_rows.sum()), changing=int((step>1e-7).sum()), replay_hits=meta, trials=[])
            history.append(trace)
            print('ADJOINT_SOLVE', attempt, trace['eligible'], trace['changing'], trace['proposed_gain'], flush=True)
            adopted = False
            for strength in (1., .75, .5, .25, .125, .0625):
                gate = incumbent + (1-incumbent)*step*strength
                rows = self.validate_black(gate, early=True)
                bad = [dict(raw=r['raw'], issues=r['violations']) for r in rows if r['violations']]
                rec = dict(strength=strength, black_failures=bad)
                trace['trials'].append(rec)
                if bad:
                    continue
                score = self.tr.fraction(rows)
                rec['black'] = score
                if score <= best_black + 1e-7:
                    continue
                street, pooled, bad = self.validate_street(gate)
                scene_score = pooled['delete']['after_fraction']
                if scene_score <= best_street + 1e-7:
                    bad.append(dict(state='delete', issues=['no_incremental_lower_gain']))
                rec.update(street=pooled, street_failures=bad)
                print('ADJOINT_REPLAY', attempt, strength, score, scene_score, bad, flush=True)
                if not bad:
                    accepted = rec, gate, rows, street
                    incumbent, best_black, best_street = gate, score, scene_score
                    adopted = True
                    break
            write(RUN/'support_search.json', dict(rounds=history, best_black=best_black, best_street=best_street))
            if not adopted:
                break
        if accepted is None:
            write(RUN/'support_verified.json', dict(status='NO_PASS', rounds=history))
            return
        chosen, gate, black, street = accepted
        np.savez_compressed(RUN/'support_accepted.npz', native_row_id=self.tr.native_ids.cpu().numpy(),
                            attenuation=gate.cpu().numpy())
        self.save_render(gate, black, street)
        assert self.tr.exp.unchanged()
        write(RUN/'support_verified.json', dict(status='PASS', frames=49, chosen=chosen, black=black, street=street,
              n_adjusted=int((gate>1e-7).sum()), cap=TOTAL_CAP, training_updates=0,
              replay_98_images=True, native_and_pvd_assets_unchanged=True, shared_rows_across_edits=True))


if __name__ == '__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        Support().run()
