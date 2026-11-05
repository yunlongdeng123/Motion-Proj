"""Guarded continuous takeover of frozen a6 native row support."""
from pathlib import Path
import sys
import numpy as np
import torch

RUN = Path(__file__).resolve().parent.parent
SWITCH = RUN.parent / 'scene008_a6_ray_switch_20261105'
RESIDUAL = RUN.parent / 'scene008_a6_ray_support_20261105'
FINE = RUN.parent / 'scene008_ray_road_colour_fine_20261105'
sys.path.insert(0, str(FINE.parent / 'scene008_ray_road_colour_20261105' / 'source'))
import calibrate

calibrate.RUN = RUN
calibrate.GRID = (11, 9)
from calibrate import (Experiment as Road, RAWS, PILOTS, merge_rows, metric, private,
                       read, save, select_rows, tag, write)
from ray_support import padded
from replay_retirement import OTHER_ALPHA, OTHER_RGB

PRIORITY = (137, 97, 112, 96, 89, 105, 113, 130)
ORDER = (*PRIORITY, *(raw for raw in RAWS if raw not in PRIORITY))
EPS = 1e-7


class Experiment(Road):
    include_accepted = False
    def __init__(self):
        super().__init__()
        self.tr = self.trial
        self.previous_black = read(SWITCH / 'result.json')
        self.previous_road = read(FINE / 'result.json')
        with np.load(FINE / 'coefficients.npz', allow_pickle=False) as f:
            assert np.array_equal(f['colour_baseline'], self.colour.cpu().numpy())
            self.road_colour = (self.colour + self.basis @ self.basis.new_tensor(f['world'])).clamp(0, 1)
        tr = self.tr
        self.options = tr.base_options['selected']
        self.open = self.options if self.include_accepted else self.options & ~self.selected
        self.zero = self.selected.float() * 0
        self.cache = {}
        for raw in RAWS:
            native = tr.latest_rows[raw] if self.include_accepted else tr.switch_rows(raw, self.selected)
            positions, lineage, code = tr.code[raw]
            already = self.selected[lineage] & (not self.include_accepted)
            metadata = padded(native)
            metadata[positions[already], 6:] = code[already]
            base = dict(native, features=metadata)
            new = self.open[lineage]
            positions, parents, code = positions[new], lineage[new], code[new]
            original = select_rows(tr.current[raw], positions)
            shader = padded(original)
            shader[:, 6:] = code
            original = dict(original, features=shader)
            patch = select_rows(tr.patch(raw, native), tr.lower)
            marks = patch['features'].new_ones((len(patch['features']), 2))
            features = torch.cat((patch['features'][:, :2], marks), 1)
            patch = dict(patch, features=padded(dict(patch, features=features)))
            self.cache[raw] = base, original, parents, patch

    def actor(self, raw, weight, extra):
        base, original, parents, _ = self.cache[raw]
        lineage = self.tr.lineage[raw]
        present = torch.zeros_like(self.open)
        present[parents] = True
        w = weight * present
        factors = (1 - w - extra)[lineage]
        base = dict(base, mask=base['mask'] & (factors > 0),
                    opacities=base['opacities'] * factors[:, None])
        appended = weight[parents]
        active = appended > 0
        if not active.any():
            return base
        source = select_rows(original, active)
        source['opacities'] *= appended[active, None]
        return merge_rows(base, source)

    def black(self, raw, weight, extra):
        actor = self.actor(raw, weight, extra)
        return private(self.tr, raw, merge_rows(actor, self.cache[raw][3]))

    def audit(self, weight, extra, *, stop=False):
        rows = []
        for raw in ORDER:
            record = self.tr.measure(raw, self.black(raw, weight, extra))
            old = self.previous_black['rows'][raw - 89]['metrics']
            m = record['metrics']
            den = old['isolated_lower_alpha']['sum']
            if den and (old['lower_pvd']['sum'] - m['lower_pvd']['sum']) / den > .01:
                record['violations'].append('lower_vs_parent')
            rows.append(record)
            if stop and record['violations']:
                break
        return sorted(rows, key=lambda r: r['raw'])

    def compare_controls(self):
        before = self.audit(self.zero, self.zero)
        for a, b in zip(before, self.previous_black['rows']):
            assert a['raw'] == b['raw'] and not a['violations']
            for key, m in a['metrics'].items():
                old = b['metrics'][key]
                assert m['count'] == old['count'] and abs(m['sum'] - old['sum']) < .001, (a['raw'], key, m, old)
        all_rows = self.audit(self.open.float(), self.zero)
        old = self.tr.check_switch(self.options, early=False)
        max_delta = 0.
        for a, b in zip(all_rows, old):
            for key in a['metrics']:
                one, two = a['metrics'][key], b['metrics'][key]
                assert one['count'] == two['count']
                max_delta = max(max_delta, abs(one['sum'] - two['sum']) / max(1, one['count']))
        assert max_delta < 5e-6, max_delta
        print('CONTROLS', self.tr.fraction(before), self.tr.fraction(all_rows), max_delta, flush=True)
        return before, all_rows, max_delta

    def search(self, before, field, groups, cap, levels, other):
        best = self.tr.fraction(before)
        trace = []
        # The first pass distributes modest strength. The reverse pass can
        # spend margin on complementary groups and back off groups if useful.
        for stage, ordering in ((0, groups), (1, groups[::-1])):
            for ids in ordering:
                if not len(ids):
                    continue
                current = float(field[ids][0])
                proposals = [v for v in levels if v <= cap and abs(v - current) > EPS]
                if stage == 0:
                    proposals = [v for v in proposals if v <= cap / 2]
                chosen = current
                for strength in proposals:
                    candidate = field.clone()
                    candidate[ids] = strength
                    args = (candidate, other) if cap == 1 else (other, candidate)
                    rows = self.audit(*args, stop=True)
                    failures = [dict(raw=r['raw'], keys=r['violations']) for r in rows if r['violations']]
                    score = self.tr.fraction(rows) if len(rows) == len(RAWS) else None
                    take = score is not None and not failures and score > best + EPS
                    trace.append(dict(stage=stage, first=int(ids[0]), n=len(ids), strength=strength,
                                      ratio=score, fail=failures, adopted=take))
                    if take:
                        chosen, best, before = strength, score, rows
                field[ids] = chosen
            print('PASS', cap, stage, best, flush=True)
        return field, before, trace

    def run(self):
        prior, complete, max_delta = self.compare_controls()
        controls = {}
        weight = self.zero.clone()
        best = prior
        for strength in (.125, .25, .5, .75):
            candidate = self.open.float() * strength
            rows = self.audit(candidate, self.zero)
            controls[str(strength)] = rows
        match_order = self.tr.base_options['order']
        match_order = match_order[self.open[match_order]]
        weight, matched, trace = self.search(best, weight, torch.tensor_split(match_order, 14), 1,
                                             (.125, .25, .5, .75, 1.), self.zero)
        # Non-matched rows form a separate control. Share small opacity
        # reductions by earlier frozen blocker/protection attribution rank.
        with np.load(RESIDUAL / 'shared_planes.npz', allow_pickle=False) as f:
            assert np.array_equal(f['native_row_id'], self.tr.native_ids.cpu().numpy())
            order = torch.as_tensor(f['order'], device='cuda')
        order = order[~self.options[order]]
        extra, after, trace_extra = self.search(matched, self.zero.clone(), torch.tensor_split(order, 16),
                                                .2, (.025, .05, .1, .2), weight)
        assert self.tr.fraction(after) >= self.tr.fraction(matched)
        np.savez_compressed(RUN / 'candidate.npz', native_row_id=self.tr.native_ids.cpu().numpy(),
                            strength=weight.cpu().numpy(), residual=extra.cpu().numpy())
        write(RUN / 'black_search.json', dict(prior=prior, full_switch=complete, controls=controls,
                 matched=matched, combined=after, trace_matched=trace, trace_residual=trace_extra,
                 endpoint_max_metric_mean_delta=max_delta,
                 scores={key:self.tr.fraction(rows) for key,rows in
                         [('prior', prior), ('full_switch', complete), ('matched', matched), ('combined', after)]},
                 training_updates=0, count_mixed=int((weight > 0).sum()), count_residual=int((extra > 0).sum())))

    def street(self, raw, delete, weight, extra, baseline=None):
        if baseline is None:
            original = self.assemble(raw, delete)
            self.colourize(original, self.road_colour)
        else:
            original = baseline
        rows, road_slice, env = original
        count_actor = len(self.cache[raw][0]['features'])
        # The parent street path starts with static, then a6, a37 if present,
        # road and PVD. Swap a6 only. Keep feature tags and clip metadata.
        static_n = len(self.tr.n.static.get_xyz)
        assert (rows['features'][static_n:static_n + count_actor, 2] == 1).all()
        actor = self.actor(raw, weight, extra)
        tagged = tag(dict(actor, features=actor['features'][:, :2]), 0)
        tagged['features'][:, 6:] = actor['features'][:, 6:]
        left = select_rows(rows, slice(0, static_n))
        right = select_rows(rows, slice(static_n + count_actor, None))
        joint = merge_rows(merge_rows(left, tagged), right)
        return self.render(raw, (joint, road_slice, env))


if __name__ == '__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        Experiment().run()
