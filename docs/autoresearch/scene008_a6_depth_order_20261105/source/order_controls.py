"""Separate retained-ray ordering effects from split/cutoff effects."""
import torch

from depth_order import RUN, RAWS, read, write
from joint_order import Evaluate


@torch.no_grad()
def main():
    e = Evaluate()
    median = read(RUN / 'black_median.json')
    mass = read(RUN / 'black_quantile.json')
    assert len(median['rows']) == len(mass['rows']) == 49
    rows = []
    for a, b in zip(median['rows'], mass['rows']):
        assert a['raw'] == b['raw'] and a['baseline'] == b['baseline']
        rows.append(dict(a, modes={**a['modes'], **b['modes']}))
    summary = {**median['summary'], **mass['summary']}
    write(RUN / 'black_order.json', dict(median, modes=mass['modes'], summary=summary, rows=rows))
    cutoff = []
    for raw in RAWS:
        trace = e.black_trace(raw)
        base = trace.base()
        controls = {}
        for label, mode in (('clip_zero_shift', 3), ('all_zero_shift', 4)):
            output, _ = trace.replay(mode, 0.)
            controls[label] = e.black_row(raw, output, e.tr.measure(raw, base))
        cutoff.append(dict(raw=raw, modes=controls))
    cutoff_summary = {k: e.tr.fraction([r['modes'][k] for r in cutoff]) for k in cutoff[0]['modes']}
    arms = ['clip_wide', 'clip_mass_wide']
    street, parity = e.evaluate_street(arms)
    write(RUN / 'order_controls.json', dict(frames=49, baseline=median['baseline'], black=summary,
        zero_shift_ratio=cutoff_summary, street=street, center_noop_max=parity))
    print('ORDER_CONTROLS', cutoff_summary, parity, {
        k:(a['pool']['delete']['after_fraction'], len(a['issues'])) for k,a in street.items()}, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
