"""Read the selected delivery ledger and independently rerender all evaluation domains."""
import hashlib

import numpy as np
import torch

from takeover import RUN, read, write
from verify_geometry import Check


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = RUN / 'delivery'
    summary = read(root / 'summary.json')
    expected = read(root / 'verify.json')
    file = root / 'accepted.npz'
    trial = Check()
    with np.load(file, allow_pickle=False) as f:
        assert bool(f['signed_relative_to_previous'])
        assert np.array_equal(f['native_row_id'], trial.tr.native_ids.cpu().numpy())
        weight, extra = (trial.zero.new_tensor(f[k]) for k in ('strength', 'residual'))
        if 'variant' in f.files:
            sel = torch.as_tensor(f['variant'], device='cuda')
            trial.prepare_replay(sel, trial.zero.new_tensor(f['normal_local']),
                                 trial.zero.new_tensor(f['cut_local_m']))
    black = trial.audit(weight, extra)
    street, stats, failed = trial.audit_joint(weight, extra)
    assert len(black) == len(street) == 49 and not failed
    assert not any(r['violations'] for r in black)
    assert black == expected['black'] and street == expected['street']
    assert stats == expected['chosen']['street'] and trial.tr.exp.unchanged()
    write(root / 'replay_receipt.json', dict(status='PASS', frames=49, method=summary['selected'],
        exact_record_replay=True, street_states=['keep', 'delete'],
        ledger_sha256=digest(file), verification_sha256=digest(root / 'verify.json')))


if __name__ == '__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
