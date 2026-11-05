"""Replay the adopted support ledger from disk on the accepted parent."""
import hashlib
import numpy as np
import torch

from depth_order import RUN, PARENT, RAWS, read, write
from support import Support


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    trial = Support()
    saved = RUN/'support_accepted.npz'
    expected_path = RUN/'support_verified.json'
    with np.load(saved,allow_pickle=False) as file:
        assert np.array_equal(file['native_row_id'],trial.tr.native_ids.cpu().numpy())
        gate = trial.zero.new_tensor(file['attenuation'])
        assert torch.isfinite(gate).all() and float(gate.min()) >= 0 and float(gate.max()) <= .25000003
    expected = read(expected_path)
    assert expected['status'] == 'PASS'
    black = trial.validate_black(gate)
    street, stats, issues = trial.validate_street(gate)
    assert len(black) == len(street) == len(RAWS) == 49 and not issues
    assert black == expected['black'] and street == expected['street']
    assert stats == expected['chosen']['street'] and trial.tr.exp.unchanged()
    write(RUN/'replay_receipt.json', dict(status='PASS', frames=49, exact_record_replay=True,
          street_states=['keep','delete'], parent_sha256=sha(PARENT/'delivery'/'accepted.npz'),
          attenuation_sha256=sha(saved), validation_sha256=sha(expected_path),
          cuda_library_sha256=sha(RUN/'libreplay_quantile.so'), training_updates=0))


if __name__ == '__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
