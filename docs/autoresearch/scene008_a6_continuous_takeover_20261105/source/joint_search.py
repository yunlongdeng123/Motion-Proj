"""Re-optimize support reallocation from the verified LiDAR geometry."""
import numpy as np
import torch

from takeover import RUN, read, write
from verify import GUARDS
from verify_geometry import Check


class Joint(Check):
    prefix = 'joint_optimized'
    round_budget = 2

    def verify_tradeoff(self, search):
        with np.load(RUN / f'{self.prefix}_candidates.npz', allow_pickle=False) as f:
            assert np.array_equal(f['native_row_id'], self.tr.native_ids.cpu().numpy())
            finalists = [(int(i), self.zero.new_tensor(w), self.zero.new_tensor(a))
                         for i, w, a in zip(f['rounds'], f['strength'], f['residual'])]
        prior = read(RUN / 'geometry_verified.json')['chosen']
        prior_street = prior['street']['delete']['after_fraction']
        chosen = None
        trials = []
        for number, weight, extra in reversed(finalists):
            for scale in (1., .75, .5, .25):
                w, a = scale * weight, scale * extra
                black = self.audit(w, a)
                score = self.tr.fraction(black)
                issues = [dict(raw=r['raw'], keys=r['violations']) for r in black if r['violations']]
                row = dict(round=number, scale=scale, black=score, black_failures=issues)
                trials.append(row)
                if issues or score <= prior['black'] + 1e-7:
                    continue
                streets, stats, failed = self.audit_joint(w, a)
                if stats['delete']['after_fraction'] <= prior_street + 1e-7:
                    failed.append(dict(state='delete', issues=['lower_vs_geometry']))
                row.update(street=stats, street_failures=failed, passes=not failed)
                print('JOINT_OPTIMIZED', number, scale, score, stats['delete']['after_fraction'], failed, flush=True)
                if not failed and (chosen is None or score > chosen[0]['black']):
                    chosen = row, w, a, black, streets
                    break
            if chosen is not None and chosen[0]['round'] == number and chosen[0]['scale'] == 1.:
                break
        write(RUN / f'{self.prefix}_trials.json', dict(search, trials=trials))
        if chosen is None:
            write(RUN / f'{self.prefix}_verified.json', dict(status='NO_PASS', trials=trials))
            return
        choice, weight, extra, black, street = chosen
        with np.load(RUN / 'geometry_candidate.npz', allow_pickle=False) as f:
            assert np.array_equal(f['variant'], self.geometry.cpu().numpy())
            normal, cut = f['normal_local'].copy(), f['cut_local_m'].copy()
        self.images(weight, extra, street, RUN / f'{self.prefix}_render')
        np.savez_compressed(RUN / f'{self.prefix}_accepted.npz', native_row_id=self.tr.native_ids.cpu().numpy(),
                            strength=weight.cpu().numpy(), residual=extra.cpu().numpy(),
                            variant=self.geometry.cpu().numpy(), normal_local=normal, cut_local_m=cut,
                            signed_relative_to_previous=True)
        assert self.tr.exp.unchanged()
        write(RUN / f'{self.prefix}_verified.json', dict(status='PASS', frames=49, chosen=choice, trials=trials,
              black=black, street=street, guards=GUARDS, training_updates=0, replay_98_images=True,
              native_and_pvd_assets_unchanged=True, shared_rows_across_edits=True))


if __name__ == '__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        joint = Joint()
        joint.load_geometry_candidate()
        joint.start_ratio = read(RUN / 'geometry_verified.json')['chosen']['black']
        joint.run_tradeoff()
