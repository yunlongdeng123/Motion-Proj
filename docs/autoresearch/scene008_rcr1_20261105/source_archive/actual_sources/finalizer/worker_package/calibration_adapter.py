"""Fresh fixed-M2 scene adapter for the unchanged 6x5 -> 11x9 solver."""
from pathlib import Path

import numpy as np
import torch

import calibration_method as method
from final_runtime import read, write, sha


def evidence(runtime, raw, delete):
    data = runtime.raw(raw)
    obs = data['obs']
    road = ((obs['road_edit'] & runtime.fixed_ground[raw-89]) if delete else obs['road']).copy()
    protect = obs['protect'] & ~road
    if delete:
        fit = road
    else:
        with np.load(runtime.road_run/'inputs/views'/f'{raw:03d}.npz', allow_pickle=False) as source:
            shadow = source['shadow_proxy'] | source['new_shadow_proxy']
        fit = road & ~shadow
    cv = lambda array: torch.as_tensor(array, device='cuda')
    target = cv(obs['road_edit_rgb']).permute(2, 0, 1).float() if delete else data['rgb']
    if float(target.max()) > 1:
        target = target/255.
    return dict(road=cv(road), fit=cv(fit), protect=cv(protect), a6=cv(obs['visible'][0]),
                a37=cv(obs['visible'][1]), true_rgb=data['rgb'], target=target)


class SceneAdapter:
    def __init__(self, runtime):
        self.runtime = runtime
        self.colour = runtime.baseline_colours
        basis, self.knots = method.world_basis(runtime.road_xy)
        self.basis = self.colour.new_tensor(basis)
        assert torch.allclose(self.basis.sum(1), torch.ones_like(self.basis[:, 0]), atol=2e-6, rtol=0)

    def assemble(self, raw, delete):
        # Every design is built against the fresh road baseline, including the
        # fine design. Fine coefficients are an absolute residual, not an
        # extra residual on top of the coarse result.
        return self.runtime.scene(raw, delete, road_colours=self.colour)

    def render(self, raw, scene):
        # cache_design uses the four group channels to accumulate arbitrary
        # basis weights. Do not interpret those temporary channels as groups.
        return self.runtime.render(raw, scene)

    def colourize(self, scene, colours):
        rows, where, _ = scene
        rows['colors_precomp'] = rows['colors_precomp'].clone()
        rows['colors_precomp'][where] = colours

    def evidence(self, raw, delete):
        return evidence(self.runtime, raw, delete)

    def check_parent(self, raw, state, base, evidence):
        # The retired adapter compared with old ray-switch/street.json. Those
        # numbers belong to another scene. Bind/check this new frozen scene;
        # do not pretend to reproduce the old clipped scene's metrics.
        self.runtime.assert_unchanged()
        assert all(bool(torch.isfinite(base[key]).all()) for key in ('rgb', 'alpha', 'depth_sum', 'extra'))
        assert base['extra'].shape[0] >= 4
        assert float((base['extra'][:4].sum(0)-base['alpha'][0]).abs().max()) < 2e-5


class CoarseExperiment(SceneAdapter, method.CoarseMethod):
    pass


class FineExperiment(SceneAdapter, method.FineMethod):
    cache_design = method.CoarseMethod.cache_design


def geometry_parity(experiment, coefficients, raws=(105, 113, 137)):
    checks = []
    for raw in raws:
        for delete in (False, True):
            scene = experiment.assemble(raw, delete)
            before = experiment.render(raw, scene)
            colours = (experiment.colour+experiment.basis@experiment.basis.new_tensor(coefficients)).clamp(0, 1)
            experiment.colourize(scene, colours)
            after = experiment.render(raw, scene)
            difference = {key: float((after[key]-before[key]).abs().max())
                          for key in ('alpha', 'depth_sum', 'extra')}
            assert max(difference.values()) == 0., (raw, delete, difference)
            checks.append(dict(raw=raw, delete=delete, maxabs=difference))
    return checks


def calibrate(runtime, output, fine=False, coarse=None, preflight=False):
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    method.RUN = output
    method.GRID = (11, 9) if fine else (6, 5)
    method.COARSE = Path(coarse).resolve() if coarse else None
    method.RAWS = (105,) if preflight else tuple(range(89, 138))
    method.HOLDOUT = () if preflight else (89, 95, 101, 105, 113, 121, 125, 130, 137)
    if fine:
        source = read(method.COARSE/'scene_binding.json')
        if (source['binding_sha256'] != runtime.binding_sha256 or source['stage'] != 'coarse6x5'
                or bool(source['preflight_only']) != bool(preflight)):
            raise ValueError('Fine anchor must be the fresh matching coarse calibration')
        with np.load(method.COARSE/'coefficients.npz', allow_pickle=False) as saved:
            assert np.array_equal(saved['colour_baseline'], runtime.baseline_colours.cpu().numpy())
        experiment = FineExperiment(runtime)
    else:
        experiment = CoarseExperiment(runtime)
    metadata = dict(binding_sha256=runtime.binding_sha256, binding=runtime.binding,
                    stage='fine11x9' if fine else 'coarse6x5', preflight_only=bool(preflight),
                    solver_source_sha256=sha(Path(__file__).parent/'calibration_method.py'),
                    baseline_road_sha256=runtime.baseline_hash,
                    coarse_source=None if not fine else dict(path=str(method.COARSE),
                              coefficients_sha256=sha(method.COARSE/'coefficients.npz')),
                    holdout=method.HOLDOUT, BUILD_frames=1 if preflight else 40,
                    old_ray_switch_or_old_coefficients_loaded=False, training_updates=0,
                    fit_method='Original exact alpha*T design + original six IRLS BVLS solves; same bounds/weights/regularization',
                    target='Original keep RGB / reobserved delete-road proxy; same independent fixed ground support',
                    quality_accepted=False)
    write(output/'scene_binding.json', metadata)
    experiment.cache_design()
    designs = sorted((output/'design').glob('*.npz'))
    assert len(designs) == (2 if preflight else 80)
    assert not any(int(path.name[:3]) in method.HOLDOUT for path in designs)
    for state in ('keep', 'delete'):
        fit = protection = 0
        for path in (output/'design').glob(f'*_{state}.npz'):
            with np.load(path, allow_pickle=False) as data:
                assert data['basis'].shape[1] == method.GRID[0]*method.GRID[1]
                assert np.isfinite(data['basis']).all() and np.isfinite(data['residual']).all()
                fit += int(data['fit'].sum())
                protection += int((~data['fit']).sum())
        assert fit > 0, (state, fit, protection)
    coefficients = experiment.solve()
    parity = geometry_parity(experiment, coefficients['world'], raws=(105,) if preflight else (105, 113, 137))
    with np.load(output/'coefficients.npz', allow_pickle=False) as restored:
        assert np.array_equal(restored['world'], coefficients['world'])
        assert np.array_equal(restored['colour_baseline'], runtime.baseline_colours.cpu().numpy())
        assert np.isfinite(restored['world']).all() and np.max(np.abs(restored['world'])) <= .09000001
    runtime.assert_unchanged(full=True)
    write(output/'calibration_receipt.json', dict(status='PASS_CALIBRATION_ENGINEERING',
          preflight_only=bool(preflight), design_files=len(designs), exact_geometry_alpha_depth_parity=parity,
          coefficient_round_trip_exact=True, coefficients_sha256=sha(output/'coefficients.npz'),
          M2_all_27_tensors_bitwise_unchanged_CPU=True, native_and_road_unchanged=True,
          calibration_parameters=int(coefficients['world'].size), gradient_training_updates=0,
          robust_linear_solves=18 if fine else 36,
          quality_accepted=False))
    return coefficients
