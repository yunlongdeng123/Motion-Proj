"""CPU mathematical checks; no renderer, model, observations or GPU imports."""
import json
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import ndtr

from halfspace_moments import split_moments, NumericalMomentsError


def recover(result):
    parent, children = result['parent'], list(result['children'].values())
    probability = sum(child['prob'] for child in children)
    mean = sum((child['prob']*child['mean'] for child in children if child['prob']), np.zeros(3))
    second = sum((child['prob']*(child['cov']+np.outer(child['mean'], child['mean']))
                  for child in children if child['prob']), np.zeros((3, 3)))
    np.testing.assert_allclose(probability, parent['prob'], rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(mean/parent['prob'], parent['mean'], rtol=2e-9, atol=3e-9)
    np.testing.assert_allclose(second/parent['prob'], parent['cov']+np.outer(parent['mean'], parent['mean']),
                               rtol=2e-9, atol=3e-9)
    np.testing.assert_allclose(sum(c['relative_prob'] for c in children), 1., atol=2e-10, rtol=0)


def same(first, second):
    for key in ('prob', 'mean', 'cov'):
        if first[key] is None or second[key] is None:
            assert first[key] is second[key]
        else:
            np.testing.assert_allclose(first[key], second[key], rtol=3e-10, atol=3e-11)


def independent_integral(rho, a, b):
    """Direct probability/first/second integrals; no Z derivatives or Hessian."""
    spread = np.sqrt(1-rho*rho)
    phi = lambda value: np.exp(-.5*value*value)/np.sqrt(2*np.pi)

    def integrand(x):
        conditional_mean = rho*x
        t = (b-conditional_mean)/spread
        probability, density = ndtr(-t), phi(t)
        y = conditional_mean*probability+spread*density
        yy = (conditional_mean**2+spread**2)*probability+spread*(conditional_mean+b)*density
        return phi(x)*np.array([probability, x*probability, y, x*x*probability, x*y, yy])

    moments, error = quad_vec(integrand, a, np.inf, epsabs=1e-13, epsrel=2e-12)
    assert error < 2e-11
    z = moments[0]
    mean = moments[1:3]/z
    second = np.array([[moments[3], moments[4]], [moments[4], moments[5]]])/z
    return z, mean, second-np.outer(mean, mean)


def main():
    cases = []
    standard = dict(mu=np.zeros(3), cov=np.eye(3), normal=np.array([0., 1., 0.]), cut=0.)
    full = split_moments(**standard)
    recover(full); cases.append('full_1d_split')
    quadrant = split_moments(**standard, inherited_normal=[1., 0., 0.], inherited_cut=0.)
    child = quadrant['children']['keep']
    np.testing.assert_allclose(child['prob'], .25, atol=2e-13)
    np.testing.assert_allclose(child['mean'], [np.sqrt(2/np.pi), np.sqrt(2/np.pi), 0.], atol=2e-12)
    np.testing.assert_allclose(child['cov'], np.diag([1-2/np.pi, 1-2/np.pi, 1.]), atol=2e-12)
    recover(quadrant); cases.append('independent_quadrant_closed_form')

    for rho in (0., .6):
        cov = np.array([[1., rho, 0.], [rho, 1., 0.], [0., 0., .7]])
        result = split_moments(np.zeros(3), cov, [0., 1., 0.], -.1, [1., 0., 0.], .25)
        z, mean, covariance = independent_integral(rho, .25, -.1)
        child = result['children']['keep']
        np.testing.assert_allclose(child['prob'], z, rtol=2e-11, atol=2e-13)
        np.testing.assert_allclose(child['mean'][:2], mean, atol=2e-11)
        np.testing.assert_allclose(child['cov'][:2, :2], covariance, atol=2e-11)
        recover(result)
    cases.append('two_independent_direct_integrals')

    mu = np.array([.2, -.3, .4])
    cov = np.array([[.7, .15, -.08], [.15, 1.2, .22], [-.08, .22, .5]])
    n, n0 = np.array([.7, -.3, .8]), np.array([-1., .4, .1])
    base = split_moments(mu, cov, n, .2, n0, -.6)
    scaled = split_moments(mu, cov, 7.3*n, 7.3*.2, .003*n0, .003*(-.6))
    reversed_plane = split_moments(mu, cov, -4*n, -4*.2, n0, -.6)
    same(base['parent'], scaled['parent'])
    for name in ('keep', 'replace'):
        same(base['children'][name], scaled['children'][name])
    same(base['children']['keep'], reversed_plane['children']['replace'])
    same(base['children']['replace'], reversed_plane['children']['keep'])
    recover(base); recover(scaled); recover(reversed_plane)
    cases.append('raw_normal_scaling_and_sign_flip')

    for inherited, inherited_cut, normal, cut in (
            ([1., 0., 0.], -.4, [3., 0., 0.], .9),
            ([2., 0., 0.], -.8, [-3., 0., 0.], -.9),
            ([1., 0., 0.], .4, [-1., 0., 0.], -.1)):
        result = split_moments(mu, cov, normal, cut, inherited, inherited_cut)
        recover(result)
        assert all(c.get('method', 'empty') != 'bivariate_intersection' for c in result['children'].values())
    assert result['children']['keep']['status'] == 'EMPTY'
    cases.append('parallel_same_opposite_and_empty_intersection')

    # Independently sample the original 3D Gaussian, then condition by raw
    # halfspaces.  This also checks covariance transport into the third axis.
    rng = np.random.default_rng(20261105)
    mc_mu = np.array([.1, -.2, .3])
    mc_cov = np.array([[1., .6, .1], [.6, 1., .2], [.1, .2, .7]])
    result = split_moments(mc_mu, mc_cov, [0., 1., 0.], -.1, [1., 0., 0.], .25)
    sample = rng.multivariate_normal(mc_mu, mc_cov, size=600000)
    keep = (sample[:, 0] >= .25) & (sample[:, 1] >= -.1)
    selected = sample[keep]
    child = result['children']['keep']
    probability_tolerance = 6*np.sqrt(child['prob']*(1-child['prob'])/len(sample))
    assert abs(keep.mean()-child['prob']) < probability_tolerance
    mean_tolerance = 6*np.sqrt(np.diag(child['cov'])/len(selected))
    assert np.all(np.abs(selected.mean(axis=0)-child['mean']) < mean_tolerance)
    np.testing.assert_allclose(np.cov(selected, rowvar=False, ddof=0), child['cov'], atol=.015, rtol=0)
    recover(result); cases.append('fixed_seed_independent_3d_monte_carlo')

    tiny = split_moments(np.zeros(3), np.eye(3), [1., 0., 0.], 9.)
    assert tiny['status'] == 'TINY_CHILD' and tiny['children']['keep']['prob'] == 0
    assert tiny['children']['keep']['status'] == 'TINY' and tiny['children']['keep']['estimated_prob'] > 0
    assert tiny['children']['keep']['mean'] is None
    for bad in (np.diag([1., 0., 1.]), np.diag([1., -1., 1.]), np.array([[1., .1, 0.], [0., 1., 0.], [0., 0., 1.]])):
        try: split_moments(np.zeros(3), bad, [1., 0., 0.], 0.)
        except ValueError: pass
        else: raise AssertionError('Invalid covariance was accepted')
    try: split_moments(np.zeros(3), np.eye(3), [1., 0., 0.], 0., [0., 1., 0.], 9.)
    except NumericalMomentsError: pass
    else: raise AssertionError('Tiny inherited parent was accepted')
    cases.append('explicit_tiny_regions_and_invalid_covariance_errors')
    print(json.dumps(dict(status='PASS', checks=cases, mc_samples=len(sample),
                         mc_probability_error=float(keep.mean()-child['prob']), gpu_used=False)))


if __name__ == '__main__':
    main()
