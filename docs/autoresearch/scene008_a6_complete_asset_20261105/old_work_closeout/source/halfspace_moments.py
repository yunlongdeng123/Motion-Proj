"""FP64 moments of a Gaussian split by one plane, with an optional old clip.

Probabilities are relative to the ORIGINAL untruncated Gaussian.  The old clip
and keep side use >=; the replace side uses < (boundaries have zero mass).
This is a moment calculation, not an opacity/projection conservation proof.
"""
import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.special import log_ndtr, ndtr

TINY_PROB = 1e-14
PARALLEL_TOL = 64*np.finfo(np.float64).eps
LIMITS = (
    'Finite mean, symmetric positive-definite covariance, and finite nonzero planes are required.',
    'Regions with probability <=1e-14 return prob=0 and status=TINY, with an estimate or upper bound; no moments are fabricated.',
    'A tiny/empty parent and non-positive or unreliable conditional covariance raise NumericalMomentsError.',
    'Bivariate quadrature truncates its standard-normal integration axis at +/-12; omitted probability is <=2*Phi(-12).',
    'Near-singular nonparallel constraints are rejected; they are not silently converted to a one-dimensional event.',
    'Means/covariances conserve Gaussian probability moments, not native raster opacity or rendering.',
)


class NumericalMomentsError(ArithmeticError):
    pass


def _plane(normal, cut):
    normal = np.asarray(normal, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(cut):
        raise ValueError('Plane requires finite normal[3] and cut')
    scale = np.max(np.abs(normal))
    if scale == 0:
        raise ValueError('Zero plane normal')
    length = np.linalg.norm(normal/scale)
    unit, bound = (normal/scale)/length, (float(cut)/scale)/length
    if not np.isfinite(bound):
        raise ValueError('Plane scaling overflow')
    return unit, bound


def _phi(value):
    return np.exp(-.5*value*value)/np.sqrt(2*np.pi)


def _zero(status, estimate=0., estimate_kind='computed'):
    return dict(prob=0., estimated_prob=float(estimate), estimate_kind=estimate_kind,
                mean=None, cov=None, status=status)


def _pack(prob, mean, cov, method, error=0.):
    cov = .5*(cov+cov.T)
    if not np.isfinite(prob) or not 0 < prob <= 1:
        raise NumericalMomentsError(f'Invalid probability {prob}')
    if not np.isfinite(mean).all() or not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min() <= 0:
        raise NumericalMomentsError('Nonfinite moments or non-positive conditional covariance; no eigenvalue clipping applied')
    return dict(prob=float(prob), estimated_prob=float(prob), estimate_kind='computed',
                mean=mean, cov=cov, status='OK', method=method, probability_error_estimate=float(error))


def _interval(mu, cov, normal, low, high):
    if low >= high:
        return _zero('EMPTY')
    variance = float(normal@cov@normal)
    sigma = np.sqrt(variance)
    a, b = (low-normal@mu)/sigma, (high-normal@mu)/sigma
    # Stable tail difference, never subtraction of two CDFs rounded near one.
    log_big, log_small = ((log_ndtr(-a), log_ndtr(-b)) if a >= 0
                          else (log_ndtr(b), log_ndtr(a)))
    probability = float(np.exp(log_big)*(-np.expm1(log_small-log_big)))
    if not np.isfinite(probability):
        raise NumericalMomentsError('Unreliable interval probability')
    if probability <= TINY_PROB:
        return _zero('TINY', max(probability, 0.))
    pa, pb = _phi(a), _phi(b)
    first = (pa-pb)/probability
    aphi, bphi = (a*pa if np.isfinite(a) else 0.), (b*pb if np.isfinite(b) else 0.)
    variance_standard = 1+(aphi-bphi)/probability-first*first
    direction = cov@normal/sigma
    return _pack(probability, mu+direction*first,
                 cov+np.outer(direction, direction)*(variance_standard-1), 'one_dimensional_interval')


def _bivariate_probability(a, b, rho, spread):
    # Integrate using the stronger lower bound; splitting at the CDF transition
    # resolves high correlations without random multivariate-CDF integration.
    if b > a:
        a, b = b, a
    lower, upper = max(float(a), -12.), 12.
    points = [lower, upper]
    points += [v for v in (-8., -4., 0., 4., 8.) if lower < v < upper]
    if rho != 0:
        transition, width = b/rho, spread/abs(rho)
        points += [v for v in (transition-8*width, transition, transition+8*width) if lower < v < upper]
    points = sorted(set(points))
    total, error = 0., 2*float(ndtr(-12.))
    with warnings.catch_warnings():
        warnings.simplefilter('error', IntegrationWarning)
        try:
            for left, right in zip(points[:-1], points[1:]):
                value, estimated = quad(lambda x: _phi(x)*ndtr((rho*x-b)/spread),
                                        left, right, epsabs=1e-300, epsrel=2e-11, limit=200)
                total += value
                error += estimated
        except IntegrationWarning as exception:
            raise NumericalMomentsError('Bivariate quadrature did not converge reliably') from exception
    if not np.isfinite(total) or not 0 <= total <= 1 or (total > TINY_PROB and error/total > 1e-8):
        raise NumericalMomentsError(f'Unreliable bivariate probability {total}, estimated error {error}')
    return total, error


def _intersection(mu, cov, normals, cuts):
    first, second = normals
    if np.linalg.norm(np.cross(first, second)) <= PARALLEL_TOL:
        low, high = cuts[0], np.inf
        if first@second > 0:
            low = max(low, cuts[1])
        else:
            high = -cuts[1]
        return _interval(mu, cov, first, low, high)
    projected = normals@cov@normals.T
    sigma = np.sqrt(np.diag(projected))
    rho = float(projected[0, 1]/np.prod(sigma))
    if not -1 < rho < 1 or 1-rho*rho <= PARALLEL_TOL:
        raise NumericalMomentsError('Nearly singular nonparallel bivariate constraints')
    a = (cuts-normals@mu)/sigma
    upper_bound = float(min(ndtr(-a[0]), ndtr(-a[1])))
    if upper_bound <= TINY_PROB:
        return _zero('TINY', upper_bound, 'upper_bound')
    spread = np.sqrt(1-rho*rho)
    probability, error = _bivariate_probability(a[0], a[1], rho, spread)
    if probability <= TINY_PROB:
        return _zero('TINY', probability)
    v = (rho*a-a[::-1])/spread
    phi_a, phi_v, cdf_v = _phi(a), _phi(v), ndtr(v)
    gradient = phi_a*cdf_v/sigma
    hessian = np.empty((2, 2), dtype=np.float64)
    hessian[np.diag_indices(2)] = (a*phi_a*cdf_v-rho*phi_a*phi_v/spread)/sigma**2
    hessian[0, 1] = hessian[1, 0] = phi_a[0]*phi_v[0]/(spread*np.prod(sigma))
    transport = cov@normals.T
    mean = mu+transport@gradient/probability
    curvature = hessian/probability-np.outer(gradient, gradient)/probability**2
    return _pack(probability, mean, cov+transport@curvature@transport.T, 'bivariate_intersection', error)


def split_moments(mu, cov, normal, cut, inherited_normal=None, inherited_cut=None):
    """Return parent and keep/replace child moments, plus numerical diagnostics.

    parent/children have prob, estimated_prob, mean, cov, status.  Each child
    additionally has relative_prob=prob/parent.prob.  EMPTY/TINY children have
    prob=0 and mean/cov=None.  TINY is explicit; it is never an exact empty set.
    """
    mu, cov = np.asarray(mu, dtype=np.float64), np.asarray(cov, dtype=np.float64)
    if mu.shape != (3,) or cov.shape != (3, 3) or not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise ValueError('Expected finite mu[3], cov[3,3]')
    if np.max(np.abs(cov-cov.T)) > 64*np.finfo(float).eps*max(1., np.max(np.abs(cov))):
        raise ValueError('Covariance is not symmetric')
    cov = .5*(cov+cov.T)
    if np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError('Covariance must be positive definite')
    n, b = _plane(normal, cut)
    if (inherited_normal is None) != (inherited_cut is None):
        raise ValueError('Both inherited_normal and inherited_cut are required together')
    if inherited_normal is None:
        parent = _pack(1., mu.copy(), cov.copy(), 'untruncated')
        children = dict(keep=_interval(mu, cov, n, b, np.inf),
                        replace=_interval(mu, cov, -n, -b, np.inf))
    else:
        old_n, old_b = _plane(inherited_normal, inherited_cut)
        parent = _interval(mu, cov, old_n, old_b, np.inf)
        if parent['status'] != 'OK':
            raise NumericalMomentsError('Inherited parent has numerically tiny/empty probability')
        children = {name:_intersection(mu, cov, np.stack((old_n, sign*n)), np.array([old_b, sign*b]))
                    for name, sign in (('keep', 1.), ('replace', -1.))}
    z = parent['prob']
    mean_sum, second_sum, probability_sum = np.zeros(3), np.zeros((3, 3)), 0.
    for child in children.values():
        child['relative_prob'] = child['prob']/z
        probability_sum += child['prob']
        if child['prob']:
            mean_sum += child['prob']*child['mean']
            second_sum += child['prob']*(child['cov']+np.outer(child['mean'], child['mean']))
    parent_second = parent['cov']+np.outer(parent['mean'], parent['mean'])
    diagnostics = dict(probability_error=float(probability_sum-z),
        relative_probability_error=float((probability_sum-z)/z),
        mean_recovery_maxabs=float(np.max(np.abs(mean_sum/z-parent['mean']))),
        second_recovery_maxabs=float(np.max(np.abs(second_sum/z-parent_second))))
    status = 'TINY_CHILD' if any(c['status'] == 'TINY' for c in children.values()) else 'OK'
    relative_error = max(abs(diagnostics['relative_probability_error']),
        diagnostics['mean_recovery_maxabs']/max(1., np.max(np.abs(parent['mean']))),
        diagnostics['second_recovery_maxabs']/max(1., np.max(np.abs(parent_second))))
    if status == 'OK' and relative_error > 2e-7:
        raise NumericalMomentsError(f'Failed split reconstruction: {diagnostics}')
    return dict(status=status, parent=parent, children=children, diagnostics=diagnostics,
                normal=n, cut=b, limits=list(LIMITS))
