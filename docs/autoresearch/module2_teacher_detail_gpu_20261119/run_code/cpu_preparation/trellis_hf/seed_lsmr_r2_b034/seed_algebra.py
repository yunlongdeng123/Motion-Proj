"""CPU-only TRELLIS HF seed algebra. No model inference, optimizer or native import."""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy import ndimage as ndi


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blur(value, sigma_cells=3., truncate=3.):
    """Same finite separable/replicate Gaussian operator as the H parameterization."""
    radius = int(np.ceil(sigma_cells * truncate))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-.5 * (x / sigma_cells)**2)
    kernel /= kernel.sum()
    out = ndi.convolve1d(np.asarray(value, np.float64), kernel, axis=1, mode='nearest')
    return ndi.convolve1d(out, kernel, axis=0, mode='nearest')


def output_from_raw(raw, mH, *, bound=.04, sigma_cells=3.):
    return np.asarray(mH, np.float64)[..., None] * ((.5 * bound) * (raw - blur(raw, sigma_cells)))


def taper(mask, width_cells):
    d = ndi.distance_transform_edt(np.asarray(mask, bool))
    return np.sin(.5 * np.pi * np.clip(d / width_cells, 0, 1))**2 * mask


def dc_box_target(raw_hf, window, weight, bound):
    """Bounded target with zero effective-field DC; no additive base-color change."""
    raw_hf = np.asarray(raw_hf, np.float64)
    w = np.asarray(window, np.float64) * np.asarray(weight, np.float64)
    if w.sum() == 0:
        return np.zeros_like(raw_hf)
    lo = raw_hf.min((0, 1)) - bound
    hi = raw_hf.max((0, 1)) + bound
    for _ in range(70):
        mid = .5 * (lo + hi)
        mass = np.sum(w[..., None] * np.clip(raw_hf - mid, -bound, bound), axis=(0, 1))
        lo = np.where(mass > 0, mid, lo)
        hi = np.where(mass > 0, hi, mid)
    return window[..., None] * np.clip(raw_hf - .5 * (lo + hi), -bound, bound)


def spectral_inverse(target, sigma_cells=3., ridge=.001):
    """Deterministic regularized inverse of I-G; this is input encoding, not photo fitting."""
    radius = int(np.ceil(3 * sigma_cells))
    pad = 2 * radius
    rhs = np.pad(target, ((pad, pad), (pad, pad), (0, 0)))
    H, W = rhs.shape[:2]
    x = np.arange(-radius, radius + 1)
    k = np.exp(-.5 * (x / sigma_cells)**2)
    k /= k.sum()
    kernel = np.zeros((H, W))
    kernel[:len(k), :len(k)] = k[:, None] * k[None, :]
    kernel = np.roll(kernel, (-radius, -radius), axis=(0, 1))
    response = np.maximum(0., 1. - np.fft.fft2(kernel).real)
    inverse = response / (response**2 + ridge**2)
    raw = np.fft.ifft2(np.fft.fft2(rhs, axes=(0, 1)) * inverse[..., None], axes=(0, 1)).real
    return raw[pad:pad + target.shape[0], pad:pad + target.shape[1]]


def field_stats(field, mask, *, sigma_cells=3.):
    f = np.asarray(field, np.float64)
    mask = np.asarray(mask, bool)
    if not mask.any():
        return dict(nodes=0, RGB_RMS=0., abs_max=0., DC_RGB=[0., 0., 0.], LF_RMS=0.)
    low = blur(f, sigma_cells)
    return dict(nodes=int(mask.sum()), RGB_RMS=float(np.sqrt(np.mean(f[mask]**2))),
        abs_max=float(np.abs(f[mask]).max()), DC_RGB=f[mask].mean(0).tolist(),
        LF_RMS=float(np.sqrt(np.mean(low[mask]**2))))


def encode_h_seed(bandlimited_rgb, prior_legal, mH_combined, *, source_valid=None,
                  spacing_m=.02, hp_sigma_m=.06, model_bound=.04, prior_bound=.02,
                  taper_m=.12, raw_seed_cap=.9, inverse_ridge=.001):
    """Return H initialization only. Full G support stays inside legal unobserved teacher domain.

    `prior_legal` must come from the new independent permission ledger, never legacy source/known.
    `mH_combined` is the actual common-method output multiplier, including observed H where allowed.
    """
    rgb = np.asarray(bandlimited_rgb, np.float64)
    legal = np.asarray(prior_legal, bool).copy()
    mH = np.asarray(mH_combined, np.float64)
    if rgb.shape[:2] != legal.shape or mH.shape != legal.shape or rgb.shape[-1] != 3:
        raise ValueError('RGB/permission/common mH shape mismatch')
    if not np.isfinite(rgb).all() or not np.isfinite(mH).all() or np.any((mH < 0) | (mH > 1)):
        raise ValueError('nonfinite RGB or invalid mH')
    if source_valid is not None:
        legal &= np.asarray(source_valid, bool)
    legal &= mH > 0
    sigma = hp_sigma_m / spacing_m
    radius = int(np.ceil(3 * sigma))
    support_radius = radius + 1  # one extra cell for the actual bilinear world-field reader
    safe = ndi.binary_erosion(legal, structure=np.ones((2 * support_radius + 1, 2 * support_radius + 1)), border_value=0)
    window = taper(safe, taper_m / spacing_m)
    target_window = taper(legal, taper_m / spacing_m)
    raw_hf = rgb - blur(rgb, sigma)
    intrinsic_target = dc_box_target(raw_hf, target_window, mH, prior_bound)
    target = mH[..., None] * intrinsic_target
    inverse_raw = spectral_inverse(intrinsic_target / (.5 * model_bound), sigma, inverse_ridge)
    raw = np.clip(inverse_raw * window[..., None], -raw_seed_cap, raw_seed_cap)
    actual = output_from_raw(raw, mH, bound=model_bound, sigma_cells=sigma)
    scale = min(1., prior_bound * (1 - 1e-6) / max(float(np.abs(actual).max()), 1e-30))
    raw *= scale
    # Float32 is the actual H parameter storage; replay its tanh rather than claiming exact injection.
    seed = np.ascontiguousarray(np.arctanh(raw), dtype=np.float32)
    replay_raw = np.tanh(seed.astype(np.float64))
    actual = output_from_raw(replay_raw, mH, bound=model_bound, sigma_cells=sigma)
    unmodulated = (.5 * model_bound) * (replay_raw - blur(replay_raw, sigma))
    if np.any(actual[~legal] != 0):
        raise AssertionError('H blur leaks outside teacher-only permission')
    naive_raw = np.clip(intrinsic_target / (.5 * model_bound), -raw_seed_cap, raw_seed_cap) * window[..., None]
    naive = output_from_raw(naive_raw, mH, bound=model_bound, sigma_cells=sigma)
    error = actual - target
    denom = max(int(legal.sum()) * 3, 1)
    report = dict(legal_nodes=int(legal.sum()), safe_raw_parameter_nodes=int(safe.sum()),
        model_H_bound=model_bound, desired_effective_prior_bound=prior_bound,
        Gaussian_blur_radius_cells=radius, bilinear_reader_halo_cells=1, raw_support_erosion_cells=support_radius,
        raw_seed_cap=raw_seed_cap, raw_seed_abs_max=float(np.abs(replay_raw).max()),
        tanh_derivative_min=float((1 - replay_raw**2).min()), seed_latent_abs_max=float(np.abs(seed).max()),
        amplitude_feasibility_scale=scale, inverse_ridge=inverse_ridge,
        clipped_inverse_components=int((np.abs(inverse_raw * window[..., None]) >= raw_seed_cap).sum()),
        actual=field_stats(actual, legal, sigma_cells=sigma), target=field_stats(target, legal, sigma_cells=sigma),
        target_MSE=float(np.sum(error[legal]**2) / denom), target_abs_max=float(np.abs(error).max()),
        naive_double_highpass_target_MSE=float(np.sum((naive[legal] - target[legal])**2) / denom),
        actual_outside_prior_permission_abs_max=float(np.abs(actual[~legal]).max()) if (~legal).any() else 0.,
        target_scope='full legal teacher boundary window before raw-support projection; not an already eroded nearzero target',
        operation='deterministic spectral inverse + full-support projection + finite tanh encoding; no photo loss or optimizer',
        exact_target_injection_claimed=False, prior_loss_added=False, model_or_native_executed=False)
    return dict(H_seed_latent=seed, raw_seed_tanh=replay_raw.astype(np.float32), actual_effective_H=actual.astype(np.float32),
        unmodulated_H=unmodulated,
        desired_effective_H=target.astype(np.float32), raw_safe_support=safe, prior_legal=legal,
        boundary_window=window.astype(np.float32), target_boundary_window=target_window.astype(np.float32),
        mH_combined=np.ascontiguousarray(mH)), report


def bilinear(field, uv):
    uv = np.asarray(uv, np.float64)
    shape = uv.shape[:-1]
    p = uv.reshape(-1, 2)
    return np.stack([ndi.map_coordinates(field[..., k], [p[:, 1], p[:, 0]], order=1, mode='constant', cval=0., prefilter=False)
                     for k in range(field.shape[-1])], axis=-1).reshape(*shape, field.shape[-1])
