"""Visible-only road-paint geometry for a single-frame inpainting adapter.

The module has no target-image or scene/actor/frame argument.  Extended paint
corridors are uncertain priors, never observations or hard composited lines.
"""
import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

CHANNELS = ['visible_white', 'visible_yellow', 'inferred_white',
            'inferred_yellow', 'tangent_cos2', 'tangent_sin2',
            'inferred_confidence', 'road_support']


def _paint(image, visible):
    lum = image @ np.asarray([.2126, .7152, .0722], np.float32)
    median = float(np.median(lum[visible])) if visible.any() else .3
    threshold = float(np.clip(median + .09, .35, .85))
    r, g, b = image.transpose(2, 0, 1)
    white = (lum > threshold) & (np.ptp(image, axis=2) < .20) & visible
    yellow = ((r-b > .055) & (g-b > .035) & (r-g < .22)
              & (lum > max(median+.035, .25)) & visible)
    return white, yellow, threshold


def _candidates(paint, height, width):
    segments = cv2.HoughLinesP(
        np.uint8(paint)*255, 1, np.pi/720,
        threshold=max(12, round(width*.018)),
        minLineLength=max(12, round(height*.10)),
        maxLineGap=max(5, round(height*.06)))
    if segments is None:
        return []
    candidates = []
    for x1, y1, x2, y2 in np.asarray(segments).reshape(-1, 4):
        dy, dx = float(y2-y1), float(x2-x1)
        length = np.hypot(dx, dy)
        # The current branch covers longitudinal paint, not stop bars or arrows.
        if abs(dy) < .30*length or abs(dy) < height*.08:
            continue
        a = dx/dy
        candidates.append((float(a), float(x1-a*y1), float(length)))
    return sorted(candidates, key=lambda x: (-x[2], x[0], x[1]))


def _fit(paint, visible, candidates):
    height, width = paint.shape
    yy, xx = np.nonzero(paint)
    fits = []
    merge_distance = max(1.5, width*.004)
    for a0, b0, _ in candidates:
        if len(fits) >= 12:
            break
        # Avoid merging adjacent parallel stripes of a double boundary.
        if any(abs(np.arctan(a0)-np.arctan(f['a'])) < np.deg2rad(4)
               and max(abs((a0-f['a'])*y+b0-f['b'])
                       for y in [.25*height, .85*height]) < merge_distance
               for f in fits):
            continue
        radius = max(1.5, width*.003)
        distance = np.abs(xx-a0*yy-b0)/np.sqrt(1+a0*a0)
        select = distance < radius
        if select.sum() < max(18, height*.09):
            continue
        y, x = yy[select].astype(float), xx[select].astype(float)
        if np.ptp(y) < height*.16:
            continue
        weights = np.ones_like(y)
        for _ in range(3):
            matrix = np.stack([y, np.ones_like(y)], 1)
            coef = np.linalg.lstsq(matrix*weights[:, None], x*weights, rcond=None)[0]
            residual = np.abs(x-matrix@coef)
            weights = np.minimum(1., radius/(residual+1e-5))
        a, b = coef.tolist()
        residual = np.abs(x-a*y-b)/np.sqrt(1+a*a)
        if np.median(residual) > radius:
            continue
        rows = np.arange(height)
        center = np.rint(a*rows+b).astype(int)
        inside = (center >= 0) & (center < width)
        observed_rows = np.zeros(height, bool)
        paint_rows = np.zeros(height, bool)
        observed_rows[inside] = visible[rows[inside], center[inside]]
        for shift in [-1, 0, 1]:
            p = np.clip(center+shift, 0, width-1)
            paint_rows[inside] |= paint[rows[inside], p[inside]]
        fraction = float(paint_rows[observed_rows].mean()) if observed_rows.any() else 0.
        confidence = float(np.clip(np.ptp(y)/height, 0, 1)
                           * np.exp(-np.median(residual)/max(radius, 1))
                           * np.sqrt(min(len(x)/(height*.35), 1.)))
        if confidence < .12:
            continue
        # This is a corridor, not an asserted solid/dashed paint realization.
        style = 'solid_supported' if fraction > .7 else 'phase_or_style_uncertain'
        if style != 'solid_supported':
            confidence *= .5
        fits.append(dict(a=float(a), b=float(b), confidence=confidence,
                         sigma_px=float(np.clip(np.percentile(residual, 80)+.7, .8, width*.008)),
                         observed_rows=np.flatnonzero(paint_rows).tolist(),
                         observed_paint_pixels=int(select.sum()),
                         visible_paint_fraction=fraction, style=style,
                         median_residual_px=float(np.median(residual))))
    return fits


def build_lane_condition(image, visible_road, hole, road_support=None):
    """Return H×W×8 float32 channels and an auditable visible-only receipt.

    `image` is already foreground/occlusion sanitized.  Hidden pixels are
    overwritten again before *all* color statistics, detection and fitting.
    `road_support` may include the task's inferred ground beneath occluders;
    it is geometry/support information, not a hidden-paint label.
    """
    image = np.asarray(image, np.float32)
    if image.max() > 1:
        image = image/255.
    visible_road, hole = np.asarray(visible_road, bool), np.asarray(hole, bool)
    h, w = hole.shape
    if image.shape != (h, w, 3) or visible_road.shape != hole.shape:
        raise ValueError('RGB and masks must share one continuous camera grid')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('expected finite normalized sRGB input')
    support = visible_road | hole if road_support is None else np.asarray(road_support, bool)
    if support.shape != hole.shape:
        raise ValueError('road support shape differs')
    visible = binary_erosion(visible_road & ~binary_dilation(hole, iterations=2), iterations=1)
    sanitized = image.copy()
    sanitized[~visible] = 0
    white, yellow, threshold = _paint(sanitized, visible)
    condition = np.zeros((h, w, len(CHANNELS)), np.float32)
    condition[..., 0], condition[..., 1] = white, yellow
    condition[..., 7] = support
    rows, cols = np.mgrid[:h, :w]
    records = []
    rejected_empty_support = 0
    destination = hole & support
    for paint_index, paint in enumerate([white, yellow]):
        for fit in _fit(paint, visible, _candidates(paint, h, w)):
            a, b = fit['a'], fit['b']
            perpendicular = np.abs(cols-a*rows-b)/np.sqrt(1+a*a)
            observed = np.asarray(fit['observed_rows'])
            if observed.size == 0:
                rejected_empty_support += 1
                continue
            extrapolation = np.maximum(np.maximum(observed.min()-rows, rows-observed.max()), 0)
            confidence = fit['confidence']*np.exp(-extrapolation/max(.45*h, 1))
            corridor = np.exp(-.5*(perpendicular/fit['sigma_px'])**2)*confidence*destination
            condition[..., 2+paint_index] = np.maximum(condition[..., 2+paint_index], corridor)
            winner = corridor > condition[..., 6]
            theta = np.arctan2(1., a)
            condition[..., 4][winner] = np.cos(2*theta)*corridor[winner]
            condition[..., 5][winner] = np.sin(2*theta)*corridor[winner]
            condition[..., 6] = np.maximum(condition[..., 6], corridor)
            records.append(dict(paint='white' if paint_index == 0 else 'yellow', **fit))
    assert np.isfinite(condition).all()
    receipt = dict(schema='visible_road_lane_condition.v1', channels=CHANNELS,
                   input_shape_hw=[h, w], threshold=threshold,
                   visible_road_pixels=int(visible.sum()), visible_white_pixels=int(white.sum()),
                   visible_yellow_pixels=int(yellow.sum()), inferred_corridor_pixels=int((condition[..., 6] > .05).sum()),
                   line_fits=records, hidden_color_used=False, hidden_paint_used=False,
                   rejected_empty_support_fits=rejected_empty_support,
                   line_scope='conservative longitudinal linear corridors; no forced junction, arrows or dash phase',
                   semantics='inferred uncertain conditioning, never observed ground truth or final painted lines')
    return condition, receipt
