"""Generic conservative visible-road effect candidates; no instance/box fallback."""
import numpy as np
from scipy.ndimage import binary_dilation,binary_closing,distance_transform_edt,label,find_objects

def conservative_effect_candidates(rgb, target, body_write, retained, retained_guard, foreground, road_mask):
    """Generic conservative dark-component proposal attached to target's lower body."""
    height, width = target.shape
    yy, xx = np.mgrid[:height, :width]
    ty, tx = np.where(target)
    x0, y0, x1, y1 = int(tx.min()), int(ty.min()), int(tx.max()+1), int(ty.max()+1)
    bw, bh = x1-x0, y1-y0
    # Only local lower road is eligible. Upper/front-car shadows are excluded.
    bounds = [max(0, int(x0-.65*bw)), max(0, int(y0+.50*bh)),
              min(width, int(x1+.65*bw)), min(height, int(y1+.30*bh))]
    region = (xx >= bounds[0]) & (xx < bounds[2]) & (yy >= bounds[1]) & (yy < bounds[3])
    prior = road_mask & ~foreground & ~retained_guard & region
    road_mask = np.asarray(road_mask, bool)
    image = np.asarray(rgb, np.float32)
    if np.issubdtype(np.asarray(rgb).dtype, np.integer) or image.max() > 1:
        image = image/255
    if image.shape != (*target.shape, 3) or road_mask.shape != target.shape:
        raise ValueError('RGB, road support and masks must share a camera grid')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('Expected finite normalized or uint8 sRGB')
    luminance = image @ np.array([.2126, .7152, .0722], np.float32)
    if prior.sum() < 128:
        empty = np.zeros(target.shape, bool)
        return empty, dict(completed=True, accepted=False, reason="insufficient local visible-road pixels", target_bbox=[x0,y0,x1,y1]), dict(visible_road_prior=prior, paint=empty, shadow=empty, superdark=empty, confidence=np.zeros(target.shape, np.float32))
    median = float(np.median(luminance[prior]))
    threshold = float(np.clip(median+.09, .35, .85))
    red, green, blue = image.transpose(2, 0, 1)
    white = (luminance > threshold) & (np.ptp(image, axis=2) < .20)
    yellow = (red-blue > .055) & (green-blue > .035) & (red-green < .22) & (luminance > .20)
    paint_visible = (white | yellow) & road_mask & ~foreground & ~retained_guard
    paint = binary_dilation(paint_visible, iterations=max(2, round(bw*.008)))
    anchor = prior & ~paint & (luminance > .055)
    if anchor.sum() < 128:
        anchor = prior & ~paint
    q75 = float(np.quantile(luminance[anchor], .75))
    dark_threshold = .72*q75
    superdark_threshold = max(.055, .38*q75)
    lower_body = target & (yy >= y0+.70*bh)
    target_distance = distance_transform_edt(~lower_body)
    other_distance = distance_transform_edt(~retained) if retained.any() else np.full(target.shape, np.inf)
    # A local pixel assigned more closely to another foreground actor cannot
    # extend this target's effect. This also stops nearby pickup-shadow leakage.
    target_nearest = target_distance <= other_distance
    reach = target_distance <= .70*bw
    legal = prior & ~paint & ~body_write & target_nearest & reach
    dark = legal & (luminance < dark_threshold)
    # Small morphology only joins camera-sampling cracks within legal road.
    join = max(1, round(bw*.006))
    candidate = binary_closing(dark, iterations=join) & legal
    contact_radius = max(24, round(.10*bw))
    contact = binary_dilation(lower_body, iterations=contact_radius) & legal
    components, count = label(candidate)
    records = []
    min_area = max(64, round(target.sum()*.0015))
    max_area = round(target.sum()*1.0)
    areas = np.bincount(components.ravel(), minlength=count+1)
    contacts = np.bincount(components[contact].ravel(), minlength=count+1)
    accepted_ids = (areas >= min_area) & (areas <= max_area) & (contacts >= max(8, round(bw*.015)))
    accepted_ids[0] = False
    selected = accepted_ids[components]
    bounds_by_component = find_objects(components)
    for index in range(1, count+1):
        area = int(areas[index])
        touching = int(contacts[index])
        accept = bool(accepted_ids[index])
        if area >= min_area or touching:
            ys, xs = bounds_by_component[index-1]
            records.append(dict(component=index, pixels=area, contact_pixels=touching, accepted=accept,
                                bbox=[int(xs.start),int(ys.start),int(xs.stop),int(ys.stop)],
                                reason="local lower-road dark component touching target" if accept else "no sufficient target contact or conservative area rejection"))
    superdark = selected & (luminance < superdark_threshold)
    shadow = selected & ~superdark
    confidence = np.zeros(target.shape, np.float32)
    contrast = np.clip((q75-luminance)/max(q75, .05), 0, 1)
    confidence[shadow] = np.clip(.25+.55*contrast[shadow], .25, .75)
    confidence[superdark] = .25  # strong darkness is not independent shadow truth
    assert not (selected & (retained_guard | paint | foreground | ~road_mask)).any()
    receipt = dict(completed=True, accepted=bool(selected.any()), target_bbox=[x0,y0,x1,y1],
                   search_bbox=bounds, target_bbox_rule="x +/- .65width; y lower .50height through +.30height below body",
                   visible_asphalt_anchor_pixels=int(anchor.sum()), visible_asphalt_q75=q75,
                   shadow_luma_threshold=dark_threshold, superdark_luma_threshold=superdark_threshold,
                   contact_radius_native_px=contact_radius, morphology_join_native_px=join,
                   candidate_pixels=int(candidate.sum()), accepted_effect_pixels=int(selected.sum()),
                   regular_dark_effect_pixels=int(shadow.sum()), superdark_uncertain_pixels=int(superdark.sum()),
                   excluded_nearer_other_actor_pixels=int((prior & ~target_nearest).sum()),
                   selected_components=sum(record["accepted"] for record in records), components=records,
                   effect_role="inferred object-effect proposal from visible road; not measured shadow GT",
                   unshadowed_color_role="local visible-road luminance q75 heuristic only, not road color under car",
                   unseen_road_GT_read=False, other_frame_RGB_read=False,
                   decision="candidate only, body/effect variants remain separate for review")
    return selected, receipt, dict(visible_road_prior=prior, paint=paint, shadow=shadow,
                                    superdark=superdark, confidence=confidence)
