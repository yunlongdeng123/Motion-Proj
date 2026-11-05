"""Frozen-order opacity derivatives on selected rays in the native raster."""

import torch


def _prefix(values):
    return torch.cat((values.new_zeros((1, *values.shape[1:])), values.cumsum(0)))


def _empty(ids, alphas, meta):
    index = ids.new_empty(0, dtype=torch.int64)
    scalar = alphas.new_empty(0, dtype=torch.float64)
    return dict(ids=index, pixels=index, rgb=scalar.reshape(0, 3), alpha=scalar,
                depth_sum=scalar, pvd=scalar, meta=meta)


@torch.no_grad()
def ray_adjoint(trace, hits, actor_count, lineage, masks):
    """Differentiate with respect to fractional decreases of current actor opacity.

    The returned indices pair persistent actor IDs with original raster pixel IDs.
    The local derivative holds the sort order and active contributor set fixed.
    """
    ids, alphas, offsets = hits
    pixels = trace.pixels.long()
    selected = torch.stack([masks[k].reshape(-1)[pixels] for k in
                            ('known', 'depth', 'outside', 'target')]).any(0)
    counts = (offsets[1:] - offsets[:-1]).long()
    slots = torch.where(selected & (counts > 0))[0]
    counts = counts[slots]
    bounds = _prefix(counts)
    hit_count = int(bounds[-1])
    meta = dict(pixels=len(slots), hits=hit_count, surviving_hits=0, actor_hits=0,
                clamped_actor_hits=0, cutoff_pixels=0)
    if not hit_count:
        return _empty(ids, alphas, meta)

    # Retain all contributors on these rays, including PVD and other fixed rows.
    ray = torch.arange(len(slots), device=ids.device).repeat_interleave(
        counts, output_size=hit_count)
    position = torch.arange(hit_count, device=ids.device)
    source = offsets[slots[ray]].long() + position - bounds[ray]
    rows = ids[source].long()
    alpha = alphas[source].double()
    log_t = _prefix(torch.log1p(-alpha))
    log_start = log_t[bounds[ray]]
    before = (log_t[:-1] - log_start).exp()
    after = (log_t[1:] - log_start).exp()
    # Native raster skips the first hit that would cross its terminal threshold.
    live = after >= 0.0001
    live_prefix = _prefix(live.long())
    live_counts = live_prefix[bounds[1:]] - live_prefix[bounds[:-1]]
    end_t = (log_t[bounds[:-1] + live_counts] - log_t[bounds[:-1]]).exp()
    actor_hits = torch.where(live & (rows < actor_count))[0]
    meta.update(surviving_hits=int(live_prefix[-1]), actor_hits=len(actor_hits),
                cutoff_pixels=int((live_counts != counts).sum()))
    if not len(actor_hits):
        return _empty(ids, alphas, meta)

    actor = rows[actor_hits]
    actor_ray = ray[actor_hits]
    actor_pixel = pixels[slots[actor_ray]]
    xy = trace.xy.reshape(-1, 2)[actor]
    conic = trace.conic.reshape(-1, 4)[actor]
    dx = xy[:, 0] - actor_pixel.remainder(trace.w).to(xy.dtype)
    dy = xy[:, 1] - actor_pixel.div(trace.w, rounding_mode='floor').to(xy.dtype)
    power = -0.5 * (conic[:, 0] * dx * dx + conic[:, 2] * dy * dy) - conic[:, 1] * dx * dy
    clamped = conic[:, 3] * power.exp() >= 0.99
    meta['clamped_actor_hits'] = int(clamped.sum())

    weight = torch.where(live, alpha * before, 0.0)
    payload = torch.cat((trace.color.reshape(-1, 3)[rows], trace.feature[rows, :1],
                         (trace.kind[rows] == 3)[:, None]), dim=1).double()
    weighted = _prefix(weight[:, None] * payload)
    downstream = weighted[bounds[actor_ray + 1]] - weighted[actor_hits + 1]

    actor_alpha = alpha[actor_hits]
    magnitude = actor_alpha.masked_fill(clamped, 0.0)
    ratio = magnitude / (1.0 - actor_alpha)
    direct = magnitude * before[actor_hits]
    tail = ratio * end_t[actor_ray]
    rgb = (ratio[:, None] * downstream[:, :3] - direct[:, None] * payload[actor_hits, :3]
           + tail[:, None] * trace.background.reshape(1, 3))
    depth = ratio * downstream[:, 3] - direct * payload[actor_hits, 3]
    return dict(ids=lineage[actor].long(), pixels=actor_pixel, rgb=rgb, alpha=-tail,
                depth_sum=depth, pvd=ratio * downstream[:, 4], meta=meta)
