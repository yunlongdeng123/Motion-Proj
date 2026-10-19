"""CPU world-coordinate road query/fusion. No neural ownership losses.

Pixel centers are integer coordinates, world units are metres, extrinsics are
world-to-camera OpenCV (+z forward). Missing visibility evidence stays UNKNOWN.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, map_coordinates, median_filter
from scipy.spatial import cKDTree

UNKNOWN, OBSERVED, PRIOR = 0, 1, 2


def transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


@dataclass
class RoadSurface:
    """Measured local height-field mesh, with a plane only for unsupported cells.

    local_to_world is rigid. Coefficients [a,b,c] define z=a*x+b*y+c.
    Residual heights use measured LiDAR neighborhoods, not vehicle RGB/depth.
    Extrapolated plane points remain marked unsupported and cannot be OBSERVED.
    """
    local_to_world: np.ndarray
    plane: np.ndarray
    measured_xyz: np.ndarray
    support_radius_m: float = 0.8

    def height(self, xy):
        xy = np.asarray(xy, np.float64)
        z = xy @ self.plane[:2] + self.plane[2]
        if len(self.measured_xyz) < 3:
            return z, np.zeros(len(xy), bool)
        tree = cKDTree(self.measured_xyz[:, :2])
        d, idx = tree.query(xy, k=min(8, len(self.measured_xyz)),
                            distance_upper_bound=self.support_radius_m)
        if d.ndim == 1:
            d, idx = d[:, None], idx[:, None]
        ok = np.isfinite(d)
        safe = np.minimum(idx, len(self.measured_xyz)-1)
        pts = self.measured_xyz[safe]
        residual = pts[..., 2] - pts[..., :2] @ self.plane[:2] - self.plane[2]
        weights = np.where(ok, 1/np.maximum(d, 0.05)**2, 0)
        z += (residual*weights).sum(1)/np.maximum(weights.sum(1), 1e-12)
        supported = ok.sum(1) >= 3
        return z, supported

    def intersect(self, origins_world, directions_world):
        inv = np.linalg.inv(self.local_to_world)
        o = transform(origins_world, inv)
        d = directions_world @ inv[:3, :3].T
        denom = d[:, 2] - d[:, :2] @ self.plane[:2]
        numer = o[:, :2] @ self.plane[:2] + self.plane[2] - o[:, 2]
        t = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=np.abs(denom)>1e-7)
        # Fixed-point ray/height intersection; reject non-convergent grazing rays.
        for _ in range(12):
            p = o + t[:, None]*d
            z, _ = self.height(np.nan_to_num(p[:, :2]))
            correction = np.divide(z-p[:, 2], denom, out=np.zeros_like(t), where=np.abs(denom)>1e-7)
            t += correction
        p = o + t[:, None]*d
        z, support = self.height(np.nan_to_num(p[:, :2]))
        valid = np.isfinite(p).all(1) & (t>0.1) & (t<120) & (np.abs(p[:, 2]-z)<0.03)
        return transform(p, self.local_to_world), valid, support

    def normals(self, world):
        local = transform(world, np.linalg.inv(self.local_to_world))
        xy = local[:, :2]
        e = 0.05
        dx = (self.height(xy+[e,0])[0]-self.height(xy-[e,0])[0])/(2*e)
        dy = (self.height(xy+[0,e])[0]-self.height(xy-[0,e])[0])/(2*e)
        n = np.c_[-dx, -dy, np.ones(len(xy))] @ self.local_to_world[:3,:3].T
        return n / np.linalg.norm(n, axis=1)[:,None]


def query_from_pixels(pixels, K, world_to_camera, surface):
    inv = np.linalg.inv(world_to_camera)
    directions = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T @ inv[:3,:3].T
    directions /= np.linalg.norm(directions, axis=1)[:,None]
    origins = np.broadcast_to(inv[:3,3], directions.shape)
    return surface.intersect(origins, directions)


def project(world, K, world_to_camera, wh):
    camera = transform(world, world_to_camera)
    p = camera @ K.T
    uv = p[:,:2]/np.maximum(p[:,2:3], 1e-12)
    valid = (camera[:,2]>0.1) & (uv[:,0]>=0) & (uv[:,0]<wh[0]-1) & (uv[:,1]>=0) & (uv[:,1]<wh[1]-1)
    return uv, camera[:,2], valid


def sample(image, uv, order=1):
    if image.ndim == 2:
        return map_coordinates(image, [uv[:,1], uv[:,0]], order=order, mode='constant', cval=0)
    return np.stack([sample(image[...,c], uv, order) for c in range(image.shape[-1])], axis=-1)


def ray_box_blocked(world, camera_origin, boxes):
    blocked = np.zeros(len(world), bool)
    for box in boxes:
        rotation, center, half = np.asarray(box['axes']), np.asarray(box['center']), np.asarray(box['half'])
        o = (camera_origin-center) @ rotation
        d = (world-camera_origin) @ rotation
        parallel = np.abs(d)<1e-10
        safe = np.where(parallel, 1, d)
        a, b = (-half-o)/safe, (half-o)/safe
        lo, hi = np.minimum(a,b), np.maximum(a,b)
        outside = parallel & (np.abs(o)>half)
        lo = np.where(parallel, -np.inf, lo)
        hi = np.where(parallel, np.inf, hi)
        near, far = lo.max(1), hi.min(1)
        blocked |= (~outside.any(1)) & (far>=np.maximum(near,0)) & (near<0.999) & (far>0)
    return blocked


def observations(world, normals, surface_supported, frame, config):
    """Frame must provide all-occluder mask, road semantics and metric depth.

    Vehicle-only masks + projected LiDAR alone cannot certify all other occluders.
    A depth match and road semantics are required; absent evidence fails closed.
    """
    rgb = frame['rgb']
    h,w = rgb.shape[:2]
    uv, depth, inside = project(world, frame['K'], frame['world_to_camera'], (w,h))
    safe_uv = np.where(np.isfinite(uv), uv, -1)
    c2w = np.linalg.inv(frame['world_to_camera'])
    directions = c2w[:3,3]-world
    distance = np.linalg.norm(directions,axis=1)
    cosine = np.abs((directions/np.maximum(distance[:,None],1e-9)*normals).sum(1))
    # Projected area of a metric texel, proportional to cos(theta)/z^2.
    area = frame['K'][0,0]*frame['K'][1,1]*config['texel_m']**2*cosine/np.maximum(depth**2,1e-6)
    blocked = binary_dilation(frame['occluder_mask'].astype(bool), iterations=config['mask_margin_px'])
    mask_clear = sample(blocked.astype(float),safe_uv)==0
    road = sample(frame['road_mask'].astype(float),safe_uv)>0.99
    observed_depth = sample(frame['depth_m'].astype(float),safe_uv)
    depth_valid = sample(frame['depth_valid'].astype(float),safe_uv)>0.99
    depth_match = depth_valid & (np.abs(observed_depth-depth)<config['depth_tolerance_m'])
    boxes_clear = ~ray_box_blocked(world,c2w[:3,3],frame.get('boxes',[]))
    quality = sample(frame['quality'].astype(float),safe_uv)
    accepted = (inside & surface_supported & mask_clear & road & depth_match & boxes_clear
                & (cosine>=config['min_cosine']) & (area>=config['min_area_px'])
                & (quality>=config['min_quality']))
    weights = np.where(accepted, cosine**2*quality/np.maximum(distance,1)**2, 0)
    colors = sample(rgb.astype(np.float64)/255, safe_uv)
    reasons = dict(in_view=inside, surface_supported=surface_supported, mask_clear=mask_clear,
                   road_semantic=road, depth_match=depth_match, boxes_clear=boxes_clear,
                   angle_ok=cosine>=config['min_cosine'], resolution_ok=area>=config['min_area_px'],
                   quality_ok=quality>=config['min_quality'])
    return colors,weights,uv,reasons


def fuse(colors, weights, source_ids, min_views=2, color_threshold=0.08):
    """Robust consensus in linear RGB, preserving exact source frame/UV evidence.

    One good image is a candidate, not falsely reported multi-view confirmation.
    """
    colors=np.asarray(colors); weights=np.asarray(weights)
    if len(source_ids)!=len(colors) or len(set(source_ids))!=len(source_ids):
        raise ValueError('unique actual source frame IDs required')
    linear=np.where(colors<=0.04045,colors/12.92,((colors+0.055)/1.055)**2.4)
    center=np.zeros(colors.shape[1:])
    for i in range(colors.shape[1]):
        keep=weights[:,i]>0
        if keep.any(): center[i]=np.median(linear[keep,i],axis=0)
    keep=(np.max(np.abs(linear-center[None]),axis=-1)<=color_threshold)&(weights>0)
    used=weights*keep
    count=keep.sum(0)
    observed=count>=min_views
    value=(linear*used[...,None]).sum(0)/np.maximum(used.sum(0)[:,None],1e-12)
    value=np.where(value<=0.0031308,12.92*value,1.055*np.maximum(value,0)**(1/2.4)-0.055)
    value[~observed]=0
    # Scale-invariant within-query confidence; provenance includes contributor mask.
    confidence=np.where(observed,np.minimum(count/3,1),0)
    return dict(rgb=value, provenance=np.where(observed,OBSERVED,UNKNOWN).astype(np.uint8),
                confidence=confidence, source_count=count, candidate_count=(weights>0).sum(0),
                contributors=keep, source_ids=source_ids)


def fill_from_donor(query_xy, fused, donor_xy, donor_rgb, donor_approved,
                    max_distance_m=12.0, donor_regions=None):
    """Explicit same-road *prior*, never promoted to real query-location evidence.

    Valid interior asphalt is resampled periodically in road coordinates; this is
    an explicit texture prior, not known hidden ground truth or a pasted RGB box.
    """
    result={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in fused.items()}
    result['donor_index']=np.full(len(query_xy),-1,np.int64)
    if not donor_approved or len(donor_xy)==0:
        return result
    distance,index=cKDTree(donor_xy).query(query_xy)
    if donor_regions is not None:
        # Choose the nearest donor region, but preserve within-road spatial texture.
        # Excluded markings are absent from the donor point table, not blurred in.
        chosen=np.asarray(donor_regions)[index]
        for region in np.unique(donor_regions):
            members=np.flatnonzero(np.asarray(donor_regions)==region)
            q=np.flatnonzero(chosen==region)
            if not len(q):continue
            points=donor_xy[members]
            lo,hi=points.min(0),points.max(0)
            extent=np.maximum(hi-lo,0.1)
            mapped=lo+np.mod(query_xy[q]-lo,extent)
            _,local_index=cKDTree(points).query(mapped)
            index[q]=members[local_index]
    # A single or inconsistent real candidate must not silently become a prior.
    missing=(result['provenance']==UNKNOWN)&(result['candidate_count']==0)&(distance<=max_distance_m)
    result['rgb'][missing]=donor_rgb[index[missing]]
    result['provenance'][missing]=PRIOR
    result['confidence'][missing]=0.1  # displayed uncertainty, not observed confidence
    result['donor_index'][missing]=index[missing]
    return result


def select_asphalt_donors(rgb, polygons, occluder_mask, road_support=None):
    """Remove bright/colored markings and their edges inside human-supplied boxes.

    This is an auditable conservative color/edge filter, not perfect semantics.
    Real preparation additionally intersects measured ground support. The RGB
    passed here must be unannotated; output overlays are never sampled as input.
    """
    from PIL import Image,ImageDraw
    h,w=rgb.shape[:2]
    regions=np.full((h,w),-1,np.int16)
    for i,polygon in enumerate(polygons):
        mask=Image.new('1',(w,h));ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon],fill=1)
        inside=binary_erosion(np.asarray(mask),iterations=2)
        regions[inside]=i
    c=rgb.astype(float)/255
    high=c.max(2);low=c.min(2);sat=(high-low)/np.maximum(high,1e-6)
    luma=c@np.array([.2126,.7152,.0722])
    local=median_filter(luma,size=9)
    yellow=(c[:,:,0]>c[:,:,2]+.1)&(c[:,:,1]>c[:,:,2]+.07)&(sat>.25)
    marking=np.zeros((h,w),bool)
    for i in range(len(polygons)):
        mask=(regions==i)&(~occluder_mask)
        if not mask.any():continue
        median=np.median(luma[mask]);mad=np.median(np.abs(luma[mask]-median))
        bright=(luma>median+max(.08,3*mad))&(sat<.35)
        edge=(luma-local>.055)&(luma>median+.04)
        marking|=mask&(bright|yellow|edge)
    marking=binary_dilation(marking,iterations=2)
    accepted=(regions>=0)&(~marking)&(~occluder_mask)
    if road_support is not None:accepted &= road_support
    return accepted,regions,marking
