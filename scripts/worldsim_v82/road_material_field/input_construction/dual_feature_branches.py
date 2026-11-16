"""Actual material/global-geometry residuals for a future matched U1 protocol.

This module does not load or train a diffusion model.  The two zero heads keep
the inherited T0 prediction unchanged until a separately authorized training run.
"""
import torch
import torch.nn.functional as F

GEOMETRY_CHANNELS = [
    'lane_visible_white', 'lane_visible_yellow', 'lane_inferred_white', 'lane_inferred_yellow',
    'lane_tangent_cos2', 'lane_tangent_sin2', 'lane_inferred_confidence', 'road_support',
    'normalized_depth', 'depth_source_confidence', 'retained_actor_K', 'uncertainty',
    'aligned_global_geometry_VAE0', 'aligned_global_geometry_VAE1',
    'aligned_global_geometry_VAE2', 'aligned_global_geometry_VAE3',
]
MATERIAL_CHANNELS = ['legal_material_VAE0', 'legal_material_VAE1',
                     'legal_material_VAE2', 'legal_material_VAE3', 'legal_material_support']


def align_global_geometry(global_latent, local_image_hw, local_latent_hw,
                          native_from_local, native_from_global, global_image_hw):
    """Map full-view VAE features to local latent centers by fixed pixel geometry.

    Matrices map image-edge coordinates (pixel centers u+.5,v+.5) to the same
    native image coordinates.  No learned crop or correspondence is fitted here.
    """
    batch = global_latent.shape[0]
    device = global_latent.device
    local_h, local_w = local_image_hw
    latent_h, latent_w = local_latent_hw
    global_h, global_w = global_image_hw
    yy, xx = torch.meshgrid(
        (torch.arange(latent_h, device=device, dtype=torch.float32)+.5)*local_h/latent_h,
        (torch.arange(latent_w, device=device, dtype=torch.float32)+.5)*local_w/latent_w,
        indexing='ij')
    points = torch.stack([xx, yy, torch.ones_like(xx)], -1).reshape(1, -1, 3).expand(batch, -1, -1)
    local_matrix = torch.as_tensor(native_from_local, device=device, dtype=torch.float32)
    global_matrix = torch.as_tensor(native_from_global, device=device, dtype=torch.float32)
    if local_matrix.ndim == 2:
        local_matrix = local_matrix[None].expand(batch, -1, -1)
    if global_matrix.ndim == 2:
        global_matrix = global_matrix[None].expand(batch, -1, -1)
    mapping = torch.linalg.solve(global_matrix, local_matrix)
    global_uv = points @ mapping.transpose(1, 2)
    assert torch.isfinite(global_uv).all()
    assert (global_uv[..., 2].abs() > 1e-8).all()
    global_uv = global_uv[..., :2]/global_uv[..., 2:3]
    grid = torch.stack([2*global_uv[..., 0]/global_w-1,
                        2*global_uv[..., 1]/global_h-1], -1).reshape(batch, latent_h, latent_w, 2)
    return F.grid_sample(global_latent.float(), grid, mode='bilinear',
                         padding_mode='zeros', align_corners=False)


class ZeroFeatureHead(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, 3, padding=1), torch.nn.SiLU(),
            torch.nn.Conv2d(32, 320, 1),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features)


class DualFeatureBranches(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.material = ZeroFeatureHead(len(MATERIAL_CHANNELS))
        self.geometry = ZeroFeatureHead(len(GEOMETRY_CHANNELS))
        self.active_material = self.active_geometry = None
        self.forward_calls = 0

    def set_conditions(self, legal_material_latent, material_support, lane8,
                       normalized_depth, depth_source_confidence, retained_K, uncertainty,
                       aligned_global_geometry):
        size = legal_material_latent.shape[-2:]
        resize = lambda x: F.interpolate(x.float(), size=size, mode='area')
        support = resize(material_support)
        depth_confidence = resize(depth_source_confidence)
        depth = resize(normalized_depth.float()*depth_source_confidence.float())/depth_confidence.clamp_min(1e-8)
        depth = torch.where(depth_confidence>0,depth,torch.zeros_like(depth))
        # No I_geo or foreground RGB enters the material feature head.
        self.active_material = torch.cat([legal_material_latent.float(), support], dim=1)
        self.active_geometry = torch.cat([
            resize(lane8), depth, depth_confidence,
            resize(retained_K), resize(uncertainty), aligned_global_geometry.float(),
        ], dim=1)
        assert self.active_material.shape[1] == 5 and self.active_geometry.shape[1] == 16
        assert self.active_geometry.shape[-2:] == size
        assert torch.isfinite(self.active_material).all() and torch.isfinite(self.active_geometry).all()

    def forward_hook(self, module, inputs, output):
        if self.active_material is None or self.active_geometry is None:
            raise RuntimeError('U1 needs actual material and aligned geometry features before every UNet call')
        material, geometry = self.active_material, self.active_geometry
        if material.shape[0] != output.shape[0]:
            assert output.shape[0] % material.shape[0] == 0
            repeat = output.shape[0]//material.shape[0]
            material = material.repeat(repeat, 1, 1, 1)
            geometry = geometry.repeat(repeat, 1, 1, 1)
        residual = self.material(material)+self.geometry(geometry)
        assert residual.shape == output.shape
        self.forward_calls += 1
        return output+residual.to(output.dtype)
