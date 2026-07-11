"""Motion-Proj 骨干网络子包。"""
from .base import BackboneCapabilities, Conditioning, DiffusionBackbone
from .registry import build_backbone, register_backbone

__all__ = ["BackboneCapabilities", "DiffusionBackbone", "Conditioning", "build_backbone", "register_backbone"]
