"""Pinned research-candidate scene with a self-contained baked new-road asset."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    return h.hexdigest()


class BakedRoadRuntime:
    def __init__(self,manifest_path,verify_dependencies=True):
        self.manifest_path=Path(manifest_path).resolve()
        self.manifest=json.loads(self.manifest_path.read_text())
        self.manifest_hash=sha(self.manifest_path)
        if self.manifest['status']!='RESEARCH_CANDIDATE_NOT_ADOPTED':
            raise ValueError('This wrapper requires an explicit research-candidate manifest')
        asset=self.manifest_path.parent/self.manifest['new_road']['file']
        assert sha(asset)==self.manifest['new_road']['sha256']
        with np.load(asset,allow_pickle=False) as z:
            assert len(z['active'])==10098 and z['active'].all()
            assert z['reference_srgb'].shape==(10098,3) and np.isfinite(z['reference_srgb']).all()
            assert z['original_source_proxy_srgb'].shape==(10098,3)
            assert str(z['material_role'])=='FITTED_VISIBLE_ROAD_MATERIAL_NOT_HIDDEN_RGB_TRUTH'
        if verify_dependencies:
            for info in self.manifest['runtime_dependencies'].values():
                p=Path(info['path']);assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],str(p)
        source=Path(self.manifest['r0_runtime_path'])
        spec=importlib.util.spec_from_file_location('_baked_pinned_r0',source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        self.base=module.R0Runtime(asset,contract=self.manifest['v44_contract_path'])
        self.rt=self.base.rt;self.extension=self.base.extension
        assert len(self.extension.ids)==10098
        assert self.rt.binding_sha256==self.manifest['frozen_v44_binding_sha256']
        self.extension.record.update(candidate_status='RESEARCH_CANDIDATE_NOT_ADOPTED',
            colour='fitted persistent visible-road material; original real-view source proxy is separately retained',
            material_role='FITTED_VISIBLE_ROAD_MATERIAL_NOT_HIDDEN_RGB_TRUTH',
            source_proxy_field='original_source_proxy_srgb',source_witness_RGB_field='source_rgb',
            parent_source_asset_sha256=self.manifest['lineage']['source_asset_sha256'],
            parent_material_sha256=self.manifest['lineage']['material_sha256'],
            inactive_rows_removed=4563)

    def scene(self,raw,delete=False):
        return self.base.scene(raw,delete=delete,extension=True,retire=False,material=False)

    def render_scene(self,raw,delete=False):
        return self.base.render(raw,delete=delete,extension=True,retire=False,material=False)

    def receipt(self):
        assert sha(self.manifest_path)==self.manifest_hash
        return dict(status='RESEARCH_CANDIDATE_NOT_ADOPTED',scene_manifest_sha256=self.manifest_hash,
            manifest=str(self.manifest_path),edit_operator='only physical A37 existence changes; all other scene inputs shared',
            material_role='observed appearance fitting; not recovered hidden RGB',runtime=self.base.receipt(),
            unresolved=self.manifest['known_unresolved'],adoption_claim=False)
