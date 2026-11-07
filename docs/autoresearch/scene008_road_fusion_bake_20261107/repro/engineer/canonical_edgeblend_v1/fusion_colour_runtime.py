"""Fixed-geometry native RGB gradient path for one shared parent colour table.

Only the supplied parent-colour tensor enters autograd. This does not modify
the frozen canonical runtime, persistent geometry, view AA or original layers.
"""
from pathlib import Path
import importlib.util
import sys
import numpy as np

PARENT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_seam_contact_20261107_r1/engineer/canonical_n2_mip_candidate_v2')


class FusionColourRuntime:
    def __init__(self,manifest_path=PARENT/'scene_manifest.json'):
        self.manifest_path=Path(manifest_path)
        source=self.manifest_path.parent/'canonical_road_runtime.py'
        spec=importlib.util.spec_from_file_location('_fusion_pinned_canonical',source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        self.model=module.CanonicalRoadRuntime(self.manifest_path,verify_dependencies=False)
        self.rt=self.model.rt;self.ids=self.model.parent_indices;self.initial=self.model.extension._rgb[::4].detach().clone()
        self.child_rows=len(self.ids);assert self.child_rows==40392 and len(self.initial)==10098
        import torch
        assert torch.equal(self.model.extension._rgb,self.initial.index_select(0,self.ids))
        self.forwards=0

    def scene(self,raw,state,parent_colours):
        import torch
        if state not in ('keep','delete'):raise ValueError('Unknown state')
        if parent_colours.shape!=(10098,3):raise ValueError('One persistent10098x3 colour table required')
        scene=self.model.scene(raw,delete=state=='delete')
        rows={key:None if value is None else value.detach() for key,value in scene['rows'].items()}
        n=self.child_rows;assert bool((rows['features'][-n:,18]==1).all()) and not bool((rows['features'][:-n,18]!=0).any())
        child_colours=parent_colours.index_select(0,self.ids)
        rows['colors_precomp']=torch.cat((rows['colors_precomp'][:-n],child_colours),0)
        feature=rows['features'].clone();feature[-n:,12:15]=child_colours;feature[-n:,19:22]=child_colours
        rows['features']=feature
        return dict(scene,rows=rows)

    def render(self,raw,state,parent_colours):
        scene=self.scene(raw,state,parent_colours);rt=self.rt
        out=rt.raster(scene['cam'],rt.pipe,0,scene['rows'],scene['bg'].detach());self.forwards+=1
        out['rgb']=out['rgb']+(1-out['alpha'])*scene['env'].detach()
        out['group_alpha_T']=out['extra'][:4];out['group_rgb_T']=out['extra'][4:16].reshape(4,3,*out['alpha'].shape[-2:])
        out['env_rgb_T']=(1-out['alpha'])*scene['env'].detach();out['newroad_alpha_T']=out['extra'][16]
        out['newroad_rgb_T']=out['extra'][17:20];out['depth_m']=out['depth_sum'][0]/out['alpha'][0].clamp_min(1e-8)/rt.scale
        with rt.torch.no_grad():
            group_error=float((out['group_alpha_T'].sum(0)-out['alpha'][0]).abs().max())
            reconstruction=out['group_rgb_T'].sum(0)+(1-out['alpha'])*scene['bg'][:,None,None]+out['env_rgb_T']
            colour_error=float((reconstruction-out['rgb']).abs().max())
            assert group_error<2e-5 and colour_error<5e-5,(group_error,colour_error)
        out['group_conservation_maxabs']=group_error;out['RGB_conservation_maxabs']=colour_error
        return out

    def receipt(self):
        assert self.model.extension.unchanged()
        return dict(original=self.model.receipt(),gradient_forwards=self.forwards,
            trainable_role='shared persistent parent road sRGB only; children inherit same parent colour',
            original_parameter_updates=0,geometry_updates=0,opacity_updates=0,view_AA_policy_updates=0,
            direct_native_raster=True,feature_ABI=24,new_per_frame_material=False,adoption_claim=False)
