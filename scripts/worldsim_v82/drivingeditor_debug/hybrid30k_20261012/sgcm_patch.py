"""确定性最小 source patches；第三方完整 source 不提交到研究仓库。

Root 将本目录的 sgcm_helpers.py materialize 为 scene/hybrid_sgcm_helpers.py，
然后调用 patch_sgcm_source 和 patch_gaussian_source。enabled=False 逐字返回。
"""
import ast


def _replace(text, old, new, label):
    if text.count(old) != 1:
        raise ValueError(f"{label}: expected one upstream anchor, got {text.count(old)}")
    return text.replace(old, new, 1)


def _newlines(text):
    return ("\r\n" if "\r\n" in text else "\n"), text.replace("\r\n", "\n")


def patch_sgcm_source(text, enabled=True):
    if not enabled:
        return text
    newline, source = _newlines(text)
    source = _replace(source, "import numpy as np\n", "import numpy as np\n\nfrom scene.hybrid_sgcm_helpers import (bind_sgcm_guard, guard_additions, select_clone_once,\n                                       mark_clone_lineages_spent)\n", "SGCM imports")
    source = _replace(source, "    max_points: int = 3000000\n", "    max_points: int = 3000000\n    guard_enabled: bool = True\n", "SGCM guard config")
    source = _replace(source, "        config = cls(**values)\n", "        config = cls(**values)\n        if type(config.guard_enabled) is not bool:\n            raise ValueError('SGCM guard_enabled must be a boolean')\n", "SGCM guard validation")
    source = _replace(source, "        self.metadata = metadata\n", "        self.metadata = metadata\n        self.training_views = ()\n", "SGCM view binding slot")
    source = _replace(source, "    def refine(self, gaussians, radii, iteration):\n", "    def refine(self, gaussians, radii, iteration, visibility_filter=None):\n", "SGCM current visibility interface")
    anchor = "                raise RuntimeError('SGCM radii must match current topology; schedule must avoid generic densification')\n"
    source = _replace(source, anchor, anchor+"            if visibility_filter is not None:\n                if (visibility_filter.shape != radii.shape or visibility_filter.dtype != torch.bool\n                        or not torch.equal(visibility_filter, radii > 0)):\n                    raise RuntimeError('SGCM current visibility must be the aligned static rendered radii>0 filter')\n", "SGCM current visibility consistency")
    anchor = "            split = inside & (distance_m <= 0.5*cfg.dr_m) & (radii > cfg.split_radius_px)\n"
    source = _replace(source, anchor, anchor+"            guard_statistics = {}\n            if cfg.guard_enabled:\n                clone, split, guard_statistics = guard_additions(self, gaussians, clone, split)\n", "SGCM guarded additions")
    old = "            if len(clone_ids):\n                clone_ids = clone_ids[torch.argsort(radii[clone_ids], descending=True, stable=True)[:budget]]\n"
    new = "            if len(clone_ids):\n                if cfg.guard_enabled:\n                    clone_ids, duplicate_count = select_clone_once(gaussians, clone_ids, radii, budget)\n                    guard_statistics['excluded_duplicate_lineage_in_step'] = duplicate_count\n                else:\n                    clone_ids = clone_ids[torch.argsort(radii[clone_ids], descending=True, stable=True)[:budget]]\n"
    source = _replace(source, old, new, "SGCM clone unique selection")
    anchor = "            parents = torch.cat((clone_ids, split_ids, split_ids))\n"
    source = _replace(source, anchor, anchor+"            if cfg.guard_enabled:\n                mark_clone_lineages_spent(gaussians, clone_ids)\n", "SGCM clone budget persistence")
    source = _replace(source, "                gaussians.densification_postfix(*children)\n", "                gaussians.densification_postfix(*children, sgcm_parent_indices=parents)\n", "SGCM append lineage")
    source = _replace(source, "                    'dr_m':cfg.dr_m,'dr_model':cfg.dr_m*self.model_scale}\n", "                    'dr_m':cfg.dr_m,'dr_model':cfg.dr_m*self.model_scale,\n                    **guard_statistics}\n", "SGCM guard counters")
    source = _replace(source, "def create_sgcm_controller(source_path, args):\n", "def create_sgcm_controller(source_path, args, training_views=None, gaussians=None):\n", "SGCM factory signature")
    anchor = "    print('SGCM_INIT '+json.dumps(metadata,sort_keys=True),flush=True)\n"
    source = _replace(source, anchor, "    bind_sgcm_guard(controller, training_views, gaussians,\n                    resuming=bool(getattr(args, 'start_checkpoint', None)))\n"+anchor, "SGCM guard binding before training")
    ast.parse(source)
    return source.replace("\n", newline)


patch_source = patch_sgcm_source


def _patch_class_method(source, class_name, method_name, mutation):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    lines = source.splitlines(keepends=True)
    before = "".join(lines[:method.lineno-1])
    body = "".join(lines[method.lineno-1:method.end_lineno])
    after = "".join(lines[method.end_lineno:])
    return before+mutation(body)+after


def patch_gaussian_source(text, enabled=True):
    if not enabled:
        return text
    newline, source = _newlines(text)
    source = _replace(source, "class GaussianModel:\n", "from scene.hybrid_sgcm_helpers import (lineage_after_append, lineage_after_prune,\n                                       capture_sgcm_optimizer, restore_sgcm_optimizer)\n\n\nclass GaussianModel:\n", "Gaussian lineage helpers")
    source = _patch_class_method(source, "GaussianModel", "capture", lambda body: _replace(body, "self.optimizer.state_dict()", "capture_sgcm_optimizer(self)", "Gaussian capture lineage"))
    source = _patch_class_method(source, "GaussianModel", "restore", lambda body: _replace(body, "        self.setup_functions()\n", "        opt_dict = restore_sgcm_optimizer(self, opt_dict)\n        self.setup_functions()\n", "Gaussian restore lineage"))
    source = _patch_class_method(source, "GaussianModel", "prune_points", lambda body: body+"        lineage_after_prune(self, valid_points_mask)\n")
    def postfix(body):
        body = _replace(body, "new_rotation, new_t, new_scaling_t, new_velocity):", "new_rotation, new_t, new_scaling_t, new_velocity, sgcm_parent_indices=None):", "Gaussian postfix lineage signature")
        return body+"        lineage_after_append(self, sgcm_parent_indices, len(new_xyz))\n"
    source = _patch_class_method(source, "GaussianModel", "densification_postfix", postfix)
    def split(body):
        old = "                                   new_t, new_scaling_t, new_velocity)\n"
        new = "                                   new_t, new_scaling_t, new_velocity,\n                                   sgcm_parent_indices=(torch.nonzero(selected_pts_mask, as_tuple=False).flatten().repeat(N)\n                                                        if getattr(self, '_sgcm_lineage_ids', None) is not None else None))\n"
        return _replace(body, old, new, "Generic split inherits lineage")
    source = _patch_class_method(source, "GaussianModel", "densify_and_split", split)
    def clone(body):
        old = "                                   new_rotation, new_t, new_scaling_t, new_velocity)\n"
        new = "                                   new_rotation, new_t, new_scaling_t, new_velocity,\n                                   sgcm_parent_indices=(torch.nonzero(selected_pts_mask, as_tuple=False).flatten()\n                                                        if getattr(self, '_sgcm_lineage_ids', None) is not None else None))\n"
        return _replace(body, old, new, "Generic clone inherits lineage")
    source = _patch_class_method(source, "GaussianModel", "densify_and_clone", clone)
    ast.parse(source)
    return source.replace("\n", newline)
