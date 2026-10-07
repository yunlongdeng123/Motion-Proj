"""沿用r47请求合同；只修正指令与先验消融的混杂。"""
from iteration14.interface import DeletionRequest, encode_references, check_request


def drop_priors(condition, drop_rgb=False, drop_geometry=False, *, drop_parameters=False):
    """条件组保留相同任务指令；CFG无条件支路必须显式另外清空指令。"""
    result = {k: v.clone() for k, v in condition.items()}
    if drop_rgb:
        for key in ('reference_latents', 'reference_valid', 'reference_pose'):
            result[key].zero_()
    if drop_geometry:
        for key in ('geometry', 'bev', 'bev_rays'):
            result[key].zero_()
        result['geometry'][:, 2] = 1
        result['bev'][:, 6] = 1
    if drop_parameters:
        result['parameters'].zero_()
    return result
