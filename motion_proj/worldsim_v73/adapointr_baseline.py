"""官方完整AdaPoinTr与稀疏真实观测任务桥，不使用旧V7.2校验包装器。"""
import importlib
from pathlib import Path
import sys
import torch
import yaml
from easydict import EasyDict


def chunked_knn(nsample,xyz,new_xyz,chunk=512):
    """保持全部输入与精确距离topk，避免原始输入的整块N×N距离驻留。"""
    values=[]
    for query in new_xyz.split(chunk,dim=1):
        distance=(query.float().square().sum(-1,keepdim=True)+xyz.float().square().sum(-1)[:,None]
                  -2*query.float()@xyz.float().transpose(1,2))
        values.append(distance.topk(nsample,dim=-1,largest=False,sorted=False).indices)
    return torch.cat(values,dim=1)


def load_official_model(repo,checkpoint):
    repo=Path(repo); sys.path.insert(0,str(repo))
    module=importlib.import_module('models.AdaPoinTr')
    utility=importlib.import_module('models.Transformer_utils')
    # 两处全局函数读取均改为同一分块算子；不缩减图像、点或邻居数。
    module.knn_point=chunked_knn; utility.knn_point=chunked_knn
    config=EasyDict(yaml.safe_load((repo/'cfgs/PCN_models/AdaPoinTr.yaml').read_text())['model'])
    model=module.AdaPoinTr(config)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(payload['base_model'])
    source={'pretrained_epoch':payload.get('epoch'),'model_config':dict(config),
            'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'execution':'official full 512-query/16384-point model; exact chunked knn preserves all raw build points',
            'input_axes':'original metric Actor canonical axes; isotropic division by known max box dimension, no GT-derived centering/alignment'}
    return model,source


def prepare_input(points,size):
    scale=size.max().clamp_min(.1)
    # 官方512个中心需要足够槽位；缺点时只重复已有输入，不产生新观测。
    if len(points)<512: points=points[torch.arange(512,device=points.device)%len(points)]
    return (points/scale)[None].contiguous(),scale


def select_surface_centers(points,count):
    from pointnet2_ops import pointnet2_utils
    with torch.no_grad():
        ids=pointnet2_utils.furthest_point_sample(points.detach()[None].float().contiguous(),count)[0].long()
    return points[ids]


def sparse_denoising_loss(denoised_coarse,denoised_fine,targets):
    """局部观测→去噪输出单向覆盖，不把稀疏标签附近的未知面全部罚为错误。"""
    with torch.no_grad():
        indices=torch.cdist(denoised_coarse.detach().float(),targets.float()).topk(min(32,len(targets)),largest=False).indices
    local_targets=targets[indices]
    local_prediction=denoised_fine.reshape(len(denoised_coarse),-1,3)
    return torch.cdist(local_targets.float(),local_prediction.float()).min(-1).values.mean()
