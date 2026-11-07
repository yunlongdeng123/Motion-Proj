"""按输入 PLY 的真实行顺序绑定身份，复用原生 parent-aware 谱系。

身份仅为元数据：不改变参数、优化器组、损失、采样或增密规则。
新检查点保存逐行 actor_id，并交叉核验初始来源映射和原生谱系。
没有可靠身份的旧检查点不能用于选择性删除；禁止几何猜标签。
"""
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch

OWNERSHIP_KEY = 'dynamic_actor_ownership_v1'
LINEAGE_KEY = 'sgcm_lineage_v1'
OPTIMIZER_INDEX = 14


def _ids(values: Iterable[int], name: str, allow_empty=False) -> Tuple[int, ...]:
    values = tuple(values)
    if (not allow_empty and not values) or any(type(v) is not int or v <= 0 for v in values):
        raise ValueError(f'{name} 必须为正整数 actor ID 集合')
    if len(set(values)) != len(values):
        raise ValueError(f'{name} 含重复身份')
    return tuple(sorted(values))


@dataclass(frozen=True)
class ActorEditQuery:
    dynamic_actor_ids: Tuple[int, ...]
    delete_actor_ids: Tuple[int, ...] = ()

    def __post_init__(self):
        dynamic = _ids(self.dynamic_actor_ids, 'dynamic_actor_ids')
        delete = _ids(self.delete_actor_ids, 'delete_actor_ids', allow_empty=True)
        if not set(delete) <= set(dynamic):
            raise ValueError('delete_actor_ids 必须是 dynamic_actor_ids 的子集')
        object.__setattr__(self, 'dynamic_actor_ids', dynamic)
        object.__setattr__(self, 'delete_actor_ids', delete)


def _validate_table(declared, source, lineage, rows):
    declared = _ids(declared, 'dynamic_actor_ids')
    for value, name in [(source, 'source_actor_ids'), (lineage, 'lineage_ids'), (rows, 'actor_ids')]:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or value.ndim != 1:
            raise ValueError(f'{name} 必须为一维 int64 Tensor')
    if not len(source) or lineage.shape != rows.shape:
        raise ValueError('身份来源/当前行数不匹配')
    if tuple(sorted(set(source.detach().cpu().tolist()))) != declared:
        raise ValueError('初始来源身份与 dynamic_actor_ids 不一致')
    if len(lineage) and bool(((lineage < 0) | (lineage >= len(source))).any()):
        raise ValueError('谱系引用未知初始来源行')
    expected = source[lineage.to(source.device)]
    if not torch.equal(expected.detach().cpu(), rows.detach().cpu()):
        raise ValueError('逐高斯 actor ID 与保存的来源谱系不一致')
    return declared


def actor_ids(dynamic):
    """每个当前高斯的明确身份；clone/split 的父来源不变，prune 保持行映射。"""
    declared = getattr(dynamic, '_dynamic_actor_ids_declared', None)
    source = getattr(dynamic, '_dynamic_actor_source_ids', None)
    lineage = getattr(dynamic, '_sgcm_lineage_ids', None)
    n = len(dynamic.get_xyz)
    if declared is None or not isinstance(source, torch.Tensor) or not isinstance(lineage, torch.Tensor):
        raise ValueError('动态模型没有可靠 actor 身份元数据；不能猜测旧权重标签')
    if source.dtype != torch.int64 or lineage.dtype != torch.int64 or lineage.shape != (n,):
        raise ValueError('动态 actor 身份与高斯拓扑不匹配')
    if source.device != dynamic.get_xyz.device or lineage.device != dynamic.get_xyz.device:
        raise ValueError('动态 actor 身份 device 不匹配')
    if len(lineage) and bool(((lineage < 0) | (lineage >= len(source))).any()):
        raise ValueError('动态谱系引用未知来源行')
    return source[lineage]


def bind_initial_ownership(dynamic, actor_source_counts, dynamic_actor_ids):
    """仅在真实 native initialize 后绑定；counts 按 loader 的实际 PLY 拼接顺序。"""
    declared = _ids(dynamic_actor_ids, 'dynamic_actor_ids')
    pairs = tuple(actor_source_counts)
    if not pairs or any(type(a) is not int or type(n) is not int or n <= 0 for a, n in pairs):
        raise ValueError('每个输入 PLY 必须具有明确 actor ID 和正整数真实行数')
    if tuple(sorted(a for a, _ in pairs)) != declared:
        raise ValueError('来源 PLY 身份必须与完整动态建模集合逐一对应')
    if sum(n for _, n in pairs) != len(dynamic.get_xyz):
        raise ValueError('输入 PLY 行数与 native 动态初始化不一致')
    if getattr(dynamic, '_dynamic_actor_source_ids', None) is not None:
        raise ValueError('禁止重绑已有动态身份')
    device = dynamic.get_xyz.device
    lineage = getattr(dynamic, '_sgcm_lineage_ids', None)
    if lineage is not None:
        raise ValueError('仅允许对尚未增密且无旧谱系的新初始化绑定身份')
    dynamic._dynamic_actor_source_ids = torch.cat([
        torch.full((n,), a, dtype=torch.int64, device=device) for a, n in pairs])
    dynamic._dynamic_actor_ids_declared = declared
    dynamic._sgcm_lineage_ids = torch.arange(len(dynamic.get_xyz), dtype=torch.int64, device=device)
    dynamic._sgcm_clone_blocked = torch.zeros(len(dynamic.get_xyz), dtype=torch.bool, device=device)
    return actor_ids(dynamic)


def install_identity_serialization(gaussian_class=None, dynamic_class=None):
    """复用已有 a6/a37 方案的原生 SGCM-aware 18tuple 保存/恢复；幂等安装。"""
    if gaussian_class is None or dynamic_class is None:
        from scene.gaussian_model import GaussianModel, Dynamic_GaussianModel
        gaussian_class, dynamic_class = GaussianModel, Dynamic_GaussianModel
    if dynamic_class.__dict__.get('_actor_identity_serialization_v1', False):
        return
    native_capture, native_restore = gaussian_class.capture, gaussian_class.restore

    def capture(self):
        rows = actor_ids(self)
        source = self._dynamic_actor_source_ids
        declared = self._dynamic_actor_ids_declared
        _validate_table(declared, source, self._sgcm_lineage_ids, rows)
        result = list(native_capture(self))
        if len(result) != 18 or not isinstance(result[OPTIMIZER_INDEX], dict):
            raise ValueError('预期原生 18tuple/Adam checkpoint 格式')
        optimizer = dict(result[OPTIMIZER_INDEX])
        if LINEAGE_KEY not in optimizer:
            raise ValueError('原生 checkpoint 未保存动态谱系')
        optimizer[OWNERSHIP_KEY] = dict(schema_version=1, dynamic_actor_ids=list(declared),
            source_actor_ids=source.detach().cpu().clone(), actor_ids=rows.detach().cpu().clone())
        result[OPTIMIZER_INDEX] = optimizer
        return tuple(result)

    def restore(self, model_args, training_args=None):
        if not isinstance(model_args, (list, tuple)) or len(model_args) != 18:
            raise ValueError('预期原生 18tuple checkpoint')
        optimizer = dict(model_args[OPTIMIZER_INDEX])
        payload = optimizer.pop(OWNERSHIP_KEY, None)
        lineage = optimizer.get(LINEAGE_KEY)
        if (not isinstance(payload, dict) or set(payload) != {'schema_version', 'dynamic_actor_ids', 'source_actor_ids', 'actor_ids'}
                or type(payload['schema_version']) is not int or payload['schema_version'] != 1
                or not isinstance(lineage, dict)):
            raise ValueError('旧动态 checkpoint 缺少可靠 actor 身份；拒绝选择性删除')
        ids = lineage.get('lineage_ids')
        if not isinstance(ids, torch.Tensor) or ids.shape != (len(model_args[1]),):
            raise ValueError('checkpoint 身份谱系长度不匹配')
        declared = _validate_table(payload['dynamic_actor_ids'], payload['source_actor_ids'], ids, payload['actor_ids'])
        clean = list(model_args)
        clean[OPTIMIZER_INDEX] = optimizer
        native_restore(self, tuple(clean), training_args=training_args)
        self._dynamic_actor_ids_declared = declared
        self._dynamic_actor_source_ids = payload['source_actor_ids'].to(self.get_xyz.device).clone()
        if not torch.equal(actor_ids(self).detach().cpu(), payload['actor_ids'].detach().cpu()):
            raise ValueError('恢复后的逐高斯 actor 身份改变')

    dynamic_class.capture, dynamic_class.restore = capture, restore
    dynamic_class._actor_identity_serialization_v1 = True
