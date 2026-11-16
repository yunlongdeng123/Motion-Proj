"""DELETE 仅过滤原生动态 rasterization mask；KEEP 走原函数，不改参数。

每个调用使用独立 globals 的同一个 native render code object，避免全局
monkeypatch 与并发查询串扰。静态与保留动态仍交给原 joint rasterizer 深度排序。
"""
import types
import torch

from .ownership import ActorEditQuery, actor_ids


def render_scene(camera, static, dynamic, args, background, *, dynamic_actor_ids,
                 delete_actor_ids=(), renderer=None, **kwargs):
    query = ActorEditQuery(dynamic_actor_ids, delete_actor_ids)
    owners = actor_ids(dynamic)
    if query.dynamic_actor_ids != tuple(dynamic._dynamic_actor_ids_declared):
        raise ValueError('查询的 dynamic_actor_ids 必须与该已训练模型一致；换删除目标无需重训')
    if renderer is None:
        import gaussian_renderer as renderer
    native_render = renderer.render
    if not query.delete_actor_ids:
        # 严格 KEEP：不构造 actor 过滤器、不改 mask、不替换 native 函数。
        return native_render(camera, static, dynamic, args, background, **kwargs)
    if kwargs.get('is_training', False):
        raise ValueError('DELETE 是查询操作，禁止作为训练分支过滤器')
    keep = owners.new_ones(owners.shape, dtype=torch.bool)
    for actor in query.delete_actor_ids:
        keep &= owners != actor
    native_prepare = native_render.__globals__.get('prepare_rasterization')
    if native_prepare is None:
        raise ValueError('native render 缺少已知 prepare_rasterization 接口')

    filtered_calls = [0]
    def selected_prepare(viewpoint, model, pipe, *pos, **named):
        prepared = native_prepare(viewpoint, model, pipe, *pos, **named)
        if named.get('dynamic', False):
            if model is not dynamic:
                raise ValueError('动态准备调用引用错误模型')
            if prepared['mask'].shape != keep.shape:
                raise ValueError('原生动态 rasterization mask 与 actor 身份长度不匹配')
            prepared = dict(prepared, mask=prepared['mask'] & keep)
            filtered_calls[0] += 1
        return prepared

    private_globals = dict(native_render.__globals__, prepare_rasterization=selected_prepare)
    selected_render = types.FunctionType(native_render.__code__, private_globals, native_render.__name__,
                                         native_render.__defaults__, native_render.__closure__)
    selected_render.__kwdefaults__ = native_render.__kwdefaults__
    result = selected_render(camera, static, dynamic, args, background, **kwargs)
    if filtered_calls[0] == 0:
        raise RuntimeError('DELETE 未进入原生 dynamic prepare；拒绝静默返回未筛选结果')
    return result
