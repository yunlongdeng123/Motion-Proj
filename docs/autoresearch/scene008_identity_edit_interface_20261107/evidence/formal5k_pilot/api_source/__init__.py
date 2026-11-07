"""完整动态建模与删除查询分离；冻结的 DrivingEditor 引擎保持不变。"""
from .ownership import ActorEditQuery, actor_ids, bind_initial_ownership, install_identity_serialization
from .render import render_scene

__all__ = ['ActorEditQuery', 'actor_ids', 'bind_initial_ownership',
           'install_identity_serialization', 'render_scene']
