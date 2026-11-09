"""完整动态建模与删除查询分离；冻结的 DrivingEditor 引擎保持不变。"""
from .ownership import ActorEditQuery, actor_ids, bind_initial_ownership, install_identity_serialization
from .render import render_scene

__all__ = ['ActorEditQuery', 'actor_ids', 'bind_initial_ownership',
           'install_identity_serialization', 'render_scene']

# Shared scene interface: explicit data manifest; legacy APIs above remain replayable.
from .v44_manifest import SceneManifest
from .v44_scene import V44Scene
from .v44_factory import from_manifest as load_editable_scene
__all__ += ['SceneManifest', 'V44Scene', 'load_editable_scene']
