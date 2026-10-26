"""复用已核验的跨帧几何算子，补齐49帧诊断标签；角色为同片段一致性。"""
import json,sys,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
source=Path('/root/autodl-tmp/prepare_crossview.py').read_text()
old="OUT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026')"
assert source.count(old)==1
source=source.replace(old,'OUT=Path('+repr(str(ROOT/'phase2_reference49'))+')')
source=source.replace('for raw in [102,129,137]:','for raw in range(89,138):')
assert 'for raw in range(89,138):' in source
(ROOT/'phase2_reference49').mkdir(exist_ok=True)
exec(compile(source,'existing_crossview_operator_parameterized','exec'),{'__name__':'__main__'})
(ROOT/'phase2_reference49/role.json').write_text(json.dumps(dict(
    purpose='49帧跨视图一致性诊断；非独立测试/道路GT',
    target_views=list(range(89,138)),source_views=list(range(138,170)),
    source_views_already_used_by_199_initialization=True,
    original_optimizer_views=[102,129,137],unknown_remains_unknown=True),indent=2))
