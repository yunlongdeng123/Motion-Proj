"""数据工厂有界修订，训练配方不变；负候选和首次几何阶段均保存。"""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,F,read,dump
p=O/'factory_amendment.json'
if not p.exists():
    old=read(F/'pair_candidates.json')['summary']
    dump(p,{'reason':'155已落盘窗/75场景经过实际SAM和精确位姿仅14候选/6训练场景，旧2.5D来源角度/尺寸和未观测动态包络限制覆盖；不是模型能力反证',
            'prior_exact_summary':old,'new_data_only':'已有完整sedan/SUV GLB只提供多视角silhouette，真实GT不改，synthetic RGB完全遮掉；无资产模型推理',
            'map_semantics':'同G1允许drivable/lane/parking，补官方carpark_area；LiDAR重新拟合独立cache，不用地图z认证接地',
            'static_semantics':'behind-A barrier/cone/rack属于真实待恢复背景；全A深度前置才能准入，前景静物/行人/未知车辆仍拒绝',
            'quality_fixed':'size72x40, bbox area .006-.18, border8, ego band64, clearance.3m, real ground support2.5m, mask normalizedIoU.8/area15%, ten-frame protected30%-80% or dense20%-70%, all influence inH',
            'bounded':'two existing shapes, global finite81 poses/window; no thresholds/asset family/seed grids; one data-set effect arm160steps',
            'training_change':False,'loss_change':False,'architecture_change':False,'model_outcomes_seen_for_this_round':False,'all_prior_outputs_retained':True,'references':['https://www.nuscenes.org/tutorials/map_expansion_tutorial.html'],
            'resource_engineering':'GLTF Y-up corrected before canonical silhouette; PyMeshLab import exited1, no decimation used, full mesh retained; raster union skips only exactly redundant unit-pixel degenerate triangles, regression checked'})
    print('factory amendment registered')
