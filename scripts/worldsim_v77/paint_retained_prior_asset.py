import sys,datetime
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/work');sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
import hybrid_retained_actor_asset as asset
from repair_common import dump
asset.ROOT=asset.BASE/'r41';reference=asset.BASE/'r40/source_rgba.png';assert not (asset.ROOT/'texture_registration.json').exists()
dump(asset.ROOT/'texture_registration.json',dict(task_id='WS-V77-HYBRID-BG-20260927',run_id='r41',stage='texture',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),shape='Fixed r41 mesh from saved SV3D side-front prior; visually closed SUV, coarse wheels/details, not yet accepted',texture_reference=str(reference),roles='Original observed actor52 front f130 is PBR appearance reference. Side/back texture inferred by existing frozen HunyuanPBR; not factual hidden truth. No new backbone, training or replacement of r18.',config=dict(seed=7740,views=6,resolution=512,use_remesh=False),stop_rule='One texture call, inspect all views and observed pose before hidden DELETE. Do not assume shape-only improvement implies texture identity.',human_verdict=None,background_input_dir=None))
asset.paint(reference)
