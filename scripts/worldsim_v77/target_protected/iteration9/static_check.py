from pathlib import Path
import sys
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected');sys.path.insert(0,str(P/'iteration8'))
import asset_factory
asset_factory.O=asset_factory.T/'r9'
import static_occupancy
static_occupancy.main()
