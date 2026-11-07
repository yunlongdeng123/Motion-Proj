"""在正式拟合前登记：两种目标使用完全相同的跨时刻道路见证。"""
from pathlib import Path
import json
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1')
record=dict(schema='TARGET_INFORMATION_MATCHING_AMENDMENT_V1',
    timing='Before final target manifest, smoke optimizer and any formal direct/fusion fit; only pre-existing baseline pixel bound read.',
    reason='BUILD40 source-valid supports >=3 frames for only 520 of 10098 parents; using cross-time observed road is necessary and is part of the authorized task.',
    source_raws='Original 199-frame FRONT road cache excluding the nine declared HELD raws: 190 frames.',
    source_excluded_raws=[89,95,101,105,113,121,125,130,137],
    direct_control='Reproject the UNFUSED shared observation/source atlas, using the same source observations and replace domain as the fused atlas.',
    fusion_control='Reproject the fused shared atlas built from the identical source observations; target fusion is the only controlled difference.',
    shared_mask='Same original BUILD40 real-road fit mask. On pixels without a reliable atlas replacement, both use unchanged actual frame RGB.',
    extra_reference='observation_rgb remains the actual original per-frame RGB for target fidelity/protection evaluation; it is not a third formal optimization branch.',
    comparison_claim='Equal information, geometry, parameter count, optimizer and fixed budget for unfused vs fused atlas writeback.',
    limitations=['The frozen prior reconstruction/material already saw all199, so this is still a same-scene development control, not an independent HELD9 test.',
                'This observed-road source fusion is not a learned completion result on a never-observed road.',
                'Poisson target continuity and fidelity must be measured separately from final 3D rendered improvement.'],
    supersedes='round_scope.json wording Direct real RGB vs fusion as the primary formal pair; all frozen-scene, optimizer and stopping rules stay in force.')
p=RUN/'contract/target_information_amendment_v1.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(p)
