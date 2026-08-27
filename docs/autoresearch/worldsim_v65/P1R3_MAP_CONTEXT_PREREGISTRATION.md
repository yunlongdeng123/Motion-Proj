# P1R3 Map/Context Preregistration

`WS-V65-H-P1R3-001` tests the unexecuted R3 family from the V6.5 Signal Atlas. It does not rescue the rejected trajectory,
Actor/time, or admission models. The V6.4 q0 trunk/logit, original 16-scene Tier-L split, point sampling, seed, capacity, and
40% evaluation coverage remain fixed.

## Observable map contract

The source is the official nuScenes map expansion v1.3 already present on the public AutoDL volume. It is extracted to an
independent research data root; no repository checksum, fingerprint, or custom semantic label is added. For each target pose,
the official devkit rasterizes the following ego-aligned 200×200 layers at the native 0.512m grid: drivable area, road
segment, lane, pedestrian crossing, walkway, carpark area, road divider, and lane divider.

Per-voxel inputs are those eight binary semantics and signed distance to the drivable boundary. Five continuous unit features
are broadcast: route mean/max curvature, route length, fraction of route samples on drivable area, and local drivable fraction.
The hard 1.5m route corridor, scene ID, stratum, and hidden truth are not model features. MapTR/VAD motivate retaining map
structure as an explicit planning context; this probe uses the smallest rasterized interface already aligned to native voxels.

## Training and evaluation

The same deterministic point sampler as P1 is rerun so R3 has the same 523,910 train and 497,892 nested-evaluation target
denominators if all source units remain available. CPU threads prefetch native/evidence and rasterize the next unit while the
current unit's q0 embedding executes on GPU. The compact cache is written once.

Arms are frozen q0 and `q0 + 14D map/context FiLM residual`, with the same low-capacity architecture and seed 0 as P1. A
within-unit shuffle of all 14 map/context rows is the perturbation control. Positive train-only support requires all of:

- AUROC gain at least 0.005;
- matched-40% fixed-route risk reduction at least 5%;
- more evaluation scenes improve than worsen;
- non-route emitted risk increases by at most 5%;
- real map/context AUROC exceeds the within-unit shuffled control.

Failure closes R3 without a fresh cohort. Success only authorizes a separately frozen fresh map/context selection cohort; it
does not alter P2's negative result.

- MapTR (ICLR 2023 official code): https://github.com/hustvl/MapTR
- VAD (ICCV 2023): https://openaccess.thecvf.com/content/ICCV2023/html/Jiang_VAD_Vectorized_Scene_Representation_for_Efficient_Autonomous_Driving_ICCV_2023_paper.html
- nuScenes map expansion: https://github.com/nutonomy/nuscenes-devkit
