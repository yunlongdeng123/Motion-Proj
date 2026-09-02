# WorldSim V7 Project-Page and Video Asset Index

## Literal first-return boundary asset

- Paper asset: `paper/figures/supplement/p21_first_return_boundary.png` and vector PDF counterpart.
- Renderer: `scripts/plot_worldsim_v7_p21_safety_boundary.py`.
- Inputs: frozen P20/P21/P22 canonical `summary.json` files plus the already reported source proxy rates
  `0.9662%/1.4362%`; no dataset, model, checkpoint, threshold, or policy is read.
- Left panel owns only the source/AV2 metric-correction comparison; right panel owns the consumed-source deletion frontier.
  Marker area encodes Chamfer penalty, and the dashed line marks one hazardous early event removed per matched hit lost.
- The AV2 bars are an already-consumed diagnostic, not fresh transfer, collision evidence, or a road-safety bound.

This index resolves the frozen qualitative evidence package without duplicating the 46 MiB binary bundle in Git.

Canonical run:

```text
run://worldsim_v7/WS-V7-P3B-AV2-CAMERA-EVIDENCE-01/20260902T150000Z__camera-evidence-s0-r1
```

Absolute AutoDL root:

```text
/root/autodl-tmp/runs/worldsim_v7/WS-V7-P3B-AV2-CAMERA-EVIDENCE-01/20260902T150000Z__camera-evidence-s0-r1
```

Machine-readable sources are `CAMERA_CASES.jsonl`, `MAIN_CAMERA_PANELS.json`, and
`SUPPLEMENT_CAMERA_PANELS.json`. Every panel and video path below is relative to the canonical run root.

## Frozen selection and presentation contract

- The ten qualitative logs are retained in their pre-registered order (`q00`--`q09`).
- Within each log, the first three eligible Actor UUIDs in lexical order are retained (`a0`--`a2`). No method metric,
  target geometry, RGB appearance, or video quality is used for case selection.
- Camera choice maximizes projected query-LiDAR visibility over the fixed ordered ring-camera list. RGB is decoded only
  after camera selection; ties use fixed camera order.
- The main-paper set is the `a0` Actor from `q00`--`q07` (8 panels). The compact supplement gallery is the `a0` Actor
  from `q00`--`q09` (10 panels). The full project-page package retains all 30 panels and all 30 videos.
- Videos are before/after temporal diagnostics with synchronized RGB and sparse depth, not photorealistic world renders.
- Actor identity, hazard label, log, camera, and path are immutable. Failed or visually weak cases remain in the package.

## Figure 1 four-quadrant teaser contract

- The `valid--safe` / `artifact--safe` pair is the first non-hazardous main-paper case, `q01-a0`:
  log `bf360aeb-1bbd-3c1e-b143-09cf83e4f2e4`, Actor `0e0562b6-0a44-4b8e-8a4d-685efcc4e9e1`,
  camera `ring_rear_left`, panel `panels/q01_a0_0e0562b6.png`.
- The `valid--hazard` / `artifact--hazard` pair is the first hazardous main-paper case, `q00-a0`:
  log `b87683ae-14c5-321f-8af3-623e7bafc3a7`, Actor `1f197615-ba7e-4e13-9301-1dcf39cc6839`,
  camera `ring_front_center`, panel `panels/q00_a0_1f197615.png`.
- “First” is evaluated in the already frozen `q00`--`q07` main-case order using only the stored hazard metadata. No RGB,
  depth, Chamfer, visibility, artifact severity, or rendering-quality choice was made for the teaser.
- Each left/right pair is a clipped view of the same frozen panel. Only the paired synthetic artifact overlay changes;
  Actor identity, trajectory, extent, camera, and hazard label remain fixed. The four crops are evidence overlays rather
  than photorealistic reconstructions, and no new binary rendering is introduced.

## Package summary

| Item | Value |
| --- | ---: |
| logs / Actors | 10 / 30 |
| hazardous / non-hazardous Actors | 13 / 17 |
| main / compact-supplement / full-package panels | 8 / 10 / 30 |
| videos | 30 |
| visible query returns, min / median / max | 14 / 82 / 4,096 |
| sparse-depth crop points, min / median / max | 870 / 2,358 / 17,286 |
| panel / video payload | 20,861,149 / 26,519,039 bytes |
| selected cameras | front-center 11; front-left 1; front-right 6; rear-left 8; rear-right 1; side-left 2; side-right 1 |

## Complete frozen case list

| Case | AV2 log | Actor UUID | Category | Hazard | Camera | Visible query | Crop depth | Role | Panel | Video |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| q00-a0 | `b87683ae-14c5-321f-8af3-623e7bafc3a7` | `1f197615-ba7e-4e13-9301-1dcf39cc6839` | REGULAR_VEHICLE | hazard | `ring_front_center` | 251 | 2287 | main+supp | `panels/q00_a0_1f197615.png` | `videos/q00_a0_1f197615.mp4` |
| q00-a1 | `b87683ae-14c5-321f-8af3-623e7bafc3a7` | `28b7a230-c6b0-4b87-ba5b-85186a7c89a8` | REGULAR_VEHICLE | non-hazard | `ring_front_center` | 18 | 2420 | supp | `panels/q00_a1_28b7a230.png` | `videos/q00_a1_28b7a230.mp4` |
| q00-a2 | `b87683ae-14c5-321f-8af3-623e7bafc3a7` | `353ec30c-f60b-4ec4-9607-1ed3a446342a` | REGULAR_VEHICLE | non-hazard | `ring_front_center` | 47 | 2358 | supp | `panels/q00_a2_353ec30c.png` | `videos/q00_a2_353ec30c.mp4` |
| q01-a0 | `bf360aeb-1bbd-3c1e-b143-09cf83e4f2e4` | `0e0562b6-0a44-4b8e-8a4d-685efcc4e9e1` | REGULAR_VEHICLE | non-hazard | `ring_rear_left` | 137 | 1770 | main+supp | `panels/q01_a0_0e0562b6.png` | `videos/q01_a0_0e0562b6.mp4` |
| q01-a1 | `bf360aeb-1bbd-3c1e-b143-09cf83e4f2e4` | `13e710cb-473b-49c4-ae6b-19d76493f5ac` | REGULAR_VEHICLE | non-hazard | `ring_front_left` | 238 | 7100 | supp | `panels/q01_a1_13e710cb.png` | `videos/q01_a1_13e710cb.mp4` |
| q01-a2 | `bf360aeb-1bbd-3c1e-b143-09cf83e4f2e4` | `16ebc621-c8b6-4c07-9128-d46127030969` | REGULAR_VEHICLE | non-hazard | `ring_rear_left` | 1130 | 6608 | supp | `panels/q01_a2_16ebc621.png` | `videos/q01_a2_16ebc621.mp4` |
| q02-a0 | `c865c156-0f26-411c-a16c-be985333f675` | `08fc423e-8c18-4a07-a5e2-3457069a4b42` | REGULAR_VEHICLE | non-hazard | `ring_side_right` | 85 | 2476 | main+supp | `panels/q02_a0_08fc423e.png` | `videos/q02_a0_08fc423e.mp4` |
| q02-a1 | `c865c156-0f26-411c-a16c-be985333f675` | `1089a9c9-384b-46f7-9ecf-5a6103b53b92` | REGULAR_VEHICLE | non-hazard | `ring_front_right` | 39 | 2186 | supp | `panels/q02_a1_1089a9c9.png` | `videos/q02_a1_1089a9c9.mp4` |
| q02-a2 | `c865c156-0f26-411c-a16c-be985333f675` | `226eb544-1f6d-471b-9d09-355ee27ee02a` | REGULAR_VEHICLE | non-hazard | `ring_front_right` | 45 | 2580 | supp | `panels/q02_a2_226eb544.png` | `videos/q02_a2_226eb544.mp4` |
| q03-a0 | `d1395998-7e8a-417d-91e9-5ca6ec045ee1` | `05cccb41-2ec0-4d84-8bc7-9a6e1c71af4c` | REGULAR_VEHICLE | hazard | `ring_side_left` | 2456 | 17286 | main+supp | `panels/q03_a0_05cccb41.png` | `videos/q03_a0_05cccb41.mp4` |
| q03-a1 | `d1395998-7e8a-417d-91e9-5ca6ec045ee1` | `1bde1918-0e30-4cce-be34-930d59e6743e` | BUS | non-hazard | `ring_front_center` | 4096 | 10193 | supp | `panels/q03_a1_1bde1918.png` | `videos/q03_a1_1bde1918.mp4` |
| q03-a2 | `d1395998-7e8a-417d-91e9-5ca6ec045ee1` | `1ce92c5b-e60d-4e85-b477-601ae8e55624` | REGULAR_VEHICLE | non-hazard | `ring_rear_left` | 152 | 1109 | supp | `panels/q03_a2_1ce92c5b.png` | `videos/q03_a2_1ce92c5b.mp4` |
| q04-a0 | `d770f926-bca8-31de-9790-73fbb7b6a890` | `036844e3-c000-4e66-8c6e-4ade42571253` | REGULAR_VEHICLE | hazard | `ring_front_right` | 292 | 2358 | main+supp | `panels/q04_a0_036844e3.png` | `videos/q04_a0_036844e3.mp4` |
| q04-a1 | `d770f926-bca8-31de-9790-73fbb7b6a890` | `04cf1201-2e7a-4110-8145-03e5e46f8e30` | REGULAR_VEHICLE | non-hazard | `ring_rear_left` | 37 | 1066 | supp | `panels/q04_a1_04cf1201.png` | `videos/q04_a1_04cf1201.mp4` |
| q04-a2 | `d770f926-bca8-31de-9790-73fbb7b6a890` | `1ffcff2e-4118-45b8-8207-f8a8a450c743` | REGULAR_VEHICLE | hazard | `ring_front_right` | 38 | 2488 | supp | `panels/q04_a2_1ffcff2e.png` | `videos/q04_a2_1ffcff2e.mp4` |
| q05-a0 | `de56b100-508b-3479-81fe-735349f8e8de` | `07695d24-f795-4d40-a5cf-b0a7c0fdc10b` | REGULAR_VEHICLE | hazard | `ring_front_center` | 14 | 2598 | main+supp | `panels/q05_a0_07695d24.png` | `videos/q05_a0_07695d24.mp4` |
| q05-a1 | `de56b100-508b-3479-81fe-735349f8e8de` | `0b12eaa1-b12a-44d8-a450-5b8435a1c183` | REGULAR_VEHICLE | non-hazard | `ring_rear_left` | 37 | 1492 | supp | `panels/q05_a1_0b12eaa1.png` | `videos/q05_a1_0b12eaa1.mp4` |
| q05-a2 | `de56b100-508b-3479-81fe-735349f8e8de` | `0d0ed06f-9df8-4d69-b2d7-fa63b726575f` | REGULAR_VEHICLE | hazard | `ring_front_center` | 19 | 2565 | supp | `panels/q05_a2_0d0ed06f.png` | `videos/q05_a2_0d0ed06f.mp4` |
| q06-a0 | `e0ea281b-6956-3605-b720-71b54ec87d25` | `08515e0d-785e-407d-8ef3-2de4759d9aee` | REGULAR_VEHICLE | non-hazard | `ring_front_right` | 79 | 2757 | main+supp | `panels/q06_a0_08515e0d.png` | `videos/q06_a0_08515e0d.mp4` |
| q06-a1 | `e0ea281b-6956-3605-b720-71b54ec87d25` | `22591769-257b-4b36-8e3a-d23d3e716252` | REGULAR_VEHICLE | hazard | `ring_front_center` | 15 | 2637 | supp | `panels/q06_a1_22591769.png` | `videos/q06_a1_22591769.mp4` |
| q06-a2 | `e0ea281b-6956-3605-b720-71b54ec87d25` | `27c2b4d9-8542-49e6-a408-f022a294a8ab` | REGULAR_VEHICLE | non-hazard | `ring_side_left` | 16 | 2087 | supp | `panels/q06_a2_27c2b4d9.png` | `videos/q06_a2_27c2b4d9.mp4` |
| q07-a0 | `e50e7698-de3d-355f-aca2-eddd09c09533` | `076c9103-ab14-421a-ae8f-e8b35745b6ba` | BUS | non-hazard | `ring_front_center` | 37 | 2064 | main+supp | `panels/q07_a0_076c9103.png` | `videos/q07_a0_076c9103.mp4` |
| q07-a1 | `e50e7698-de3d-355f-aca2-eddd09c09533` | `10980c47-891f-46cc-a86a-dd24860ce567` | REGULAR_VEHICLE | non-hazard | `ring_rear_left` | 781 | 3474 | supp | `panels/q07_a1_10980c47.png` | `videos/q07_a1_10980c47.mp4` |
| q07-a2 | `e50e7698-de3d-355f-aca2-eddd09c09533` | `1fcf1d7b-9eff-4508-a6af-d75e5541f8bd` | REGULAR_VEHICLE | hazard | `ring_front_right` | 28 | 2280 | supp | `panels/q07_a2_1fcf1d7b.png` | `videos/q07_a2_1fcf1d7b.mp4` |
| q08-a0 | `f1275002-842e-3571-8f7d-05816bc7cf56` | `08432e5c-f1d2-446f-85e2-e8052d64a0b7` | REGULAR_VEHICLE | hazard | `ring_front_center` | 231 | 2101 | supp | `panels/q08_a0_08432e5c.png` | `videos/q08_a0_08432e5c.mp4` |
| q08-a1 | `f1275002-842e-3571-8f7d-05816bc7cf56` | `16a8cf78-5490-47bc-94d9-f797876c53f6` | REGULAR_VEHICLE | hazard | `ring_front_center` | 170 | 2216 | supp | `panels/q08_a1_16a8cf78.png` | `videos/q08_a1_16a8cf78.mp4` |
| q08-a2 | `f1275002-842e-3571-8f7d-05816bc7cf56` | `1e9b3160-7aad-4e02-91e1-ff98eeedd230` | REGULAR_VEHICLE | hazard | `ring_rear_left` | 928 | 4391 | supp | `panels/q08_a2_1e9b3160.png` | `videos/q08_a2_1e9b3160.mp4` |
| q09-a0 | `f668074d-d6c6-3ea7-a7b5-aad0a1203b03` | `6d55ff95-10af-43d1-a6b4-00d12244f007` | LARGE_VEHICLE | non-hazard | `ring_front_center` | 38 | 1630 | supp | `panels/q09_a0_6d55ff95.png` | `videos/q09_a0_6d55ff95.mp4` |
| q09-a1 | `f668074d-d6c6-3ea7-a7b5-aad0a1203b03` | `75b1c75d-7221-44e4-ba17-6ea373ac7079` | REGULAR_VEHICLE | hazard | `ring_rear_right` | 269 | 870 | supp | `panels/q09_a1_75b1c75d.png` | `videos/q09_a1_75b1c75d.mp4` |
| q09-a2 | `f668074d-d6c6-3ea7-a7b5-aad0a1203b03` | `a5a7675d-dc8a-48f0-bca3-21cd6c2afd74` | REGULAR_VEHICLE | hazard | `ring_rear_left` | 115 | 990 | supp | `panels/q09_a2_a5a7675d.png` | `videos/q09_a2_a5a7675d.mp4` |

## Suggested project-page grouping

1. Lead with q00-a0, q03-a0, q05-a0, and q07-a0 to span dense frontal, side-view, far/sparse, and exposed
   Chamfer-worsening evidence without changing the frozen case set.
2. Present the remaining six `a0` cases as the compact gallery in pre-registered order.
3. Expose all `a1/a2` panels and videos under an “all frozen cases” accordion, retaining weak-visibility and failure cases.
4. Caption every video as a diagnostic visualization; do not describe it as a photorealistic reconstruction or closed-loop rollout.
