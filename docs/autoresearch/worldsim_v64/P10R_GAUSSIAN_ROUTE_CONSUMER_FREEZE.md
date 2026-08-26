# P10R Bounded Gaussian Route Consumer Freeze

Date: 2026-08-26

## Objective

Bind the supported target-free Gaussian BEV states to a bounded logged-route semantic consumer without claiming collision or planning truth.

Task: `WS-V64-P10R-GAUSSIAN-ROUTE-CONSUMER-01`

Run: `20260826T183000Z__gaussian-route-consumer-s0-r1`

Hypothesis: `WS-V64-H-P10R-001`

## Frozen route contract

For each of the 96 case target frames, read the next 20 lidar poses at 10 Hz, transform their origins from world coordinates into the target lidar frame, and form a 1.5 metre horizontal corridor. Overlay that corridor on the P10G C0 and M0 BEV density arrays at the already frozen support threshold 0.05.

The run reports route length, corridor cells, support cells, density exposure mass, and whether each arm has any route interception. The input is limited to P10G packages and processed `lidar_pose` text files. Target evidence, the risk model, collision ground truth, and a planner are not read.

## Gates and claim boundary

Only two gates are used: all 96 cases are consumed, and aggregate M0 route support cells are strictly greater than C0. There is no horizon, corridor, threshold, scene, or seed sweep and no extra smoke/regression matrix.

A pass supports additional semantic exposure on logged future-route corridors. It does not establish physical collision, free-space truth, counterfactual route validity, planning utility, closed-loop behavior, or safety.
