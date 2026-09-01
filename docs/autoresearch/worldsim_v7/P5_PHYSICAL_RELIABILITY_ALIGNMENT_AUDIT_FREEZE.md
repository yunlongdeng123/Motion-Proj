# P5 Physical--Reliability Alignment Audit Freeze

Date: 2026-09-02

## Question

Can V7 P4 physical Actor evidence be connected to the frozen V6.7 P346 multi-horizon reliability stack without letting scene or
Actor identity become a shortcut?

P4 identifies Actors by nuScenes scene name and instance token. V6.7 P109/P346 rows identify them by DriveStudio scene index and
numeric Actor id. The audit uses the official nuScenes scene-table ordering plus DriveStudio `instances_info.json[id]` to form an
exact semantic join. It reads no images, LiDAR, AV2, model outputs, or target quality beyond already retained rows.

## Frozen decision

Direct lightweight joint fitting is allowed only if P4's nuScenes **train** role aligns to at least three V6.7 source scenes and
twenty Actors. Three scenes is the minimum for a nontrivial scene-heldout split; twenty Actors is only a capacity floor for the
already small low-capacity interface. These two conditions replace exploratory training, not add a metric matrix.

If either fails, P5 does not train a new joint head. It may only execute a frozen descriptive interface audit on aligned retained
rows, with P346 and P4 weights unchanged. This is motivated by autonomous-driving domain-generalization evidence that scene and
sensor shifts are substantive, and by spurious-group work showing average ERM can hide minority-group failure.

References:

- https://openaccess.thecvf.com/content/ICCV2023/html/Sanchez_Domain_Generalization_of_3D_Semantic_Segmentation_in_Autonomous_Driving_ICCV_2023_paper.html
- https://proceedings.mlr.press/v139/liu21f.html
- https://github.com/ziyc/drivestudio/blob/main/docs/NuScenes.md

The output is descriptive alignment coverage by P4 role, scene, Actor, V6.7 source row count, and existing V6.7 scene fold. No
hash, checksum, fingerprint, model fit, calibration, or AV2 read is performed.
