# V43 shared semantic diagnostics template

CPU-only observational visualization, 2026-10-27. No checkpoint or renderer-owner changes.

Existing reviewed target masks, actual source LiDAR and camera calibration feed CPU projection/depth candidate checks, then semantic x original renderer branch colors, then frame/row inspection. GPU alpha*T visibility attribution and dense semantic segmentation remain pending. The architecture is documented in the local semantic-audit report.

Use the same template for every scene. Required scene files: scene.js, cloud.js, identity.js (all source vehicle boxes; target orange, others blue), edit_geometry.js, foreign_clouds.js, semantic_observations.js. Identity boxes are read-only audit overlays; never create mask or point semantic labels from GT boxes. Do not treat all static rows as background.

Run: python build_diagnostic_template.py --template-dir THIS_DIRECTORY --scene-dir EXISTING_SCENE_EXPORT_DIRECTORY

The shared parent viewer.js is generated from scene_viewer.js + scene_edit_geometry.js + scene_sampling.js + scene_semantics.js. Dynamic imports support file:// via local JS wrappers; no web service is required. Full-row sampling still uses cloud_full_manifest.js and cloud_full/*.js from the existing exports.

Semantic map: 0 unknown, 1 vehicle, 2 road, 3 sidewalk, 4 building, 5 vegetation, 6 sky, 7 pole/sign, 8 person/cycle. Only 0/1 have observations in this delivery; remaining IDs are reserved. Current vehicle hints cover only the reviewed target mask, not every visible vehicle. Unknown is not background. A parked vehicle can legitimately be static.

Colors in joint mode: green vehicle x dynamic, magenta target-mask vehicle x static (candidate overlap, not proven ownership error), slate unknown x static, light gray unknown x dynamic. Semantics use original observations and original per-row geometry, then remain attached to rows during edit visualization. Foreign-asset semantics currently abstain.

Default diagnostic: target mask center hit, nearest actual LiDAR <=2 pixels, abs z difference <=0.5m, max principal sigma <=0.5m. GUI supports mask-only projection (weak), radius 0..5px, depth tolerance and sigma controls. Neither profile changes trusted 0/1/U labels or model parameters. No counterfactual LiDAR is synthesized.

Rule audit: three development scenes, 49 frames each, exact saved display samples (18K static/10K dynamic each), three profiles. Only 263/714/225 of the respective 10K dynamic samples can be active across two fixed time bins; applying repeated sparse depth support rejects all dynamic positives. This invalidates using the fixed-bin rule as general per-row identity admission. Counts are sample statistics, not pixel contribution or semantic precision. The strict z-only proxy is weaker than the original 3D-distance test; original full-population results are preserved.

Next: validate projected semantics against actual alpha*T and occlusion, retain instance identity as a separate field, evaluate evidence within each row's effective lifetime plus cross-time instance/lineage evidence, then manually calibrate before learning unknown labels. Do not relax thresholds to achieve a desired unknown fraction. Dense segmentation inference requires a separately recorded model/version and evidence checks.

Host remains no-GPU and heartbeat paused. No GPU experiments, training, model downloads or completion were started. Quota at tool check: codex weekly remaining19%, quick-finish rule active. Do not restart waiting controllers from stale PID receipts.
