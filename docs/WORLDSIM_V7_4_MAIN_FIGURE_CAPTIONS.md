# Figure captions

## fig06_shared_low_return_witness

**Shared first-return conflict in official geometry models.** The same measured heldout beam (8.564 m) intersects reconstructed triangles 0.413–0.666 m earlier in four official geometry models under the declared calibration and BUILD-scale diagnostic. RGB locates the target and low-return region; the side profile and range bars expose the otherwise nearly coincident image projections. The GT endpoint lies 2.65 cm below the annotated box bottom and is not asserted to be a car-body return. A single-beam inverse sensor model moves the occupied endpoint toward the sensor and leaves the occluded interval unknown. No closed-loop or false-safe outcome is measured.

## fig08_legacy_beam_counterexample

**A successful control retained from the original badcase.** The original legacy beam is 0.323 m early in an adapted VGGT–LiDAR fusion system. All four primary official models are within the fixed ±0.20 m Hit band on this same beam under the primary diagnostic. Native LiDAR reconstruction references use additional information and are shown separately. This retained counterexample prevents transferring a legacy system failure to current official models.

## fig03_white_car_evidence_vs_surface_count

**Input evidence and output surface count are separate interventions.** For the same 752 heldout returns, increasing RGB inputs from six to twelve is compared at a common six-map readout and at the full twelve-map readout. DVGT-1 and Pi3X improve Hit while reducing Early in the common-map control. Their additional Early errors appear after unioning more output surfaces. Since unioning a fixed superset can only move the first intersection closer, this plot does not establish an inevitable intrinsic completion trade-off. The common-map control is post-hoc discovery analysis.

## fig02_cross_scene_official_models

**Official-model first-return diagnostics across scene contexts.** Rows show metadata-selected near-static targets from four distinct discovery logs, plus the original white target. All columns use the same RGB crop. Red polygons are actual triangles causing >0.20 m early intersections on BUILD-supported heldout returns. Counts retain the full object denominator. Empty red regions and difficult visibility conditions are retained; 3D-box association does not certify semantic car-body ownership. Omega uses the user-provided original-512 mirror, not the 416 reproduction checkpoint.

## fig04_cross_log_early_rates

**Cross-log rates with explicit denominators.** Early-return rates use the same fixed calibrated-depth grid and global BUILD-background scale. All five discovery logs are retained, including the weak-reference log with only 25 rays. Two scenes belong to one log. These rates describe the exposed discovery window, not population prevalence or an independent confirmatory benchmark.

## fig05_confidence_controls

**Ordinary confidence filtering.** Per-view confidence retention is fixed at 100, 90, 75 and 50 percent. Each curve uses identical calibration and scale, with equal-log aggregation. Filtering reduces Early but also removes surface-point coverage. All settings are shown; no threshold is selected to maximize a failure claim. Coverage is vertex-neighborhood recall and depends on sampling density.

## fig07_dggt_native_rendering

**DGGT native Gaussian reconstruction and depth.** Official nuScenes weights reconstruct the input views through Gaussian RGB+expected-depth rasterization and the official sky model, with a predicted semantic sky mask replacing the dataset GT mask. Depth colors share a scale within each row. This is an input-view reconstruction check, not novel-view synthesis or physical LiDAR evaluation. All twelve depth/pose predictions exactly match VGGT, so the depth-grid results do not constitute an independent geometry failure.

## fig09_secondary_cross_scene

**Second-batch reconstruction references with explicit information budgets.** DVGT-2 RGB predictions use the declared calibrated-depth grid. NKSR and NoKSR receive additional BUILD LiDAR and PCA normals and retain their native learned surfaces. Red marks actual early-intersection triangles on BUILD-supported heldout rays. The same metadata-selected objects and full QUERY denominators are retained. Missing method inputs are not counted as successful predictions.

## fig01_legacy_white_car_story

**Legacy adapted-system example, retained as historical context.** Actual RGB, projected early-hit triangles, BEV rays, a metric range zoom and a single-beam occupancy diagnostic show a 0.323 m early intersection. The source is an adapted VGGT DPT plus LiDAR fusion surface, not official VGGT. The endpoint shift is measured; false-safe, obstacle-level BEV performance and closed-loop planning are not evaluated.

## architecture_components

**Components and information flow.** BUILD RGB enters frozen official geometry models. Saved native predictions feed a declared surface adapter and controlled first-hit raycasting. Known cameras and actor poses plus BUILD-only metric anchors are extra information; QUERY LiDAR is used only by the evaluator. Occupancy is a single-beam diagnostic and closed-loop planning remains untested. NKSR/NoKSR and DGGT rendering follow separately declared input and representation contracts.