LOD replay package

complete_a6.npz is the canonical persistent sampled asset. Loading it with its
ordinary .rows keeps the dense default raster; that alone does not give the LOD
screenshots. SurfaceLODAsset combines its fixed final material F with posterior
surface S and the camera-frequency sampler. The current camera changes render
sampling density/filtering; keep/delete never changes S/F or the asset rows for
that same camera. The 2778 retained PVD rows are carried without modification.

The bundle deliberately excludes the large frozen native scene/checkpoints and
CUDA environment. Their existing AutoDL absolute dependencies are in manifest.
REPLAY_COMMAND.txt produces solo, keep and delete native images + raw floats and
a receipt in a fresh output directory. No training is involved. Three-view and
replay smoke receipts state the actual validation; this is a research rendering
candidate, not an adoption or 360-degree/hidden-geometry accuracy assertion.
