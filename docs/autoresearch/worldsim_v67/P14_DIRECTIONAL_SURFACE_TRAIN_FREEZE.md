# V6.7 P14 differentiable directional surface training freeze

P14 trains on the consumed V65 P10V six-scene cohort and performs one selection read on the consumed V65 P3C six-scene
cohort. P13 fresh targets are excluded. The model consumes the frozen 273-D native boundary representation plus 11 target-free
motion-compensated nearest-hit/ray features.

The P4R analytic support is immutable. The network is a `512/256/128` implicit residual field and can only rescue an analytic
rejection; it cannot delete analytic support or remove an Actor. Training uses 600 GPU epochs with an asymmetric conflict weight
and differentiable false-rescue penalty. A single probability threshold is derived only from the 99th percentile of training
conflict scores (lower bounded by 0.5), then frozen before selection.

Selection requires conflict reduction >=0.50, overall/clean retention >=0.40, clean-retention gain over the analytic core >=0.01,
at least 100 rescued clean points, and preservation of the analytic core/Actor existence. No architecture, loss or threshold sweep
is allowed on this selection cohort.
