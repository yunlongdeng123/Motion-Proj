# P6R selective MLP development-training closeout

Date: 2026-08-26

Canonical run:

`run://worldsim_v64/WS-V64-P6R-SELECTIVE-MLP-01/20260826T134500Z__selective-mlp-s0-r1`

The single frozen run trained on 786,054 sampled native-boundary points from the 16 consumed development scenes. Inputs were all 273 native dimensions; 59,867 labels were hidden-FREE (prevalence 0.0761614). Focal loss decreased from 0.0337864 to 0.0251443 over the fixed 20 epochs. Development AUROC was 0.8811503 and is descriptive only.

GPU fit took 10.1545 seconds; total wall time including one data-read pass, descriptive scoring, and artifact write was 34.1934 seconds. Peak process RSS was 3.6301 GiB. The model artifact is 177 KiB and contains no hash, checksum, or fingerprint.

Verdict: `development_training_complete`. It is not calibration support. The model is now frozen; no refit or parameter sweep is allowed. The former eight confirmation scenes remain target-unread and may now be consumed once as independent calibration under `p6r_case_calibration_v1.yaml`. The new eight-scene confirmation cohort remains unread.
