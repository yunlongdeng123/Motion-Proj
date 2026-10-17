# R15 restored-capacity instance/road/shadow backend

Active entry: run.py train -> shadow_rebuild.train.run -> imported shadow_rebuild.objective.objective. The historical R14 objective definition retained earlier in train.py is inactive. Actual logs expose legacy_global_rgb/static_rgb/sky_rgb/dynamic_rgb and old100x visible mask plus new constraints.

Reuses original SH evaluator, EnvLight cube axes/kernel, hybrid/rigid losses, SSIM and guarded SGCM; rigid/evidence G-buffer APIs adapt actual owner/source/Adam rows. Static time/velocity freedom is removed; road albedo and edited Gaussian solar visibility explain RGB. Raw inputs and GT remain required; no pretrained GN11 parameters are loaded.

Production requires SH3, cube1024,3M capacity, dense measured static seeds, actual sky/box/SGCM/reference input paths. Fixture-only low capacity cannot run real scenes. Use configs/scene0008_restored.json; old simplified R14 profiles are not R15 configs.

CPU checks: python -m unittest discover -s tests -v (22 migrated checks in this package). The constant-sky CPU synthetic fixture is not the production cube kernel; that kernel was compared exactly against the actual old EnvLight on GPU. Original unused R14 tests/profiles are retained in historical snapshots, not listed as passing this suite.

Training45k=G30k/M10k/J5k. Generic topology: strictly after500 and before15000, interval100 (first600,last14900). Original SGCM offset50 avoids overlap. Native reset3000 causes a transient3k low-alpha preview. Train shadow visibility grids16/8/4 affect only the new light calculation; full RGB/SH/cube resolution is retained, final images use per-pixel integration.

Campaign scripts are bound to the preserved remote data hierarchy. They are not a self-contained dataset release. No automatic success verdict or shutdown is included in the worker.
