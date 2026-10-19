# Actual code snapshot for AI review

Native baseline core is stored once under code/base. Apply one of code/overlays/{full,real_only,prior_only,ownership_only} to obtain an exact current engine core; code/materialize.py only copies and verifies files. code/source_map.json contains all per-experiment actual hashes. Baseline has no new ownership/real-road/prior-road modules. Actual configs/statuses are under run_metadata; original absolute paths are retained as provenance. No weights, datasets or CUDA binaries are committed here.
