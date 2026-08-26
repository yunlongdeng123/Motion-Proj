# P10R4 I/O Shard Recovery Freeze

Date: 2026-08-27  
Failure: `V64-F26`  
Recovery raw run: `20260827T021800Z__test-raw-shard-recovery-s4-r1`

The first raw-only producer found all 14,437 required members absent from the persistent catalog and launched ten concurrent
full `.tgz` scans. After about four minutes every archive was only 4--10% complete, the shared NVMe workers were mostly in
page wait, and no complete scene had reached the GPU feeder. At that rate the single GPU would remain idle for roughly an hour.

The blocker was searched before changing execution. CPython tarfile still requires a sequential compressed-stream walk for
selected members. `ratarmount`/`rapidgzip` can create gzip seek-point indices, but building ten new indices is itself a full
archive pass and does not help this one exact-once cohort. The existing 71,555-entry semantic member-to-shard catalog offers a
cheaper migration: capture-prefix peers already map each selected scene unambiguously to its archive.

Seven scenes resolve directly from exact capture prefixes: `1084/1081->10`, `0462->05`, `0820->08`,
`0534/0598/0527->06`. Scene `0668` is the unresolved adjacent capture in the trainval07 scene range; already extracted files
with its exact prefix and neighboring catalogued temporal scenes support shard `07`. The recovery therefore scans only
`05,06,07,08,10`, preserving complete files already atomically written by the stopped all-shard attempt. Only process-scoped
`.partial.<pid>` files may be removed after the original workers stop.

References:

- `https://docs.python.org/3/library/tarfile.html`
- `https://github.com/mxmlnkn/ratarmount`
- `https://github.com/mxmlnkn/rapidgzip`

The scene cohort, model, M0/M1 policy, target frames, route, denominator, tail, gates, run IDs after raw preparation, and test
quality-unread state do not change. The original raw-only run remains an operations-failed record. The already-running feeder
remains the sole preprocess/native producer and resumes from complete raw files. This is an I/O recovery only; it creates no
new hashes, checksums, fingerprints, smoke suite, or regression matrix.
