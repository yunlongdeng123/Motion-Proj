# Native CPU build attempt 01: authorized compile only

This attempt supersedes the frozen candidate's earlier "do not install now" note only for private dependency preparation and compilation. It may extract compatible required packages into `../deps/prefix`, read existing system compiler/TBB/OpenMP, and compile the unchanged author's core with at most one build worker. It may not run maintainer scripts or install/update packages globally, alter compiler/CUDA/PyTorch/profile/symlinks, construct NativeBackend, initialize the detector, process a real scan, publish labels or train.

The frozen `../handoff_manifest.json` SHA256 is `50f603c82ac62d834ccd10f9e37ddf73a2392fe5451bd2e434c32b895f94a47d`. All20 vendor source/config files must remain byte-identical. Any necessary portability change is confined to this attempt's build or adapter boundary and must be recorded separately. Author event math, thresholds, clustering, grid, buffer and defaults remain unchanged.

Prerequisite storage on this host is80,393,879,552 free bytes (about74.9GiB), satisfying the30GiB minimum. Abort if free disk falls below30GiB, private dependency download exceeds3GiB or attempt/prefix growth exceeds8GiB. No native object construction is used to assess RAM: author's dense arrays still require a separately reviewed resource audit before execution.

Existing read-only dependency facts: Ubuntu22.04.1, GNU g++11.3.0, system TBB2021.5.0-7ubuntu2 and libgomp12.3.0-1ubuntu1~22.04. Private package receipts record cached apt version/source metadata, configured public repository URI, expected/actual SHA256 and bytes, extraction destination and resource observations. `apt-get download` and `dpkg-deb --extract` do not register/install packages or execute their maintainer scripts. No apt update/install command is run.

After preparation, the effective CMake prefix is `../deps/prefix/usr`, reflecting the extracted Debian package layout. The build compiles a static library only. A successful build is not an algorithm/effect check, does not create genuine M-detector labels and does not make the unresolved scan-start/midpoint/scan-end reference contract ready.
