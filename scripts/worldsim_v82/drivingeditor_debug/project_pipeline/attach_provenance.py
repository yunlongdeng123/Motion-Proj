#!/usr/bin/env python3
"""Attach independently verified download/video identities to prepared inputs."""
import json
from pathlib import Path

base = Path("/root/autodl-tmp/drivingeditor_repro")
raw_hashes = {
    "0017085": "4b438ab67fb159704f25e16b4524957d2e4132d10758d79c961398c8aba973d4",
    "0145050": "5ef9ffe94920e1d0dab5bd90ba13c73cba3c53e25a3968c8e926beedd180b3fc",
    "0147030": "45e08afc6745fd64ecd8d1359b9a824841149d78ba370ecaa4ca4876addd6823",
    "0158150": "34e7bd088f1947864a476ef4eb16e9dba504bb98f90fd77c2301c5ccb71a3bf3",
}
for scene in ["0017085", "0017085_demo_000_020", "0145050", "0147030", "0158150"]:
    path = base / "data/waymo_scenes" / scene / "preparation_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["raw_sha256"] = raw_hashes[scene[:7]]
    manifest["raw_integrity_evidence"] = "Independent full-file SHA256 matches pinned mirror LFS record (data_audit)"
    if scene == "0017085_demo_000_020":
        manifest["source_demo"] = {
            "url": "https://github.com/user-attachments/assets/b990916b-9f0a-4ec8-a2f2-bdd152d4b468",
            "name": "20241214-010221.mp4", "video_fps": 8, "video_frames": 21,
            "verified_raw_camera": "FRONT", "verified_raw_frame_start": 0,
            "verified_raw_frame_end_inclusive": 20,
            "method": "Exhaustive RGB MAE against all 199 raw frames; unique best match for all 21 video frames; first/middle/last visually checked",
            "matching_evidence": str(base / "references/0017085_demo_match/frame_match.json"),
            "scope": "Exact video input images; author checkpoint and dynamic actor-selection settings are not supplied by the repository",
        }
    path.write_text(json.dumps(manifest, indent=2))
print("Attached raw SHA256 provenance to 5 scene inputs and exact-video matching provenance to demo")
