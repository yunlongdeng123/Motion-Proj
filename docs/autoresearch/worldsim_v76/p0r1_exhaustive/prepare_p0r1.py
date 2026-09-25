"""创建独立 exhaustive 复现 run，只复用既有图像/SIFT，不复用匹配、几何或权重。"""
import json
import shutil
import sqlite3
from pathlib import Path
import yaml

root = Path("/root/autodl-tmp/external/worldsim_v75/VAD-GS")
old = Path("/root/autodl-tmp/runs/v76_ego_view/VADGS-P0-000")
new = old.parent / "VADGS-P0R1-000"
config = root / "configs/v76/nuscenes_000_repro_exhaustive.yaml"
assert not new.exists(), f"refuse overwrite: {new}"
assert not config.exists(), f"refuse overwrite: {config}"
new.mkdir()
colmap = new / "colmap"
colmap.mkdir()
src = sqlite3.connect(f"file:{old / 'colmap/database.db'}?mode=ro", uri=True)
dst = sqlite3.connect(colmap / "database.db")
src.backup(dst)
# 仅清理新副本，旧 run 保持完整；匹配从全图 exhaustive 重新开始。
for table in ("matches", "two_view_geometries"):
    dst.execute(f"DELETE FROM {table}")
dst.commit()
dst.execute("VACUUM")
counts = {t: dst.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
          for t in ("images", "keypoints", "descriptors", "matches", "two_view_geometries")}
assert counts == {"images":305, "keypoints":305, "descriptors":305, "matches":0, "two_view_geometries":0}
dst.close()
src.close()
shutil.copytree(old / "colmap/created", colmap / "created")
assert (colmap / "created/sparse/model/points3D.txt").stat().st_size == 0
for name in ("train_imgs", "mask"):
    (colmap / name).symlink_to(old / "colmap" / name, target_is_directory=True)
for name in ("id_names.txt", "cam_rigid_config.json"):
    shutil.copy2(old / "colmap" / name, colmap / name)
(colmap / "triangulated/sparse/model").mkdir(parents=True)
cfg = yaml.safe_load((root / "configs/v76/nuscenes_000_train.yaml").read_text())
cfg["exp_name"] = "vadgs_official_000_repro_exhaustive"
cfg["data"]["colmap_matching"] = "exhaustive"
cfg["resume"] = False
cfg["model_path"] = str(new)
cfg["record_dir"] = str(new / "records")
config.write_text(yaml.safe_dump(cfg, sort_keys=False))
manifest = {"run_id":"VADGS-P0R1-000", "seed":0, "config":str(config),
    "source_path":cfg["source_path"], "previous_run":str(old),
    "reuse":"only same official RGB images, SIFT keypoints/descriptors, input calibration/poses",
    "new":"exhaustive matches, triangulation, corrected name-based track mapping, fresh initialization and all weights",
    "database_initial_counts":counts, "colmap":"3.7 CPU, exhaustive matcher, 12 threads, seed 0",
    "training":"30000 iterations; official cameras 0-4 temporal split; camera 5 extrapolation separate",
    "failure_ledger_refs":["V76-F01"], "failure_ledger_delta":"none; evaluates the engineering correction",
    "shutdown_authorization":"2026-09-26 user asked autonomous continuation, then AutoDL shutdown after sufficient completion and saving results"}
(new / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(manifest, ensure_ascii=False))
