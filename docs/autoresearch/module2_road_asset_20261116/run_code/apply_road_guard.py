"""以真实五帧保留前景道路贡献建立完整世界行回退资产。"""
import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82")
from drivingeditor_dynamic_edit.road_asset import apply_persistent_road_guard


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--base-road", required=True, type=Path)
    parser.add_argument("--guard", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base, candidate = arrays(args.base_road), arrays(args.asset / "road.npz")
    guard_data = arrays(args.guard)
    guarded, receipt = apply_persistent_road_guard(base, candidate, guard_data["guard"])
    np.savez_compressed(args.output / "road.npz", **guarded)
    provenance = arrays(args.asset / "provenance.npz")
    guard = guard_data["guard"]
    provenance["guarded_world_rows"] = guard
    provenance["effective_row_rgb"] = guarded["reference_srgb"]
    provenance["effective_active"] = guarded["active"]
    provenance["effective_sigma_m"] = guarded["gaussian_sigma_m"]
    provenance["effective_model_xyz"] = guarded["model_xyz"]
    provenance["planned_material_updated"] = provenance["material_updated"].copy()
    provenance["material_updated"][guard] = False
    provenance["preserved_legacy_rgb"][guard] = True
    for key in ["coverage_added", "coverage_repaired"]:
        provenance["planned_" + key] = provenance[key].copy()
        provenance[key][guard] = False
    if "observed_surface_activated" in provenance:
        provenance["planned_observed_surface_activated"] = provenance["observed_surface_activated"].copy()
        provenance["observed_surface_activated"][guard] = False
        provenance["coverage_trusted"] = provenance["trusted_geometry"] & ~provenance["observed_surface_activated"]
    np.savez_compressed(args.output / "provenance.npz", **provenance)
    for source in args.asset.iterdir():
        if source.is_file() and source.name not in ["road.npz", "provenance.npz", "receipt.json"]:
            shutil.copy2(source, args.output / source.name)
    previous = json.loads((args.asset / "receipt.json").read_text())
    previous.update(output_road=str(args.output / "road.npz"), persistent_guard=dict(**receipt,
        evidence_npz=str(args.guard), evidence_json=str(args.guard.with_suffix(".json")),
        source_asset=str(args.asset), parent_road=str(args.base_road)))
    previous["active_nodes"] = int(guarded["active"].sum())
    previous["material_updated_nodes"] = int(provenance["material_updated"].sum())
    previous["preserved_legacy_rgb_nodes"] = int(provenance["preserved_legacy_rgb"].sum())
    (args.output / "receipt.json").write_text(json.dumps(previous, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
