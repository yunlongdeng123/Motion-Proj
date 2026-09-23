"""从常驻公共包恢复必要的场景/日志表和地图JSON，不恢复大规模RGB/LiDAR。"""
import argparse
import json
from pathlib import Path
import tarfile
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    required = {"scene.json", "log.json"}
    if not all((out / name).exists() for name in required):
        with tarfile.open("/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval_meta.tgz", "r|gz") as archive:
            for member in archive:
                name = Path(member.name).name
                if name in required and member.isfile():
                    (out / name).write_bytes(archive.extractfile(member).read())
                    required.remove(name)
                    if not required:
                        break
    scenes = json.loads((out / "scene.json").read_text())
    logs = {row["token"]: row for row in json.loads((out / "log.json").read_text())}
    selection = {row["name"]: {"scene_token": row["token"], "location": logs[row["log_token"]]["location"]}
                 for row in scenes if row["name"] in ["scene-0230", "scene-0242", "scene-0255"]}
    assert len(selection) == 3
    with zipfile.ZipFile("/root/autodl-tmp/nuScenes-map-expansion-v1.3.zip") as archive:
        for location in {row["location"] for row in selection.values()}:
            matches = [name for name in archive.namelist() if name.endswith(f"/{location}.json") or name == f"{location}.json"]
            assert len(matches) == 1, matches
            destination = out / f"{location}.json"
            if not destination.exists():
                destination.write_bytes(archive.read(matches[0]))
            data = json.loads(destination.read_text())
            print(json.dumps({"location": location, "layers": list(data), "lane_divider_example": data.get("lane_divider", [])[:1]}), flush=True)
    (out / "scene-map-index.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection), flush=True)


if __name__ == "__main__":
    main()
