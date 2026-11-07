"""R0.0: read-only V44 coverage audit; at most eight initial CUDA forwards.

Uses the archived V44 runtime in its own interpreter. Does not train, extend the
road, rewrite source contracts, or manage any other process or power state.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np

CAM = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
V44 = CAM / "scene008_v44_rgb_asset_20261107_r1"
FULL_GEOMETRY = CAM / "r15_gn11_fullclip_cpu_20261026/frames"
PILOT = (105, 113, 130, 137)
FIXED720 = (230, 204, 270, 222)
SILVER1600 = (210, 213, 290, 233)
SILVER4992 = (202, 192, 298, 244)
GROUPS = ("asset", "a37", "road", "native_static")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def stat(value, mask):
    selected = np.asarray(value, np.float64)[mask]
    if not selected.size:
        return dict(pixels=0, sum=0., mean=None, minimum=None, maximum=None)
    return dict(pixels=int(selected.size), sum=float(selected.sum()),
                mean=float(selected.mean()), minimum=float(selected.min()),
                maximum=float(selected.max()))


def rgb_stat(value, mask):
    return [stat(channel, mask) for channel in value]


def roi(box):
    result = np.zeros((320, 480), bool)
    x0, y0, x1, y1 = box
    result[y0:y1, x0:x1] = True
    return result


def bbox(mask):
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if x.size else None


def masks_for(obs, raw):
    fixed = roi(FIXED720)
    box = np.asarray(obs["box_candidate"][0], bool)
    band = np.zeros_like(fixed)
    bounds = bbox(box)
    if bounds:
        x0, y0, x1, y1 = bounds
        band[min(320, y1 + 2):min(320, y1 + 22), x0 + 2:max(x0 + 2, x1 - 2)] = True
    masks = dict(r0_fixed720=fixed, under_tail_band=band,
                 observed_road=np.asarray(obs["road"], bool),
                 road_or_edit_outside_a6_box=(np.asarray(obs["road"], bool) |
                     np.asarray(obs["road_edit"], bool)) & ~box,
                 a6_visible=np.asarray(obs["visible"][0], bool),
                 a6_box_unknown=box & ~np.asarray(obs["visible"][0], bool))
    if raw == 137:
        masks["v44_silver1600"] = roi(SILVER1600)
    if raw == 130:
        masks["v44_silver4992"] = roi(SILVER4992)
    return masks


def statistics(arrays, masks):
    groups = arrays["group_alpha_T"]
    result = {}
    for name, mask in masks.items():
        item = dict(pixels=int(mask.sum()), bbox_xyxy=bbox(mask),
            groups_alpha_T={g: stat(a, mask) for g, a in zip(GROUPS, groups)},
            environment_T=stat(1 - arrays["alpha"], mask),
            joint_RGB=rgb_stat(arrays["rgb"], mask),
            joint_depth_m=stat(arrays["depth_m"], mask),
            joint_road_lt001=int((mask & (groups[2] < .01)).sum()),
            joint_asset_lt001_road_lt001_env_gt05=int((mask & (groups[0] < .01) &
                (groups[2] < .01) & ((1 - arrays["alpha"]) > .5)).sum()))
        if "road_solo_alpha" in arrays:
            solo = arrays["road_solo_alpha"]
            item.update(road_solo_alpha=stat(solo, mask),
                road_solo_RGB=rgb_stat(arrays["road_solo_rgb"], mask),
                road_solo_depth_m=stat(arrays["road_solo_depth_m"], mask & (solo > 1e-8)),
                road_solo_zero=int((mask & (solo == 0)).sum()),
                road_solo_lt001=int((mask & (solo < .01)).sum()),
                solo_positive_but_joint_road_lt001=int((mask & (solo > .01) & (groups[2] < .01)).sum()))
        if "group_rgb_T" in arrays:
            item["actual_groups_RGB_T"] = {g: rgb_stat(c, mask) for g, c in zip(GROUPS, arrays["group_rgb_T"])}
            item["actual_environment_RGB_T"] = rgb_stat(arrays["env_rgb_T"], mask)
        result[name] = item
    return result


def image(a):
    from PIL import Image
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = a.transpose(1, 2, 0)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, 2)
    return Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255 + .5))


def panel(values, labels, path, crop=None, factor=1):
    from PIL import Image, ImageDraw
    tiles = []
    for value, label in zip(values, labels):
        im = image(value)
        if crop:
            im = im.crop(crop)
        im = im.resize((im.width * factor, im.height * factor), Image.Resampling.NEAREST)
        tile = Image.new("RGB", (im.width, im.height + 24), "#eef3f6")
        tile.paste(im, (0, 24))
        ImageDraw.Draw(tile).text((3, 5), label, fill="#102a39")
        tiles.append(tile)
    out = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height), "#eef3f6")
    x = 0
    for tile in tiles:
        out.paste(tile, (x, 0)); x += tile.width
    out.save(path)


def configuration(args):
    contract = read(args.contract)
    paths = {k: Path(v) for k, v in contract["paths"].items()}
    source = args.runtime_source
    tracked = [Path(__file__).resolve(), source, args.contract,
               source.parent.parent / "contract/v44_contract.py",
               paths["road_checkpoint"], paths["road_run"] / "inputs/road_reference.npz",
               paths["fine"] / "coefficients.npz", paths["fine"] / "design.json",
               paths["asset"], paths["posterior"]]
    return paths, {str(p): sha(p) for p in tracked}


def fine_index(xy, ref):
    keys = ref["keys"]
    low = keys.min(0)
    shape = keys.max(0) - low + 1
    lookup = np.full(tuple(shape), -1, np.int32)
    ij = keys - low
    lookup[ij[:, 0], ij[:, 1]] = np.arange(len(keys))
    ij = np.floor(xy / float(ref["cell_m"])).astype(np.int64) - low
    inside = (ij >= 0).all(-1) & (ij < shape).all(-1)
    safe = np.clip(ij, [0, 0], shape - 1)
    ids = lookup[safe[..., 0], safe[..., 1]]
    ids[~inside] = -1
    return ids


def cpu(args):
    """Saved V44 floats and cached measured geometry only; no torch/CUDA import."""
    from scipy.spatial import cKDTree
    from PIL import Image, ImageDraw
    paths, before = configuration(args)
    dest = args.output / "cpu_all49"
    dest.mkdir(parents=True, exist_ok=False)
    ref = load_npz(paths["road_run"] / "inputs/road_reference.npz")
    tree = cKDTree(ref["local_xyz"][ref["active"], :2])
    rows, overview, source_files = [], [], {}
    yy, xx = np.mgrid[:320, :480]
    for raw in range(89, 138):
        obs_path = paths["observations"] / f"{raw:03d}.npz"
        saved_path = args.saved_results / f"{raw:03d}/delete_float.npz"
        obs, arrays = load_npz(obs_path), load_npz(saved_path)
        arrays["alpha"] = arrays["alpha"].squeeze()
        masks = masks_for(obs, raw)
        view_path = paths["road_run"] / f"inputs/views/{raw:03d}.npz"
        geometry_path = FULL_GEOMETRY / f"{raw:03d}.npz"
        view, geo = load_npz(view_path), load_npz(geometry_path)
        depth, uid = view["depth_m"], view["unit_id"]
        rays = np.stack((xx + .5, yy + .5, np.ones_like(xx)), 2) @ np.linalg.inv(geo["K"]).T
        inverse = np.linalg.inv(geo["w2l"])
        local = (rays @ inverse[:3, :3].T) * depth[:, :, None] + inverse[:3, 3]
        valid = (uid >= 0) & np.isfinite(depth) & (depth > 0)
        indices = fine_index(np.nan_to_num(local[:, :, :2]), ref)
        member = valid & (indices >= 0)
        active = member & ref["active"][np.maximum(indices, 0)]
        nearest = tree.query(np.nan_to_num(local[:, :, :2]).reshape(-1, 2))[0].reshape(320, 480)
        world = {}
        for name, mask in masks.items():
            chosen = mask & valid
            world[name] = dict(pixels=int(mask.sum()), measured_geometry_rays=int(chosen.sum()),
                geometry_unknown=int((mask & ~valid).sum()), fine_candidate=int((mask & member).sum()),
                fine_active=int((mask & active).sum()), fine_absent=int((chosen & ~member).sum()),
                nearest_active_xy_m=stat(nearest, chosen),
                local_min_m=local[chosen].min(0).tolist() if chosen.any() else None,
                local_max_m=local[chosen].max(0).tolist() if chosen.any() else None)
        row = dict(raw=raw, state="delete", statistics=statistics(arrays, masks), world=world,
                   source_float_sha256=sha(saved_path))
        rows.append(row)
        source_files[str(obs_path)] = sha(obs_path)
        source_files[str(view_path)] = sha(view_path)
        source_files[str(geometry_path)] = sha(geometry_path)
        groups = arrays["group_alpha_T"]
        failure = masks["road_or_edit_outside_a6_box"] & (groups[0] < .01) & (groups[2] < .01) & ((1-arrays["alpha"]) > .5)
        overlay = arrays["rgb"].transpose(1, 2, 0).copy()
        overlay[failure] = .2 * overlay[failure] + .8 * np.array([1., .15, .05])
        tile = image(overlay).resize((240, 160))
        tile2 = Image.new("RGB", (240, 182), "#eef3f6"); tile2.paste(tile, (0, 22))
        ImageDraw.Draw(tile2).text((3, 4), f"raw{raw} fixed720 road={world['r0_fixed720']['fine_active']}", fill="#102a39")
        overview.append(tile2)
        if raw in PILOT:
            np.savez_compressed(dest / f"{raw:03d}_world_maps.npz", local_xyz=local.astype(np.float32),
                geometry_valid=valid, fine_id=indices, fine_active=active,
                nearest_active_xy_m=nearest.astype(np.float32), **masks)
            footprint = np.zeros((320, 480, 3), np.float32)
            footprint[valid] = [1., .15, .05]; footprint[member] = [1., .7, 0]; footprint[active] = [.1, .8, .2]
            panel([arrays["rgb"], footprint, groups[2], 1-arrays["alpha"]],
                ["V44 saved delete", "Measured ray: absent red / active green", "Road alpha*T", "Environment T"],
                dest / f"{raw:03d}_world_panel.png")
        print(json.dumps(dict(phase="cpu", raw=raw, fixed720=world["r0_fixed720"])), flush=True)
    sheet = Image.new("RGB", (240*7, 182*7), "#eef3f6")
    for i, tile in enumerate(overview): sheet.paste(tile, (i % 7 * 240, i // 7 * 182))
    sheet.save(dest / "all49_coverage_overview.jpg", quality=92)
    result = dict(status="PASS_CPU_ALL49_SAVED_V44_COVERAGE", frames=49, fresh_forwards=0,
        road_reference_schema={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in ref.items()},
        world_domain=dict(all_candidates=len(ref["keys"]), active=int(ref["active"].sum()),
            local_min_m=ref["local_xyz"].min(0).tolist(), local_max_m=ref["local_xyz"].max(0).tolist()),
        contracts=dict(fixed720=dict(xyxy=FIXED720, pixels=720, role="R0 independent fixed rectangle"),
            silver1600=dict(raw=137, xyxy=SILVER1600, pixels=1600, role="V44 full silver rectangle; different denominator"),
            geometry="Cached measured view depth/unit_id with original fullclip K,w2l; absent unit_id is unknown geometry, not empty road",
            gap="Observed road/edit excluding A6 box AND asset alphaT<.01 AND road alphaT<.01 AND environmentT>.5",
            target="Original visible RGB/proxy only; no independent hidden delete GT"), rows=rows,
        tracked_sha256_before=before, tracked_sha256_after={p:sha(p) for p in before},
        source_input_sha256=source_files, updates=0, training=False, cuda_imported=False)
    assert result["tracked_sha256_before"] == result["tracked_sha256_after"]
    write(dest / "cpu_all49.json", result)


def gpu_snapshot():
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True).strip()
    pids = [line.split(",")[0].strip() for line in processes.splitlines() if line]
    commands = subprocess.check_output(["ps", "-p", ",".join(pids), "-o", "pid=,args="], text=True).strip() if pids else ""
    return dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), gpu=gpu,
                compute_processes=processes, process_commands=commands, own_pid=os.getpid())


def gate_gpu(snapshot):
    fields = snapshot["gpu"].splitlines()[0].split(",")
    free = int(fields[-2])
    external = []
    for line in snapshot["compute_processes"].splitlines():
        if line and int(line.split(",")[0]) != os.getpid(): external.append(line)
    unknown = [line for line in external if int(line.split(",")[0]) != 13030]
    known_command = "scene008_full_dynamic_edit_20261107_r1/source/train_full5.py --mode formal" in snapshot["process_commands"]
    if free < 20*1024 or unknown or (external and not known_command):
        raise RuntimeError("GPU_BUSY_DIAGNOSTIC_NOT_STARTED: " + json.dumps(snapshot))


def gpu(args):
    snapshot = gpu_snapshot(); gate_gpu(snapshot)
    paths, before = configuration(args)
    dest = args.output / "gpu_initial8"
    dest.mkdir(parents=True, exist_ok=False)
    write(dest / "preflight_gpu.json", snapshot)
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.30)
    sys.path.insert(0, str(args.runtime_source.parent))
    spec = importlib.util.spec_from_file_location("v44_runtime", args.runtime_source)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    started = time.time(); ledger = []; rows = []

    def count_forward(raw, kind):
        if len(ledger) >= 8: raise RuntimeError("Eight-forward initial diagnostic budget exhausted")
        ledger.append(dict(raw=raw, kind=kind, ordinal=len(ledger)+1, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        write(dest / "forward_ledger.json", dict(maximum_initial_forwards=8, launched=ledger))

    with torch.no_grad():
        rt = module.V44Runtime(args.contract)
        initial_versions = rt.tensor_versions(); initial_colours = rt.colour_versions()
        for raw in PILOT:
            data = rt.raw(raw); camera = data["cam"]; obs = data["obs"]
            road = rt.road_kwargs(camera, rt.exp.repair.road, rt.pipe, sh_channels=1, precomputed_color=True)
            road["colors_precomp"] = rt.road_colours
            count_forward(raw, "road_isolated")
            solo = rt.raster(camera, rt.pipe, 0, road, rt.n.bg.new_zeros(3))
            count_forward(raw, "V44_delete_joint")
            joint = rt.render_scene(raw, delete=True)
            to_np = lambda value: value.detach().cpu().numpy()
            alpha = to_np(joint["alpha"])[0]; solo_alpha = to_np(solo["alpha"])[0]
            arrays = dict(rgb=to_np(joint["rgb"]), alpha=alpha,
                depth_m=to_np(joint["depth_sum"])[0]/np.maximum(alpha, 1e-8)/rt.scale,
                group_alpha_T=to_np(joint["group_alpha_T"]), group_rgb_T=to_np(joint["group_rgb_T"]),
                env_rgb_T=to_np(joint["env_rgb_T"]), road_solo_alpha=solo_alpha,
                road_solo_rgb=to_np(solo["rgb"]),
                road_solo_depth_m=to_np(solo["depth_sum"])[0]/np.maximum(solo_alpha, 1e-8)/rt.scale)
            assert all(np.isfinite(a).all() for a in arrays.values())
            xyz = to_np(road["means3D"]).astype(np.float64)
            transform = obs["model_to_camera"]; cc = xyz @ transform[:3,:3].T + transform[:3,3]
            uvh = cc @ obs["K"].T; uv = uvh[:,:2]/cc[:,2:]-.5; radii=to_np(solo["radii"])
            overlaps = {}
            masks = masks_for(obs, raw)
            for name, box in [("r0_fixed720",FIXED720)] + ([("v44_silver1600",SILVER1600)] if raw==137 else []):
                x0,y0,x1,y1=box
                hit=(radii>0)&(uv[:,0]+radii>=x0)&(uv[:,0]-radii<x1)&(uv[:,1]+radii>=y0)&(uv[:,1]-radii<y1)
                overlaps[name]=int(hit.sum())
            saved=load_npz(args.saved_results/f"{raw:03d}/delete_float.npz")
            errors={k:float(np.max(np.abs(arrays[k]-saved[k].squeeze() if k=="alpha" else arrays[k]-saved[k]))) for k in ("rgb","alpha","depth_m","group_alpha_T")}
            row=dict(raw=raw, state="delete", statistics=statistics(arrays,masks),
                source_scene_meta=joint["meta"], road_rows=len(xyz), active_road_rows=int(to_np(road["mask"]).sum()),
                positive_camera_depth_road_rows=int((cc[:,2]>.2).sum()), nonzero_solo_radii=int((radii>0).sum()),
                splat_AABB_ROI_overlap=overlaps, saved_V44_float_parity_maxabs=errors,
                group_alpha_conservation_maxabs=joint["group_conservation_maxabs"],
                group_RGB_reconstruction_maxabs=joint["color_reconstruction_maxabs"])
            assert max(errors[k] for k in ("rgb","alpha","group_alpha_T")) < 2e-5
            assert errors["depth_m"] < 5e-5
            rows.append(row)
            np.savez_compressed(dest/f"{raw:03d}_fresh.npz", **arrays, road_centres_uv=uv.astype(np.float32), road_radii=radii)
            image(arrays["rgb"]).save(dest/f"{raw:03d}_delete.png")
            values=[arrays["rgb"],solo_alpha,arrays["group_alpha_T"][2],arrays["group_alpha_T"][3],1-alpha]
            labels=["V44 joint delete", "Road solo alpha", "Joint road alpha*T", "Native static alpha*T", "Environment T"]
            panel(values,labels,dest/f"{raw:03d}_coverage.png")
            panel(values,labels,dest/f"{raw:03d}_r0_fixed720.png",crop=FIXED720,factor=5)
            if raw==137:
                panel(values,labels,dest/"137_v44_silver1600.png",crop=SILVER1600,factor=4)
                annotated=image(arrays["rgb"]).resize((960,640))
                from PIL import ImageDraw
                draw=ImageDraw.Draw(annotated)
                for box,colour in [(FIXED720,"#ff8c00"),(SILVER1600,"#00ffff")]:
                    x0,y0,x1,y1=box;draw.rectangle((x0*2,y0*2,x1*2-1,y1*2-1),outline=colour,width=3)
                draw.rectangle((0,0,900,42),fill="#eef3f6")
                draw.text((5,5),"Orange R0 fixed720 [230,204,270,222); cyan V44 silver1600 [210,213,290,233)",fill="#102a39")
                draw.text((5,23),"Different fixed denominators. Neither rectangle is independent hidden delete ground truth.",fill="#102a39")
                annotated.save(dest/"137_distinct_ROIs.png")
            assert rt.tensor_versions()==initial_versions and rt.colour_versions()==initial_colours
            rt.assert_unchanged(full=True)
            write(dest/f"{raw:03d}_metrics.json",row)
            print(json.dumps(dict(phase="gpu",raw=raw,forwards=len(ledger),fixed720=row["statistics"]["r0_fixed720"])),flush=True)
            del data, road, solo, joint, arrays
            rt._data.clear(); torch.cuda.empty_cache()
        runtime=rt.receipt()
        assert len(ledger)==8 and rt.render_calls==4
        receipt=dict(status="PASS_V44_R0_INITIAL8_READ_ONLY", raws=list(PILOT), fresh_forwards=len(ledger),
            joint_forwards=rt.render_calls, isolated_road_forwards=4, training=False, no_grad=True,
            neural_updates=0, geometry_updates=0, colour_fit_updates=0, calibration_solves=0,
            new_road_rows=0, state="delete", baseline_runtime_source=str(args.runtime_source),
            runtime_receipt=runtime, tracked_sha256_before=before,
            tracked_sha256_after={p:sha(p) for p in before}, rows=rows,
            peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            torch_allocator_memory_fraction=.30,
            torch_allocator_budget_bytes=int(torch.cuda.get_device_properties(0).total_memory*.30),
            concurrency_policy=dict(allowed_other_gpu_pids=[13030],
                known_task="scene008_full_dynamic_edit_20261107_r1 formal5k", minimum_preflight_free_MiB=20*1024),
            seconds=time.time()-started,
            preflight_gpu=snapshot, final_gpu=gpu_snapshot(), shutdown_performed=False,
            unrelated_processes_modified=False, interpretation="Isolated coverage versus joint contribution; parameter freeze does not imply composited appearance freeze; no hidden truth claim")
        assert receipt["tracked_sha256_before"]==receipt["tracked_sha256_after"]
        write(dest/"receipt.json",receipt)


def package(args):
    files={str(p.relative_to(args.output)):dict(bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(args.output.rglob("*")) if p.is_file() and p.name not in ("delivery.tar.gz","delivery_manifest.json")}
    write(args.output/"delivery_manifest.json",dict(files=files,source_sha256=sha(Path(__file__).resolve()),power_action="none"))
    with tarfile.open(args.output/"delivery.tar.gz","w:gz") as archive:
        for name in [*files,"delivery_manifest.json"]: archive.add(args.output/name,arcname=name)
    print(json.dumps(dict(archive=str(args.output/"delivery.tar.gz"),sha256=sha(args.output/"delivery.tar.gz"),files=len(files))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("cpu","gpu","package"))
    parser.add_argument("--contract",type=Path,default=V44/"contract/input_contract.json")
    parser.add_argument("--runtime-source",type=Path,default=V44/"source/v44_runtime.py")
    parser.add_argument("--saved-results",type=Path,default=V44/"results/full49_r1/v44")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    {"cpu":cpu,"gpu":gpu,"package":package}[args.mode](args)


if __name__=="__main__": main()
