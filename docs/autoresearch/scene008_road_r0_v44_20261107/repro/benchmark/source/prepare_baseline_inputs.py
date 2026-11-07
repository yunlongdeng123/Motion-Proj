"""Add legal FRONT actor guards to a fixed51 subset, with no target RGB read.

Only the trusted input builder reads raw annotation metadata. Each estimator
later sees one copied track allowlist, never the original TFRecord or evaluator.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from PIL import Image
sys.dont_write_bytecode=True
import road_patch_benchmark as b


def main():
    p=argparse.ArgumentParser();p.add_argument("--benchmark",type=Path,required=True);p.add_argument("--settings",type=Path,required=True)
    p.add_argument("--frame-budget",choices=("fixed51","full199"),default="fixed51")
    args=p.parse_args();root=args.benchmark;t=time.time()
    raws=list(range(199)) if args.frame_budget=="full199" else list(range(0,199,4))
    if 198 not in raws:raws.append(198)
    inputs_root=root/"baseline_inputs"/("full199" if args.frame_budget=="full199" else "")
    reader=b.raw_reader();pb=reader.protobuf();crc=reader.crc_function();extraction=b.read(b.FULL/"extraction.json")
    guards={};annotation_rows=[]
    with Path(extraction["raw_path"]).open("rb") as stream:
        for raw in raws:
            payload=b.record_at(stream,extraction["frames"][raw],crc);frame=pb.Frame.FromString(payload)
            cal=next(c for c in frame.context.camera_calibrations if c.name==1)
            guard=np.zeros((int(cal.height),int(cal.width)),np.uint8);n=0
            for group in frame.camera_labels:
                if group.name!=1:
                    continue
                for label in group.labels:
                    box=label.box
                    x0=max(0,int(np.floor(box.center_x-box.length/2))-16);x1=min(cal.width,int(np.ceil(box.center_x+box.length/2))+17)
                    y0=max(0,int(np.floor(box.center_y-box.width/2))-16);y1=min(cal.height,int(np.ceil(box.center_y+box.width/2))+17)
                    if x1>x0 and y1>y0:
                        guard[y0:y1,x0:x1]=255;n+=1
            # Conservative occupancy: any native guard within a prepared pixel.
            guards[raw]=np.asarray(Image.fromarray(guard).resize((480,320),Image.Resampling.BOX))>0
            annotation_rows.append(dict(raw=raw,FRONT_label_rectangles=n,guard_pixels=int(guards[raw].sum()),payload_CRC_passed=True))
    exports=[]
    for track in ("appearance_blind","geometry_RGB_blind"):
        source=root/"isolated_inputs"/track;dest=inputs_root/track
        if dest.exists():
            raise FileExistsError("Preserve earlier baseline inputs")
        dest.mkdir(parents=True)
        original=b.read(source/"input_manifest.json")
        for sub in ("rgb","geometry","camera","guard"):
            (dest/sub).mkdir()
        for raw in raws:
            for sub,suffix in (("geometry",".npz"),("camera",".json"),("rgb",".png"),("rgb","_valid.npz")):
                name=f"{raw:03d}"+suffix;path=source/sub/name;rel=str(path.relative_to(source))
                if b.sha(path)!=original["input_files"][rel]["sha256"]:
                    raise ValueError("Isolated source changed")
                shutil.copyfile(path,dest/sub/name)
            np.savez_compressed(dest/"guard"/f"{raw:03d}.npz",vehicle_guard=guards[raw])
        patch_manifest=b.read(source/"patch_manifest.json");patch_manifest["source_blinded_raw_ids"]=patch_manifest["raw_ids"]
        patch_manifest["raw_ids"]=raws
        b.write(dest/"patch_manifest.json",patch_manifest)
        shutil.copyfile(args.settings,dest/"baseline_settings.json")
        files={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=b.sha(p)) for p in sorted(dest.rglob("*")) if p.is_file()}
        b.write(dest/"input_manifest.json",dict(schema="scene008.R1.baseline_isolated_inputs.v1",track=track,raw_ids=raws,
            declared_cameras=[1],input_files=files,source_isolated_manifest_sha256=b.sha(source/"input_manifest.json"),
            settings_sha256=b.sha(args.settings),only_this_directory_may_be_mounted=True,
            actor_guard_source="Original FRONT 2D annotation rectangles +16nativepx; all annotated actors, no target road marking or height labels",
            old_scene_and_other_track_and_sealed_reference_access_allowed=False,OS_strong_isolation_certified=False))
        exports.append(dict(track=track,directory=str(dest),input_manifest_sha256=b.sha(dest/"input_manifest.json"),input_files=len(files)))
    receipt_name="baseline_input_preparation_full199.json" if args.frame_budget=="full199" else "baseline_input_preparation.json"
    b.write(root/"audit"/receipt_name,dict(status="PASS_LEGAL_"+args.frame_budget.upper()+"_FRONT_BASELINE_INPUTS",raw_ids=raws,
        target_RGB_or_sealed_references_read=False,only_raw_annotation_metadata_added=True,annotation_rows=annotation_rows,
        tracks=exports,settings_sha256=b.sha(args.settings),source_FRONT199_blinding_unchanged=True,
        target_geometry_not_used_to_define_new_mask=True,seconds=time.time()-t,GPU_used=False,neural_updates=0))
    print({"status":"PASS_LEGAL_"+args.frame_budget.upper()+"_FRONT_BASELINE_INPUTS","frames":len(raws),"seconds":time.time()-t})


if __name__=="__main__":
    main()
