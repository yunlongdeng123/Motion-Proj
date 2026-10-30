"""Create private frozen-evaluation Scene so the native loader cannot rewrite the training Scene."""
import argparse,json,shutil,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--event",type=Path,required=True);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
event=json.loads(args.event.read_text());scene=Path(event["config"]["source_path"]);args.output.mkdir(parents=True,exist_ok=False);copy=args.output/"query_scene";copy.mkdir()
for source in scene.iterdir():
 if source.name in ("transform_matrix.txt","preparation_manifest.json"):shutil.copy2(source,copy/source.name)
 else:(copy/source.name).symlink_to(source,target_is_directory=source.is_dir())
original_model=Path(event["model_path"]);shutil.copy2(original_model/"cameras.json",args.output/"cameras.json")
transform=scene/"transform_matrix.txt";provenance=dict(frozen_weight_event=str(args.event.resolve()),original_query_scene=str(scene),original_training_model_path=str(original_model),original_transform_sha256=hashlib.sha256(transform.read_bytes()).hexdigest(),original_transform_mtime_ns=transform.stat().st_mtime_ns,private_scene=str(copy),requires_shared_gauge_install_before_Scene=True)
event["frozen_weight_event"]=str(args.event.resolve());event["model_path"]=str(args.output);event["config"]["model_path"]=str(args.output);event["config"]["source_path"]=str(copy)
(args.output/"probe_event.json").write_text(json.dumps(event,indent=2));(args.output/"query_provenance.json").write_text(json.dumps(provenance,indent=2));print(args.output/"probe_event.json")
