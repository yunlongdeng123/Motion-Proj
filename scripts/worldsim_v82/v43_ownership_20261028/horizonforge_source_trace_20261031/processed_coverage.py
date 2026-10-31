import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import re
import subprocess
import time
from collections import Counter

ROOT=Path('/root/autodl-tmp')
OUT=ROOT/'codex_tmp/horizon_waymo_inventory_20261031_r1'
initial=json.loads((OUT/'list_sources_and_mapping.json').read_text())
raw=json.loads((OUT/'raw_file_inventory.json').read_text())['files']
targets=[4,20,29,35,65,67,79,81,86,113,140]
regex=re.compile(r'(?:segment-)?(\d+_\d+_\d+_\d+_\d+)(?:_with_camera_labels)?')
sources=[];contexts=set()
for source in initial['sources']:
    lines=Path(source['path']).read_text().splitlines()
    entries=[];explicit={}
    for n,line in enumerate(lines,1):
        match=regex.search(line)
        if not match:continue
        row=dict(source_line=n,raw=line,context_id=match.group(1))
        fields=line.split(',')
        if len(fields)>=2 and fields[0].strip().isdigit():
            row['explicit_scene_id']=int(fields[0]);explicit[int(fields[0])]=row
        entries.append(row)
    rows=[]
    for i in targets:
        row=explicit.get(i) if explicit else entries[i] if i<len(entries) else None
        rows.append(dict(scene_number=f'{i:03}',index=i,status='candidate' if row else 'index_outside_local_list',
            **(row or {})))
        if row:contexts.add(row['context_id'])
    sources.append(dict(path=source['path'],sha256=source['sha256'],context_entry_count=len(entries),
        index_semantics='explicit_scene_id_column' if explicit else 'zero_based_file_line_no_resort',
        candidate_rows=rows,explicit_ids_missing_in_range=[i for i in range(max(explicit)+1) if i not in explicit] if explicit else None,
        horizonforge_identity='unverified_list_conditioned_candidate_only'))
(OUT/'canonical_list_mapping.json').write_text(json.dumps(dict(target_numbers=[f'{i:03}' for i in targets],sources=sources,
    warning='Same 3-digit scene number has different context IDs across lists; no HorizonForge list inheritance demonstrated.'),ensure_ascii=False,indent=2)+'\n')

started=time.monotonic()
roots=[ROOT/'data',ROOT/'datasets',ROOT/'drivingeditor_repro',ROOT/'streetcrafter_work',ROOT/'street_crafter/output',
       ROOT/'runs',ROOT/'models',ROOT/'dggt',ROOT/'retained']
roots=[p for p in roots if p.exists()]
args=['rg','--files']+[str(p) for p in roots]
process=subprocess.Popen(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
matches={context:[] for context in contexts};metadata=[];count=0
for line in process.stdout:
    name=line.rstrip('\n');count+=1
    # 只枚举路径；未扫描RGB、点云、raw或checkpoint的内容。
    for context in contexts:
        if context in name:matches[context].append(name)
    filename=Path(name).name
    if filename in ('sensor_preparation.json','preparation_manifest.json','preview_metadata.json','scene_metadata.json',
                    'meta_info_train.json','meta_info_val.json','manifest.json'):
        metadata.append(name)
stderr=process.stderr.read();returncode=process.wait()

def category(name):
    path=Path(name);parts=set(path.parts);suffix=path.suffix.lower()
    if suffix in ('.pth','.pt','.ckpt','.safetensors'):return 'checkpoint_file'
    if suffix in ('.tfrecord','.tfrecords','.parquet'):return 'raw_container'
    source=any(s in name for s in ('motion_scene_inputs_v1/','motion_scene_inputs_full','/data/waymo_','/official_scenes/','/processed/','/training_set_processed/','/validation_set_processed/'))
    if suffix in ('.jpg','.jpeg','.png') and source and parts.intersection({'image','images','rgb'}):return 'original_processed_rgb'
    if parts.intersection({'calib','calibration','intrinsics','extrinsics'}) and suffix in ('.txt','.json','.npy'):return 'camera_calibration'
    if parts.intersection({'pose','ego_pose','poses','ego_poses'}) and suffix in ('.txt','.json','.npy'):return 'pose'
    if parts.intersection({'lidar','velodyne','lidars'}) and suffix in ('.bin','.npy','.npz','.ply'):return 'lidar'
    if parts.intersection({'annotations','instances','track','tracks'}) and suffix in ('.txt','.json','.pkl','.npz'):return 'annotation_track'
    if suffix=='.ply':return 'pointcloud_or_export_not_necessarily_GS'
    if suffix in ('.json','.yaml','.yml','.txt','.csv','.log'):return 'metadata_or_log'
    if suffix in ('.jpg','.png','.jpeg'):return 'other_image_not_accepted_as_original_RGB'
    return 'other'

metadata_hits=[]
for name in metadata:
    path=Path(name)
    if not path.is_file() or path.stat().st_size>2*1024*1024:continue
    try:text=path.read_text()
    except (OSError,UnicodeError):continue
    hit=[c for c in contexts if c in text]
    if hit:metadata_hits.append(dict(path=name,matching_contexts=hit,bytes=path.stat().st_size))

coverage=[]
for context in sorted(contexts):
    paths=matches[context];bins={}
    for name in paths:bins.setdefault(category(name),[]).append(name)
    containers=[r for r in raw if r['context_id']==context]
    checkpoints=[]
    for name in bins.get('checkpoint_file',[]):
        path=Path(name)
        checkpoints.append(dict(path=name,bytes=path.stat().st_size,loaded=False))
    coverage.append(dict(context_id=context,raw_files=containers,filename_match_count=len(paths),
        category_counts={k:len(v) for k,v in bins.items()},
        representative_paths={k:v[:12] for k,v in bins.items()},checkpoint_files=checkpoints,
        metadata_context_matches=[row['path'] for row in metadata_hits if context in row['matching_contexts']],
        original_rgb_candidates=bins.get('original_processed_rgb',[])[:12],
        source_identity='candidate_list_mapping_not_HorizonForge_confirmation'))
(OUT/'context_coverage.json').write_text(json.dumps(dict(schema='local_waymo_candidate_context_coverage_v1',
    coverage=coverage,search=dict(command=args,file_paths_enumerated=count,returncode=returncode,stderr=stderr,
    wall_seconds=time.monotonic()-started),limitations=['Counts derived from file paths plus bounded small metadata reads; no model/raw/RGB payload scanning.',
    'Existing processed subsets and model checkpoints do not prove HorizonForge source matching.',
    'Unidentified numeric directories cannot be assigned to a context without metadata.']),ensure_ascii=False,indent=2)+'\n')
(OUT/'matching_file_paths.json').write_text(json.dumps(matches,ensure_ascii=False,indent=2)+'\n')
(OUT/'metadata_context_hits.json').write_text(json.dumps(metadata_hits,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(context_candidates=len(contexts),paths_enumerated=count,wall_seconds=time.monotonic()-started,
    positive=[dict(context_id=r['context_id'],category_counts=r['category_counts'],raw_files=len(r['raw_files'])) for r in coverage if r['filename_match_count'] or r['metadata_context_matches']]),ensure_ascii=False))
