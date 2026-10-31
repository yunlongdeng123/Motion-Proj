import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import re
import subprocess
from collections import Counter
import urllib.parse

ROOT=Path('/root/autodl-tmp');OUT=ROOT/'codex_tmp/horizon_waymo_inventory_20261031_r1'
pattern=re.compile(r'(?:segment-)?(\d+_\d+_\d+_\d+_\d+)(?:_with_camera_labels)?')
roots=[ROOT/'data',ROOT/'datasets',ROOT/'drivingeditor_repro',ROOT/'streetcrafter_work',ROOT/'street_crafter',
    ROOT/'runs',ROOT/'models',ROOT/'dggt',ROOT/'retained']
args=['rg','--files']+[str(p) for p in roots if p.exists()]+['-g','*.tfrecord','-g','*.tfrecords','-g','*.parquet']
result=subprocess.run(args,text=True,capture_output=True)
files=[]
for name in result.stdout.splitlines():
    path=Path(name);match=pattern.search(name)
    files.append(dict(path=name,context_id=match.group(1) if match else None,bytes=path.stat().st_size,
        suffix=path.suffix,raw_payload_sha256_recomputed=False))
(OUT/'all_raw_inventory.json').write_text(json.dumps(dict(schema='waymo_existing_raw_stat_inventory_v2',
    files=files,search=dict(command=args,returncode=result.returncode,stderr=result.stderr),
    note='Supersedes narrow initial raw_file_inventory search; includes campaign raw_scene0029. File existence/size is not a new full SHA verification.'),ensure_ascii=False,indent=2)+'\n')

def run(cmd):
    r=subprocess.run(cmd,text=True,capture_output=True)
    return r.stdout.strip() if r.returncode==0 else None

provenance=[]
for name,path in [('DriveStudio',ROOT/'third_party/drivestudio'),('DGGT',ROOT/'dggt'),
                   ('StreetCrafter',ROOT/'street_crafter')]:
    remote=run(['git','-C',str(path),'remote','get-url','origin'])
    if remote and remote.startswith('http'):
        u=urllib.parse.urlsplit(remote);remote=urllib.parse.urlunsplit((u.scheme,u.netloc.split('@')[-1],u.path,'',''))
    provenance.append(dict(project=name,checkout=str(path),git_head=run(['git','-C',str(path),'rev-parse','HEAD']),
        origin=remote,list_worktree_status=run(['git','-C',str(path),'status','--porcelain','--','data/waymo_train_list.txt','data/waymo_val_list.txt','data/waymo_example_scenes.txt']),
        omniRe_relation='DriveStudio official implementation correspondence requires upstream evidence',
        list_role_not_inferred_from_filename=True))
(OUT/'local_project_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')

alias=ROOT/'drivingeditor_repro/campaign/followup_scene_prep_20261013/inputs/scene0029_069_117_red_cpu_v1'
alias_rows=[]
if (alias/'preparation_manifest.json').is_file():
    data=json.loads((alias/'preparation_manifest.json').read_text())
    filelist=run(['rg','--files',str(alias)]).splitlines()
    counts=Counter()
    for p in filelist:
        parts=set(Path(p).parts)
        for field,folders in [('rgb',{'image','images','rgb'}),('camera',{'calib','intrinsics','extrinsics'}),
            ('pose',{'pose','poses','ego_pose'}),('lidar',{'velodyne','lidar'}),('annotations',{'annotations','tracks'})]:
            if parts.intersection(folders):counts[field]+=1
    alias_rows.append(dict(context_id='10526338824408452410_5714_660_5734_660',
        source_manifest=str(alias/'preparation_manifest.json'),processed_root=str(alias),
        observed_counts=dict(counts),source_context_reference_found_in_manifest='10526338824408452410_5714_660_5734_660' in json.dumps(data),
        identity_role='DriveStudio train029 alternative candidate, not confirmed HorizonForge029'))
(OUT/'processed_alias_coverage.json').write_text(json.dumps(alias_rows,ensure_ascii=False,indent=2)+'\n')

canonical=json.loads((OUT/'canonical_list_mapping.json').read_text())['sources']
val=next(s for s in canonical if s['path']=='/root/autodl-tmp/waymo_val_list.txt')
coverage=json.loads((OUT/'context_coverage.json').read_text())['coverage']
bycontext={r['context_id']:r for r in coverage};matrix=[]
for row in val['candidate_rows']:
    context=row['context_id'];present=[f for f in files if f['context_id']==context and f['suffix'] in ('.tfrecord','.tfrecords')]
    cov=bycontext.get(context,{})
    matrix.append(dict(scene_number=row['scene_number'],context_id=context,
        numbering='zero_based_existing_root_waymo_val_list_file_order',full_raw_files_present=present,
        complete_raw_filename_present=bool(present),
        prepared_original_rgb_count=cov.get('category_counts',{}).get('original_processed_rgb',0),
        prepared_camera_file_count=cov.get('category_counts',{}).get('camera_calibration',0),
        prepared_pose_file_count=cov.get('category_counts',{}).get('pose',0),
        prepared_lidar_file_count=cov.get('category_counts',{}).get('lidar',0),
        prepared_annotation_file_count=cov.get('category_counts',{}).get('annotation_track',0),
        GS_checkpoint_files_located=cov.get('checkpoint_files',[]),
        newly_extracted_preview_does_not_count_as_complete_dataset=True,
        HF_correspondence='root_matcher_evidence_separate_not_inferred_from_inventory'))
(OUT/'sorted_val_source_coverage.json').write_text(json.dumps(dict(schema='sorted_val11_source_component_inventory_v1',
    rows=matrix,scope='component paths in inspected roots; unknown numeric aliases not falsely assigned',
    first_record_range_receipts='range_sorted_val/network_receipt.json'),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(all_raw_files=len(files),tfrecord_files=sum(f['suffix'] in ('.tfrecord','.tfrecords') for f in files),
    tfrecord_unique_contexts=len({f['context_id'] for f in files if f['suffix'] in ('.tfrecord','.tfrecords') and f['context_id']}),
    target_sorted_val_full_raw=sum(bool(r['full_raw_files_present']) for r in matrix),alias_rows=alias_rows),ensure_ascii=False))
