from pathlib import Path
import argparse,hashlib,io,json,tarfile,time
ROOT=Path(__file__).resolve().parent.parent
API=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit')
ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['smoke','formal5k','formal10k'],required=True)
ap.add_argument('--scope',choices=['pilot','all49'],default='pilot');ap.add_argument('--kind',choices=['edit','keep'],default='edit');a=ap.parse_args()
tag=a.stage+('_keep_' if a.kind=='keep' else '_')+a.scope
evaluation=ROOT/tag;model=ROOT/a.stage
summary=json.loads((evaluation/'summary.json').read_text())
expected=('PASS_NATIVE_KEEP_RECONSTRUCTION_PILOT' if a.scope=='pilot' else 'PASS_NATIVE_KEEP_RECONSTRUCTION_ALL49') if a.kind=='keep' else ('PASS_NATIVE_IDENTITY_EDIT_PILOT_ENGINEERING' if a.scope=='pilot' else 'PASS_NATIVE_IDENTITY_EDIT_ALL49_ENGINEERING')
assert summary['status']==expected and summary['source_integrity']=='PASS_UNCHANGED_BYTES_MTIME_SHA256'
assert summary['optimization_updates']==0 and summary['parameters_unchanged']
if a.kind=='edit':
    assert summary['keep_native_max_abs_difference']<=1e-6 and summary['keep_again_max_abs_difference']<=1e-6
    assert summary['cuda_joint_target_contribution_zero'] and summary['visible_owner37_witness']
else:
    assert summary['identity_partition_max_abs_error']<=1e-5 and summary['delete_actor_ids']==[]
    assert summary['full49_fixed_reference']['full_pixels']==121128 and summary['full49_fixed_reference']['upper_pixels']==25906
alias='full_dynamic_edit/'+evaluation.name;archive=ROOT/(evaluation.name+'_light.tar.gz')
paths=[]
for path in evaluation.rglob('*'):
    if path.is_file() and path.suffix in ['.json','.jsonl','.csv','.png','.npz']:
        paths.append((path,alias+'/'+str(path.relative_to(evaluation))))
for path in ROOT.glob('*.json'):paths.append((path,alias+'/run_receipts/campaign/'+path.name))
for path in list(model.glob('*.json'))+list(model.glob('*.jsonl')):paths.append((path,alias+'/run_receipts/model/'+path.name))
event=model/f"checkpoint_events/iteration_{summary['iteration']:06d}.json";paths.append((event,alias+'/native_checkpoint_event.json'))
if a.stage=='smoke':
    gate=json.loads((ROOT/'smoke_lifecycle_gate.json').read_text())
    assert gate['status']=='PASS_REAL_LONG540_FULL_DYNAMIC_SAVE_PILOT_FULL_RESTORE_NEXT_ENTRY'
    for path in list((ROOT/'smoke_resume2step').glob('*.json'))+list((ROOT/'smoke_resume2step').glob('*.jsonl')):
        paths.append((path,alias+'/run_receipts/resume2step/'+path.name))
for path in (ROOT/'source').glob('*.py'):paths.append((path,alias+'/source/'+path.name))
for path in API.glob('*.py'):paths.append((path,alias+'/api_source/'+path.name))
if a.stage=='formal10k' and a.kind=='keep':
    for path in (ROOT/'cli_keep10k').rglob('*'):
        if path.is_file() and path.suffix in ['.json','.png']:
            paths.append((path,alias+'/public_cli_original/'+str(path.relative_to(ROOT/'cli_keep10k'))))
manifest={}
with tarfile.open(archive,'w:gz') as tar:
    for path,name in sorted(set(paths),key=lambda row:row[1]):
        payload=path.read_bytes();item=tarfile.TarInfo(name);item.size=len(payload);item.mtime=int(path.stat().st_mtime)
        tar.addfile(item,io.BytesIO(payload));manifest[name]=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    payload=(json.dumps(dict(status='ACCEPTED_NATIVE_QUERY_STAGE_LIGHT_EVIDENCE',stage=a.stage,scope=a.scope,files=manifest),indent=2)+'\n').encode()
    item=tarfile.TarInfo(alias+'/retrieval_source_manifest.json');item.size=len(payload);tar.addfile(item,io.BytesIO(payload))
receipt=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),stage=a.stage,scope=a.scope,kind=a.kind,files=len(manifest)+1)
(ROOT/(evaluation.name+'_archive_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
