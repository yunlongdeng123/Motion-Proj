from pathlib import Path
import hashlib,json,os,tarfile
os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full5_road_integration_20261107_r1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=[]
for folder in ['adapter_parity4_retry1','C4_retry1']:
 assert (R/folder/'receipt.json').is_file()
 files += [p for p in (R/folder).rglob('*.json') if p.is_file()]
 if folder=='C4_retry1':files += list((R/folder/'frames').rglob('*.png'))
files += list((R/'contract').glob('*.json'))+list((R/'source').glob('*.py'))
for p in [R/'adapter_parity4/kernel_progress.json',R/'adapter_C4_queue_status.json',R/'adapter_C4_retry1_queue_status.json']:
 if p.is_file():files.append(p)
files=list(dict.fromkeys(files));D=R/'delivery';ledger={'status':'DIAGNOSTIC_BRANCH_CANCELLED_PRESERVED_NO_TRAINING','native_public_renderer_calls':8,'native_public_split_kernels':16,'initial_adapter_failed_kernel_forwards':1,'retry_adapter_joint_forwards':8,'C_joint_forwards':8,'total_logical_joint_calls':25,'actual_total_raster_kernel_forwards':41,'backwards':0,'optimizer_updates':0,'Full5_semantic_probe_launched':False,'all49_launched':False,'material_training_launched':False,'GPU_power':'KEEP_ON'}
lp=D/'Full5_diagnostic_execution_ledger.json';assert not lp.exists();lp.write_text(json.dumps(ledger,indent=2)+'\n');files.append(lp)
entries=[{'path':str(p.relative_to(R)),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]
mf=D/'Full5_adapter_diagnostic_manifest.json';assert not mf.exists();mf.write_text(json.dumps({'files':entries,'actual_compute':ledger},indent=2)+'\n')
archive=D/'Full5_adapter_diagnostic.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname='evidence/'+str(p.relative_to(R)),recursive=False)
 t.add(mf,arcname='evidence/Full5_adapter_diagnostic_manifest.json',recursive=False)
receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files)+1};(D/'Full5_adapter_diagnostic_archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
