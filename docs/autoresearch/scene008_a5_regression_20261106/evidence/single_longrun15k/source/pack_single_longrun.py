from pathlib import Path
import argparse,hashlib,io,json,re,tarfile,time
ROOT=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=[15000,20000,30000],required=True);opt=p.parse_args()
step=opt.step;tag=f'{step//1000}k';model=ROOT/f'continuation{tag}';evaluation=ROOT/f'a5_evaluation_{tag}'
def read(p):return json.loads(p.read_text())
summary=read(evaluation/'summary.json');assert summary['status']=='PASS_FRESH_NATIVE49_A5_EVALUATION'
assert summary['source_integrity']=='PASS_UNCHANGED_BYTES_MTIME_SHA256' and summary['fixed_denominator']['pixels']==121128
assert summary['replay']['complete_all49_keep_delete_replay'] and summary['replay']['max_uint8_difference']<=1
assert read(model/'restore_audit.json')['differences']==[]
alias=f'single_longrun{tag}';paths=[]
for path in list(evaluation.glob('*.json'))+list(evaluation.glob('*.csv'))+list((evaluation/'a5_float').glob('*.npz'))+list((evaluation/'anchors').glob('*.png')):
    paths.append((path,alias+'/'+str(path.relative_to(evaluation))))
for raw in [89,97,105,109]:
    for path in (evaluation/f'frames/{raw:03d}').glob('*.png'):paths.append((path,alias+'/'+str(path.relative_to(evaluation))))
for path in list(model.glob('*.json'))+list(model.glob('*.jsonl')):paths.append((path,alias+'/run_receipts/'+path.name))
for name in ['longrun_contract.json','launch_receipt.json','queue_status.json',f'node{step}_ready.json','resume_entry_schedule_audit.json','continuation_accounting.json','source_integrity_after.json']:
    path=ROOT/name
    if path.exists():paths.append((path,alias+'/run_receipts/campaign/'+name))
for path in (ROOT/'source').glob('*.py'):paths.append((path,alias+'/source/'+path.name))
event=model/f'checkpoint_events/iteration_{step:06d}.json';paths.append((event,alias+'/native_checkpoint_event.json'))
log=(ROOT/f'train{tag}.log').read_text(errors='replace')
pattern=rf'\[ITER {step}\] Evaluating train: L1 ([^ ]+) PSNR ([^ ]+) SSIM ([^ ]+) LPIPS ([^\r\n]+)'
rows=re.findall(pattern,log);assert len(rows)==1
l1,psnr,ssim,lpips=map(float,rows[0]);native=read(model/f'eval/train_{step}_render/metrics.json')
assert all(abs(native[k]-value)<1e-10 for k,value in [('psnr',psnr),('ssim',ssim),('lpips',lpips)])
metrics=dict(frame_count=49,l1=l1,psnr=psnr,ssim=ssim,lpips=lpips,extra_render_forward_passes=0,source='Existing native all49 evaluation')
archive=ROOT/f'{alias}_light.tar.gz'
with tarfile.open(archive,'w:gz') as packed:
    for path,name in sorted(paths,key=lambda x:x[1]):packed.add(path,arcname=name,recursive=False)
    for name,text in [('global49.json',json.dumps(metrics,indent=2)+'\n'),('native_global49_log_line.txt',re.search(pattern,log).group(0)+'\n')]:
        payload=text.encode();item=tarfile.TarInfo(alias+'/'+name);item.size=len(payload);item.mtime=int(time.time());packed.addfile(item,io.BytesIO(payload))
    manifest={name:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path,name in paths}
    payload=(json.dumps(dict(status='PACKED_ACCEPTED_FRESH49',iteration=step,files=manifest),indent=2)+'\n').encode()
    item=tarfile.TarInfo(alias+'/retrieval_source_manifest.json');item.size=len(payload);packed.addfile(item,io.BytesIO(payload))
receipt=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),step=step,files=len(paths)+3)
(ROOT/f'{alias}_archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
