"""取回道路资产和最终完整状态；相同来源文件用 tar 硬链接去重，不改变源文件。"""
from pathlib import Path
import argparse,hashlib,json,os,tarfile

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main(run,output):
    run,output=Path(run),Path(output)
    if output.exists():raise FileExistsError('preserve previous bundle')
    result=json.loads((run/'results.json').read_text());assert result['completed']
    output.mkdir();stage=output/'files';stage.mkdir();sources=[];dedup={}
    def collect(source,rel):
        source=Path(source);target=stage/rel
        if not source.is_file():raise FileNotFoundError(source)
        h=sha(source);target.parent.mkdir(parents=True,exist_ok=True)
        os.link(dedup.get(h,source),target);dedup.setdefault(h,target)
        sources.append(dict(source=str(source),archive_path=rel,SHA256=h,bytes=source.stat().st_size))
    for name in ['results.json','source_freeze.json','physical_ledger.json','run_state.json','common_initial_ci.npz','common_initial_field.pt','common_loss_snapshot.json']:
        collect(run/name,name)
    for arm,row in result['arms'].items():
        base=run/arm
        for p in sorted((base/'asset').rglob('*')):
            if p.is_file():collect(p,arm+'/asset/'+p.relative_to(base/'asset').as_posix())
        collect(row['authority'],arm+'/'+Path(row['authority']).name)
        collect(base/'endpoint.json',arm+'/endpoint.json')
        steps=row['extra'].get('logical_steps',row['extra'].get('accepted_steps'))
        collect(base/f'checkpoint_{steps:04d}.pt',arm+f'/checkpoint_{steps:04d}.pt')
    manifest=dict(schema='native_capacity_asset_takeback.v1',files=sources,original_bytes=sum(x['bytes']for x in sources),
        unique_bytes=sum(p.stat().st_size for p in dedup.values()),identical_file_deduplication='tar hardlinks; original source bytes unchanged',
        scene_dependencies='Road assets and final optimizer states included. Original static/dynamic/environment checkpoints and input data remain referenced by source scene manifests.',
        production_scene_modified=False)
    (stage/'asset_bundle_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    archive=output/'road_assets_and_final_state.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1,dereference=False)as tf:tf.add(stage,arcname='road_assets_and_final_state',recursive=True)
    receipt=dict(completed=True,archive=str(archive),SHA256=sha(archive),bytes=archive.stat().st_size,files=len(sources),
        original_bytes=manifest['original_bytes'],unique_bytes=manifest['unique_bytes'],GPU_operations=0,optimizer_updates=0)
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.run,a.output)
