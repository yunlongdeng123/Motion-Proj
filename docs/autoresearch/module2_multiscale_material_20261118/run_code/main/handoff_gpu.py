from pathlib import Path
import argparse,json,subprocess
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
a=argparse.ArgumentParser();a.add_argument('--to',choices=['stage_qa','static_handoff','none'],required=True);args=a.parse_args()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
if apps:raise RuntimeError('上一个GPU进程尚未退出，暂不移交：'+apps)
if args.to=='stage_qa':
    r=json.loads((P/'formal_r1/results.json').read_text());assert r['formal_physical_updates']==1400
scope={'stage_qa':'matched independentRGB actual4-update lifecycle then fresh800 formal, no extra epochs',
 'static_handoff':'registered zero-update TRELLIS2/A0mid two native columns only',
 'none':'all own GPU work ended; instance remains ON'}[args.to]
target=P/'gpu_lease.json';temporary=target.with_suffix('.json.next')
temporary.write_text(json.dumps(dict(gpu_authorized=True,active=args.to!='none',owner=args.to,scope=scope,keep_GPU_on=True),indent=2)+'\n')
temporary.replace(target);print(json.dumps(dict(owner=args.to,active=args.to!='none',GPU_keep_on=True)))
