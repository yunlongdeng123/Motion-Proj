"""Only prune owned intermediate .pth after verified complete30K and disk pressure."""
from pathlib import Path
import gc,hashlib,json,os,shutil,time
from resource_control import stop_requested
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
NATIVE=ROOT/'native_scene_training_v1'
PREFIXES=['chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state']

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def prune(scene):
    run=scene/'formal30k';ret=run/'returned.json';receipt=scene/'checkpoint_retention.json'
    if receipt.exists():return True
    if not ret.exists():return False
    returned=json.loads(ret.read_text())
    if returned.get('reason')!='native_returned' or returned.get('iteration')!=30000:return False
    events={step:json.loads((run/'checkpoint_events'/f'iteration_{step:06d}.json').read_text()) for step in range(5000,30001,5000)}
    final=events[30000];assert final['iteration']==30000 and final['status']=='complete'
    assert set(final['checkpoints'])=={'static','dynamic','environment','training_state'}
    usage=shutil.disk_usage(ROOT);pressure=usage.used/usage.total>=.90 or usage.free<30*1024**3
    if not pressure:return False
    # Verify the retained full bundle and its complete49-frame evaluation first.
    import torch
    for prefix in PREFIXES:
        p=run/f'{prefix}30000.pth';assert p.is_file() and p.stat().st_size>0
        record=next(v for v in final['checkpoints'].values() if Path(v['path'])==p)
        assert p.stat().st_size==record['bytes']
        value=torch.load(p,map_location='cpu')
        if prefix=='training_state':
            assert value['iteration']==30000 and all(k in value for k in ['python_rng','numpy_rng','torch_rng','cuda_rng','hybrid_sampler','sgcm'])
        else:assert value[1]==30000 and isinstance(value[0],(tuple,list,dict))
        del value;gc.collect()
    assert len(final['frames'])==49
    assert all(Path(p).is_file() for f in final['frames'] for p in f['channels'].values())
    candidates=[]
    for step in range(5000,30000,5000):
        assert events[step]['iteration']==step and events[step]['status']=='complete'
        for prefix in PREFIXES:
            p=run/f'{prefix}{step}.pth'
            assert p.resolve().parent==run.resolve() and p.is_file() and not p.is_symlink()
            candidates.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
    data=dict(status='planned_after_complete30k_and_disk_pressure',scope=str(run),final_full_bundle_cpu_readable=True,final_evaluation_frames=49,usage_before=dict(total=usage.total,used=usage.used,free=usage.free),files=candidates,expected_reclaimed=sum(x['bytes'] for x in candidates),preserved_final_iteration=30000,original_checkpoint_events_preserved=True)
    receipt.write_text(json.dumps(data,indent=2))
    for row in candidates:
        if stop_requested():break
        Path(row['path']).unlink();row['deleted']=True
        receipt.write_text(json.dumps(data,indent=2))
    data.update(status='pruned_owned_intermediate_checkpoints' if all(x.get('deleted') for x in candidates) else 'partial_prune_resource_stop',actual_reclaimed=sum(x['bytes'] for x in candidates if x.get('deleted')))
    receipt.write_text(json.dumps(data,indent=2));return True

if __name__=='__main__':
    while not stop_requested():
        for scene in NATIVE.iterdir():
            if scene.is_dir():prune(scene)
        time.sleep(30)
