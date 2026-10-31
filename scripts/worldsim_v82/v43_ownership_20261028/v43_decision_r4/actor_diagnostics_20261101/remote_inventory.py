import json,pathlib
base=pathlib.Path('/root/autodl-tmp/codex_tmp/v43_actor_diagnostics_20261101')
cfg=json.loads((base/'scenes.json').read_text())
roots=[pathlib.Path('/root/autodl-tmp/datasets/waymo_v1_4_3/individual_files')]
files=[p for r in roots for p in r.glob('*/*.tfrecord')]
rows=[]
for s in cfg['scenes']:
    matches=[p for p in files if s['context_id'] in p.name]
    if len(matches)!=1: raise RuntimeError((s['context_id'],matches))
    p=matches[0];st=p.stat()
    rows.append(dict(context_id=s['context_id'],scene_key=s['scene_key'],path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
(base/'raw_inventory.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(dict(scenes=len(rows),bytes=sum(x['bytes'] for x in rows))))
