"""Copy the minimal fixed run inputs needed for local Blender 5.2 rendering."""
import json
import pathlib
import shutil

RUN=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
STAGE=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/blender_input')
STAGE.mkdir(parents=True,exist_ok=True)
shutil.copy2(RUN/'registration.json',STAGE/'registration.json')
reg=json.loads((RUN/'registration.json').read_text())
for spec in reg['scenes']:
    name=spec['name']
    stage=STAGE/name
    stage.mkdir(exist_ok=True)
    for filename in ['instances_snapshot.json','selection.json']:
        shutil.copy2(RUN/name/filename,stage/filename)
    actor=stage/'actor';actor.mkdir(exist_ok=True)
    for filename in ['actor_pbr.obj','actor_pbr.mtl','actor_pbr.jpg',
                     'actor_pbr_metallic.jpg','actor_pbr_roughness.jpg']:
        shutil.copy2(RUN/name/'actor'/filename,actor/filename)
    source=pathlib.Path(spec['root'])/'extrinsics'
    dest=STAGE/'source'/name/'extrinsics';dest.mkdir(parents=True,exist_ok=True)
    for row in spec['frames']:
        f=row['frame']
        for cam in range(6):
            filename=f'{f:03d}_{cam}.txt'
            shutil.copy2(source/filename,dest/filename)
print(STAGE)
