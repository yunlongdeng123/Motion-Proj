from pathlib import Path
import json, shutil

ROOT=Path('/root/autodl-tmp/models/worldsim_v82_v42_freeinpaint_20261024')
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026')
C.mkdir(exist_ok=True)
code=C/'freeinpaint_code'
if not code.exists():shutil.copytree(ROOT/'code',code,symlinks=True)
plan=json.loads((ROOT/'download_plan.json').read_text())
paths={k:ROOT/'hub'/('models--'+k.replace('/','--'))/'snapshots'/v for k,v in plan['revisions'].items()}
if not all(p.exists() for p in paths.values()):
    print('snapshot paths missing', paths);raise RuntimeError('inspect weight tree')
for file in [code/'guidance/inpainting.py']:
    text=file.read_text()
    for repo,path in paths.items():text=text.replace('"'+repo+'"',repr(str(path)))
    text=text.replace('local_files_only=False','local_files_only=True')
    file.write_text(text)
checkpoint=paths['jyproris/FreeInpaint']/'checkpoints'
cfg=code/'config/test.yaml'
text=cfg.read_text().replace('checkpoints/freeinpaint.pth',str(checkpoint/'freeinpaint.pth')).replace('checkpoint_dir: checkpoints/ppt-v1','checkpoint_dir: '+str(checkpoint/'ppt-v1'))
text=text.replace('cache_dir: cache/reference','cache_dir: '+str(C/'reference_cache'))
cfg.write_text(text)
# Supplied masks and no benchmark metrics: LangSAM/PyIQA are not executed in this harness.
req=(code/'requirements.txt').read_text().splitlines()
(C/'inference_requirements.txt').write_text('\n'.join(l for l in req if not l.startswith(('torch','gsplat @','lang-sam @','pyiqa @')))+'\ngsplat==1.5.3\n')
(C/'runtime_binding.json').write_text(json.dumps(dict(upstream_commit='9ac03f14b652c8d03d65f770bdf61da253da0fef',weights={k:str(v) for k,v in paths.items()},adaptations=['local frozen weight paths','supplied masks, no LangSAM','no benchmark IQA imports'],gsplat='1.5.3 pending exact upstream revision installation'),indent=2))
print(C)
