"""补齐原运行时通过系统路径继承的依赖；只添加新环境缺少的包。"""
import os, shutil, json, re, subprocess, sys
from pathlib import Path

E=Path('/root/autodl-tmp/envs/splatad-impact')
sources=[Path(sys.argv[1])] if len(sys.argv)>1 else [Path('/root/autodl-tmp/envs/'+n+'/lib/python3.10/site-packages') for n in ['nksr-v74','motionproj']]
dest=E/'lib/python3.10/site-packages'
def norm(n): return re.sub(r'[-_.]+','-',n).lower()
def link_or_copy(a,b):
    try: os.link(Path(a).resolve(),b)
    except OSError: shutil.copy2(a,b)
for source in sources:
    installed={norm(p.name.split('-')[0]) for p in dest.glob('*.dist-info')}
    copied=[]
    for p in source.iterdir():
        name=p.name
        if name.startswith(('pip','gsplat','__editable__')) or name.endswith(('.pth','.egg-link')) or (dest/name).exists(): continue
        if name.endswith('.dist-info') and norm(name.split('-')[0]) in installed: continue
        if p.is_dir(): shutil.copytree(p,dest/name,copy_function=link_or_copy,symlinks=False)
        else: link_or_copy(p,dest/name)
        copied.append(name)
    (E/('runtime_inherited_'+source.parents[2].name+'.json')).write_text(json.dumps({'source':str(source),'copied':copied},indent=2))
    print('COPIED_MISSING',source.parents[2].name,len(copied),flush=True)
subprocess.run([str(E/'bin/python'),'/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1/scripts/probe_splatad_dependencies.py'],check=True)
