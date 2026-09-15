"""隔离复用兼容CUDA运行时；硬链接复制包目录，避免修改旧环境。"""
import os,subprocess,shutil,json
from pathlib import Path
E=Path('/root/autodl-tmp/envs/splatad-impact');source=Path('/root/autodl-tmp/envs/hugsim-impact/lib/python3.10/site-packages')
if E.exists():raise RuntimeError('Environment exists; inspect before modifying')
subprocess.run(['/root/autodl-tmp/envs/hugsim-impact/bin/python','-m','venv',str(E)],check=True)
dest=E/'lib/python3.10/site-packages';copied=[]
def link_or_copy(a,b):
 try:os.link(a,b,follow_symlinks=True)
 except OSError:shutil.copy2(a,b)
for p in source.iterdir():
 name=p.name
 if name.startswith(('pip','setuptools','_distutils_hack','gsplat','__editable__')) or name.endswith(('.pth','.egg-link')) or (dest/name).exists():continue
 target=dest/name
 if p.is_dir():shutil.copytree(p,target,copy_function=link_or_copy,symlinks=False)
 else:link_or_copy(str(p.resolve()),str(target))
 copied.append(name)
(E/'runtime_seed.json').write_text(json.dumps({'source':str(source),'copied':copied,'mode':'File hardlink copy of package trees; separate venv package namespace; omit all editable links and gsplat'},indent=2))
print('RUNTIME_READY',len(copied),flush=True)
