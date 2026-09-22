import email,json,zipfile
from pathlib import Path
from prune_storage import ROOT,AUDIT,describe

assets=ROOT/'runs/worldsim_simimpact/WS-SIM-SPARSE-NATIVE-01/20260915-r1/assets'
site=ROOT/'envs/sparsedrive-impact/lib/python3.9/site-packages'
rows=[]
for p in assets.glob('*.whl'):
 if p.name.startswith('flash_attn'):continue
 with zipfile.ZipFile(p) as z:
  name=next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
  wheel=email.message_from_bytes(z.read(name));installed=site/name
  assert installed.is_file()
  current=email.message_from_bytes(installed.read_bytes())
  assert wheel['Name']==current['Name'] and wheel['Version']==current['Version']
 rows.append(describe(p,'installed_package_download',{'installed_metadata':str(installed),'package':wheel['Name'],'version':wheel['Version'],'wheel':p.name,'operation':'在 Linux CPython 3.9 环境通过 pip download --no-deps 包名==版本 重新下载；环境本体保留'}))
(AUDIT/'wheel-plan.json').write_text(json.dumps({'files':rows},ensure_ascii=False,indent=2))
print(json.dumps({'files':len(rows),'GiB':sum(r['allocated'] for r in rows)/2**30}),flush=True)
