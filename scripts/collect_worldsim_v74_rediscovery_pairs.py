import json,pathlib,ast,tarfile,os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch,numpy as np
torch.set_num_threads(1)
P=pathlib.Path('/root/autodl-tmp/motion_proj');B=pathlib.Path('/root/autodl-tmp/runs/worldsim_v73')
R=pathlib.Path('/root/autodl-tmp/runs/worldsim_v74_h2/WS-V74-H2-REDISCOVERY-01/20260915-cpu-r1')
T=R/'paired';T.mkdir(exist_ok=True)
tree=ast.parse((P/'scripts/forensic_worldsim_v73_paper_surfaces.py').read_text())
models=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODELS' for t in n.targets))
models['Joint-r7']='WS-V73-Q-V2-01/20260909T131000Z__open-charts-joint-first-surface-s7304-r7'
summaries={}
for f in (R/'sources/docs/autoresearch/worldsim_v73/paper_forensics').glob('*/summary.json'):
 d=json.loads(f.read_text());summaries.update(d['selected'])
data=B/'WS-V73-M2-GLOBAL-DATA-01/20260907T180000Z__window-rigid-population-r2'
entries=json.loads((data/'index.json').read_text())['cases']
records=[]
for label in ['VGGT-native','Joint-r7','First-r6']:
 old=summaries[label];e=next(v for v in entries if v['owner']==old['actor']['owner'] and v['scene']==old['actor']['scene'])
 c=torch.load(data/e['file'],map_location='cpu',weights_only=False)
 a={'source_selection_method':label,'actor':e,'case_keys':list(c),'ray_frames':[],'surfaces':{}}
 for f in c['rays']:
  desc={k:v for k,v in f.items() if isinstance(v,(int,str,float,bool,type(None)))}
  desc['keys']=list(f)
  for k in ['positive_actor','ambiguous_owner','observed_first_range_m']:
   if k in f:desc[k+'_count']=int(f[k].sum()) if f[k].dtype==torch.bool else len(f[k])
  a['ray_frames'].append(desc)
 for name,run in models.items():
  path=B/run/(e['owner']+'_surface.pt')
  if not path.exists():path=B/run/(e['scene']+'__'+e['owner']+'_surface.pt')
  s=torch.load(path,map_location='cpu',weights_only=False)
  target=T/(label+'__'+name+'.npz')
  np.savez_compressed(target,vertices=s['vertices_actor_m'].numpy(),faces=s['faces'].numpy())
  a['surfaces'][name]={'path':str(path),'export':target.name,'vertices':len(s['vertices_actor_m']),'faces':len(s['faces'])}
 records.append(a)
 print(json.dumps(a,ensure_ascii=False),flush=True)
(T/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
with tarfile.open(R/'paired.tar','w') as tar:tar.add(T,arcname='paired')
