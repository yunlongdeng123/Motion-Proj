import json
from pathlib import Path
from prune_storage import ROOT,AUDIT,describe

base=ROOT/'runs/worldsim_v4'; runs=[]
for task in base.iterdir():
 if not task.is_dir():continue
 for run in task.iterdir():
  summary=run/'summary.json'
  if not summary.is_file():continue
  r=json.loads(summary.read_text())
  if r.get('status')=='done' and r.get('mode') in {'profile100','formal'}:runs.append((run,r))
rows=[];recovery=[]
for run,r in runs:
 if r.get('mode')!='profile100':continue
 method='adgs' if 'adgs' in run.name else 'streetgs'
 finals=[(p,s) for p,s in runs if s.get('mode')=='formal' and s.get('scene')==r.get('scene') and method in p.name]
 paths=[]
 for p,s in finals:
  paths.extend(p.rglob('checkpoint_final.pth') if method=='streetgs' else p.glob('model/point_cloud/iteration_60000/*'))
 assert paths,run
 for p in paths:assert p.is_file()
 stagefiles=sorted((run/'stages').glob('*.json'))
 stages=[json.loads(p.read_text()) for p in stagefiles]
 train=[s for s in stages if s.get('command') and s.get('mode')=='profile100']
 if not train: train=[s for s in stages if s.get('command') and 'train' in s.get('stage','')]
 assert train,run
 for s in train:assert Path(s['command'][0]).is_file()
 assert (run/'resolved.yaml').is_file() and (run/'source_snapshot').is_dir()
 targets=list(run.rglob('checkpoint_final.pth')) if method=='streetgs' else list(run.glob('model/point_cloud/iteration_100/*'))
 targets=[p for p in targets if p.is_file() and p.suffix in {'.pth','.ply'}]
 record={'run':str(run),'scene':r['scene'],'method':method,'preserved_formal_checkpoints':[str(p) for p in paths],'config':str(run/'resolved.yaml'),'source_snapshot':str(run/'source_snapshot'),'commands':[s['command'] for s in train],'restore_note':'仅 100 步预检；如要复现使用保存的命令与配置，把输出改为新的 run 目录。需要 GPU/原环境，不恢复旧任务状态；正式结果 checkpoint 全部保留。'}
 recovery.append(record)
 for p in targets:rows.append(describe(p,'legacy_profile_checkpoint',{'recovery_record_run':str(run),'recipe':'profile-plan.json:recovery'}))
(AUDIT/'profile-plan.json').write_text(json.dumps({'recovery':recovery,'files':rows},ensure_ascii=False,indent=2))
print(json.dumps({'runs':len(recovery),'files':len(rows),'GiB':sum(r['allocated'] for r in rows)/2**30}),flush=True)
