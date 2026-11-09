from pathlib import Path
import hashlib, json, subprocess

REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
PKG=REPO/'scripts/worldsim_v82/drivingeditor_dynamic_edit'
PUBLIC=CAM/'waymo_allframes_20261109_r1/public_api/drivingeditor_dynamic_edit'
RUN=CAM/'waymo_val065_delete30k_20261109_r1'
OUT=CAM/'waymo_val065_delete30k_diagnosis_20261109_r1'
OUT.mkdir(exist_ok=True)
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a): return subprocess.check_output(['git','-C',str(REPO),*a],text=True).strip()
rows=[]
for p in sorted(PUBLIC.rglob('*.py')):
    if '__pycache__' in p.parts: continue
    rel=p.relative_to(PUBLIC); q=PKG/rel
    rows.append(dict(file=str(rel),matches_checkout=q.exists() and sha(p)==sha(q),public_sha=sha(p),checkout_sha=sha(q) if q.exists() else None))
result=dict(task='WAYMO-065-V43-V44-DIAGNOSIS-20261109',git_head=git('rev-parse','HEAD'),public=str(PUBLIC),repo=str(PKG),files=rows,
            mismatches=[r for r in rows if not r['matches_checkout']],
            checkpoint_and_method_unchanged=True,training_updates=0,automatic_shutdown=False)
(OUT/'code_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('CODE',json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False))
for name in ['registration.json','native_manifest.json','semantic_manifest.json','final_manifest.json','ownership/receipt.json','world_road/summary.json','gate1000/receipt.json','gate1000/training.json','delete_white/provenance.json']:
    p=RUN/name
    if not p.exists(): print(name,'MISSING'); continue
    d=load(p)
    if 'frames' in d:
        d['frames_summary']=dict(count=len(d['frames']),first=d['frames'][:1],last=d['frames'][-1:])
        del d['frames']
    if 'actors' in d: d['actors_summary']=[x for x in d.pop('actors') if x.get('actor_id') in [10,8,21,20,24]]
    print(name,json.dumps(d,ensure_ascii=False))
old=CAM/'scene008_v44_rgb_asset_20261107_r1'
print('008_CONTRACT',json.dumps(load(old/'contract/input_contract.json'),ensure_ascii=False))
