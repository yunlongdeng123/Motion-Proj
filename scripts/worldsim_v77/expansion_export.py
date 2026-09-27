from pathlib import Path
import sys,shutil,tarfile
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');B=T/'review_bundle';B.mkdir(exist_ok=True);M=B/'media';M.mkdir(exist_ok=True);E=B/'evidence';E.mkdir(exist_ok=True)
for s in read(T/'r1/registration.json')['new_scenes']:
    p=T/'r1'/s['name']/f'cam{s["primary_camera"]}'/'review';assert (p/'summary.json').exists();shutil.copytree(p,M/s['name'],dirs_exist_ok=True)
shutil.copytree(T/'r4/review',M/'data',dirs_exist_ok=True)
for a,b in [('r3/footprints.jpg','footprints.jpg'),('r6/candidates.jpg','move_candidates.jpg'),('r5/view_candidates.jpg','source12_views.jpg'),('r4/candidate_review.jpg','candidate_review.jpg')]:shutil.copy2(T/a,M/b)
for rid,names in {'r1':['registration.json','identity_metadata.json','mask_state.json','mask_admission.json','drive_state.json','review_index.json'],'r2':['registration.json','state.json'],'r3':['proposal_registration.json','selected_command.json','review/summary.json','actor_layers/render_summary.json'],'r4':['registration.json','summary.json','manifest.json','data_audit.json'],'r5':['source_selection.json','view_candidates.json','registration.json','mask_state.json','shape_state.json','paint_state.json'],'r6':['registration_results.json']}.items():
    for name in names:
        p=T/rid/name
        if p.exists():dest=E/rid/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
if (T/'r1/guard/summary.json').exists():shutil.copy2(T/'r1/guard/summary.json',E/'r1/guard_summary.json')
with tarfile.open(T/'review_bundle.tar','w') as tar:
    for p in B.rglob('*'):
        if p.is_file():tar.add(p,arcname=p.relative_to(B))
print('EXPORTED',sum(p.stat().st_size for p in B.rglob('*') if p.is_file()))
