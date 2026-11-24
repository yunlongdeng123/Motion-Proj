from pathlib import Path
import sys,json,numpy as np,torch
R4=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_native30k_red_rebuild_20261125_r4')
root=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(R4/'code'))
from native30k_adapter import Native30KAdapter
from prepare_region_retirement_v5 import subset,load
rt=Native30KAdapter(R4/'config/native30k.json');source=rt.owners(110);p=rt.packet(110)
remove=load(root/'red_takeover_v5/retirement.npz')['remove_global_row_ids'];removed=torch.zeros(rt.native_row_count,device='cuda',dtype=torch.bool);removed[torch.as_tensor(remove,device='cuda')]=True
ids=torch.nonzero(rt.native_visible_mask(110,p)&~removed).reshape(-1);p=subset(p,ids);s=source[ids];groups=[-1]+sorted(set(s[s>=0].cpu().tolist()))
f=p['features'].new_zeros((len(ids),24));f[:,:2]=p['features'][:,:2]
for i,g in enumerate(groups):f[:,i+2]=(s==g).float()
with torch.no_grad():o=rt.raster_packet(110,dict(p,features=f))
m=load(R4/'masks/110/native.npz');red=torch.as_tensor(m['red_rebuild'],device='cuda');a9=torch.as_tensor(m['A9'],device='cuda')
rows=[]
for i,g in enumerate(groups):
    x=o['extra'][i];rows.append(dict(source_owner=g,red_alpha_mass=float(x[red].sum()),A9_alpha_mass=float(x[a9].sum()),samples=[float(x[y,xp])for xp,y in [(600,650),(580,545),(600,575),(550,700),(710,600)]]))
(root/'red_contributor_groups.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
