from pathlib import Path
from PIL import Image,ImageDraw
w=Path('work/v43_sim_mechanism_20261030');b=w/'m2_repair/evidence'
for h in (False,True):
 c=Image.new('RGB',(4*480,4*345),'white');d=ImageDraw.Draw(c)
 for row,e in enumerate(['none','delete','shift','rotate']):
  p=[b/('heldout_' if h else '')/e]
  paths=[b/('heldout_refit_assignment' if h else 'refit_assignment')/e/'000008_final.png',b/('heldout_refit_split' if h else 'refit_split')/e/'000008_final.png',b/('heldout_recall_position_scale_split' if h else 'recall_position_scale_split')/e/'000008_final.png']
  if h:ref=w/'heldout_edit_refs/heldout_edit_refs_r1'/e/'000008.png'
  else:ref=next(Path('outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030').rglob('dataset_r2/privileged/references/'+e+'/shadow_on/rgb/000008.png'))
  for col,p in enumerate(paths+[ref]):
   d.text((col*480+5,row*345+4),e+' '+['M1 refit200','M2 split refit200','M2 split frozen','reference'][col],fill='black');c.paste(Image.open(p).convert('RGB'),(col*480,row*345+25))
 c.save(w/'heldout_edit_refs'/('refit_heldout_contact.png' if h else 'refit_train_contact.png'))
