from pathlib import Path
from PIL import Image,ImageDraw
w=Path('work/v43_sim_mechanism_20261030'); m1=w/'m1_assignment/evidence';m2=w/'m2_repair/evidence';m0=Path('outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030/evidence/edit_r3_10k')
for held in (False,True):
 canvas=Image.new('RGB',(6*480,4*345),'white');d=ImageDraw.Draw(canvas)
 for row,e in enumerate(['none','delete','shift','rotate']):
  paths=[m0/e/'000008_final.png',m1/'recall_verified'/e/'000008_final.png']+[m2/a/e/'000008_final.png' for a in ['recall_position','recall_position_scale','recall_position_scale_split']]+[w/'heldout_edit_refs/heldout_edit_refs_r1'/e/'000008.png']
  if held: paths=[m1/('heldout_m0_'+e)/'final.png',m1/('heldout_recall' if e=='delete' else 'heldout_m1_'+e)/'final.png']+[m2/('heldout_'+a)/e/'000008_final.png' for a in ['recall_position','recall_position_scale','recall_position_scale_split']]+[w/'heldout_edit_refs/heldout_edit_refs_r1'/e/'000008.png']
  else:
   refs=list(Path('outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030').rglob('dataset_r2/privileged/references/'+e+'/shadow_on/rgb/000008.png'))
   if refs: paths[-1]=refs[0]
  for col,p in enumerate(paths):
   d.text((col*480+5,row*345+5),f'{e}: '+['M0','M1 recall','M2 pos','M2 pos+scale','M2 split','reference'][col],fill='black')
   if p.exists():canvas.paste(Image.open(p).convert('RGB'),(col*480,row*345+25))
   else:d.text((col*480+5,row*345+45),'MISSING '+str(p),fill='red')
 out=w/'heldout_edit_refs'/('heldout_contact.png' if held else 'train_contact.png');canvas.save(out)
