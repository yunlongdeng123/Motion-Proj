from pathlib import Path
from PIL import Image,ImageDraw
w=Path('work/v43_sim_mechanism_20261030');old=Path('outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030/evidence/edit_r3_10k');new=w/'evidence_reconstruction_r1/evidence/edit_samequery'
c=Image.new('RGB',(4*480,4*345),'white');d=ImageDraw.Draw(c)
for r,e in enumerate(['none','delete','shift','rotate']):
 ref=next(Path('outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030').rglob('dataset_r2/privileged/references/'+e+'/shadow_on/rgb/000008.png'))
 for col,(p,label) in enumerate([(ref,'independent reference'),(old/e/'000008_final.png','old native M0'),(new/e/'000008_final.png','new native hybrid'),(new/e/'000008_white.png','new native white')]):
  d.text((col*480+4,r*345+4),e+' '+label,fill='black');c.paste(Image.open(p).convert('RGB'),(col*480,r*345+25))
c.save(w/'visibility_control_r1/native_10k_samequery_contact.png')
