from pathlib import Path
from PIL import Image,ImageDraw
w=Path('work/v43_sim_mechanism_20261030');old=Path('outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030/evidence/edit_r3_10k');new=w/'evidence_reconstruction_r1/evidence';c=Image.new('RGB',(1440,2070),'white');d=ImageDraw.Draw(c)
for row,(f,e,h) in enumerate([('000000','delete',False),('000015','delete',False)]+[('000008',e,True) for e in ['none','delete','shift','rotate']]):
 if h:paths=[w/'heldout_edit_refs/heldout_edit_refs_r1'/e/'000008.png',w/'m1_assignment/evidence'/('heldout_m0_'+e)/'final.png',new/'heldout_samequery'/e/'final.png']
 else:paths=[next(Path('outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030').rglob('dataset_r2/privileged/references/'+e+'/shadow_on/rgb/'+f+'.png')),old/e/(f+'_final.png'),new/'edit_samequery'/e/(f+'_final.png')]
 for col,p in enumerate(paths):d.text((col*480+4,row*345+4),f+' '+e+(' heldout ' if h else ' common ') +['ref','old native','new native'][col],fill='black');c.paste(Image.open(p).convert('RGB'),(col*480,row*345+25))
c.save(w/'visibility_control_r1/native_10k_otherquery_contact.png')
