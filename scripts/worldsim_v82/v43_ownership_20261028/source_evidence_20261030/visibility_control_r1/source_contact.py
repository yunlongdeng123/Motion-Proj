from pathlib import Path
from PIL import Image,ImageDraw
p=Path('work/v43_sim_mechanism_20261030/source_control_r1/evidence/run_r2_color_maps/source_000008');ref=next(Path('outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030').rglob('dataset_r2/privileged/references/delete/shadow_on/rgb/000008.png'))
files=[ref]+[p/(n+'.png') for n in ['baseline_delete_final','remove_top_error_32_final','top128_color_only_reference_moment_final','baseline_delete_white']]+sorted(p.glob('top*_alphaT.png'))
c=Image.new('RGB',(5*480,3*345),'white');d=ImageDraw.Draw(c)
for i,f in enumerate(files):
 x=(i%5)*480;y=(i//5)*345;d.text((x+5,y+5),('independent delete ref' if i==0 else f.stem),fill='black');c.paste(Image.open(f).convert('RGB'),(x,y+25))
c.save('work/v43_sim_mechanism_20261030/visibility_control_r1/source_contact.png')
