"""打包三列审核媒体与真实证据。原始模型产物继续保留远端。"""
from nine_common import *
from PIL import Image,ImageDraw
import shutil,tarfile
reg=read(ROOT/'registration.json');bundle=ROOT/'review_bundle';media=bundle/'media';evidence=bundle/'evidence';media.mkdir(parents=True,exist_ok=False);evidence.mkdir()
for name in ['registration.json','effective_registration.json','engineering_amendment_01.json','display_registration.json','render_registration.json','mask_state.json','drive_state.json','omega_state.json','sequence_state.json','validation.json']:
    shutil.copy2(ROOT/name,evidence/name)
for s in reg['scenes']:
    name=s['name'];base=ROOT/name;dst=media/name;shutil.copytree(ROOT/'review'/name,dst)
    shutil.copy2(base/'actor.glb',dst/'actor.glb');shutil.copy2(base/'asset/source_rgba.png',dst/'source_rgba.png')
    sheet=Image.new('RGB',(1024,768),(23,33,46))
    for i,az in enumerate([0,45,90,135,180,225,270,315]):
        im=Image.open(base/'asset_views'/f'az{az:03}.png').convert('RGBA');bg=Image.new('RGBA',im.size,(120,120,120,255));bg.alpha_composite(im);sheet.paste(bg.convert('RGB').resize((256,384)),((i%4)*256,(i//4)*384))
        # 保持原纵横比，四列256x192，上下留白仅用于标签。
    clean=Image.new('RGB',(1024,432),(23,33,46));d=ImageDraw.Draw(clean)
    for i,az in enumerate([0,45,90,135,180,225,270,315]):
        pic=Image.open(base/'asset_views'/f'az{az:03}.png').convert('RGBA');bg=Image.new('RGBA',pic.size,(120,120,120,255));bg.alpha_composite(pic);x=(i%4)*256;y=(i//4)*216;clean.paste(bg.convert('RGB').resize((256,192)),(x,y+24));d.text((x+5,y+4),f'{name} az{az}',fill='white')
    clean.save(dst/'asset_views.jpg',quality=95)
    ev=evidence/name;ev.mkdir()
    shutil.copy2(base/'orientation_contact.jpg',dst/'orientation_contact.jpg')
    for a,b in [('asset/registration.json','asset_registration.json'),('asset/shape_state.json','shape_state.json'),('asset/paint_state.json','paint_state.json'),('actor_layers/render_summary.json','render_summary.json'),('actor_layers/placement_checks.json','placement_checks.json'),('canonical.json','canonical.json'),('orientation.json','orientation.json'),('scene_factual.json','scene_factual.json'),('scene_delete.json','scene_delete.json')]:shutil.copy2(base/a,ev/b)
with tarfile.open(ROOT/'review_bundle.tar','w') as tar:
    for p in sorted(bundle.rglob('*')):
        if p.is_file():tar.add(p,arcname=str(p.relative_to(bundle)))
print('REVIEW_EXPORTED_BYTES',(ROOT/'review_bundle.tar').stat().st_size,flush=True)
