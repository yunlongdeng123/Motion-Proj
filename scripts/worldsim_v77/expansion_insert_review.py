"""INSERT分离无深度覆盖和Ω遮挡，逐帧保存可复核读出。"""
from pathlib import Path
import sys,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump
from delete_full_query import Writer,compose_actor
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928');R=T/'r3';review=R/'review';assert not review.exists();review.mkdir();rows=[];panels=[]
writers={f'cam{c}_{kind}':Writer(review/f'cam{c}_{kind}.mp4',(688,384)) for c in [5,4] for kind in ['source','no_occlusion','omega_occlusion','pure_geometry']}
for f in range(10):
    panel=Image.new('RGB',(1376,410),(20,30,40));d=ImageDraw.Draw(panel)
    for c in [5,4]:
        world=T/'r2'/f'f{f:03}';bg=np.array(Image.open(world/f'hybrid_cam{c}.png').convert('RGB'));pure=np.array(Image.open(world/f'geometry_cam{c}.png').convert('RGB'));z=np.load(world/f'z_cam{c}.npy');layer=R/'actor_layers'/f'f{f:03}_cam{c}';rgba=np.array(Image.open(layer.with_suffix('.png')).convert('RGBA'));az=np.load(layer.with_name(layer.name+'_depth.npy'))
        simple,rawstats=compose_actor(bg,rgba,az,np.full(z.shape,np.inf));comp,stats=compose_actor(bg,rgba,az,z);geo,_=compose_actor(pure,rgba,az,z)
        assert not np.any((comp!=bg).any(-1)&(rgba[...,3]==0))
        row=dict(frame=f,camera=c,**stats,no_occlusion_asset_pixels=rawstats['visible_asset_pixels']);rows.append(row)
        views=dict(source=bg,no_occlusion=simple,omega_occlusion=comp,pure_geometry=geo)
        for kind,im in views.items():
            pic=Image.fromarray(im);ImageDraw.Draw(pic).text((5,5),f'INSERT scene0255/25 into official000 | CAM{c} f{f} | {kind}',fill='yellow');writers[f'cam{c}_{kind}'].write(pic)
            if c==5:panel.paste(Image.fromarray(im).resize((344,384)),(list(views).index(kind)*344,26));d.text((list(views).index(kind)*344+4,5),f'f{f} {kind}',fill='white')
        Image.fromarray(comp).save(review/f'cam{c}_f{f:03}.png')
    # 联系表保持图像纵横比，四列各344x192。
    fixed=Image.new('RGB',(1376,218),(20,30,40));fixed.paste(panel.crop((0,0,1376,26)),(0,0))
    for j in range(4):fixed.paste(panel.crop((j*344,26,j*344+344,410)).resize((344,192)),(j*344,26))
    panels.append(fixed)
for w in writers.values():w.close()
sheet=Image.new('RGB',(1376,218*len(panels)))
for i,p in enumerate(panels):sheet.paste(p,(0,218*i))
sheet.save(review/'all10.jpg',quality=95)
dump(review/'summary.json',dict(task_id=T.name,run_id='r3',operation='INSERT',frames=10,cameras=[5,4],rows=rows,video_frame_counts={k:w.count for k,w in writers.items()},render='Fixed Blender world+sun, no lighting fit/contact shadow; original asset unchanged. ΩB_t+RGB hole fallback; actor z test tolerance.15m.',scope='No factual reconstruction of original target12. Adjacent CAM4 can be empty/clipped. Geometric candidate screening30f; video1s only.',human_verdict=None))
print('INSERT_REVIEW_DONE',rows)
