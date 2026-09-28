"""真实读取同一B_t和GLB执行DELETE；三列同时间、同相机。"""
from nine_common import *
import argparse,numpy as np
from PIL import Image,ImageDraw
from geometry import project_bbox
from delete_full_query import compose_actor,Writer
from delete_full_common import apply_delete
p=argparse.ArgumentParser();p.add_argument('--scene');args=p.parse_args()
reg=read(ROOT/'registration.json');display={r['scene']:r['camera'] for r in read(ROOT/'display_registration.json')}
for s in reg['scenes']:
    if args.scene and s['name']!=args.scene:continue
    base=ROOT/s['name'];frames=read(base/'camera_frames.json');out=ROOT/'review'/s['name'];out.mkdir(parents=True,exist_ok=True)
    assert not (out/'summary.json').exists()
    assert all((base/'background_world'/f'{f:03}/summary.json').exists() for f in range(30))
    assert (base/'actor_layers/render_summary.json').exists()
    manifest=dict(schema='v77-explicit-scene/2',scene=s['name'],fps=10,frames=30,background=[dict(frame=f,points=f'background_world/{f:03}/background_points.npz',origin_world=fr['origin_world']) for f,fr in enumerate(frames)],
        actors=[dict(actor_id=s['actor'],asset='actor.glb',visible=True,yaw=0,poses=[dict(frame=fr['frame'],**next(b for b in fr['all_boxes'] if b['actor_id']==s['actor'])) for fr in frames])],
        render=dict(camera_source='GT',point_splat_radius=1,depth_tolerance_m=.15,default='pure_geometry',optional='hybrid inputRGB fallback only where no geometry',lighting='fixed world+sun,no fitted shadows'),human_verdict=None)
    edited=apply_delete(manifest,s['actor']);dump(base/'scene_factual.json',manifest);dump(base/'scene_delete.json',edited);assert edited['actors'][0]['visible'] is False
    dump(out/'command.json',dict(operation='DELETE',actor_id=s['actor'],same_asset_and_background=True,neural_calls_during_query=0,legality='No new occupied volume; target identity/mask quality remains subject to review',human_verdict=None))
    kinds=['original','factual','delete','hybrid_factual','hybrid_delete','completion','scope'];writers={k:Writer(out/f'{k}.mp4',(688,384)) for k in kinds}
    writers.update({f'six_{k}':Writer(out/f'six_{k}.mp4',(1032,384)) for k in ['original','factual','delete','hybrid_factual','hybrid_delete']})
    rows=[];contacts=[];stages=[];main=display[s['name']];poster_frame=int(np.argmax(s['streams'][main]['projected_areas']))
    # 同一GT轨迹外包范围用于三列共同放大；固定裁剪，避免逐帧跟随造成假稳定。
    bbs=[]
    for fr in frames:
        ac=next(b for b in fr['all_boxes'] if b['actor_id']==s['actor']);ec,ik=camera(fr,main,(384,688));bb=project_bbox(ac['pose'],ac['size_lwh'],ec,ik,(384,688))
        if bb:bbs.append(bb)
    bbs=np.array(bbs);cx=(bbs[:,0].min()+bbs[:,2].max())/2;cy=(bbs[:,1].min()+bbs[:,3].max())/2
    cw=max(160,(bbs[:,2].max()-bbs[:,0].min())*1.6,(bbs[:,3].max()-bbs[:,1].min())*1.6*688/384);cw=min(688,cw);ch=cw*384/688
    x0=int(np.clip(cx-cw/2,0,688-cw));y0=int(np.clip(cy-ch/2,0,384-ch));zoom_box=(x0,y0,int(x0+cw),int(y0+ch))
    writers.update({f'zoom_{k}':Writer(out/f'zoom_{k}.mp4',(688,384)) for k in ['original','factual','delete','hybrid_factual','hybrid_delete']})
    extra_cameras=[v['camera'] for v in s['streams'] if v['active'] and v['camera']!=main]
    for c in extra_cameras:
        writers.update({f'cam{c}_{k}':Writer(out/f'cam{c}_{k}.mp4',(688,384)) for k in ['original','factual','delete','hybrid_factual','hybrid_delete']})
    def label(a,text):
        im=Image.fromarray(a);d=ImageDraw.Draw(im);d.rectangle((0,0,im.width,22),fill=(20,30,40));d.text((4,4),text,fill='white');return im
    for fr in frames:
        f=fr['frame'];world=base/'background_world'/f'{f:03}';views=[];actor=next(b for b in fr['all_boxes'] if b['actor_id']==s['actor'])
        for c in range(6):
            stream=base/f'cam{c}';original=np.array(Image.open(stream/'rgb'/f'{f:05}.png').convert('RGB').resize((688,384),Image.Resampling.BICUBIC));bginput=np.array(Image.open(stream/'background'/f'{f:05}.png').convert('RGB').resize((688,384),Image.Resampling.BICUBIC));pure=np.array(Image.open(world/f'geometry_cam{c}.png'));hybrid=np.array(Image.open(world/f'hybrid_cam{c}.png'));z=np.load(world/f'z_cam{c}.npy')
            lp=base/'actor_layers'/f'f{f:03}_cam{c}.png';expected=s['streams'][c]['projected_areas'][f]>0;assert lp.exists()==expected,(s['name'],f,c)
            if expected:
                rgba=np.array(Image.open(lp).convert('RGBA'));az=np.load(lp.with_name(lp.stem+'_depth.npy'));factual,vis=compose_actor(pure,rgba,az,z);hfact,hvis=compose_actor(hybrid,rgba,az,z)
                assert not ((factual!=pure).any(-1)&(rgba[...,3]==0)).any()
            else:factual=pure.copy();hfact=hybrid.copy();vis=dict(asset_pixels=0,visible_asset_pixels=0,occluded_asset_pixels=0)
            # DELETE只切visible，不运行生成器，也不改背景几何。
            deletion=pure.copy() if not edited['actors'][0]['visible'] else factual
            hdelete=hybrid.copy() if not edited['actors'][0]['visible'] else hfact
            assert np.array_equal(deletion,pure)
            kcam,kk=camera(fr,c,(384,688));bb=project_bbox(actor['pose'],actor['size_lwh'],kcam,kk,(384,688))
            marked=Image.fromarray(original)
            if bb:ImageDraw.Draw(marked).rectangle(bb,outline='#ffd24a',width=2)
            scope=original.copy();mp=stream/'model_mask'/f'{f:05}.png';core_count=0
            if mp.exists():
                mask=np.array(Image.open(mp).resize((688,384),Image.Resampling.NEAREST))>0;write=np.array(Image.open(stream/'write_mask'/f'{f:05}.png').resize((688,384),Image.Resampling.NEAREST))>0
                scope[mask]=(.5*scope[mask]+.5*np.array([60,160,255])).astype('uint8');scope[write]=(.5*scope[write]+.5*np.array([255,190,0])).astype('uint8');core_count=int(write.sum())
            views.append(dict(original=np.array(marked),factual=factual,delete=deletion,hybrid_factual=hfact,hybrid_delete=hdelete,completion=bginput,scope=scope))
            rows.append(dict(frame=f,camera=c,gt_projected_area=s['streams'][c]['projected_areas'][f],write_pixels=core_count,geometry_coverage=float(np.isfinite(z).mean()),**vis))
        for k in kinds:writers[k].write(label(views[main][k],f"{s['name']} / actor{s['actor']} | CAM{main} f{f:02} {f/10:.1f}s | {k}"))
        if f==poster_frame:
            for k in ['original','factual','delete']:label(views[main][k],f"{s['name']} actor{s['actor']} | CAM{main} f{f:02} | {k}").save(out/f'poster_{k}.jpg',quality=94)
        for c in extra_cameras:
            for k in ['original','factual','delete','hybrid_factual','hybrid_delete']:
                writers[f'cam{c}_{k}'].write(label(views[c][k],f"{s['name']} actor{s['actor']} | CAM{c} f{f:02} | {k}"))
        for k in ['original','factual','delete','hybrid_factual','hybrid_delete']:
            zoom=np.array(Image.fromarray(views[main][k]).crop(zoom_box).resize((688,384),Image.Resampling.LANCZOS))
            writers[f'zoom_{k}'].write(label(zoom,f"{s['name']} actor{s['actor']} | CAM{main} f{f:02} | fixed ROI {k}"))
        for k in ['original','factual','delete','hybrid_factual','hybrid_delete']:
            mosaic=Image.new('RGB',(1032,384))
            for c in range(6):mosaic.paste(label(views[c][k],f'CAM{c} f{f:02}').resize((344,192)),((c%3)*344,(c//3)*192))
            writers[f'six_{k}'].write(mosaic)
        if f in np.linspace(0,29,10).round().astype(int):
            panel=Image.new('RGB',(1032,214),(20,30,40));d=ImageDraw.Draw(panel)
            for j,k in enumerate(['original','factual','delete']):panel.paste(Image.fromarray(views[main][k]).resize((344,192)),(j*344,22));d.text((j*344+4,4),f'f{f:02} CAM{main} {k}',fill='white')
            contacts.append(panel)
            rawpath=base/f'cam{main}/native'/f'{f:05}.png';native=np.array(Image.open(rawpath).resize((688,384))) if rawpath.exists() else views[main]['completion']
            # 全帧输入/补景/几何分开保存，便于判断先在哪一段变坏。
            stage=Image.new('RGB',(1376,214),(20,30,40));d=ImageDraw.Draw(stage)
            for j,(name,im) in enumerate([('mask',views[main]['scope']),('DriveEditor native',native),('writeback',views[main]['completion']),('Omega pure',views[main]['delete'])]):stage.paste(Image.fromarray(im).resize((344,192)),(j*344,22));d.text((j*344+4,4),f'f{f:02} {name}',fill='white')
            stages.append(stage)
    for w in writers.values():w.close();assert w.count==30
    for name,panels,width in [('contact',contacts,1032),('stages',stages,1376)]:
        sheet=Image.new('RGB',(width,214*len(panels)))
        for i,im in enumerate(panels):sheet.paste(im,(0,214*i))
        sheet.save(out/f'{name}.jpg',quality=94)
    visible_rows=[r for r in rows if r['gt_projected_area']>0]
    summary=dict(scene=s['name'],actor_id=s['actor'],primary_camera=main,frames=30,fps=10,views=6,actual_layer_renders=len(read(base/'actor_layers/render_summary.json')['rows']),
        mask_missing_when_gt_visible=sum(r['write_pixels']==0 for r in visible_rows),gt_visible_view_times=len(visible_rows),rows=rows,video_frames={k:w.count for k,w in writers.items()},
        original_reconstruction='pure B_t point render + generated target GLB at original GT pose',edit='same exact B_t, actor visible=false',optional_hybrid='input RGB only at no-geometry pixels, not proof of complete reconstruction',
        camera_GT=True,lidar_scale=True,temporal_world_consistent=False,strict_generalization=False,poster_frame=poster_frame,zoom_box_xyxy=zoom_box,zoom_rule='GT trajectory union plus context; fixed across time and all three columns; display only',human_verdict=None)
    dump(out/'summary.json',summary);progress('query',scene=s['name'],state='complete',frames=30)
