"""建立目标可辨认的审计图、视频与离线页；保留原实验视频。"""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import argparse,json,math,shutil,subprocess,html
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--work-root',type=Path,default=Path.cwd()/'work')
B=p.parse_args().work_root.resolve();A=B/'v77_actor_audit';O=B.parent/'outputs'/'v77-actor-audit';O.mkdir(parents=True,exist_ok=True)
FONT='C:/Windows/Fonts/msyh.ttc'
def font(n=20):return ImageFont.truetype(FONT,n)
def txt(d,xy,s,fill='#153248',size=20):d.text(xy,s,font=font(size),fill=fill)
def arrow(d,p,q,color,width=4):
    d.line([p,q],fill=color,width=width);ang=math.atan2(q[1]-p[1],q[0]-p[0])
    d.polygon([q,(q[0]-13*math.cos(ang-.45),q[1]-13*math.sin(ang-.45)),(q[0]-13*math.cos(ang+.45),q[1]-13*math.sin(ang+.45))],fill=color)
def resize_crop(im,box,w=688,h=290):
    x0,y0,x1,y1=box;cx=(x0+x1)/2;cy=(y0+y1)/2;cw=max(x1-x0+40,(y1-y0+25)*w/h);ch=cw*h/w
    return im.crop((cx-cw/2,cy-ch/2,cx+cw/2,cy+ch/2)).resize((w,h),Image.Resampling.LANCZOS)
def basepath(s,f):return B/'v77_review_input'/s/f'{f:03d}'
def getraw(s,f,c):return Image.open(basepath(s,f)/f'raw_cam{c}.jpg').convert('RGB')
def getbg(s,f,c):return Image.open(basepath(s,f)/f'pixel_bg_cam{c}.png').convert('RGBA')
def composite(bg,p):return Image.alpha_composite(bg,Image.open(p).convert('RGBA')).convert('RGB')
plan=json.loads((A/'view_plan.json').read_text());registration=json.loads((B/'v77_blender_input'/'registration.json').read_text())
records=[]
for s,config in plan.items():
    out=O/s;out.mkdir(exist_ok=True);audit=json.loads((A/f'{s}_audit.json').read_text());proj=json.loads((A/f'{s}_projection.json').read_text())
    actor=config['actor_id'];ref=audit['reference_frame'];cam=audit['reference_camera'];row=next(x for x in config['frames'] if x['frame']==ref)
    spec=next(x for x in registration['scenes'] if x['name']==s)['spec']
    fa=json.loads((B/'v77_blender_input'/s/'instances_snapshot.json').read_text())
    def project(center,f,c):
        c2w=np.loadtxt(B/'v77_blender_input'/'source'/s/'extrinsics'/f'{f:03d}_{c}.txt')
        p=np.linalg.inv(c2w)@np.r_[center,1];v=spec['views'][c];K=np.array(v['intrinsics']);sx=688/v['original_wh'][0];sy=384/v['original_wh'][1]
        if p[2]<=.1:return None
        q=K@p[:3];return (q[0]/q[2]*sx+(sx-1)/2,q[1]/q[2]*sy+(sy-1)/2)
    def annotate(im,f,c,box,mask=True,move=True):
        im=im.convert('RGBA');d=ImageDraw.Draw(im)
        if mask:
            m=Image.open(A/s/f'mask_f{f:03d}_cam{c}.png').convert('L').resize((688,384),Image.Resampling.NEAREST)
            tint=Image.new('RGBA',im.size,(0,222,201,0));tint.putalpha(m.point(lambda p:round(p*.20)));im=Image.alpha_composite(im,tint)
            edge=m.filter(ImageFilter.MaxFilter(5));edge=np.maximum(np.asarray(edge).astype(int)-np.asarray(m).astype(int),0).astype('uint8')
            tint=Image.new('RGBA',im.size,(0,255,218,0));tint.putalpha(Image.fromarray(edge));im=Image.alpha_composite(im,tint);d=ImageDraw.Draw(im)
        x0,y0,x1,y1=box;d.rectangle(box,outline='#00ffda',width=2)
        label=f'目标 actor {actor}';ty=max(4,y0-28);d.rectangle((x0,ty,x0+165,ty+26),fill='#063d40');txt(d,(x0+6,ty+1),label,'#ffffff',17)
        r=audit['old_move']['frames'][f]
        pose=np.array(r['target_pose']);src=project(pose[:3,3],f,c);dst=project(pose[:3,3]+audit['old_move']['delta_world_m'],f,c)
        if move and src and dst:
            arrow(d,src,dst,'#ff607b',5)
        return im.convert('RGB')
    raw=getraw(s,ref,cam);annotated=annotate(raw,ref,cam,row['box'])
    # Identity overview plus readable crop. Cyan is the queried target only.
    hero=Image.new('RGB',(1376,426),'#eaf0f4');hero.paste(annotated,(0,42));hero.paste(resize_crop(raw,row['box'],688,384),(688,42));d=ImageDraw.Draw(hero)
    txt(d,(15,10),f'{s} · CAM{cam} · f{ref:03d} · 目标 actor {actor}',size=18);txt(d,(705,10),'同一原车放大（未生成）',size=18)
    hero.save(out/'target.jpg',quality=93)
    # BEV exact translation swept polygon in actor-relative coordinates.
    worst=ref if s=='scene_0230' else 77
    r=audit['old_move']['frames'][worst];pose=np.array(r['target_pose']);center=pose[:2,3];rot=pose[:2,:2]
    scale=42;cx,cy=420,430
    def xy(p):
        local=(np.array(p)-center)@rot
        return (cx-local[1]*scale,cy-local[0]*scale)
    bev=Image.new('RGBA',(840,860),'#f4f7f9');d=ImageDraw.Draw(bev)
    for i in range(-9,10):
        d.line((cx+i*scale,80,cx+i*scale,800),fill='#dbe4e9');d.line((25,cy+i*scale,815,cy+i*scale),fill='#dbe4e9')
    txt(d,(24,16),f'{s} / f{worst:03d}：旧 MOVE 平移扫掠检查',size=24)
    txt(d,(24,52),'俯视，目标车头朝上；网格 1 m。不是可行驶区地图。',size=17)
    for p in r['peers']:
        points=[xy(q) for q in p['footprint_world']];cen=np.mean(points,axis=0)
        if not 35<cen[0]<805 or not 95<cen[1]<770:continue
        hit=p['swept_overlap_m2']>1e-5;color='#df7b85' if hit else '#a7b7c4'
        d.polygon(points,fill=color,outline='#344e62',width=2);txt(d,(cen[0]-24,cen[1]-10),p['actor_id'],'#102b3b',18)
    layer=Image.new('RGBA',bev.size);dl=ImageDraw.Draw(layer);dl.polygon([xy(q) for q in r['swept_footprint_world']],fill=(244,128,40,65),outline='#e09536',width=2)
    bev=Image.alpha_composite(bev,layer);d=ImageDraw.Draw(bev)
    src=[xy(q) for q in r['source_footprint_world']];dst=[xy(q) for q in r['destination_footprint_world']]
    d.polygon(src,outline='#008f86',width=5);d.polygon(dst,outline='#c6325b',width=5)
    arrow(d,xy(center),xy(center+np.array(audit['old_move']['delta_world_m'])[:2]),'#c6325b',4)
    arrow(d,(cx,cy),(cx,cy-65),'#008f86',4)
    txt(d,(cx+12,cy-22),f'目标 {actor}',size=17)
    txt(d,(24,789),'青框：原位　红框：终点　橙色：整段平移扫掠区域',size=18)
    txt(d,(24,820),'红填充：相交邻车；灰填充：其他 GT actor（数字为 ID）',size=16)
    bev.convert('RGB').save(out/'bev.jpg',quality=93)
    # All variants use the identical crop window, not a content-dependent zoom.
    bg=getbg(s,ref,cam);yaw0=composite(bg,A/f'{s}_yaw0.png');yaw180=composite(bg,A/f'{s}_yaw180.png')
    variants=[('原车 RGB',raw),('同 GLB / yaw 0°',yaw0),('同 GLB / yaw 180°',yaw180)]
    comp=Image.new('RGB',(2064,326),'#eef3f7');d=ImageDraw.Draw(comp)
    for i,(label,im) in enumerate(variants):comp.paste(resize_crop(im,row['box']),(688*i,36));txt(d,(688*i+12,6),label,size=20)
    comp.save(out/'yaw_control.jpg',quality=93)
    corrected=composite(bg,A/s/f'f{ref:03d}_cam{cam}_corrected.png');uniform=composite(bg,A/s/'uniform_scale.png')
    comp=Image.new('RGB',(2064,326),'#eef3f7');d=ImageDraw.Draw(comp)
    for i,(label,im) in enumerate([('原车 RGB',raw),('原逐轴尺寸匹配 + 正确朝向',corrected),('保留 GLB 比例：按车长统一缩放',uniform)]):
        comp.paste(resize_crop(im,row['box']),(688*i,36));txt(d,(688*i+12,6),label,size=20)
    comp.save(out/'scale_control.jpg',quality=93)
    # Three actor-focused videos, same camera schedule and timestamps.
    for kind,title in [('raw','原视频 · 青框为目标'),('corrected','修正放置的 GLB 原位 · 非 MOVE'),('delete','DELETE · 原位置补景')]:
        fd=A/s/f'video_{kind}';fd.mkdir(exist_ok=True)
        for i,v in enumerate(config['frames']):
            f,c,box=v['frame'],v['camera'],v['box'];base=getraw(s,f,c) if kind=='raw' else getbg(s,f,c).convert('RGB')
            if kind=='corrected':base=composite(base.convert('RGBA'),A/s/f'f{f:03d}_cam{c}_corrected.png')
            overview=annotate(base,f,c,box,mask=(kind=='raw'),move=False)
            img=Image.new('RGB',(1376,428),'#12283a');img.paste(overview,(0,44));img.paste(resize_crop(base,box,688,384),(688,44));d=ImageDraw.Draw(img)
            txt(d,(14,9),f'{title} | actor {actor} | f{f:03d} / {f/10:.1f}s / CAM{c}','white',18)
            txt(d,(850,9),'右：同一目标窗口放大；相机可能切换','#94ebd8',16)
            img.save(fd/f'{i:03d}.jpg',quality=92)
        fps=2 if s=='scene_0230' else 1
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(fps),'-i',str(fd/'%03d.jpg'),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out/f'{kind}.mp4')],check=True)
        shutil.copy2(fd/'000.jpg',out/f'{kind}_poster.jpg')
    shutil.copy2(A/f'{s}_audit.json',out/'command_audit.json');shutil.copy2(A/f'{s}_projection.json',out/'projection_audit.json')
    shutil.copy2(A/f'{s}_render_validation.json',out/'render_validation.json')
    records.append({'name':s,'actor_id':actor,'fps':2 if s=='scene_0230' else 1,'frames':config['frames'],'yaw_correction_deg':config['yaw_correction_deg'],'human_verdict':None})
    print('AUDIT_MEDIA_DONE',s,flush=True)
(O/'review_data.json').write_text(json.dumps({'task_id':'WS-V77-ACTOR-COMMAND-AUDIT-20260926','scenes':records,'training_steps':0,'new_model_forwards':0,'human_verdict':None},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
