"""CPU连续规划诊断：真实Y、车辆洞上界、中灰masked-Y、世界位姿；不合成X。"""
import argparse,html,json,subprocess,sys
from pathlib import Path
import cv2,numpy as np,imageio_ffmpeg
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import read,dump,footprint
from data_contract import masked_condition_from_x
from planning import inputs,silhouettes,normalized_masks

KINDS=['original','hole','condition','world']
def encode(folder,kind):
    exe=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([exe,'-v','error','-y','-framerate','10','-i',str(folder/f'%03d_{kind}.jpg'),'-c:v','libx264','-threads','2','-preset','medium','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/f'{kind}.mp4')],check=True)
    reader=imageio_ffmpeg.read_frames(str(folder/f'{kind}.mp4'),pix_fmt='rgb24');meta=next(reader);count=sum(1 for _ in reader)
    assert count==10 and meta['size']==(1024,576)
    return {'kind':kind,'frames':count,'size':list(meta['size'])}

def world_frame(pose,frame):
    im=Image.new('RGB',(1024,576),'#152237');draw=ImageDraw.Draw(im);center=np.array(pose['actor']['translation'])[:2]
    xy=lambda q:(512+(float(q[0])-center[0])*14,288-(float(q[1])-center[1])*14)
    for x in range(22,1024,70):draw.line([(x,0),(x,576)],fill='#334258')
    for y in range(8,576,70):draw.line([(0,y),(1024,y)],fill='#334258')
    for j,ob in enumerate(frame['actors']):
        draw.polygon([xy(q) for q in footprint(ob).exterior.coords],outline='#35e69b',width=3)
        draw.text(xy(ob['translation']),f'Protected {chr(66+j)}',fill='#35e69b')
    draw.polygon([xy(q) for q in footprint(pose['actor']).exterior.coords],outline='#ffbd30',width=4)
    draw.text(xy(center),'A DELETE proposal',fill='#ffbd30')
    camera=xy(np.array(frame['camera_to_world'])[:2,3]);draw.ellipse([camera[0]-5,camera[1]-5,camera[0]+5,camera[1]+5],fill='white')
    draw.text(camera,'camera',fill='white');draw.line([camera,xy(center)],fill='white',width=2)
    draw.text((12,12),'WORLD XY / 5m grid / centered on A; world center follows A',fill='white')
    return np.asarray(im)

def main(parent,root):
    cv2.setNumThreads(1);root.mkdir(exist_ok=True,parents=True);out=root/'review';out.mkdir(exist_ok=True)
    sources,donors,donor_root,pairs,qa=inputs(parent)
    config={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r5','parent_run':str(parent),'cpu_only':True,
            'scope':'same four r4 preplans; continuous planning visualization and exact-mask admission infrastructure',
            'fixed_rules':'same r4 timestamps/poses/donor/6px hole bound; no new search or changed thresholds',
            'actual_synthetic_cases':0,'GPU_calls':0,'training_ready':0,'human_verdict':None}
    if (root/'run_config.json').exists():assert read(root/'run_config.json')==config
    dump(root/'run_config.json',config);cases=[]
    for pair in pairs:
        sid=pair['source_id'];c=sources[sid];donor=donors[pair['donor_source_id']];_,_,shapes=silhouettes(pair,donor,donor_root)
        dest=out/'assets'/sid;dest.mkdir(exist_ok=True,parents=True);frames=[]
        mask_metrics=normalized_masks([a for a,h in shapes],[p['box'] for p in pair['frames']])
        for i,(f,pose,(a,h)) in enumerate(zip(c['frames'],pair['frames'],shapes)):
            with Image.open(parent/'native10_factory/rgb'/f['filename']) as im:y=np.asarray(im.convert('RGB').resize((1024,576),Image.Resampling.LANCZOS))
            assert f['timestamp']==pose['timestamp']
            conditional=masked_condition_from_x(y[None],h[None])[0]
            altered=y.copy();altered[h]=255-altered[h]
            assert np.array_equal(conditional,masked_condition_from_x(altered[None],h[None])[0])
            cp=np.rint((conditional+1)*127.5).clip(0,255).astype('uint8')
            overlay=y.copy();overlay[h]=(.5*overlay[h]+.5*np.array([255,180,30])).astype('uint8')
            labelled=Image.fromarray(y);ld=ImageDraw.Draw(labelled);hole=Image.fromarray(overlay);hd=ImageDraw.Draw(hole)
            for j,ob in enumerate(f['actors']):
                box=ob['projection']['box_xyxy']
                for draw in [ld,hd]:
                    draw.rectangle(box,outline='#35e69b',width=2);draw.text((box[0],max(0,box[1]-14)),f'B {ob["instance_token"][:7]} / GT envelope',fill='#35e69b')
            hd.rectangle(pose['box'],outline='#ffbd30',width=2);hd.text((pose['box'][0],max(0,pose['box'][1]-14)),'A DELETE / planned hole bound',fill='#ffbd30')
            panels=dict(original=np.asarray(labelled),hole=np.asarray(hole),condition=cp,world=world_frame(pose,f))
            for kind,arr in panels.items():Image.fromarray(arr).save(dest/f'{i:03}_{kind}.jpg',quality=94)
            # 无损规划轮廓可复核，不导出为训练X/Y或最终H。
            Image.fromarray(a.astype('uint8')*255).save(dest/f'{i:03}_planned_A.png')
            Image.fromarray(h.astype('uint8')*255).save(dest/f'{i:03}_hole_upper_bound.png')
            frames.append({'frame':i,'timestamp_us':f['timestamp'],'relative_ms':(f['timestamp']-c['frames'][0]['timestamp'])/1000,
                           'delta_ms':f['delta_ms'],'hole_fraction':float(h.mean()),'ego_bottom_guard_pass':True,
                           'hidden_pixel_probe_pass':True,'planned_silhouette_continuity':mask_metrics[i],
                           'images':{k:f'assets/{sid}/{i:03}_{k}.jpg' for k in KINDS}})
        decoded=[encode(dest,k) for k in KINDS]
        cases.append({'case_id':sid,'scene':c['scene'],'camera':c['camera'],'donor':donor['source_id'],
                      'protected_instances':pair['required_protected_instances'],'type':'single_actor_preplan',
                      'preflight_note':qa[sid]['note'],'frames':frames,'decoded_videos':decoded,
                      'videos':{k:f'assets/{sid}/{k}.mp4' for k in KINDS},'human_verdict':None,'training_ready':False,
                      'status':'planning_only_pending_true_protected_masks'})
        print('CONTINUOUS_PREVIEW',sid,'frames',len(frames),flush=True)
    manifest=config|{'cases':cases,'case_count':len(cases),'scene_count':len({c['scene'] for c in cases}),
                     'frame_count':sum(len(c['frames']) for c in cases),'video_count':4*len(cases),
                     'decoded_video_frames':sum(d['frames'] for c in cases for d in c['decoded_videos']),
                     'actual_final_H_verified':False,'precise_protected_occlusion_verified':False,
                     'no_actual_synthetic_X_rendered':True,'new_independent_visual_review':'not_repeated; existing r4 fixed-frame source QA reused; temporal visualization requires human inspection'}
    dump(out/'review_manifest.json',manifest);dump(root/'summary.json',{k:v for k,v in manifest.items() if k!='cases'})
    labels=['真实视频 Y · 绿色 B 定位框','黄色 A 删除洞规划上界','masked-Y 条件规划 · 洞内中灰','米制位姿 · 相机 / A / B']
    cards=[]
    for c in cases:
        videos=''.join(f'<div><h3>{labels[j]}</h3><video controls muted loop preload="metadata" src="{c["videos"][k]}"></video><img class="still" data-kind="{k}" src="{c["frames"][0]["images"][k]}" alt="{c["case_id"]} {k} 第0帧"></div>' for j,k in enumerate(KINDS))
        cards.append(f'<section class="case" data-case="{c["case_id"]}"><h2>{c["case_id"]} · {c["scene"]} · {c["camera"]}</h2><p>A来源 {c["donor"]}；真实保护车 B：{", ".join(t[:12] for t in c["protected_instances"])}。{html.escape(c["preflight_note"])}</p><button class="play">同步播放／暂停</button><label>逐帧 <input class="slider" type="range" min="0" max="9" value="0" step="1"></label><span class="info"></span><div class="cols">{videos}</div></section>')
    boxes=''.join(f'<rect x="{10+240*i}" y="10" width="205" height="75" rx="8"/><text x="{112+240*i}" y="40">{a}<tspan x="{112+240*i}" dy="23">{b}</tspan></text>' for i,(a,b) in enumerate([('真实 train Y','原 RGB + GT/LiDAR'),('r4 固定位姿','已有连续 A 轮廓'),('连续洞规划','逐帧曝光/尺度检查'),('条件清零探针','40 帧 / 16 个视频'),('等待 GPU SAM2','真实 B mask → 原关卡')]))
    arrows=''.join(f'<path d="M{218+i*240} 47h25" marker-end="url(#arrow)"/>' for i in range(4))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 r5 连续洞规划</title><style>body{background:#101827;color:#edf3fb;font:16px/1.7 system-ui;max-width:1800px;margin:25px auto;padding:0 20px}section{padding:20px;margin:20px 0;background:#1b293d;border-radius:10px}a{color:#8cd2ff}.cols{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}video,img,svg{width:100%;height:auto}h3{font-size:14px}button{padding:8px 12px;margin-right:14px}svg rect{fill:#244b70;stroke:#639dcc}svg text{fill:white;text-anchor:middle;font-size:16px}svg path{stroke:#8cc5e8;fill:none}.info{margin-left:14px}small{color:#aec0d6}@media(max-width:1000px){.cols{grid-template-columns:repeat(2,1fr)}}</style><h1>v77 r5：连续洞规划与 GPU 后准入入口</h1><p><a href="../v77-target-protected-r4/index.html">r4 来源与负对照</a> · <a href="../v77-target-protected-r3/data_review.html">r3 实际数据人工全检</a></p><section><svg viewBox="0 0 1210 105" aria-label="CPU规划到精确mask准入"><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8" style="fill:#8cc5e8"/></marker></defs>BOXES ARROWS</svg></section><section><p>沿用 r4 的 4 个候选、10 次真实曝光和原位姿，新增连续规划预览。共 40 帧、16 个视频，160 帧视频实解码。黄色区域是 A 真实供体轮廓投影加 6px 的规划上界，绿色框只是 B 定位用的 GT 包络。第三栏按归一化 0 显示为中灰；它用于说明遮挡条件，当前还没有实际 synthetic-X 或最终删除 H。</p><p>每段约 1 秒，MP4 按 10fps 播放；逐帧处显示真实曝光时间。世界图以 A 为中心，网格间距 5m，中心随 A 移动。滑块切换下方静帧，并把四段视频移到同一帧；视频播放时仍需人眼判断是否稳定。</p><p>精确准入入口只接受来源匹配、PNG完整、连续性通过并且独立审核通过的真实 SAM2 保护车 mask，再调用既有 A→B 遮挡／可见比例关卡。当前四例均停在缺 mask，训练准入 0。复用 r4 的 Sol 来源抽帧结论，本轮没有重复 AI 看图，也没有用单帧认证时序。</p></section>CARDS<script>const DATA=MANIFEST;
for(const section of document.querySelectorAll('.case')){const c=DATA.cases.find(c=>c.case_id===section.dataset.case);const slider=section.querySelector('.slider');const vids=[...section.querySelectorAll('video')];function frame(){const f=c.frames[Number(slider.value)];section.querySelector('.info').textContent=`第 ${f.frame} 帧 / 真实时间 +${f.relative_ms.toFixed(1)}ms / 洞上界 ${(f.hole_fraction*100).toFixed(1)}%`;for(const im of section.querySelectorAll('.still')){im.src=f.images[im.dataset.kind];im.alt=`${c.case_id} ${im.dataset.kind} 第${f.frame}帧`;}for(const v of vids){v.pause();if(v.readyState>=1)v.currentTime=f.frame/10;}}slider.addEventListener('input',frame);section.querySelector('.play').addEventListener('click',()=>{if(vids[0].paused){const t=vids[0].currentTime;for(const v of vids){v.currentTime=t;v.play().catch(()=>{});}}else vids.forEach(v=>v.pause());});frame();}
</script></html>'''
    page=page.replace('BOXES',boxes).replace('ARROWS',arrows).replace('CARDS',''.join(cards)).replace('MANIFEST',json.dumps(manifest,ensure_ascii=False).replace('</','<\\/'))
    (out/'index.html').write_text(page);print('CPU_R5_REPORT',manifest['case_count'],manifest['video_count'],manifest['decoded_video_frames'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.parent,a.root)
