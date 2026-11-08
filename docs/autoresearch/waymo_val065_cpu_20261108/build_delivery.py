"""整理 scene-065 的 CPU 输入证据与下一阶段合同，不发布伪编辑结果。"""
from pathlib import Path
import json
import shutil
import hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'outputs/waymo_val065_preparation'
SOURCE=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-31/g/outputs/V43_R4_GPU_20261031/actor_candidates/scenes/15021599536622641101_556_150_576_150')
REMOTE='/root/autodl-tmp/codex_tmp/waymo_val065_cpu_20261108_r1'


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def write(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main():
    selection=read(OUT/'selected_case.json')
    verification=read(OUT/'evidence/inputs/verification.json')
    prompts=read(OUT/'evidence/inputs/point_prompt_verification.json')
    background=read(OUT/'evidence/inputs/background_nominations.json')
    status=read(OUT/'evidence/cpu_status.json')
    assert status['stage']=='CPU_PREPARATION_COMPLETE_GPU_NOT_STARTED'
    assert verification['status']=='PASS_CPU_BANK'
    media=read(OUT/'media.json')
    preserve='KOsYCOB6W_n0DzZsyDdJ-w'
    potential_reveal='bzFbWRrSGOgEaXxnuGV0tw'
    assert media['actors'][preserve]['alias']=='a7' and media['actors'][potential_reveal]['alias']=='a20'
    # 这份合同只绑定新数据；模型、Adam状态、车辆曲面、补路资产均不从029复制。
    contract=dict(schema='waymo.case.cpu_handoff.v1',task_id='WAYMO-VAL065-A9-CPU-20261108',
        scope='CPU preparation only; GPU stages not started',context_id=selection['context_id'],
        target=selection['target'],preserve=dict(description='右侧相邻车道的白色皮卡',alias='a7',track_id=preserve,native_actor_id=8),
        potential_reveal=dict(description='目标白车前方、画面中位于其上方的深色车辆',alias='a20',track_id=potential_reveal,native_actor_id=21,confirmed_surface_correspondence=False),
        camera='FRONT',source_wh=[1920,1280],render_wh=[480,320],build=selection['build'],holdout=selection['holdout'],
        pilot_raws=selection['pilot_raws'],render_window=[0,198],time='(raw-99)*0.02; physical timestamps retained; heldout gap preserved',
        CPU_inputs=REMOTE+'/inputs',engine_template='/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val029_v44_20261108_r1/engine',
        native_config_template='/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val029_v44_20261108_r1/config.json',
        new_native_root='/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val065_native10k_20261108_r1',
        native_control=dict(name='V1 无编辑直通',formal_updates=10000,seed=0,start_checkpoint=None,learning_rate_horizon=30000,
            rendering='call original gaussian_renderer.render directly',delete_actor_ids=[],
            semantic_rebinding=False,restored_vehicle_insert=False,world_road_insert=False,image_composite=False,
            architecture='fresh native static + dynamic + environment; original joint rasterizer',
            inherited_losses='same native Full5 GN11: full scene RGB1, static RGB1, full raw dynamic alpha MSE100; inherited SGCM/NPS and active regularizers'),
        GPU_entry_gate=['four-frame source segmentation QA; full target silhouette and neighbor boundary',
            'all159 BUILD instance and sky/semantic inputs; no HELD fitting',
            '600s real forward/backward/finite-loss/memory/data smoke',
            'save full checkpoint; evaluate original no-edit and target delete; exact parameters/Adam/RNG/sampler/SGCM/EMA restore; two completed resumed updates and next-entry check',
            'only then fresh formal10K; smoke updates accounted separately'],
        edit_comparisons=['source-ID a9 deletion from native10K',
            'current method shared ownership and surface/road handoff; new-scene assets only, separately tracked from V1'],
        export_requirements=['V1 no-edit KEEP; current-method KEEP; a9 DELETE on aligned camera/time',
            'RGB,alpha,depth_sum, target/static/other-actor/road alphaT separately; KEEP-delete-KEEP repeat',
            'allBUILD frame regional denominators; HELD images only decoded after fit for evaluation',
            'draggable HTML; target and preserved regions; source mapping and provenance'],
        unresolved_inputs=['instance masks and sky/semantic maps require GPU',
            'background nominations await target silhouette and road semantics',
            'hidden dark-vehicle correspondence not yet verified; do not force reuse of029 SUV recovery'],
        scene029=dict(native_no_edit_model='/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val029_v44_20261108_r1/formal10k',
            formal_updates_already_completed=10000,new_training_updates=0,
            route='unmodified native renderer; existing KEEP API is a direct return of that function',
            original_pngs_retrieved=158,raw_window=[119,197],missing_V1_heldout_raws=list(range(109,119)),keep_and_red_delete_separate=True,pending_gpu_exports=['10 HELD original V1 renders and source-ID red deletion'],camera_manifest_CPU_pose_maxabs=2.4091839634365897e-14),
        GPU_queries=0,new_training_updates=0,render_calls=0,human_verdict=None,failure_ledger_refs=[],failure_ledger_delta='none')
    write(OUT/'experiment_contract.json',contract)
    (OUT/'assets/input').mkdir(exist_ok=True)
    frames=[]
    for raw in selection['build']:
        src=SOURCE/f'frames/{raw:06d}.jpg';dest=OUT/f'assets/input/{raw}.jpg';shutil.copy2(src,dest)
        f=media['frames'][raw]
        frames.append(dict(raw=raw,timestamp_micros=f['timestamp_micros'],file=f'assets/input/{raw}.jpg',
            sha256=hashlib.sha256(src.read_bytes()).hexdigest(),boxes=[dict(
            role=role,label=label,box=next(b['projection_comparison']['official_box'] for b in f['boxes'] if b['track']==track))
            for role,label,track in [('target','删除目标：前方白车',selection['target']['track_id']),('preserve','保持：右侧白色皮卡',preserve)]
            if any(b['track']==track and b['official_FRONT_visible'] for b in f['boxes'])]))
    data=dict(frames=frames,pilot_raws=selection['pilot_raws'])
    (OUT/'assets/source.js').write_text('window.CPU_CASE='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';',encoding='utf8')
    svg='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 270"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10" fill="#59738b"/></marker></defs><rect width="1120" height="270" rx="12" fill="#f4f8fb"/><g stroke="#59738b" stroke-width="2" fill="none" marker-end="url(#arrow)"><path d="M184 107H220"/><path d="M407 107H443"/><path d="M650 107H687"/><path d="M891 107H930"/><path d="M1018 156V173H595V187"/><path d="M1018 173H952V187"/></g><g fill="#e2eef7" stroke="#9cb6ca"><rect x="15" y="56" width="168" height="100" rx="10"/><rect x="221" y="56" width="186" height="100" rx="10"/><rect x="444" y="56" width="206" height="100" rx="10"/></g><g fill="#fef3dc" stroke="#bba67d" stroke-dasharray="5 3"><rect x="688" y="56" width="203" height="100" rx="10"/><rect x="931" y="56" width="174" height="100" rx="10"/></g><g fill="#fef3dc" stroke="#bba67d" stroke-dasharray="5 3"><rect x="444" y="188" width="306" height="52" rx="8"/><rect x="800" y="188" width="305" height="52" rx="8"/></g><g font-family="Microsoft YaHei, sans-serif" font-size="16" fill="#233f54" text-anchor="middle"><text x="99" y="89">原始驾驶录像</text><text x="99" y="118">199帧 · 图像与测距</text><text x="314" y="89">先划分再解码</text><text x="314" y="118">构建159 / 留出40</text><text x="547" y="89">CPU真实输入</text><text x="547" y="118">标定 · 测距 · 分割提示</text><text x="790" y="89">GPU阶段（未启动）</text><text x="790" y="118">分割QA → 600秒smoke</text><text x="1018" y="89">原生完整训练10K</text><text x="1018" y="118">同一份持久三维场景</text><text x="597" y="220">V1 无编辑直通渲染</text><text x="952" y="220">编辑通路：KEEP／删除白车</text><text x="121" y="28" font-size="18" font-weight="bold">输入、训练与两条输出通路</text></g></svg>'''
    (OUT/'architecture.svg').write_text(svg,encoding='utf8')
    best=background['pairs'][0]
    # 真实照片上的同一世界回波提名；未用分割或虚构删除图。
    canvas=Image.new('RGB',(1200,670),'#f4f8fb')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    count=min(16,len(best['points']))
    selected=[best['points'][i] for i in np.linspace(0,len(best['points'])-1,count,dtype=int)]
    for i,(role,raw,key) in enumerate([('参考实拍',best['reference_raw'],'reference_uv_native'),('待删白车所在实拍',best['query_raw'],'query_uv_native')]):
        im=Image.open(SOURCE/f'frames/{raw:06d}.jpg').convert('RGB')
        pixels=np.array([p[key] for p in selected])/2
        draw=ImageDraw.Draw(im)
        box=next(b['projection_comparison']['official_box'] for b in media['frames'][raw]['boxes'] if b['track']==selection['target']['track_id'])
        draw.rectangle(box,outline='#ff5656',width=3)
        for n,(x,y) in enumerate(pixels):
            draw.ellipse((x-4,y-4,x+4,y+4),outline='#13f5ad',width=2)
            if n%3==0:draw.text((x+5,y-15),str(n+1),font=small,fill='#13f5ad',stroke_width=1,stroke_fill='#102a33')
        canvas.paste(im.resize((600,400)),(i*600,36))
        label=ImageDraw.Draw(canvas)
        label.text((i*600+10,6),f'{role} · FRONT · 原始帧{raw} · {raw/10:.1f}秒',font=font,fill='#233f54')
        lo=np.maximum(pixels.min(0)-[25,30],[0,0]);hi=np.minimum(pixels.max(0)+[25,30],[960,640])
        crop=im.crop((*lo,*hi));crop.thumbnail((580,180))
        canvas.paste(crop,(i*600+10,462))
        label.text((i*600+10,439),'同号绿点：同一真实回波的世界坐标投影；红框仅定位白车',font=small,fill='#365a70')
    canvas.save(OUT/'assets/background_correspondence.png')
    metrics=f"{verification['build_frames']} 构建帧，{verification['heldout_frames']} 留出帧；{verification['measured_points']:,} 个真实测距点；{prompts['target_positive_frames']}/{verification['build_frames']} 帧有白车测距提示。"
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Waymo scene-065 · 白车删除 CPU 准备</title><style>body{margin:0;background:#f4f7fa;color:#203649;font:16px/1.65 system-ui,"Microsoft YaHei"}main{max-width:1180px;margin:auto;padding:32px 24px}h1{font-size:30px}h2{font-size:22px;margin-top:30px}p{max-width:1050px}.panel{background:white;border:1px solid #d8e3eb;border-radius:10px;padding:20px;margin:18px 0}.row{display:flex;gap:16px;align-items:center;flex-wrap:wrap}.row input{flex:1;min-width:200px}button,select{font:inherit;padding:5px 12px;border:1px solid #b8c9d7;background:white;border-radius:5px}canvas{width:100%;height:auto;background:#0b1422;display:block}.grid{display:grid;grid-template-columns:1.6fr 1fr;gap:18px}.note{font-size:14px;color:#5c7385}img{max-width:100%;height:auto}code{word-break:break-all;font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #dae4ed}.pill{display:inline-block;background:#e1eee6;padding:3px 10px;border-radius:5px}a{color:#146798}@media(max-width:750px){.grid{grid-template-columns:1fr}main{padding:14px}}</style><main><span class="pill">CPU准备完成 · GPU未启动</span><h1>删除前方白车，先看清输入与无编辑参照</h1><p>这是用户指定的 Waymo 验证集 scene-065。待删除车辆是同车道正前方的白色车，右侧白色皮卡用于检查保持。它在199帧都有前相机标注，中后段明显大于上一街段的被遮挡小车。较晚时车体下缘接近或超过画面边界，因此保留早期完整入镜时刻。</p><p>@@METRICS@@ 留出原始帧79–118未解码为训练图像或点云，也未参与分割提示与背景搜索。</p><img src="architecture.svg" alt="从199帧原始观测到159帧CPU输入、后续分割smoke和独立无编辑10K的组件图"><section class="panel"><h2>逐帧实拍：只展示构建时刻</h2><p class="note">以下没有分割或删除结果。彩色框是原始标注的二维投影，不代表真实车体轮廓。左图完整视野，右图放大待删除白车。</p><div class="row"><button id="prev">←</button><input id="frame" type="range" min="0" max="158" value="60" step="1" aria-label="构建帧时间轴"><button id="next">→</button><strong id="label"></strong></div><div class="row"><span>分割核验帧：</span><button data-raw="30">30</button><button data-raw="60">60</button><button data-raw="150">150</button><button data-raw="198">198</button></div><div class="grid"><canvas id="full" width="960" height="640"></canvas><canvas id="crop" width="480" height="480"></canvas></div><p id="source" class="note"></p></section><h2>“完整重建”采用无编辑直通</h2><p>同一份构建观测从头训练原生完整场景10K，直接调用原渲染器输出。该参照绕过对象重绑、恢复车辆插入、道路补片和图像拼接。原生重建、当前方法保留场景和删除白车分别保存，不能用编辑后的KEEP结果冒充直通参照。seed=0，学习率仍沿用30K时钟，正式更新在10K停止，smoke另记。</p><p>scene-029 已有这一独立原生10K运行。本轮确认79个构建时刻的KEEP与删红车共158张原PNG，已分别接入现有子页。另10个留出时刻的旧导出经过共享归属与行打包，已从V1列排除并留空，需要GPU另行直通导出；没有为展示重复训练，也没有把原生删车称为无编辑上限。</p><h2>背景显露证据仍需分割核验</h2><p>搜索只使用构建时刻的原始测距点。先检查参考时刻的相机深度与其他框遮挡，再把同一个世界坐标点投回白车所在位置。@@BACKGROUND@@ 抽样图中不少点落在车体两侧，框内投影不能证明被车身遮挡。这些近地回波尚未通过白车真实轮廓和道路语义核验，不能作为独立删除真值。白车前方深色车辆的完整表面对应也尚未确认。</p><h2>进入GPU阶段之前已固定什么</h2><table><tr><th>项</th><th>固定内容</th></tr><tr><td>目标与保持</td><td>前方白车a9删除；右侧白色皮卡a7保持，原始轨迹作为映射键。</td></tr><tr><td>数据</td><td>原始前相机1920×1280；拟合/渲染480×320；159构建帧、40连续留出帧。</td></tr><tr><td>分割与smoke</td><td>先核验30/60/150/198四帧，再补齐全构建帧分割；600秒真实smoke必须覆盖保存、评价、精确恢复和下一阶段入口。</td></tr><tr><td>正式预算</td><td>fresh10K，无旧checkpoint；原生无编辑直通与删除查询分开，额外恢复/补路计算另记。</td></tr><tr><td>尚缺</td><td>实例与天空/道路语义掩码、GPU接线验收、真实删除结果；本轮均未生成。</td></tr></table><details class="panel"><summary>原始身份与可重跑资料</summary><p>context：<code>15021599536622641101_556_150_576_150</code><br>白车a9：<code>NTjq-OhZ_97OMJ4xwJAxXg</code><br>皮卡a7：<code>KOsYCOB6W_n0DzZsyDdJ-w</code></p><p><a href="experiment_contract.json">冻结合同</a> · <a href="evidence/inputs/verification.json">CPU验收</a> · <a href="evidence/inputs/background_nominations.json">背景候选</a> · <a href="target_timeline.json">目标逐帧登记</a></p><p class="note">本页只核验输入，没有跨场景编辑成功或人工质量验收。完整数据在远端隔离目录，本地保留元数据、提示和影像证据。</p></details></main><script src="assets/source.js"></script><script src="assets/viewer.js"></script></html>'''
    html=html.replace('@@METRICS@@',metrics).replace('@@BACKGROUND@@',f"稀疏提名最多（不等于已确认的显露支持）的是查询帧{best['query_raw']}、参考帧{best['reference_raw']}，共{best['returns']}个回波、{best['cells_2px_render']}个两像素格。")
    html=html.replace('<h2>进入GPU阶段之前已固定什么</h2>','<img src="assets/background_correspondence.png" alt="同一世界测量点在参考和查询实拍中的投影提名，仍需轮廓与道路语义核验"><h2>进入GPU阶段之前已固定什么</h2>')
    (OUT/'index.html').write_text(html,encoding='utf8')
    js="""(()=>{'use strict';const D=window.CPU_CASE,$=id=>document.getElementById(id);let index=60,epoch=0;const full=$('full'),crop=$('crop');function draw(){const turn=++epoch,f=D.frames[index],im=new Image();$('frame').value=index;$('label').textContent=`原始帧${f.raw} · ${(f.raw/10).toFixed(1)}秒`;$('prev').disabled=index===0;$('next').disabled=index===D.frames.length-1;$('source').textContent='同帧实拍JPEG预览；79–118连续留出帧不加载。';for(const c of [full,crop]){const ctx=c.getContext('2d');ctx.fillStyle='#0b1422';ctx.fillRect(0,0,c.width,c.height);}im.onload=()=>{if(turn!==epoch)return;const ctx=full.getContext('2d');ctx.drawImage(im,0,0,960,640);ctx.font='18px Microsoft YaHei';for(const b of f.boxes){const [x0,y0,x1,y1]=b.box;ctx.strokeStyle=b.role==='target'?'#ff716d':'#48dced';ctx.lineWidth=3;ctx.strokeRect(x0,y0,x1-x0,y1-y0);ctx.fillStyle=ctx.strokeStyle;ctx.fillText(b.label,x0,Math.max(22,y0-7));}const box=f.boxes.find(b=>b.role==='target').box,pad=25,x=Math.max(0,box[0]-pad),y=Math.max(0,box[1]-pad),w=Math.min(960,box[2]+pad)-x,h=Math.min(640,box[3]+pad)-y,scale=Math.min(480/w,480/h);crop.getContext('2d').drawImage(im,x,y,w,h,(480-w*scale)/2,(480-h*scale)/2,w*scale,h*scale);};im.onerror=()=>{if(turn===epoch)$('source').textContent='当前帧图像缺失，未借用其他帧。';};im.src=f.file;}function set(i){index=Math.max(0,Math.min(D.frames.length-1,Math.round(i)));draw();}$('frame').oninput=e=>set(Number(e.target.value));$('prev').onclick=()=>set(index-1);$('next').onclick=()=>set(index+1);document.querySelectorAll('[data-raw]').forEach(b=>b.onclick=()=>set(D.frames.findIndex(f=>f.raw===Number(b.dataset.raw))));document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.key==='ArrowLeft'){e.preventDefault();set(index-1);}if(e.key==='ArrowRight'){e.preventDefault();set(index+1);}});window.CpuCaseViewer={set,getRaw:()=>D.frames[index].raw};draw();})();"""
    (OUT/'assets/viewer.js').write_text(js,encoding='utf8')
    print(json.dumps(dict(status='BUILT_CPU_PREPARATION_DELIVERY',entry=str(OUT/'index.html'),source_images=len(frames),GPU_started=False),ensure_ascii=False))


if __name__=='__main__':
    main()
