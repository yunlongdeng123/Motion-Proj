"""复用既有报告结构；CPU准备和GPU结果共用入口，未跑结果明确留空。"""
from common import *
import html, shutil, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ARCH='''<svg viewBox="0 0 1060 200" role="img" aria-label="r48模块与数据流">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#7cabc8"/></marker></defs>
<g fill="#18334a" stroke="#5683a3"><rect x="10" y="20" width="200" height="55" rx="6"/><rect x="10" y="110" width="200" height="55" rx="6"/><rect x="270" y="20" width="215" height="55" rx="6"/><rect x="270" y="110" width="215" height="55" rx="6"/><rect x="545" y="65" width="220" height="60" rx="6"/><rect x="820" y="65" width="225" height="60" rx="6"/></g>
<g fill="#e7f0f8" font-size="15" font-family="Arial, Microsoft YaHei" text-anchor="middle"><text x="110" y="44">邻帧 RGB · 剔除A</text><text x="110" y="65">LiDAR / pose / tracks</text><text x="110" y="134">目标 RGB + 固定H</text><text x="110" y="155">同一参数化指令</text><text x="378" y="44">冻结 VAE + RGB 注意力</text><text x="378" y="65">BEV / 2D几何编码</text><text x="378" y="134">官方 DriveEditor</text><text x="378" y="155">主干 / 3D支路冻结</text><text x="655" y="89">r47小条件分支 · 继续训练</text><text x="655" y="113">64步 → 验证 → 最多128步</text><text x="932" y="89">原生 DELETE → 固定α写回</text><text x="932" y="113">真实DEV + 已知GT对照</text></g>
<g fill="none" stroke="#7cabc8" stroke-width="2" marker-end="url(#a)"><path d="M210 47H270"/><path d="M210 137H270"/><path d="M485 47H515V82H545"/><path d="M485 137H515V108H545"/><path d="M765 95H820"/></g></svg>'''

JS='''function play(b){const vs=[...b.closest('article').querySelectorAll('video')];vs.forEach(v=>{v.currentTime=0;v.play().catch(()=>{})})}
function pause(b){b.closest('article').querySelectorAll('video').forEach(v=>v.pause())}
function seek(b){b.closest('article').querySelectorAll('video').forEach(v=>{v.pause();v.currentTime=Number(b.value)/10})}
const key='v77-r48-human';let scores={};try{scores=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
document.querySelectorAll('[data-score]').forEach(e=>{e.value=scores[e.dataset.score]?.score||'';e.onchange=()=>store(e)});
document.querySelectorAll('[data-note]').forEach(e=>{e.value=scores[e.dataset.note]?.note||'';e.onchange=()=>store(e)});
function store(e){const id=e.dataset.score||e.dataset.note;const card=e.closest('article');scores[id]={score:card.querySelector('[data-score]').value,note:card.querySelector('[data-note]').value};localStorage.setItem(key,JSON.stringify(scores))}
function exportScores(){const rows=Object.entries(scores).map(([id,x])=>({case_id:id.split('@')[0],checkpoint_step:Number(id.split('@')[1]),...x}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({run:'r48',rows},null,2)],{type:'application/json'}));a.download='r48_human_review.json';a.click();URL.revokeObjectURL(a.href)}'''


def review_sentence(value, field):
    if isinstance(value, str):return value
    if field=='comparison':return value['ranking']
    if field=='risk':
        level={'critical':'严重','high':'高','medium':'中','low_to_medium':'低至中'}.get(value['level'],value['level'])
        return level+'：'+value['basis']
    if field=='overall':return value['judgment']
    if field=='hypothesis':return value['name']+'。'+value['why']
    raise ValueError(field)


def encode(folder, dest, native=False):
    import imageio_ffmpeg
    source=folder/'native' if native else folder
    frames=np.stack([np.asarray(Image.open(source/f'{f:05}.png').convert('RGB')) for f in range(10)])
    ff=imageio_ffmpeg.get_ffmpeg_exe()
    command=[ff,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1024x576','-r','10',
        '-i','-','-an','-c:v','libx264','-threads','1','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
    r=subprocess.run(command,input=frames.tobytes(),capture_output=True);assert r.returncode==0,r.stderr.decode()
    probe=subprocess.run([ff,'-v','error','-i',str(dest),'-map','0:v:0','-f','null','-'],capture_output=True)
    assert probe.returncode==0,probe.stderr.decode()


def main():
    assert read(O/'preflight.json')['CPU_ready']
    plan=read(O/'manifest.json');checks={r['case_id']:r for r in read(O/'input_checks.json')['rows']}
    final_state=O/'evaluation/step_0128/state.json'
    completed=final_state.exists() and read(final_state)['stage']=='complete'
    if completed:
        from summarize import main as summarize
        summarize()
    summary=read(O/'summary.json') if (O/'summary.json').exists() else None
    metrics={r['case_id']:r for r in summary['evaluation']} if summary else {}
    assistant_path=O/'assistant_image_review.json'
    assistant=read(assistant_path) if assistant_path.exists() else None
    assistant_cases={r['case_id']:r for r in assistant['cases']} if assistant else {}
    byid=cases();user=read(O/'user_review/human_review.json');scores={(r['version'],r['case_id']):r for r in user['records']}
    out=O/'review';out.mkdir(exist_ok=True);cards=[];new_videos=[]
    for cid in EVAL_128+TRAIN_IDS:
        c=byid[cid];dest=out/'assets'/cid;dest.mkdir(parents=True,exist_ok=True)
        source=PARENT/'review/assets'/cid
        for pattern in ('reference_*.png','original_*.jpg','target_*.png','mask_*.png','geometry_axes_*.png','bev_*.png'):
            for p in source.glob(pattern):shutil.copy2(p,dest/p.name)
        refs=read(PARENT/'inputs'/cid/'references.json')['references']
        slots=checks[cid]['reference_slots'];token=sum(r['valid_attention_tokens'] for r in slots if r['role']=='protected_actor_appearance')
        body=f'<h2>{cid} · {html.escape(c["scene"])}</h2><p>{c["split"]} · 原查询索引 {c["frame_indices"]} · 10帧/10fps，一秒固定窗。</p>'
        if cid in EVAL_128:
            for version in ('r46','r47'):
                score=scores.get((version,cid))
                if score:
                    body+=f'<p><b>{version} 原人工分：{score["human_score"] if score["human_score"] is not None else "未填"}</b>；RGB足够：{score["RGB_prior_sufficient"] or "未填"}；OCC足够：{score["OCC_prior_sufficient"] or "未填"}。{html.escape(score["note"] or "")}</p>'
            if cid=='A022':body+='<p>r46人工2分正控制；r47未评分。本轮必须检查回退。</p>'
            old=PARENT/'review/gpu_assets'/cid
            for name in ('original','masked','baseline','baseline_native','RGB_and_geometry','RGB_and_geometry_native'):
                shutil.copy2(old/f'{name}.mp4',dest/f'{name}.mp4')
            def video(name,label):
                return f'<figure><figcaption>{label}</figcaption><video controls muted playsinline preload="metadata" src="assets/{cid}/{name}.mp4"></video></figure>'
            available=[]
            for step in (64,128):
                folder=O/'evaluation'/f'step_{step:04}'/cid
                if not (folder/'result.json').exists():continue
                available.append(step)
                for native in (False,True):
                    name=f'r48_{step}'+('_native' if native else '')
                    path=dest/(name+'.mp4')
                    if not path.exists():encode(folder,path,native);new_videos.append(str(path))
            third=video(f'r48_{available[-1]}',f'r48 · {available[-1]}步') if available else '<figure class="pending"><figcaption>r48 · 本轮结果待生成</figcaption><p>此case尚无本轮采样，不能展示新效果。64与128步均固定验证。</p></figure>'
            body+='<div class="three">'+video('baseline','r46 · 官方原权重 + r21完整SAM')+video('RGB_and_geometry','r47 · RGB + BEV + 2D几何')+third+'</div>'
            body+='<p><button onclick="play(this)">同步播放</button> <button onclick="pause(this)">暂停</button> 帧 <input type="range" min="0" max="9" value="0" oninput="seek(this)"></p>'
            body+='<details><summary>原RGB、实际遮洞输入、原生／写回对照</summary><div class="three">'+video('original','原RGB · 蓝H边界')+video('masked','实际输入 · H内先擦除')+'</div><div class="three">'+video('baseline_native','r46原生 DELETE')+video('RGB_and_geometry_native','r47原生 DELETE')+'</div>'
            for step in available:body+='<div class="three">'+video(f'r48_{step}',f'r48 {step}步 · 固定α写回')+video(f'r48_{step}_native',f'r48 {step}步 · 原生 DELETE')+'</div>'
            body+='</details>'
            if 'f05_native_final' in checks[cid]:
                evidence=checks[cid]['f05_native_final']
                body+=f'<p>保存PNG精确核验：r47 f05洞内α=1区域占 {evidence["alpha_one_H_fraction"]:.1%}，这里原生与写回完全相同。它帮助区分融合，但不能单独证明哪个模型模块产生残影。</p>'
            if available:
                score_id=f'{cid}@{available[-1]}'
                body+=f'<p>r48 {available[-1]}步人工分 <select data-score="{score_id}"><option value="">未评</option><option>0</option><option>1</option><option>2</option></select> <input data-note="{score_id}" placeholder="后车结构/目标残影/幻觉/时序"></p>'
                frame=5;req=request(c);yy,xx=np.where(req.edit_mask[frame])
                box=(max(0,int(xx.min())-60),max(0,int(yy.min())-45),min(1024,int(xx.max())+61),min(576,int(yy.max())+46))
                views=[('Original / GT',images(c,'target')[frame] if c['kind']=='synthetic' else req.target_rgb[frame])]
                for label,folder in [('r46',PARENT/'evaluation'/cid/'baseline'),('r47',PARENT/'evaluation'/cid/'RGB_and_geometry')]+[
                    (f'r48 {step}',O/'evaluation'/f'step_{step:04}'/cid) for step in available]:
                    views.append((label,np.asarray(Image.open(folder/f'{frame:05}.png').convert('RGB'))))
                strip=Image.new('RGB',(320*len(views),245),'#102131');draw=ImageDraw.Draw(strip)
                for j,(label,array) in enumerate(views):
                    draw.text((j*320+8,6),cid+' f05 '+label,fill='white')
                    crop=ImageOps.contain(Image.fromarray(array).crop(box),(320,215))
                    strip.paste(crop,(j*320+(320-crop.width)//2,28+(215-crop.height)//2))
                strip.save(dest/'f05_compare.jpg',quality=94)
                body+=f'<details open><summary>f05 固定同位置放大 · 单帧不能证明时序</summary><img loading="lazy" src="assets/{cid}/f05_compare.jpg"></details>'
            if cid in assistant_cases:
                row=assistant_cases[cid]
                body+='<details open><summary>GPT-5.6 Sol · xhigh · 独立图像 review（非人工分）</summary><ul>'
                body+=''.join('<li>'+html.escape(str(v))+'</li>' for v in row['observations'])
                body+='</ul><p><b>视觉对比：</b>'+html.escape(review_sentence(row['visual_comparison'],'comparison'))+'</p>'
                body+='<p><b>回退风险：</b>'+html.escape(review_sentence(row['regression_risk'],'risk'))+'</p>'
                body+='<p><b>边界：</b>'+html.escape(str(row['uncertainty']))+'</p></details>'
            if cid in metrics and c['kind']=='synthetic':
                body+='<details><summary>辅助记录：合成 GT 像素误差，不能代替图像 review</summary><p>真实GT洞内MAE（0–1，10帧合并）：'+ '；'.join(
                    r['version']+f' {r["GT_H_MAE"]:.6f}' for r in metrics[cid]['versions'])+'。这些数值不用于本轮视觉质量排序。</p></details>'
        body+=f'<p>实际可用保护车参考 token 合计 {token}；这是接收范围，不是同一主B的清晰车身面积。绿O仍是GT框proxy，灰U不等于空地。</p>'
        if cid in ('A034','A061_w08'):body+='<p>可靠外观检查组：slot0/1提供主要后车部分视角；其他slot可能属于不同保护车或场景上下文，不能把总token当同一实体。</p>'
        if cid=='A048':body+='<p>弱参考组：slot3主要是墙面，实际有效保护token为0；其余参考仍很局部。</p>'
        if cid=='A041_w10':body+='<p>用户标RGB足够；当前进入模块的保护参考仅12 token、局部车身。两种口径分别保留，尚不能据此认定失败只因RGB不足。</p>'
        body+='<details open><summary>实际参考 · 保比例、先剔除A、灰padding无效</summary><div class="six">'
        for r in refs:
            slot=r['reference_slot'];detail=slots[slot]
            body+=f'<figure><figcaption>slot{slot} · {html.escape(r["camera"])} · {html.escape(r["role"])}<br>有效token {detail["valid_attention_tokens"]}'+(' · 无效占位' if r['padding'] else '')+f'</figcaption><img loading="lazy" src="assets/{cid}/reference_{slot:02}.png"></figure>'
        body+='</div></details><details><summary>f00 / f05 / f09 · 原图、独立H、几何、BEV</summary>'
        for f in (0,5,9):
            body+='<div class="four">'+''.join(f'<figure><figcaption>f{f:02} · {label}</figcaption><img loading="lazy" src="assets/{cid}/{name}_{f:02}.{ext}"></figure>' for name,ext,label in (
                ('original','jpg','黄A包络 / 蓝H'),('mask','png','真实编辑H'),('geometry_axes','png','绿保留框 / 蓝实测 / 灰未知'),('bev','png','BEV proxy')) )+'</div>'
        body+='</details>'
        diagnostic=O/'diagnostics'/cid
        if (diagnostic/'reference_encoding/result.json').exists():
            target=dest/'reference_encoding';shutil.copytree(diagnostic/'reference_encoding',target,dirs_exist_ok=True)
            body+='<details open><summary>冻结VAE：输入参考 → 编码后重构</summary><div class="four">'+''.join(
                f'<figure><figcaption>slot{s} · {name}</figcaption><img src="assets/{cid}/reference_encoding/{name}_{s:02}.png"></figure>'
                for s in (0,1) for name in ('source','decoded'))+'</div></details>'
        if (O/'diagnostics/state.json').exists() and cid in DIAGNOSTIC_IDS:
            body+='<details open><summary>r47权重下的修正消融：相同指令／几何，RGB正确、错配、移除</summary><div class="four">'
            module_rows=[]
            for label in plan['diagnostics']:
                folder=diagnostic/label;name='probe_'+label
                path=dest/(name+'.mp4')
                if not path.exists():encode(folder,path);new_videos.append(str(path))
                body+=video(name,label)
                native_name=name+'_native';native_path=dest/(native_name+'.mp4')
                if not native_path.exists():encode(folder,native_path,True);new_videos.append(str(native_path))
                body+=f'<details><summary>{label} 原生DELETE</summary>'+video(native_name,label+' · 未写回')+'</details>'
                response=read(folder/'result.json')
                shutil.copy2(folder/'result.json',dest/(name+'.json'))
                for r in response['module_response']:
                    if r['CFG']=='C':
                        module_rows.append(f'<tr><td>{label}</td><td>{r["sample_call"]}</td><td>{r["block"]}</td><td>{r["rgb_head_rms"]:.5g}</td><td>{r["geometry_head_rms"]:.5g}</td><td>{r["delta_to_activation"]:.5g}</td></tr>')
            body+='</div><p>错配只用于敏感性探针，不是合法条件或可推广结果；全未知C仍保留任务指令，UC清空新指令。</p><table><tr><th>条件</th><th>采样调用</th><th>UNet block</th><th>RGB head RMS</th><th>几何 head RMS</th><th>门控Δ/激活 RMS</th></tr>'+''.join(module_rows)+'</table><p>标量用于定位编码/融合响应；响应变强或视频像素变化不能直接证明保护车恢复成功。</p></details>'
        cards.append(f'<article id="{cid}">{body}</article>')
    phase='GPU结果 · 待人工审核' if completed else 'CPU准备完成 · 等待用户开启GPU'
    controller=read(O/'controller_state.json')
    if not completed and controller['stage']=='running':
        names={'diagnose':'参考编码与RGB响应','train64':'64步训练','eval64':'64步验证',
               'train128':'续训到128步','eval128':'128步验证','report':'视频报告生成'}
        phase='GPU短循环正在运行 · '+names.get(controller.get('phase'),'本轮计算')
    if controller['stage']=='stopped_on_error':phase='GPU工程中止 · 保留现场'
    counts=read(O/'preflight.json')
    image_summary=''
    if assistant:
        image_summary='<article><h2>独立图像 review · GPT-5.6 Sol / xhigh / 未启用 fast</h2><p>'+html.escape(review_sentence(assistant['overall'],'overall'))+'</p><p><b>下一模块假设：</b>'+html.escape(review_sentence(assistant['next_module_hypothesis'],'hypothesis'))+'</p><p>本节只依据抽帧图像，人工评分与视频时序判断仍留空。M003/M006与A022只用于验证，不进入训练。默认保留r46，本轮128步后已停止。</p><a href="assistant_image_review.json">完整独立评审记录</a></article>'
    page=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>v77 r48</title><style>body{{background:#0c1721;color:#e7edf5;font:15px/1.6 Arial,Microsoft YaHei;margin:22px}}main{{max-width:1500px;margin:auto}}a{{color:#8dd4ff}}article{{border:1px solid #345367;border-radius:8px;padding:18px;margin:20px 0}}figure{{margin:0;min-width:0}}figcaption{{padding:7px 0;color:#c4d9e6}}img,video{{width:100%;background:#111}}.three,.four,.six{{display:grid;gap:12px;margin:12px 0}}.three{{grid-template-columns:repeat(3,1fr)}}.four{{grid-template-columns:repeat(4,1fr)}}.six{{grid-template-columns:repeat(6,1fr)}}.pending{{background:#18334a;padding:12px}}summary,button{{cursor:pointer}}button,input,select{{font:inherit}}input[data-note]{{width:55%}}table{{border-collapse:collapse}}td,th{{border:1px solid #345367;padding:8px}}svg{{width:100%;max-height:240px}}@media(max-width:900px){{.three,.four,.six{{grid-template-columns:repeat(2,1fr)}}}}</style><main><h1>v77 r48：{phase}</h1><p>唯一run WS-V77-TARGET-PROTECTED-20260929/r48。旧r46/r47视频原样保留；此页不会把旧视频标为本轮新推理。</p>{ARCH}<p>CPU核验 {counts["cases"]}例 / {counts["frames"]}帧；从r47 step320继续，只训练M010/M013/M018/M042。主干冻结，原查询H/α、seed42、25采样步固定；真实DEV不进入训练。</p><p>消融已修正：所有条件组保持相同指令，CFG无条件支路另行清空。旧r47全未知组同时改变指令，旧数值仅作历史，不能用作纯先验增量。</p><p>先比较可靠参考经过冻结VAE后是否可用、四处融合残差和正确／错配／无RGB响应；64步立即验A034、A061、A022、M003/M006，工程/数值正常才继续到128步，加A048/A041。128后停止，不自动追加数据或训练。</p><p>判断重点：保护车结构、前景残影、目标缺席及A022回退。像素变化不是语义收益；单帧不能证明时序；本轮未引入最终未曝光测试。</p>{image_summary}<p><a href="preflight.json">CPU检查</a> · <a href="input_checks.json">输入/原生与写回证据</a> · <a href="human_review.json">新版人工原记录</a> · <a href="manifest.json">固定方案</a></p><nav>'+ ' · '.join(f'<a href="#{cid}">{cid}</a>' for cid in EVAL_128+TRAIN_IDS)+'</nav><p><button onclick="exportScores()">导出本轮人工评分</button></p>'+''.join(cards)+f'<script>{JS}</script></main></html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    for name in ('preflight.json','input_checks.json','manifest.json'):
        shutil.copy2(O/name,out/name)
    shutil.copy2(O/'user_review/human_review.json',out/'human_review.json')
    if assistant:
        shutil.copy2(assistant_path,out/'assistant_image_review.json')
    # 所有本地媒体/证据链接必须存在；新编码已实际解码。
    import re
    links=re.findall(r'(?:src|href)="([^"#]+)"',page)
    for link in links:assert (out/link).is_file(),link
    dump(O/'report_state.json',{'stage':'GPU_review_ready' if completed else 'CPU_review_ready',
        'media_and_evidence_links':len(links),'new_videos_decoded':new_videos,'human_verdict':None})
    print('REPORT_READY',len(links),flush=True)


if __name__=='__main__':main()
