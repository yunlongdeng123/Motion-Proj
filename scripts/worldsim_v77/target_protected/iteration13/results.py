"""r45 固定三臂结果交付：保留原生输出、同步视频和空白人工判定。"""
from common import *
import html, shutil, subprocess
import numpy as np
import cv2
from PIL import Image, ImageDraw

ARMS = ['adapter_off', 'all_unknown', 'conditioned']
LABELS = {'original': '原视频 / 合成任务真实 Y', 'input': '实际遮洞输入',
          'adapter_off': '原始 DriveEditor（分支关闭）',
          'all_unknown': '训练后分支 · 全未知条件', 'conditioned': '训练后分支 · O/N/U/Q 条件'}


def encode(frames, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    p = subprocess.run([ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{w}x{h}', '-r', '10', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2',
        '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
        input=np.stack(frames).astype('uint8').tobytes(), capture_output=True)
    if p.returncode: raise RuntimeError(p.stderr.decode())


def load_frames(path):
    files = sorted(path.glob('*.png'))
    assert len(files) == 10, (path, len(files))
    return np.stack([np.asarray(Image.open(p).convert('RGB')) for p in files])


def card_video(cid, key, caption):
    return f'<figure><figcaption>{caption}</figcaption><video preload="metadata" muted playsinline controls src="assets/{cid}/{key}.mp4"></video></figure>'


def main():
    plan = read(O/'manifest.json'); state = read(O/'evaluation/state.json')
    assert state['stage'] == 'complete_pending_visual_review' and len(state['completed']) == 36
    out = O/'review'; out.mkdir(exist_ok=True)
    if not (out/'conditions.html').exists(): shutil.copy2(out/'index.html', out/'conditions.html')
    old = (out/'conditions.html').read_text()
    architecture = old[old.index('<svg '):old.index('</svg>')+6]
    cases = [c for c in plan['cases'] if c['split'] != 'train']
    inventory = {c['case_id']: c for c in read(O/'condition_inventory.json')['cases']}
    observations = read(O/'visual_review.json') if (O/'visual_review.json').exists() else {'cases': []}
    notes = {c['case_id']: c for c in observations['cases']}
    cards=[]; rows=[]
    for c in cases:
        cid=c['case_id']; dest=out/'assets'/cid; dest.mkdir(parents=True,exist_ok=True)
        x=images(c,'rgb'); h=images(c,'hole')>0
        original=images(c,'target') if c['kind']=='synthetic' else x
        masked=x.copy(); masked[h]=127
        marked=original.copy()
        for i in range(10):
            contours,_=cv2.findContours(h[i].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(marked[i],contours,-1,(255,198,46),2)
        clips={'original': marked, 'input': masked}
        row={'case_id':cid,'kind':c['kind'],'scene':c['scene'],'metrics':{},'human_verdict':None}
        for arm in ARMS:
            path=O/'evaluation'/cid/arm
            clips[arm]=load_frames(path);clips[arm+'_native']=load_frames(path/'native')
            row['metrics'][arm]=read(path/'result.json')
        row['condition_vs_unknown_hole_rgb_MAE']=float(np.abs(clips['conditioned'].astype(float)-clips['all_unknown']).mean(-1)[h].mean()/255)
        row['output_change_checks']={}
        for left,right in [('conditioned','all_unknown'),('conditioned','adapter_off'),('all_unknown','adapter_off')]:
            delta=np.abs(clips[left+'_native'].astype(float)-clips[right+'_native'])
            row['output_change_checks'][left+'_vs_'+right]={'hole_rgb_MAE':float(delta.mean(-1)[h].mean()/255),
                'changed_hole_pixel_fraction':float((delta.max(-1)[h]>0).mean()),'max_channel_difference':float(delta[h].max())}
        for key,frames in clips.items():
            video=dest/(key+'.mp4')
            if not video.exists(): encode(frames,video)
        # 固定 f5 的等比例局部对照；同一个 crop 用于全部臂，不拉伸车型。
        ys,xs=np.where(h[5]); x0=max(0,int(xs.min())-80);x1=min(1024,int(xs.max())+81)
        y0=max(0,int(ys.min())-80);y1=min(576,int(ys.max())+81)
        tiles=[]
        for key in ['original','input']+ARMS:
            im=Image.fromarray(clips[key][5,y0:y1,x0:x1]);im.thumbnail((384,300))
            tile=Image.new('RGB',(384,330),(20,27,36));tile.paste(im,((384-im.width)//2,28+(300-im.height)//2))
            ImageDraw.Draw(tile).text((8,8),key,fill='white');tiles.append(tile)
        contact=Image.new('RGB',(384*5,330))
        for j,tile in enumerate(tiles):contact.paste(tile,(384*j,0))
        contact.save(dest/'f05_compare.jpg',quality=94)
        cov=inventory[cid]['hole_condition_fraction']
        note=notes.get(cid,{}).get('observation','待助手固定帧检查；人工 verdict 留空。')
        metrics=''
        if c['kind']=='synthetic':
            metrics='<p>洞内 MAE（越低越好）：'+' / '.join(f'{LABELS[a]} {row["metrics"][a]["hole_MAE"]:.4f}' for a in ARMS)+'</p>'
        verdicts=''.join(f'<label>{LABELS[a]} <select data-arm="{a}"><option value="">未评</option><option value="0">0 · 失败</option><option value="1">1 · 不太好</option><option value="2">2 · 可接受</option></select></label>' for a in ARMS)
        cards.append(f'''<article id="{cid}" data-case="{cid}"><h2>{cid} · {html.escape(c['scene'])} · {'真实 DELETE' if c['kind']=='real' else '有真实 GT 的合成遮挡'}</h2>
        <p>{html.escape(c['type'])}。黄色轮廓为本轮 H；真实任务删除该区域内指定目标，合成任务恢复被遮住的原视频内容。
        洞内 O {cov['O']:.1%} / N {cov['N']:.3%} / U {cov['U']:.1%}。</p>
        <div class="inputs">{card_video(cid,'original',LABELS['original'])}{card_video(cid,'input',LABELS['input'])}</div>
        <div class="controls"><button onclick="syncPlay(this)">五列同步播放</button><button onclick="syncPause(this)">暂停</button><label>帧 <input type="range" min="0" max="9" step="1" value="0" oninput="seekAll(this)"><output>0</output></label><label>速度 <select onchange="rateAll(this)"><option value="1">1×</option><option value="0.5">0.5×</option></select></label></div>
        <div class="outputs">{''.join(card_video(cid,a,LABELS[a]) for a in ARMS)}</div>{metrics}
        <p class="note">助手固定 f5 观察：{html.escape(note)} 单帧不判定时序通过；请同步播放及逐帧审核。</p>
        <details><summary>固定 f5 同位置局部对照</summary><img loading="lazy" src="assets/{cid}/f05_compare.jpg"></details>
        <details class="native"><summary>原生 diffusion 输出（未写回原图）</summary><div class="outputs">{''.join(card_video(cid,a+'_native',LABELS[a]+' · 原生') for a in ARMS)}</div></details>
        <div class="scores">{verdicts}<label>人工备注 <input data-note type="text" placeholder="人工评分和意见只由你填写"></label></div></article>''')
        rows.append(row)
    summary={'cases':rows,'human_verdict':None,'case_count':12,'windows':36,'videos':96,'frames_per_video':10,'fps':10}
    syn=[r for r in rows if r['kind']=='synthetic']
    summary['synthetic_case_equal_MAE']={a:float(np.mean([r['metrics'][a]['hole_MAE'] for r in syn])) for a in ARMS}
    summary['synthetic_protected_case_equal_MAE']={a:float(np.mean([r['metrics'][a]['protected_hole_MAE'] for r in syn if r['metrics'][a]['protected_hole_MAE'] is not None])) for a in ARMS}
    dump(O/'results_summary.json',summary);dump(out/'results_summary.json',summary)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 r45 · 条件分支三组对照</title>
    <style>body{font:16px/1.65 system-ui;background:#121922;color:#e4ecf5;margin:auto;padding:24px;max-width:1700px}h1,h2{line-height:1.3}a{color:#8bc5ff}article{padding:20px;background:#1c2633;margin:24px 0;border-radius:10px}video,img{width:100%;display:block}figure{margin:0}figcaption{min-height:2em}.inputs,.outputs{display:grid;gap:10px}.inputs{grid-template-columns:repeat(2,1fr);max-width:1120px;margin:auto}.outputs{grid-template-columns:repeat(3,1fr)}.note{background:#293649;padding:12px}.controls,.scores,nav{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:14px 0}button,select,input{font:inherit}button,select{padding:5px}svg{width:100%;height:auto}summary{cursor:pointer}details{margin:14px 0}small{color:#b9c6d5}@media(max-width:800px){.inputs,.outputs{grid-template-columns:1fr}}</style>
    <h1>r45：小条件分支能否帮助真实 DELETE？</h1><p>160 步训练完成，只训练 157,888 个 Adapter 参数，DriveEditor 主干冻结。固定 4 个合成开发例 + 8 个真实 DELETE 开发例，三组共 36 窗。最终160步 checkpoint；seed42、25步、10帧，同输入、mask 和写回规则。</p>'''
    page+=architecture
    reused=sum('reused_from' in v for r in rows for v in r['metrics'].values())
    page+=f'<p>对照总计36窗：{36-reused}窗本轮新增推理，{reused}窗复用已核对输入相同且不依赖条件的原模型结果。复用来源记录在逐例JSON中。</p>'
    if (O/'time_fix_audit.json').exists():
        page+='<p class="note">本轮先修复关键帧时间边界：关键帧直接取关联 sample 标注，中间帧按官方 SDK 插值。A013、A007 原首帧全灰是标注被漏掉；修复后已恢复。A022 首帧洞内没有可靠背景返回，仍保持未知。实际比较发现训练条件也改变，因此按相同预算重新训练 160 步；旧 r45 结果保留，不能作为完整条件的负结果。</p>'
    if observations.get('conclusion'):
        page+='<p class="note"><strong>本轮结果：</strong>'+html.escape(observations['conclusion'])+'</p>'
    page+='<p>4个合成DEV的case等权洞内MAE（0–1，越低越好）：'+' / '.join(f'{LABELS[a]} {summary["synthetic_case_equal_MAE"][a]:.6f}' for a in ARMS)+'</p>'
    page+='''<p class="note">先比较右侧「实际条件」与中间「全未知」，判断几何提示本身的增量；再与左侧原模型比较。全未知是同一训练分支的输入消融，不是另训的等容量模型。原生输出不是 factual 原位重建。真实 DELETE 没有去车 GT，不能用合成误差代替真实任务收益。</p>'''
    n_fractions=[inventory[c['case_id']]['hole_condition_fraction']['N'] for c in cases if c['kind']=='real']
    page+=f'<p>O：GT 车辆包络保守内核；N：实测 LiDAR 背景返回；U：未知；Q：启发式置信度。GT 相机与车辆框是本轮 POC 的辅助输入。N 占各真实 case 删除洞的 {min(n_fractions):.3%}–{max(n_fractions):.3%}，本轮对背景约束很弱，不能据此否定充分背景条件。全部评测属于已曝光 DEV。</p>'
    page+='''
    <p><a href="conditions.html">保留 CPU 条件图和全部24例输入说明</a> · <a href="results_summary.json">逐例原始指标</a> · <button onclick="exportScores()">导出人工评分 JSON</button></p>'''
    page+='<nav>'+''.join(f'<a href="#{c["case_id"]}">{c["case_id"]}</a>' for c in cases)+'</nav>'+''.join(cards)
    page+='''<script>
    const storageKey='v77-r45-human-review-v1';let saved={};try{saved=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}
    function primary(el){return [...el.closest('article').querySelectorAll('video')].filter(v=>!v.closest('.native'))}
    function syncPlay(el){const vv=primary(el);const t=vv[0].currentTime>=.9?0:vv[0].currentTime;vv.forEach(v=>{v.currentTime=t;v.play().catch(()=>{})})}
    function syncPause(el){primary(el).forEach(v=>v.pause())}
    function seekAll(el){primary(el).forEach(v=>{v.pause();v.currentTime=Number(el.value)/10+.001});el.nextElementSibling.value=el.value}
    function rateAll(el){primary(el).forEach(v=>v.playbackRate=Number(el.value))}
    document.querySelectorAll('article[data-case]').forEach(a=>{const id=a.dataset.case;const s=saved[id]||{};a.querySelectorAll('[data-arm]').forEach(e=>{e.value=s[e.dataset.arm]??'';e.onchange=()=>{saved[id]||={};saved[id][e.dataset.arm]=e.value===''?null:Number(e.value);localStorage.setItem(storageKey,JSON.stringify(saved))}});const n=a.querySelector('[data-note]');n.value=s.note||'';n.oninput=()=>{saved[id]||={};saved[id].note=n.value;localStorage.setItem(storageKey,JSON.stringify(saved))}});
    function exportScores(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({run:'r45',reviewer:'human',scores:saved},null,2)],{type:'application/json'}));a.download='v77_r45_human_review.json';a.click();URL.revokeObjectURL(a.href)}
    </script><small>WS-V77-TARGET-PROTECTED-20260929 / r45 · 10fps；每窗10帧约1秒；人工结果保存在当前浏览器。</small></html>'''
    (out/'index.html').write_text(page.replace('r45',plan['run_id']))
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
