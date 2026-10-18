"""Assemble completed ID render matrices. Incomplete old5 remains explicitly pending."""
from pathlib import Path
import argparse,json,shutil,subprocess,html

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path(__file__).parent/'renders')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'outputs/identity_comparison_20261018')
    p.add_argument('--videos',action='store_true');a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    ready={};frames=['0008%03d'%r for r in range(89,138)]
    for model in ['old3','old5','r15']:
        src=a.input/('renders_'+model);receipt=src/'render_receipt.json'
        if not receipt.is_file():continue
        data=json.loads(receipt.read_text(encoding='utf-8'))
        assert data['frames']==49 and data['parameter_updates']==0 and data['checkpoint_file_unchanged']
        assert data['actual_ID_ownership'] and all(r['delete_target_visible_alpha_exact_zero'] for r in data['records'])
        assert len(data['records'])==49*len(data['actors'])
        dest=out/'assets'/model;dest.mkdir(parents=True,exist_ok=True)
        for fid in frames:
            assert (src/fid).is_dir()
            (dest/fid).mkdir(exist_ok=True)
            for im in (src/fid).glob('*.png'):shutil.copy2(im,dest/fid/im.name)
            for actor in data['actors']:
                assert all((dest/fid/('a%02d_%s.png'%(actor,s))).is_file() for s in ['isolated','alpha','visible','delete'])
        shutil.copy2(receipt,dest/'render_receipt.json');ready[model]=data
    if a.videos:
        ff=shutil.which('ffmpeg')
        if not ff:raise FileNotFoundError('ffmpeg unavailable; PNG assets retained')
        vid=out/'videos';vid.mkdir(exist_ok=True)
        for model,r in ready.items():
            for actor in r['actors']:
                for mode in ['isolated','delete']+(['delete_no_shadow'] if model=='r15' else []):
                    target=vid/('%s_a%02d_%s.mp4'%(model,actor,mode))
                    if target.exists():continue
                    pattern=str((out/'assets'/model/('0008%03d')/('a%02d_%s.png'%(actor,mode))).resolve())
                    cmd=[ff,'-hide_banner','-loglevel','error','-threads','2','-filter_threads','1','-framerate','10',
                        '-start_number','89','-i',pattern,'-frames:v','49','-c:v','libx264','-threads','2','-crf','18',
                        '-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
                    subprocess.run(cmd,check=True)
    styles='''body{font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif;background:#0d1723;color:#dfebf8;margin:0 auto;max-width:1600px;padding:24px}a{color:#8ad1ff}h1{font-size:29px}h2{font-size:21px}select,button{font:inherit;background:#1c3249;color:inherit;border:1px solid #536f88;padding:7px;border-radius:6px;margin:6px}.note{background:#142b41;border:1px solid #405970;border-radius:8px;padding:15px}.row{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.panel{background:#101e2e;border:1px solid #354f6a}img{width:100%;display:block}.panel p{font-size:13px;padding:8px;margin:0}.small{font-size:14px;color:#a8bfd5}.na{min-height:150px;padding:20px;color:#b0bdd0}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #3a5067}svg{width:100%;height:auto}video{max-width:100%;width:480px;background:black}.videogrid{display:flex;flex-wrap:wrap;gap:20px}.videogrid figure{margin:0}@media(max-width:850px){.row{grid-template-columns:1fr 1fr}body{padding:12px}}'''
    arch='''<svg viewBox="0 0 1100 225" role="img" aria-label="旧基线补齐与R15冻结对照组件图"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#9ab4cf"/></marker></defs><g fill="#173149" stroke="#6c89a7"><rect x="10" y="18" width="190" height="66" rx="8"/><rect x="250" y="18" width="250" height="66" rx="8"/><rect x="550" y="18" width="240" height="66" rx="8"/><rect x="10" y="140" width="190" height="66" rx="8"/><rect x="250" y="140" width="250" height="66" rx="8"/><rect x="860" y="75" width="225" height="90" rx="8"/></g><g stroke="#9ab4cf" stroke-width="2" fill="none" marker-end="url(#a)"><path d="M200 51H250M500 51H550M790 51H825V100H860M200 172H250M500 172H825V140H860"/></g><g fill="#edf7ff" text-anchor="middle" font-family="sans-serif" font-size="16"><text x="105" y="45">同一 GN11 30K</text><text x="105" y="68">原始 LiDAR</text><text x="375" y="45">旧法：5 身份 / 原二值监督</text><text x="375" y="68">全新实例种子 + 原 30K 协议</text><text x="670" y="45">补齐 a6、a37 的旧基线</text><text x="670" y="68">保留历史 3 身份检查点</text><text x="105" y="166">R15 冻结 4K</text><text x="105" y="190">原三值监督</text><text x="375" y="166">同一真实 owner 独立 / 删除</text><text x="375" y="190">已验证 FP64 阴影适配器</text><text x="972" y="104">49 帧身份对照</text><text x="972" y="129">RGB / alpha / 删除</text><text x="972" y="153">三值状态与几何审计</text></g></svg>'''
    rows=''
    for model,label in [('old3','历史旧 3 身份 30K'),('old5','补齐旧 5 身份 30K'),('r15','冻结 R15 4K')]:
        r=ready.get(model);status=f"完成：49 × {len(r['actors'])} 身份／帧，参数更新 0" if r else '训练／渲染进行中，未展示未完成结果'
        rows+=f'<tr><td>{label}</td><td>{status}</td><td>{"、".join("a"+str(x) for x in r["actors"]) if r else "a5、a26、a18、a6、a37（待完成）"}</td></tr>'
    geometry_html=''
    geometry_path=a.input.parent/'tristate/old5_geometry/geometry.json'
    if 'old5' in ready and geometry_path.is_file():
        geo=json.loads(geometry_path.read_text(encoding='utf-8'));gr=''
        for actor in [6,37,5,26,18]:
            o=next(r for r in geo['actors'] if r['actor']==actor);r=next(r for r in geo['R15_frozen4k_geometry'] if r['actor']==actor)
            no=o['centers_outside_common_box_with_tolerance'];nr=r['centers_outside_common_box_with_tolerance']
            gr+=f"<tr><td>a{actor}</td><td>{o['old_owner_index']} ↔ {actor}</td><td>{o['initial_points']:,} → {o['final_points']:,}</td><td>{r['points']:,}</td><td>{no:,} / {o['final_points']:,} ({no/o['final_points']:.2%})</td><td>{nr:,} / {r['points']:,} ({nr/r['points']:.2%})</td></tr>"
        geometry_html='<h2>真实身份与几何对照</h2><table><tr><th>同一车辆</th><th>旧5 owner ↔ R15 owner</th><th>旧5 初始 → 30K 点数</th><th>原 R15 4K 点数</th><th>旧5 框外中心</th><th>原 R15 框外中心</th></tr>'+gr+'</table><p class="small">两版均以同一物理米坐标、同一 train46 中位真实车框统计，使用 0.00001m 容差避免归一化/FP32 的边界舍入被计为物理越界。精确无容差与旧原生框统计也保存在凭据中。旧版固定拓扑；两版初始化和训练长度不同，点数、框外比例均不是道路污染或资产质量真值。<a href="tristate/old5_geometry/geometry.json">全量统计与归属映射</a> · <a href="tristate/index.html">旋转查看旧5与原R15的Gaussian中心</a></p>'
        if all((out/f'a{actor:02d}_completed_old5_review.png').is_file() for actor in [6,37]):
            geometry_html+='<h2>同一 a6／a37：旧版也有资产外溢和删除残影</h2><p>下图固定 raw105／113／121，所有模型使用同一输入掩码确定局部展示窗口。旧5的 a6 独立 RGB／alpha 中可见连续的大面积下部支撑和亮白、蓝色条带；原 R15 的下部表现为零散碎片。a37 删除后两版都未留下干净道路，旧5有明亮车形残留，原 R15 关阴影后仍有较暗车形。这里是三个固定锚帧的工程观察，不给全49帧人工质量评分。</p><p>例如旧5的 a6 仅约3.73%中心越出共同车框，但其最大主轴σ达到约3.29m；少数大尺度点也能产生明显外溢。框外率低不能证明独立资产更干净。旧5是新增两个真实owner后的基线扩展，不能倒推历史旧3检查点已具备同样可分离归属。</p>'+''.join(f'<figure><a href="a{actor:02d}_completed_old5_review.png"><img src="a{actor:02d}_completed_old5_review.png" alt="a{actor}旧5与原R15同帧独立、alpha及删除对照"></a><figcaption>a{actor}：依次为观测RGB、旧5独立、原R15独立、旧5 alpha、原R15 alpha、旧5删除、原R15关阴影删除。点击放大；完整原帧和全49帧视频见下方。</figcaption></figure>' for actor in [6,37])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>旧30K＋30K与R15：五身份对照</title><style>STYLES</style>
<h1>旧 GN11 30K＋实例 30K，与 R15 4K 的五身份对照</h1>
<p>按同一真实身份核对 a6、a37、a5、a26、a18。<a href="tristate/index.html">进入 0／1／unknown 交互可视化与谱系审计</a>。</p>
<div class="note">历史旧版只有 a5、a26、a18 三个独立 owner，a6／a37 由静态/PVG 分支表达。新增五身份基线复用同一个 GN11 30K，按旧原始 LiDAR 方法重新初始化并训练实例 30K；它是<strong>补齐归属的基线扩展</strong>，不冒称历史检查点原本就有这两个 owner。R15 只读取冻结 4K。</div>
ARCH
<table><tr><th>模型</th><th>实际状态</th><th>可按真实身份独立／删除</th></tr>STATUS</table>
<h2>训练与输入范围</h2><p>新增五身份沿用旧版 100×完整可见实例 mask MSE、框约束、固定拓扑、固定姿态和 seed=0。背景与实例共同优化（background lr scale=0.1），环境冻结。a5／a26／a18 的初始位置和 RGB 与旧种子逐位一致；a6／a37 使用经 49 帧身份核对的原生完整 SAM 提议。R15 的融合三值掩码保持原样；两版的输入覆盖差异必须结合三值页面解释。未重新训练 GN11，未加载 R15 或旧实例已学习权重作为新增实例种子。</p>
GEOMETRY_TABLE
<h2>逐身份、逐帧检查</h2><label>身份 <select id="actor"><option value="6">a6</option><option value="37">a37</option><option value="5">a5</option><option value="26">a26</option><option value="18">a18</option></select></label><label>原始帧 <select id="frame"></select></label><button id="prev">上一帧</button><button id="next">下一帧</button>
<p class="small">每行从左到右：完整场景、仅该 owner 的 RGB、独立 alpha、全景中可见 alpha*T、删除该 owner。所有图片都是原始整帧输出，没有乘 GT mask 或按高度裁剪。点击图可查看原分辨率。旧版使用原生 SH 输出域，R15 线性 RGB 转 sRGB 显示；跨实现外观差异不单独作为质量评分。</p><div id="matrix"></div><div id="off"></div>
<h2>49 帧视频</h2><p class="small">固定 10 fps，原始 raw89—137，循环播放。视频只压缩 PNG 预览；FP32 原始渲染结果保留在远端任务目录。</p><div id="videos" class="videogrid"></div>
<h2>验证边界</h2><p>全部已完成删除检查均确认目标 owner 可见 alpha 严格为 0，其他身份几何不变。R15 删除同时移除目标太阳遮挡体，并重新计算剩余阴影，另给出关闭阴影的删除图。历史旧版没有这项物理阴影。49 帧是同一重建窗口的诊断，不是新的独立测试集；没有配对编辑 GT，human_verdict=null，不提前宣称碎片或静态残影已解决。</p><p id="receipts"></p>
<script id="ready" type="application/json">@@READY@@</script><script>const READY=JSON.parse(document.getElementById('ready').textContent),actors=document.getElementById('actor'),frames=document.getElementById('frame'),names={old3:'历史旧3身份30K',old5:'补齐旧5身份30K',r15:'冻结R15 4K'};for(let n=89;n<=137;n++){let o=document.createElement('option');o.value='0008'+String(n).padStart(3,'0');o.textContent=n;frames.append(o)}frames.value='0008113';function panel(url,label){return `<div class="panel"><a href="${url}"><img loading="lazy" src="${url}" alt="${label}"></a><p>${label}</p></div>`}function update(){let a=Number(actors.value),aa='a'+String(a).padStart(2,'0'),fid=frames.value,html='';for(const model of ['old3','old5','r15']){let r=READY[model];html+=`<h3>${names[model]}</h3>`;if(!r){html+='<div class="note">完整30K训练与渲染尚未结束。</div>';continue}let b=`assets/${model}/${fid}/`;if(!r.actors.includes(a)){html+=`<div class="row">${panel(b+'full.png','原全景')}${panel(b+'static.png','原静态/PVG：包含此车外观')}<div class="panel na">没有该身份的独立 owner。独立RGB、alpha、删除：不适用。</div></div>`;continue}html+='<div class="row">'+[['full.png','完整场景'],[aa+'_isolated.png','仅真实 owner 的 RGB'],[aa+'_alpha.png','独立 alpha'],[aa+'_visible.png','全景可见 alpha*T'],[aa+'_delete.png','删除该 owner']].map(([f,l])=>panel(b+f,l)).join('')+'</div>'}document.getElementById('matrix').innerHTML=html;document.getElementById('off').innerHTML=READY.r15?'<h3>R15：删除并关闭物理阴影</h3><div class="row">'+panel(`assets/r15/${fid}/${aa}_delete_no_shadow.png`,'删除目标后，关闭阴影')+'</div>':'';let v='';for(const model of ['old3','old5','r15'])if(READY[model]&&READY[model].actors.includes(a))for(const mode of ['isolated','delete']){let src=`videos/${model}_${aa}_${mode}.mp4`;if(AVAILABLE_VIDEOS.includes(src))v+=`<figure><video controls loop muted preload="metadata" src="${src}"></video><figcaption>${names[model]} · ${mode==='isolated'?'独立身份':'删除身份'}</figcaption></figure>`}document.getElementById('videos').innerHTML=v||'<p>PNG 已保存，视频尚未组装。</p>'}actors.onchange=frames.onchange=update;document.getElementById('prev').onclick=()=>{frames.selectedIndex=Math.max(0,frames.selectedIndex-1);update()};document.getElementById('next').onclick=()=>{frames.selectedIndex=Math.min(48,frames.selectedIndex+1);update()};document.getElementById('receipts').innerHTML=Object.keys(READY).map(m=>`<a href="assets/${m}/render_receipt.json">${names[m]} 渲染凭据</a>`).join(' · ');const AVAILABLE_VIDEOS=@@VIDEOS@@;update();</script></html>'''
    compact={k:{'actors':v['actors'],'frames':v['frames']} for k,v in ready.items()}
    videos=['videos/'+f.name for f in (out/'videos').glob('*.mp4')] if (out/'videos').exists() else []
    replacements={'STYLES':styles,'ARCH':arch,'STATUS':rows,'GEOMETRY_TABLE':geometry_html,'@@READY@@':json.dumps(compact),'@@VIDEOS@@':json.dumps(videos)}
    import re
    page=re.sub('|'.join(map(re.escape,replacements)),lambda m:replacements[m.group()],page)
    page=page.replace("for(const mode of ['isolated','delete'])", "for(const mode of ['isolated','delete',...(model==='r15'?['delete_no_shadow']:[])])")
    page=page.replace("mode==='isolated'?'独立身份':'删除身份'", "mode==='isolated'?'独立身份':mode==='delete_no_shadow'?'删除身份并关闭阴影':'删除身份'")
    if 'old5' in ready:
        page=page.replace('<h1>','<div class="note"><strong>已更新：补齐旧五身份实例 30K 已完成。</strong> 三版独立／删除图共 637 个身份帧，五辆车均可直接核对。新增修复 R15 的 4K 已通过 GPU 自检并开始训练，本页比较的是原冻结 R15 4K。</div><h1>',1)
    (out/'index.html').write_text(page,encoding='utf-8')
    (out/'completion_status.json').write_text(json.dumps(dict(task_id='WS-V82-R15-IDENTITY-COMPARISON-20261018',models_completed=list(ready),
        old5_30k_complete='old5' in ready,render_cases={m:len(r['records']) for m,r in ready.items()},
        original_checkpoints_preserved=True,artifact='interim' if 'old5' not in ready else 'all identity renders assembled',human_verdict=None),indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),completed=list(ready),videos=len(videos))))

if __name__=='__main__':main()
