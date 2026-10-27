from pathlib import Path
from html import escape
from html.parser import HTMLParser
from datetime import datetime,timezone,timedelta
import json,shutil,subprocess
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/Persistent_World_Autoresearch_20261027';DATA=OUT/'delivery'
font=lambda size,bold=False:ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc' if bold else 'C:/Windows/Fonts/msyh.ttc',size)
OUT.mkdir(parents=True,exist_ok=True)
shutil.copy2(Path(__file__).with_name('architecture.svg'),OUT/'architecture.svg')
evaluation=json.loads((DATA/'phase1_native/pipeline_evaluation49.json').read_text())
snapshot=json.loads((DATA/'milestone_snapshot.json').read_text())
native=json.loads((DATA/'phase1_native/summary.json').read_text())
operations=json.loads((DATA/'phase2_operations/summary.json').read_text())

def contact(name,panels,raw=102):
    im=Image.new('RGB',(3*480,2*366),'#101c2b');draw=ImageDraw.Draw(im)
    for i,(label,path) in enumerate(panels):
        x=(i%3)*480;y=(i//3)*366
        draw.text((x+10,y+10),label,font=font(21),fill='#e8eff5')
        image=Image.open(DATA/path/f'{raw:03d}.png').convert('RGB');im.paste(image,(x,y+42))
    im.save(OUT/name)
contact('pipeline_raw102.jpg',[
    ('原生完整重建',Path('phase1_native/native/native_rgb')),
    ('原静态删车',Path('phase1_native/delete_all_unrepaired/native_rgb')),
    ('保背景修复 · 原生三维',Path('phase1_native/delete_all_repaired/native_rgb')),
    ('修复后仍整洞生成',Path('phase1_native/delete_all_repaired/propainter/model_png')),
    ('保留来源证据后生成',Path('phase1_native/delete_all_evidence_routed/propainter/model_png')),
    ('直接参考锚点 · 强控制',Path('phase1_native/delete_all_direct_reference/propainter/model_png'))])
contact('operations_raw102.jpg',[
    ('保留全部对象',Path('phase2_operations/retain_actors/native_rgb')),
    ('选择删除 · 归属未完善',Path('phase2_operations/delete_selected/native_rgb')),
    ('目标车辆侧移1m',Path('phase2_operations/move_selected/native_rgb')),
    ('同场景复用插入 · 侧移2m',Path('phase2_operations/insert_reused/native_rgb')),
    ('自车相机侧移0.5m',Path('phase2_operations/camera_shift/native_rgb')),
    ('新相机再显露',Path('phase2_operations/camera_revisit/native_rgb'))])

# 生成本地H.264，已有远端编码则直接复用。
ffmpeg=shutil.which('ffmpeg')
direct=DATA/'phase1_native/delete_all_direct_reference'
if ffmpeg and not (direct/'generated_h264.mp4').exists():
    subprocess.run([ffmpeg,'-v','error','-y','-framerate','10','-start_number','89','-i',str(direct/'propainter/model_png/%03d.png'),'-frames:v','49','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(direct/'generated_h264.mp4')],check=True)
def video(path,title):
    return f'<article><h3>{escape(title)}</h3><video controls preload="metadata" playsinline src="delivery/{path}"></video></article>'
names={'unrepaired_native':'原静态删车','repaired_native':'保背景原生修复','unrepaired_video':'原静态＋整洞视频补全','repaired_video':'原生修复＋整洞视频补全','evidence_routed_video':'原生修复＋证据路由','direct_reference_video':'直接参考锚点＋同视频后端'}
train=evaluation['groups']['prototype_optimized_views']['weighted_MAE'];transfer=evaluation['groups']['other_same_sequence_views']['weighted_MAE']
rows=''.join(f'<tr><td>{names[k]}</td><td>{train[k]:.5f}</td><td>{transfer[k]:.5f}</td><td>{v:.5f}</td></tr>' for k,v in evaluation['weighted_labeled_MAE'].items())
now=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M SGT')
training=snapshot.get('training_status',{})
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>持久世界状态 · 自主研究进展</title><style>
body{{font-family:"Microsoft YaHei",sans-serif;color:#19384b;background:#f4f7fa;margin:0;line-height:1.65}}main{{max-width:1380px;margin:auto;padding:28px}}section{{background:white;border:1px solid #d6e1e8;border-radius:12px;padding:22px;margin:22px 0}}h1{{font-size:30px}}h2{{font-size:23px}}h3{{font-size:18px}}a{{color:#126f9a}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #d6e1e8}}th{{background:#edf4f7}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}video{{width:100%;background:#101c2b;border-radius:6px}}.note{{background:#fff3dc;padding:16px;border-left:4px solid #ba9447}}.good{{background:#e2f2e8;padding:16px}}code{{overflow-wrap:anywhere}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}main{{padding:12px}}}}
</style><main><h1>证据感知的持久可编辑世界状态</h1>
<p>实际执行快照 · {now} · 研究继续进行 · 质量待用户评审</p>
<p><a href="../4D_Editing_Research_Roadmap_20261027/index.html">完整框架与HorizonForge对标路线</a> · <a href="delivery/phase1_native/pipeline_evaluation49.json">逐帧原始指标</a> · <a href="delivery/phase1_native/evidence_registration.json">世界证据登记</a></p>
<section><h2>当前推进到哪里</h2><p>最小原生三维→预训练视频补全闭环已实际运行。已保存49帧×5种原生输出、三个视频策略与直接参考强控制；六项对象/相机操作也已各渲染49帧。本段是原单场景历史机制证据；最新多场景结果见上方链接，不能把操作数或帧数计为独立场景。</p>
<p>历史保背景训练快照为 <code>{escape(training.get('phase','unknown'))}</code>；其smoke和fresh10K已完成，当前不追加新修复训练。新增开发路段正在逐场景准备原生30K基座，多编辑复用；见下方最新评测报告。本文保留历史机制证据，属于静态快照。</p>
<img src="architecture.svg" alt="输入、编辑、背景联合修复、按需生成与持久世界状态"/></section>
<section><h2>第一条实验证据：整洞生成会覆盖已有有效结果</h2>
<p>在原型优化过的三个视图上，修复原生MAE为0.00769，整洞视频补全后为0.05012。固定后端保留有来源支持的内容，能避免这部分覆盖。46个其他同片段视图同时改善，但仍需独立场景和几何验证。</p>
<table><tr><th>相同预训练ProPainter后端</th><th>原型优化3视图</th><th>其他46视图</th><th>49帧加权</th></tr>{rows}</table>
<p class="note">口径：PNG/clamp RGB与跨帧参考的MAE↓。三视图有16921参考像素，46视图有299258参考像素；来源138–169已参与199帧初始化。这里衡量跨视图一致性，不是独立编辑GT，也不是HorizonForge统一benchmark。无参考的目标区域不计作成功。</p>
<p><b>强控制限制：</b>只把同来源RGB锚点直接交给视频模型，也能达到0.01400，已解释大部分收益。因此暂不能把改善归因于学得的高斯修复或持久记忆。后续重点是三维正确性、换视角、反复编辑与减少重复生成。证据路由的3个锚帧已被使用，不能在这些输入像素上声称泛化。</p>
<img src="pipeline_raw102.jpg" alt="raw102同视角六种输出"/></section>
<section><h2>仍需解决的真实失败</h2><p>实际图像仍有左车黑色残留、近处条带；选择删车与移动时也有明显车体残影。视频参考域平均误差下降不能代替这些未解决区域。</p>
<p>另一次49帧只读检查发现，旧原型三优化视图中，贡献加权期望深度与受支持道路曲面的平均绝对差仍为3.557m，小于0.3m的比例仅1.466%；其他46视图为3.402m/2.032%。期望深度不是首交面，参考曲面也不是独立密集GT，因此先把它作为颜色与几何脱节的诊断。</p>
<p>下一组已排队：固定原训练合同，仅增加分支无关的表面深度矩约束；等待当前健康10K结束后，先独立真实10分钟smoke再fresh10K，并自动接续49帧原生与同后端视频评价。这是由实际失败触发的简单几何原型，尚未确认效果或创新。</p></section>
<section><h2>生成前和生成后的49帧视频</h2><div class="grid">'''
for path,label in [
 ('phase1_native/delete_all_unrepaired/native_h264.mp4','原静态删车 · 原生3D'),
 ('phase1_native/delete_all_repaired/native_h264.mp4','保背景修复 · 原生3D'),
 ('phase1_native/delete_all_repaired/generated_h264.mp4','修复后整洞生成'),
 ('phase1_native/delete_all_unrepaired/generated_h264.mp4','原静态整洞生成'),
 ('phase1_native/delete_all_evidence_routed/generated_h264.mp4','证据路由视频'),
 ('phase1_native/delete_all_direct_reference/generated_h264.mp4','直接参考锚点强控制')]:body+=video(path,label)
body+='''</div><p>视频结果作为独立生成假设保存，未写回观测记录或自动转成三维几何。源PNG保留在delivery中。</p></section>
<section><h2>操作链路已运行，归属仍是实际瓶颈</h2>
<p>当前原B未保存逐Gaussian对象ID，先用七时刻真实轨迹框投票建立可审查基线；未决867871/1030499行保留。这不是精确语义分离。下面展示真实输出，不能仅凭成功调用接口称编辑质量达标。</p>
<img src="operations_raw102.jpg" alt="选择删车、侧移、复用插入和相机变化原生输出"/><div class="grid">'''
for op,label in [('delete_selected','选择删除'),('move_selected','侧移1m'),('insert_reused','同场景复用插入'),('camera_shift','自车侧移0.5m'),('camera_revisit','新视角再显露')]:body+=video(f'phase2_operations/{op}/native_h264.mp4',label)
body+='''</div><p>相机变化和插入暂无配对GT；同场景复用插入不能代替外部资产迁移。静态分支中的车辆残留也不会随动态操作自动移动。</p></section>
<section><h2>持久状态目前真正保存的内容</h2>
<p>26088个世界网格候选逐一保存来源帧：7848观测支持候选、13778有几何支撑但未满足当前颜色资格、4462未知。修复版本、编辑请求、视频生成假设分别登记。来源支持候选和被优化过的修复都不等于已核实的正确表面。</p>
<p>原生完整通路编辑前后最大RGB差0；raw102独立重载后复渲染差0。它们是工程隔离和重载验收，尚不是持久世界记忆的算法收益。</p></section>
<section><h2>后续实验按失败推进</h2><ol>
<li>完成真实smoke与fresh 10K基线，分开记录更新和显存；对新状态做49帧/新相机评价。</li>
<li>补严格道路撤源、保留对象影响、重复请求及跨相机复用测试；撤源必须早于初始化与编码。</li>
<li>固定来源预算和生成后端，比较直接参考锚点、普通缓存和带来源的世界状态；再决定②需要哪项核心算法。</li>
<li>DiffuEraser权重补齐后接显式三维prior；几何未知、对象身份与外观生成分别定位，按实测需求迭代。</li>
<li>接入独立路段并按场景划分。240例仍是目标，当前没有完成跨场景benchmark。</li></ol>
<p>每30分钟自主接续，正常状态安静，只在实质进展、异常或需要用户处理时通知。单轮负结果保留并用于下一轮设计，整个研究持续推进。</p>
<p>来源：<a href="https://github.com/lixiaowen-xw/DiffuEraser">DiffuEraser官方实现</a>；现阶段使用其已有ProPainter组件，扩散大权重仍在补齐。<a href="https://github.com/sczhou/ProPainter">ProPainter官方实现</a>。</p>
</section></main></html>'''
# 保留历史机制证据，但当前执行方向必须随最新用户纠偏更新。
body=body.replace('<section><h2>当前推进到哪里</h2>', '<section><h2>当前推进到哪里</h2><p class="note"><strong>2026-10-27更新：阶段二多场景、多操作诊断。</strong>两个新增场景均已完成30K与完整CPU验收；首场景11原生变体、6固定视频、6次连续原生修订及连续后端3推理+2普通缓存完成。第二场景11原生变体/6固定视频和6连续原生修订已完成，第三场景真实smoke正在恢复。首套八类原生31帧与4固定视频完成，含真实跨场景资产；新增两场景分层诊断，正式对标与质量仍待验收。33段全传感器输入与264八类请求已物化，178项输入筛查通过，首个新八类7族GPU单帧入口适配差0，完整序列待完成；正式成绩仍为0。每5K保存四件，多编辑复用。新②训练与③切换仍延后，固定方法建立失败矩阵。<a href="../4D_Editing_Evaluation_20261027/index.html">查看最新流水、输入质检与实际评测</a>。持久状态是框架目标，不预定创新。周额度剩余低于20%快速保存收尾并关AutoDL；此前继续研究。</p>')
body=body.replace('下一组已排队：固定原训练合同，仅增加分支无关的表面深度矩约束；等待当前健康10K结束后，先独立真实10分钟smoke再fresh10K，并自动接续49帧原生与同后端视频评价。这是由实际失败触发的简单几何原型，尚未确认效果或创新。', '该深度矩候选已完成3985额外smoke更新并保存；正式10K未启动，已撤下。当前保背景基线10K与两个其他开发路段的只读评价已完成；先补多场景评测，再决定②改进。')
body=body.replace('DiffuEraser权重补齐后接显式三维prior；几何未知、对象身份与外观生成分别定位，按实测需求迭代。', 'DiffuEraser权重已完整下载且CPU导入通过，GPU后端切换延后；阶段二固定ProPainter。')
body=body.replace('接入独立路段并按场景划分。240例仍是目标，当前没有完成跨场景benchmark。', '36个context已完成RGB/标注准备，两个其他历史开发路段已完成49帧固定基线；封存23段未跑编辑评测。完整六类任务、道路GT及严格撤源仍需补齐。')
body=body.replace('现阶段使用其已有ProPainter组件，扩散大权重仍在补齐。', '现阶段固定使用其已有ProPainter组件；扩散大权重已完整下载，后端切换延后。')
(OUT/'index.html').write_text(body,encoding='utf-8')

class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['src','href'] and value and not value.startswith(('http:','https:','#')):self.paths.append(value)
parser=Links();parser.feed(body)
broken=[x for x in parser.paths if not (OUT/x.split('#')[0]).exists()]
for path in OUT.glob('*.jpg'):
    with Image.open(path) as image:image.verify()
validation=dict(local_link_count=len(parser.paths),broken_links=broken,video_count=body.count('<video '),
    png_count=len(list(DATA.rglob('*.png'))),report_status='ongoing_research_snapshot',browser_render_qa=False,
    quality_accepted=False,human_verdict=None)
(OUT/'validation.json').write_text(json.dumps(validation,indent=2))
assert not broken,broken
print(json.dumps(validation,indent=2))
