"""Build the four-frame v44/Full5 integration report from actual local evidence.

No renderer, checkpoint loader, network, or optimizer is used. A candidate image
is displayed only if its native480x320 PNG exists; missing results stay pending.
"""
import hashlib
import html
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

RAWS = [105, 113, 130, 137]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def h(value):
    return html.escape(str(value), quote=True)


class Builder:
    def __init__(self, root):
        self.root = root
        self.inputs = set()
        self.image_sources = []

    def read(self, relative):
        path = self.root/relative
        if not path.is_file():
            return None
        self.inputs.add(path)
        return json.loads(path.read_text(encoding='utf-8-sig'),
                          parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))

    def link(self, relative, label):
        if not (self.root/relative).is_file():
            return '<span class="pending">'+h(label)+'：待取回</span>'
        return '<a href="'+quote(relative, safe='/._-')+'">'+h(label)+'</a>'

    def image(self, relative):
        path = self.root/relative
        if not path.is_file():
            return None
        self.inputs.add(path)
        with Image.open(path) as im:
            if im.size != (480, 320):
                raise ValueError('Expected native480x320 PNG: '+relative)
            image = im.convert('RGB').copy()
        self.image_sources.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha(path), size=[480, 320]))
        return image

    def figures(self, index):
        folder = self.root/'figures'
        folder.mkdir(exist_ok=True)
        baseline = index.get('baseline_frames', 'evidence/v44_formal_reference')
        candidate = index.get('candidate_frames', 'evidence/pilot_candidate/frames')
        case_id = index.get('case_id', 'pilot_all3')
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17) if Path('C:/Windows/Fonts/arial.ttf').is_file() else ImageFont.load_default()
        rows = []
        for state in ('keep', 'delete'):
            board = Image.new('RGB', (1510, 1450), '#f3f7f9')
            draw = ImageDraw.Draw(board)
            for col, label in enumerate(('Observed (a37 present)', 'v44 '+state.upper()+' - frozen', 'Integrated '+state.upper())):
                draw.text((40+490*col, 16), label, font=font, fill='#193b4a')
            for row, raw in enumerate(RAWS):
                source_paths = [f'{baseline}/{raw:03d}/observed.png',
                                f'{baseline}/{raw:03d}/{state}.png',
                                f'{candidate}/{raw:03d}/{state}.png']
                draw.text((0, 198+350*row), str(raw), font=font, fill='#193b4a')
                for col, relative in enumerate(source_paths):
                    im = self.image(relative)
                    origin = (30+490*col, 50+350*row)
                    if im is None:
                        draw.rectangle((*origin, origin[0]+480, origin[1]+320), fill='#e3ebee')
                        draw.text((origin[0]+130, origin[1]+145), 'PENDING ACTUAL RESULT', font=font, fill='#657985')
                    else:
                        board.paste(im, origin)  # Native pixels, never resized.
                    rows.append(dict(raw=raw, state=state, column=col, path=relative if im is not None else None))
            board.save(folder/(case_id+'_'+state+'_four_frames.png'))
            crop = Image.new('RGB', (600, 214), '#f3f7f9')
            draw = ImageDraw.Draw(crop)
            for col, (label, relative) in enumerate([
                    ('Observed', f'{baseline}/105/observed.png'),
                    ('v44 '+state.upper(), f'{baseline}/105/{state}.png'),
                    ('Integrated '+state.upper(), f'{candidate}/105/{state}.png')]):
                draw.text((8+200*col, 12), label, font=font, fill='#193b4a')
                im = self.image(relative)
                if im is not None:
                    crop.paste(im.crop((0, 160, 175, 320)), (8+200*col, 45))
                else:
                    draw.rectangle((8+200*col, 45, 183+200*col, 205), fill='#e3ebee')
                    draw.text((18+200*col, 112), 'PENDING', font=font, fill='#657985')
            crop.save(folder/(case_id+'_raw105_a5_'+state+'.png'))
        return rows

    def qa(self):
        class Parser(HTMLParser):
            def __init__(self):
                super().__init__(); self.links = []
            def handle_starttag(self, tag, attrs):
                self.links.extend(v for k, v in attrs if k in ('href', 'src') and v)
        parser = Parser(); parser.feed((self.root/'index.html').read_text(encoding='utf-8'))
        links, images = [], []
        for value in parser.links:
            parts = urlsplit(value)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            path = self.root/unquote(parts.path)
            if not path.is_file():
                raise FileNotFoundError('Broken report link: '+value)
            links.append(value)
            if path.suffix == '.png':
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    images.append(dict(path=unquote(parts.path), size=list(im.size)))
            elif path.suffix == '.svg':
                element = ET.parse(path).getroot()
                images.append(dict(path=unquote(parts.path), size=[int(element.attrib['width']), int(element.attrib['height'])]))
        return dict(status='PASS', checked_relative_links=len(links), missing=[], images=images)

    def all49(self, index):
        frames = index.get('full49_frames','evidence/full49_actor5/frames')
        summary = self.read(index.get('full49_summary','evidence/full49_actor5/summary.json'))
        metrics = self.read(index.get('full49_metrics','evidence/full49_actor5/paired_metrics.json'))
        integrity = self.read(index.get('full49_integrity','evidence/full49_actor5/source_integrity.json'))
        visual = self.read(index.get('full49_visual_review','evidence/full49_actor5/root_visual_gate.json'))
        crop_contract = self.read('evidence/a5_contact_crop_contract.json') or {}
        baseline='evidence/v44_all49_reference/frames'
        count=sum((self.root/frames/f'{raw:03d}'/(state+'.png')).is_file()
                  for raw in range(89,138) for state in ('keep','delete'))
        data=dict(candidate_PNGs_present=count,expected_candidate_PNGs=98,images_complete=count==98,
                  summary=summary,metrics=metrics,source_integrity=integrity,root_visual_gate=visual,
                  contact_scope='24 actually visible original a5 masks raw89..112, not49 independent roof views')
        numeric_html=''
        if metrics and metrics.get('evaluated_a5_and_upper_summary'):
            rows=[]
            for state,values in metrics['evaluated_a5_and_upper_summary'].items():
                for key,label in [('a5_RGB','全a5可见域'),('a5_upper_RGB','固定包围盒高度上25%')]:
                    value=values[key];delta=value['delta'];relative=delta/value['baseline']
                    rows.append('<tr><td>'+state.upper()+'</td><td>'+label+'</td><td>'+f'{value["pixels"]:,}'+'</td><td>'+f'{value["baseline"]:.6f}'+'</td><td>'+f'{value["candidate"]:.6f}'+'</td><td class="'+('worse' if delta>0 else '')+'">'+f'{delta:+.6f} ({relative:+.2%})'+'</td></tr>')
            per_frame=[]
            comparisons=metrics['comparisons']
            for raw in range(89,113):
                pair=[next(row for row in comparisons if row['raw']==raw and row['state']==state) for state in ('keep','delete')]
                deltas=[row['delta']['a5_RGB'] for row in pair]
                per_frame.append('<tr><td>'+str(raw)+'</td><td>'+str(pair[0]['baseline']['a5_RGB']['pixels'])+'</td>'+''.join('<td class="'+('worse' if value>0 else '')+'">'+f'{value:+.6f}'+'</td>' for value in deltas)+'</tr>')
            worsening=[row['raw'] for row in comparisons if row['state']=='keep' and row['baseline']['a5_RGB']['pixels'] and row['delta']['a5_RGB']>0]
            gates=metrics.get('numeric_gates',[])
            numeric_html='<section><h2>49结果：保护门通过，a5全片质量未通过</h2><p>原'+str(len(gates))+'项门有'+str(sum(row['passed'] for row in gates))+'项通过，但a5门只检查raw105，未覆盖其余可见帧。KEEP回退帧为'+h(worsening)+'；保留近处结构改善，也保留远处黑糊回退，不据平均上部误差改善宣称成功。</p><table><thead><tr><th>状态</th><th>原始固定域</th><th>像素帧</th><th>正式v44</th><th>混合仅a5</th><th>ΔMAE / 相对变化</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table><p>RGB通道MAE来自已保存float，单位0–1，按固定像素帧数加权；没有从PNG反算或按新alpha筛域。49视图中的a5可见帧只有24个；upper不是按像素数取上25%，也不是独立车顶标注。</p><details><summary>24个可见帧逐帧a5全域变化（正数为回退）</summary><table><thead><tr><th>raw</th><th>固定像素</th><th>KEEP ΔMAE</th><th>DELETE ΔMAE</th></tr></thead><tbody>'+''.join(per_frame)+'</tbody></table></details><p>'+self.link(index.get('full49_metrics','evidence/full49_actor5/paired_metrics.json'),'独立CPU原始数值、19门与全部逐帧分母')+'</p></section>'
            data['a5_KEEP_worsened_raws']=worsening
        if count!=98:
            return numeric_html+'<section><h2>49视图主评价</h2><p>实际候选PNG已取回'+str(count)+'/98。结果未齐时不展示全片占位，不将四帧推断为全49通过。中央viewer尚未发布新版本。</p></section>',data
        paths={}
        for raw in range(89,138):
            record={'observed':f'{baseline}/{raw:03d}/observed.png','baseline':{},'integrated':{}}
            for state in ('keep','delete'):
                record['baseline'][state]=f'{baseline}/{raw:03d}/{state}.png'
                record['integrated'][state]=f'{frames}/{raw:03d}/{state}.png'
            for path in [record['observed'],*record['baseline'].values(),*record['integrated'].values()]:
                assert self.image(path) is not None
            paths[str(raw)]=record
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16) if Path('C:/Windows/Fonts/arial.ttf').is_file() else ImageFont.load_default()
        visible=[row for row in crop_contract.get('frames',[]) if row['pixels']]
        assert [row['raw'] for row in visible]==list(range(89,113))
        for state in ('keep','delete'):
            board=Image.new('RGB',(1800,1440),'#f3f7f9');draw=ImageDraw.Draw(board)
            for i,row in enumerate(visible):
                x=(i%3)*600;y=(i//3)*180
                draw.text((x+6,y+3),'raw'+str(row['raw'])+' / '+state.upper(),font=font,fill='#193b4a')
                record=paths[str(row['raw'])]
                for col,(label,path) in enumerate([('GT',record['observed']),('v44',record['baseline'][state]),('Integrated',record['integrated'][state])]):
                    draw.text((x+8+195*col,y+27),label,font=font,fill='#193b4a')
                    im=self.image(path).crop(tuple(row['crop_xyxy']))
                    scale=min(180/im.width,120/im.height)
                    display=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)
                    board.paste(display,(x+8+195*col+(180-display.width)//2,y+51+(120-display.height)//2))
            board.save(self.root/'figures'/('all49_actor5_a5_visible_'+state+'.png'))
        encoded=json.dumps(paths,ensure_ascii=True,separators=(',',':'))
        initial=paths['105'];metrics_path=index.get('full49_metrics','evidence/full49_actor5/paired_metrics.json')
        block=f'''<section><h2>49视图：双状态真实PNG</h2><p>共98张候选原生PNG。以下只显示已保存RGB，不借用旧Gaussian或depth面板；Observed中a37在场，不能作为DELETE隐藏道路真值。</p><div class="toolbar"><button id="allPrev">←</button><button id="allPlay">播放</button><input id="allRaw" type="range" min="89" max="137" value="105"><strong id="allLabel">raw105</strong><select id="allState"><option value="delete">仅DELETE a37</option><option value="keep">完整KEEP</option></select><button id="allNext">→</button></div><div class="three"><div><h3>Observed（a37在场）</h3><a id="allObservedLink" href="{initial['observed']}"><img id="allObserved" src="{initial['observed']}" alt="原始观测"></a></div><div><h3>冻结v44</h3><a id="allBaseLink" href="{initial['baseline']['delete']}"><img id="allBase" src="{initial['baseline']['delete']}" alt="冻结v44原生PNG"></a></div><div><h3>仅a5最小接入</h3><a id="allNewLink" href="{initial['integrated']['delete']}"><img id="allNew" src="{initial['integrated']['delete']}" alt="集成候选原生PNG"></a></div></div><p>点击任一图可看原480×320文件。全片数值与发布门仍按实际收据与root图检分别记录。{self.link(metrics_path,'49视图配对指标')}</p><details><summary>全部24个a5可见帧：GT / v44 / 集成，同原mask裁剪</summary><p>逐帧用原始GT可见mask包围盒外扩6像素，三列使用相同裁剪和显示缩放；各帧缩放可以不同。仅为显示，指标仍在原像素域。只判断实际可见部分，不推断被遮挡或出画表面。</p><h3>KEEP</h3><a href="figures/all49_actor5_a5_visible_keep.png"><img src="figures/all49_actor5_a5_visible_keep.png" alt="24可见a5帧KEEP三列联系图"></a><h3>DELETE a37</h3><a href="figures/all49_actor5_a5_visible_delete.png"><img src="figures/all49_actor5_a5_visible_delete.png" alt="24可见a5帧DELETE三列联系图"></a></details></section>
<script>(()=>{{const frames={encoded};const $=id=>document.getElementById(id);let timer=null;function show(){{const raw=Number($('allRaw').value),state=$('allState').value,row=frames[String(raw)];$('allLabel').textContent='raw'+raw+' · '+(raw-88)+'/49';for(const [image,link,path] of [['allObserved','allObservedLink',row.observed],['allBase','allBaseLink',row.baseline[state]],['allNew','allNewLink',row.integrated[state]]]){{$ (image).src=path;$(link).href=path;}}}}function step(d){{let raw=Number($('allRaw').value)+d;if(raw<89)raw=137;if(raw>137)raw=89;$('allRaw').value=raw;show();}}$('allPrev').onclick=()=>step(-1);$('allNext').onclick=()=>step(1);$('allRaw').oninput=show;$('allState').onchange=show;$('allPlay').onclick=()=>{{if(timer){{clearInterval(timer);timer=null;$('allPlay').textContent='播放';}}else{{timer=setInterval(()=>step(1),200);$('allPlay').textContent='暂停';}}}};show();}})();</script>'''
        data.update(frame_paths=paths,contact_crop_contract=crop_contract)
        diagnostic=''
        if (self.root/'figures/full49_actor5_raw094_keep_GTmask_crop.png').is_file():
            diagnostic='<section><h2>最坏回退示例：raw94</h2><p>三列使用原a5可见mask包围盒外扩6像素(119,167)–(191,224)，仅显示放大。冻结v44仍有较清楚的远处车体，混合版车顶/挡风玻璃明显黑糊。</p><img src="figures/full49_actor5_raw094_keep_GTmask_crop.png" alt="raw94 GT/v44/混合版最坏回退"><details><summary>原480×320完整图与原生Full5已有图的额外对照</summary><a href="figures/full49_actor5_raw094_keep_full.png"><img src="figures/full49_actor5_raw094_keep_full.png" alt="raw94原像素全图三列"></a><img src="figures/full49_actor5_raw094_native_control.png" alt="raw94原Full5整体KEEP与混合仅a5的对照"><p>原Full5整体KEEP虽软，但没有相同黑糊表现；混合相容性尚未唯一归因。没有据此声称原Full5整模型坏，也没有把SH桥接控制通过当作混合质量通过。</p></details></section>'
        return numeric_html+diagnostic+block,data

    def build(self):
        findings = self.read('evidence/v44_baseline_contract_findings.json')
        if not findings:
            raise FileNotFoundError('Frozen v44 reference findings required')
        index = self.read('evidence/evidence_index.json') or {}
        case_id=index.get('case_id','pilot_all3')
        all3_gate=self.read('evidence/pilot_all3/root_visual_gate.json')
        candidate_path = index.get('candidate_summary', 'evidence/pilot_candidate/summary.json')
        metrics_path = index.get('metrics_summary', 'evidence/pilot_metrics/summary.json')
        visual_path = index.get('visual_review', 'evidence/root_visual_review.json')
        replay_path = index.get('baseline_replay', 'evidence/baseline_replay_verification.json')
        candidate = self.read(candidate_path)
        metrics = self.read(metrics_path)
        visual = self.read(visual_path)
        replay = self.read(replay_path)
        candidate_integrity_path=index.get('candidate_integrity','evidence/pilot_candidate/source_integrity.json')
        candidate_integrity=self.read(candidate_integrity_path)
        figures = self.figures(index)
        complete_images = all(row['path'] for row in figures)
        state = 'FOUR_FRAME_IMAGES_AWAIT_ROOT_VISUAL_GATE' if complete_images else 'FOUR_FRAME_CANDIDATE_PENDING'
        if case_id=='pilot_actor5' and not complete_images:
            state='PILOT_ALL3_VISUAL_FAIL_ACTOR5_RETRY_PENDING'
        pilot_visual_passed=str((visual or {}).get('status','')).startswith('PASS_ROOT_PILOT4')
        if pilot_visual_passed:
            state='FOUR_FRAME_ROOT_GATE_PASSED_ALL49_RESULTS_PENDING'
        all49_section,all49_data=self.all49(index)
        if all49_data['images_complete']:
            state='ALL49_IMAGES_RETRIEVED_AWAIT_NUMERICAL_AND_ROOT_REVIEW'
        summary49=(all49_data.get('metrics') or {}).get('evaluated_a5_and_upper_summary',{})
        quality_hold=summary49.get('keep',{}).get('a5_RGB',{}).get('delta',0)>0
        if quality_hold:
            state='HOLD_FULL49_A5_REGRESSION_NOT_PUBLISHED'
        native_links = []
        for raw in RAWS:
            paths = [(f'evidence/v44_formal_reference/{raw:03d}/observed.png', 'Observed'),
                     (f'evidence/v44_formal_reference/{raw:03d}/keep.png', 'v44 KEEP'),
                     (f'evidence/v44_formal_reference/{raw:03d}/delete.png', 'v44 DELETE'),
                     (f"{index.get('candidate_frames','evidence/pilot_candidate/frames')}/{raw:03d}/keep.png", '集成KEEP'),
                     (f"{index.get('candidate_frames','evidence/pilot_candidate/frames')}/{raw:03d}/delete.png", '集成DELETE')]
            native_links.append('<tr><td>'+str(raw)+'</td><td>'+' · '.join(self.link(p,label) for p,label in paths)+'</td></tr>')
        style='body{margin:0;background:#f3f7f9;color:#193b4a;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1400px;margin:auto;padding:28px}section{background:white;border:1px solid #dce7ec;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:29px}h2{font-size:22px}img{display:block;max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce6eb;text-align:left}th{background:#edf4f7}a{color:#006b83}pre{white-space:pre-wrap;word-break:break-word;background:#f4f7f9;padding:12px;font-size:12px}.pending{color:#946322}.worse{color:#ad3636}.notice{background:#fff4de;border-left:4px solid #c99435;padding:12px}summary{cursor:pointer;font-weight:600}footer,figcaption{font-size:13px;color:#607783}.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.toolbar input{min-width:240px;flex:1}.three{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}@media(max-width:800px){.three{grid-template-columns:1fr}}'
        image_sections=''
        for name, label in [('keep','KEEP完整车体'),('delete','仅DELETE a37')]:
            image_sections+='<section><h2>'+label+'：四帧同源对照（'+h(case_id)+'）</h2><p>Observed是a37仍在的真实观测，不是删后隐藏道路真值。原图以480×320像素直接拼接；点击图版可看原尺寸。</p><a href="figures/'+case_id+'_'+name+'_four_frames.png"><img src="figures/'+case_id+'_'+name+'_four_frames.png" alt="105 113 130 137 Observed、正式v44与集成候选'+name+'"></a><h3>raw105 a5同尺度裁剪</h3><img src="figures/'+case_id+'_raw105_a5_'+name+'.png" alt="raw105原像素a5车顶固定裁剪"><p>固定窗口(0,160)–(175,320)，没有放大或增强。指标保留原始mask全域，不用裁剪图或渲染alpha选域。</p></section>'
        all3_section=''
        if all3_gate:
            all3_section='<section><h2>保留的负结果：同时接入5 / 18 / 26</h2><p>初试raw105的a5上部结构恢复，但raw113/130保留车辆明显变暗、变糊，raw137左车局部偏暗。root四帧视觉判定未通过，没有放行49帧。原图与数据保留；下一候选只接actor5，18/26继续使用原v44表示，不覆盖这次初试。</p><p>'+self.link('evidence/pilot_all3/root_visual_gate.json','root初试FAIL记录')+'</p><details><summary>初试三actor四帧原始对照板</summary><img src="figures/pilot_all3_keep_four_frames.png" alt="保留的三actor初试KEEP负结果"><img src="figures/pilot_all3_delete_four_frames.png" alt="保留的三actor初试DELETE负结果"><img src="figures/pilot_all3_raw105_a5_delete.png" alt="三actor初试raw105 a5上部改善"></details></section>'
        page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene008：v44删车效果与Full5完整车体接入</title><style>{style}</style></head><body><main>
<h1>Scene008：v44删车效果与Full5完整车体接入</h1><p class="notice">{'全49已发现a5全域误差回退与早帧黑糊，停止发布；raw105改善不代表全片成功。原19项守护门没有覆盖全时段a5质量。' if quality_hold else '先验证105、113、130、137四帧。'+('仅a5四帧已通过root图检，49结果以实际取回与守护为准。' if pilot_visual_passed else '候选PNG已到，等待root实际图检与指标门。' if complete_images else '候选尚未完整取回，图中灰格保留待测。')} 没有将接线或拼接通过写成最终成功；完整49实测与审阅完成后才发布用户旧viewer的新可选版本。</p>
<p>目标是保住正式v44的道路去蓝、完整C2 a6与仅删a37效果，同时恢复a5车顶。三actor初试未通过后，当前最小接入只从Full5 dynamic按保存ID取actor5；18/26沿用原v44表示，不加载新Full5 static/env。原v44背景、道路、C2、a37与环境保持。这是混合表示；旧static中潜在重复车体支撑仍须核对。</p>
<section><h2>组件与数据流</h2><img src="architecture.svg" alt="冻结v44与Full5保留身份同次深度排序，四帧门控后49实测发布"></section>
{all49_section}
{all3_section}
<section><h2>四帧合同与验收边界</h2><p>正式v44源comparison SHA为<code>8201fa883df9ac43274d52984c750f92513118c1152628fa0518e69cf66042e3</code>，与用户#fullclip默认v44_20261107精确同源。不是较新的未采用R0候选，也不是去蓝前current。</p><table><thead><tr><th>守护项</th><th>本次要求</th></tr></thead><tbody><tr><td>a5车顶/挡风玻璃</td><td>四帧中只有raw105含a5：固定原始可见域10,773像素，上部2,091；其余三帧该域为0，不能当四个a5视角。</td></tr><tr><td>仅删a37</td><td>KEEP/DELETE之间仅物理移除a37；保持C2、保留身份、背景、道路、环境与相机时间。</td></tr><tr><td>完整a6与道路去蓝</td><td>C2几何/材质/SurfaceLOD冻结，以正式v44为守护基准；旧road/gates索引绑定原RCR static拓扑，不移植到新Full5 static。</td></tr><tr><td>原评价域</td><td>原蓝ROI有效分母为0则保持null。固定原观测道路域与delete road proxy分开；银色残片在旧合同中明确延期。</td></tr><tr><td>发布</td><td>先root四帧图检，获准后才49实测；全部完成后作为新可选v44_full5_20261107接入#fullclip，旧v44保留。</td></tr></tbody></table><p>{self.link('evidence/v44_baseline_contract_findings.json','完整基准合同/路径/逐图SHA')} · {self.link(replay_path,'原链四帧重放核对')} · {self.link(candidate_path,'实际候选工程收据')} · {self.link(metrics_path,'实际四帧指标')} · {self.link(visual_path,'root图检门')}</p></section>
{image_sections}
<section><h2>未resize的原始PNG</h2><table><thead><tr><th>raw</th><th>原生480×320文件</th></tr></thead><tbody>{''.join(native_links)}</tbody></table><p>原始Observed含a37，不能用于声称DELETE的隐藏表面真值。human_verdict仍为空；实际代理图检单独记录。</p></section>
<section><h2>指标与工程记录</h2><p>四帧候选的实际指标未到时不填值；先前49帧v44数字仅是已冻结参考，不冒充本次新结果。七组alpha与三个新增actor的RGB贡献由同次原生forward输出；不为旧四组捏造RGB归因，也不把身份贡献等同于车顶质量。</p><details><summary>实际指标与root视觉记录</summary><pre>{h(json.dumps({'metrics':metrics,'visual':visual},ensure_ascii=False,indent=2))}</pre></details><details><summary>原链重放、候选工程与来源</summary><pre>{h(json.dumps({'baseline_replay':replay,'candidate':candidate,'source_integrity':candidate_integrity},ensure_ascii=False,indent=2))}</pre></details><p>{self.link(candidate_integrity_path,'候选源文件守恒')} · {self.link('source/build_report.py','可重复构建源码')} · {self.link('report_data.json','报告数据与manifest')}</p></section>
<footer>状态：{state}。更新：{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}。本报告不启动GPU、训练或发布旧viewer；GPU保持开机。</footer></main></body></html>'''
        (self.root/'index.html').write_text(page,encoding='utf-8')
        (self.root/'report_data.json').write_text('{}\n',encoding='utf-8')
        qa=self.qa()
        self.inputs.add(self.root/'source/build_report.py')
        self.inputs.add(self.root/'architecture.svg')
        inputs=[dict(path=p.relative_to(self.root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(self.inputs)]
        data=dict(status=state,generated_utc=datetime.now(timezone.utc).isoformat(),pilot_raws=RAWS,active_case=case_id,
                  frozen_v44_findings=findings,candidate=candidate,metrics=metrics,root_visual_review=visual,
                  baseline_replay=replay,candidate_source_integrity=candidate_integrity,all3_visual_gate=all3_gate,figure_sources=figures,input_manifest=inputs,
                  candidate_images_complete=complete_images,all49=all49_data,new_all49_result_claimed=all49_data['images_complete'],
                  old_viewer_modified=False,all49_a5_quality_hold=quality_hold,human_verdict=None,quality_accepted=False,qa=qa)
        serialized=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        (self.root/'report_data.json').write_text(serialized,encoding='utf-8')
        print(json.dumps(dict(status=state,qa=qa),ensure_ascii=True))


if __name__ == '__main__':
    Builder(Path(__file__).absolute().parent.parent).build()
