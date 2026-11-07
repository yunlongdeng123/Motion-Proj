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
        raws=index.get('pilot_raws',RAWS)
        assert len(raws)==4 and len(set(raws))==4 and all(89<=raw<=137 for raw in raws)
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17) if Path('C:/Windows/Fonts/arial.ttf').is_file() else ImageFont.load_default()
        rows = []
        for state in ('keep', 'delete'):
            board = Image.new('RGB', (1510, 1450), '#f3f7f9')
            draw = ImageDraw.Draw(board)
            for col, label in enumerate(('Observed (a37 present)', 'v44 '+state.upper()+' - frozen', 'Integrated '+state.upper())):
                draw.text((40+490*col, 16), label, font=font, fill='#193b4a')
            for row, raw in enumerate(raws):
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
        case_id=index.get('full49_case_id','full49_actor5')
        case_label=index.get('full49_label','仅source5加法（未采用）')
        figure_prefix='all49_actor5' if case_id=='full49_actor5' else case_id
        frames = index.get('full49_frames','evidence/full49_actor5/frames')
        summary = self.read(index.get('full49_summary','evidence/full49_actor5/summary.json'))
        metrics = self.read(index.get('full49_metrics','evidence/full49_actor5/paired_metrics.json'))
        integrity = self.read(index.get('full49_integrity','evidence/full49_actor5/source_integrity.json'))
        visual = self.read(index.get('full49_visual_review','evidence/full49_actor5/root_visual_gate.json'))
        crop_contract = self.read('evidence/a5_contact_crop_contract.json') or {}
        baseline='evidence/v44_all49_reference/frames'
        count=sum((self.root/frames/f'{raw:03d}'/(state+'.png')).is_file()
                  for raw in range(89,138) for state in ('keep','delete'))
        data=dict(case_id=case_id,label=case_label,candidate_PNGs_present=count,expected_candidate_PNGs=98,images_complete=count==98,
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
            only105=any('raw105' in row['name'] for row in gates) and not any('raw94' in row['name'] for row in gates)
            coverage_note='其中a5门只检查raw105，未覆盖其余可见帧。' if only105 else '原94/105的full/upper严格门保持原判定，不以结构图检或加权平均覆盖局部FAIL。'
            numeric_html='<section><h2>49数值：'+h(case_label)+'</h2><p>原'+str(len(gates))+'项门有'+str(sum(row['passed'] for row in gates))+'项通过。'+coverage_note+' KEEP全a5域回退帧为'+h(worsening)+'；全域、上部、逐帧变化分别保留，不宣称逐帧无回退。</p><table><thead><tr><th>状态</th><th>原始固定域</th><th>像素帧</th><th>正式v44</th><th>当前诊断</th><th>ΔMAE / 相对变化</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table><p>RGB通道MAE来自已保存float，单位0–1，按固定像素帧数加权；没有从PNG反算或按新alpha筛域。49视图中的a5可见帧只有24个；upper不是按像素数取上25%，也不是独立车顶标注。</p><details><summary>24个可见帧逐帧a5全域变化（正数为回退）</summary><table><thead><tr><th>raw</th><th>固定像素</th><th>KEEP ΔMAE</th><th>DELETE ΔMAE</th></tr></thead><tbody>'+''.join(per_frame)+'</tbody></table></details><p>'+self.link(index.get('full49_metrics','evidence/full49_actor5/paired_metrics.json'),'独立CPU原始门与全部逐帧分母')+'</p></section>'
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
            board.save(self.root/'figures'/(figure_prefix+'_a5_visible_'+state+'.png'))
        encoded=json.dumps(paths,ensure_ascii=True,separators=(',',':'))
        initial=paths['105'];metrics_path=index.get('full49_metrics','evidence/full49_actor5/paired_metrics.json')
        block=f'''<section><h2>49视图：{h(case_label)}双状态真实PNG</h2><p>共98张候选原生PNG。以下只显示已保存RGB，不借用旧Gaussian或depth面板；Observed中a37在场，不能作为DELETE隐藏道路真值。</p><div class="toolbar"><button id="allPrev">←</button><button id="allPlay">播放</button><input id="allRaw" type="range" min="89" max="137" value="105"><strong id="allLabel">raw105</strong><select id="allState"><option value="delete">仅DELETE a37</option><option value="keep">完整KEEP</option></select><button id="allNext">→</button></div><div class="three"><div><h3>Observed（a37在场）</h3><a id="allObservedLink" href="{initial['observed']}"><img id="allObserved" src="{initial['observed']}" alt="原始观测"></a></div><div><h3>冻结v44</h3><a id="allBaseLink" href="{initial['baseline']['delete']}"><img id="allBase" src="{initial['baseline']['delete']}" alt="冻结v44原生PNG"></a></div><div><h3>{h(case_label)}</h3><a id="allNewLink" href="{initial['integrated']['delete']}"><img id="allNew" src="{initial['integrated']['delete']}" alt="集成候选原生PNG"></a></div></div><p>点击任一图可看原480×320文件。全片数值与发布门仍按实际收据与root图检分别记录。{self.link(metrics_path,'49视图配对指标')}</p><details><summary>全部24个a5可见帧：GT / v44 / 集成，同原mask裁剪</summary><p>逐帧用原始GT可见mask包围盒外扩6像素，三列使用相同裁剪和显示缩放；各帧缩放可以不同。仅为显示，指标仍在原像素域。只判断实际可见部分，不推断被遮挡或出画表面。</p><h3>KEEP</h3><a href="figures/{figure_prefix}_a5_visible_keep.png"><img src="figures/{figure_prefix}_a5_visible_keep.png" alt="24可见a5帧KEEP三列联系图"></a><h3>DELETE a37</h3><a href="figures/{figure_prefix}_a5_visible_delete.png"><img src="figures/{figure_prefix}_a5_visible_delete.png" alt="24可见a5帧DELETE三列联系图"></a></details></section>
<script>(()=>{{const frames={encoded};const $=id=>document.getElementById(id);let timer=null;function show(){{const raw=Number($('allRaw').value),state=$('allState').value,row=frames[String(raw)];$('allLabel').textContent='raw'+raw+' · '+(raw-88)+'/49';for(const [image,link,path] of [['allObserved','allObservedLink',row.observed],['allBase','allBaseLink',row.baseline[state]],['allNew','allNewLink',row.integrated[state]]]){{$ (image).src=path;$(link).href=path;}}}}function step(d){{let raw=Number($('allRaw').value)+d;if(raw<89)raw=137;if(raw>137)raw=89;$('allRaw').value=raw;show();}}$('allPrev').onclick=()=>step(-1);$('allNext').onclick=()=>step(1);$('allRaw').oninput=show;$('allState').onchange=show;$('allPlay').onclick=()=>{{if(timer){{clearInterval(timer);timer=null;$('allPlay').textContent='播放';}}else{{timer=setInterval(()=>step(1),200);$('allPlay').textContent='暂停';}}}};show();}})();</script>'''
        data.update(frame_paths=paths,contact_crop_contract=crop_contract)
        diagnostic=''
        if case_id=='full49_actor5' and (self.root/'figures/full49_actor5_raw094_keep_GTmask_crop.png').is_file():
            diagnostic='<section><h2>最坏回退示例：raw94</h2><p>三列使用原a5可见mask包围盒外扩6像素(119,167)–(191,224)，仅显示放大。冻结v44仍有较清楚的远处车体，混合版车顶/挡风玻璃明显黑糊。</p><img src="figures/full49_actor5_raw094_keep_GTmask_crop.png" alt="raw94 GT/v44/混合版最坏回退"><details><summary>原480×320完整图与原生Full5已有图的额外对照</summary><a href="figures/full49_actor5_raw094_keep_full.png"><img src="figures/full49_actor5_raw094_keep_full.png" alt="raw94原像素全图三列"></a><img src="figures/full49_actor5_raw094_native_control.png" alt="raw94原Full5整体KEEP与混合仅a5的对照"><p>原Full5整体KEEP虽软，但没有相同黑糊表现；混合相容性尚未唯一归因。没有据此声称原Full5整模型坏，也没有把SH桥接控制通过当作混合质量通过。</p></details></section>'
        return numeric_html+diagnostic+block,data

    def build(self):
        from final_report_content import build_final
        return build_final(self)


if __name__ == '__main__':
    Builder(Path(__file__).absolute().parent.parent).build()
