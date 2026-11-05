"""Publish local, evidence-bound a6 artifact diagnosis; no network or GPU.

Run again after collection of combined_v1/observed_fit. Optional final manifest:
  {"receipt":"surface_fit/combined_v1/observed_fit/receipt.json",
   "asset":"surface_fit/combined_v1/observed_fit/complete_a6.npz",
   "evaluation":".../receipt.json",
   "figures":[{"path":".../raw113/black_HR_comparison.png"}]}
All manifest paths are local, relative to --artifacts, or absolute local paths.
The final section is emitted only when a real receipt AND real images exist.
Original image bytes and JSON/code are copied for download. Images in the HTML
are initial data URIs; there is no JavaScript or external resource dependency.
"""
import argparse,base64,hashlib,html,io,json,mimetypes,re,shutil,sys
from datetime import datetime,timezone
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from frame_compare import load_validate as load_frame_compare, publish as publish_frame_compare
from frame_compare import scientific_baseline, inject_preserving_baseline, BEGIN as FC_BEGIN, END as FC_END

WORKSPACE=Path(__file__).resolve().parents[3]
MAX_HTML_BYTES=8*1024*1024
MAX_IMAGE_URI_BYTES=7_300_000
FINAL_RAW_IDS=(106,113,130)
FINAL_FIGURE_NAMES=('black_HR_comparison.png','black_native480_comparison.png','delete_crop480_comparison.png')
FINAL_FIGURE_DESCRIPTIONS={
    'black_HR_comparison.png':('1920实际黑底：thin M2 / 旧粗糙深度formal / 新C2+matte initial / 新formal',4),
    'black_native480_comparison.png':('同480实际单体：新native / 新area控制 / 旧native / 旧area控制 / thin M2',5),
    'delete_crop480_comparison.png':('同480实际街景（delete a37）：新native / 新area控制 / 旧native / 旧area控制 / thin M2',5)}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def esc(x):return html.escape(str(x))
def dig(x,*keys):
    for k in keys:
        if not isinstance(x,dict) or k not in x:return None
        x=x[k]
    return x
def fmt(x,n=6):
    if x is None:return '—'
    if isinstance(x,bool):return str(x).lower()
    if isinstance(x,int):return f'{x:,}'
    if isinstance(x,float):return f'{x:.{n}f}'
    return esc(x)
def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def strings(x):
    if isinstance(x,str):yield x
    elif isinstance(x,dict):
        for v in x.values():yield from strings(v)
    elif isinstance(x,list):
        for v in x:yield from strings(v)

class Publisher:
    def __init__(self,art,out):
        self.art=art;self.out=out;self.sections=[];self.evidence=[];self.images=[];self.uri_bytes=0
        out.mkdir(parents=True,exist_ok=True)
    def local(self,value):
        p=Path(value)
        return p if p.is_absolute() else self.art/p
    def copy(self,p,category='data'):
        p=p.resolve()
        try:rel=p.relative_to(self.art)
        except ValueError:rel=Path(p.name)
        target=self.out/category/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.is_file() or sha(target)!=sha(p):shutil.copy2(p,target)
        return target.relative_to(self.out).as_posix()
    def evidence_file(self,rel,label=None):
        p=self.local(rel)
        if not p.is_file():return ''
        href=self.copy(p)
        if not any(x['source']==str(p.resolve()) for x in self.evidence):
            self.evidence.append(dict(label=label or p.name,source=str(p.resolve()),href=href,sha256=sha(p),bytes=p.stat().st_size))
        return '<a href="'+esc(href)+'">'+esc(label or p.name)+'</a>'
    def figure(self,rel,caption):
        p=self.local(rel)
        if not p.is_file():return ''
        original=self.copy(p,'images/originals');data=p.read_bytes();mime=mimetypes.guess_type(p.name)[0] or 'image/png'
        # Main evidence boards remain original bytes. Only oversized optional
        # preview images get a display derivative; their originals stay linked.
        derivative=False
        if len(data)>850_000:
            im=Image.open(io.BytesIO(data)).convert('RGB');im.thumbnail((2200,1500),Image.Resampling.LANCZOS)
            b=io.BytesIO();im.save(b,'JPEG',quality=92,subsampling=0);data=b.getvalue();mime='image/jpeg';derivative=True
        uri='data:'+mime+';base64,'+base64.b64encode(data).decode()
        if self.uri_bytes+len(uri)>MAX_IMAGE_URI_BYTES:
            return '<p>'+esc(caption)+'：'+self.evidence_file(p,'查看完整原图')+'</p>'
        self.uri_bytes+=len(uri)
        with Image.open(io.BytesIO(data)) as im:size=list(im.size)
        self.images.append(dict(source=str(p.resolve()),original_href=original,source_sha256=sha(p),embedded_bytes=len(data),display_derivative=derivative,wh=size,caption=caption))
        return '<figure><figcaption>'+esc(caption)+'</figcaption><a href="'+esc(original)+'"><img src="'+uri+'" alt="'+esc(caption)+'"></a></figure>'
    def add(self,anchor,title,body):
        if body:self.sections.append((anchor,title,body))
    def load(self,rel,label=None):
        p=self.local(rel)
        if not p.is_file():return None
        j=read(p);self.evidence_file(p,label);return j


def architecture():
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 300" role="img" aria-label="2D材质proposal与冻结几何先验进入持久3D资产和native联合渲染"><defs><marker id="arr" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#36596d"/></marker></defs><g stroke="#7995a5" stroke-width="1.5" fill="#fff"><rect x="14" y="24" width="215" height="78" rx="9"/><rect x="14" y="148" width="215" height="90" rx="9"/><rect x="292" y="70" width="235" height="125" rx="9"/><rect x="587" y="87" width="200" height="92" rx="9"/><rect x="848" y="87" width="200" height="92" rx="9"/><rect x="1093" y="87" width="95" height="92" rx="9"/><rect x="665" y="231" width="305" height="54" rx="9"/></g><g font-family="system-ui,sans-serif" font-size="18" fill="#18394e" text-anchor="middle"><text x="121" y="53">actual strongv2 2D</text><text x="121" y="80">RGB proposal + matte</text><text x="121" y="179">冻结PVD深度先验</text><text x="121" y="208">真实BUILD RGB / LiDAR</text><text x="410" y="103">表面与材质表示</text><text x="410" y="135">连续深度 / alpha</text><text x="410" y="168">逐项对照与颜色拟合</text><text x="687" y="121">持久 a6 S/F资产</text><text x="687" y="150">表面 + 材质</text><text x="948" y="121">视口采样 / native排序</text><text x="948" y="151">同相机 HR / LR</text><text x="1140" y="121">单体</text><text x="1140" y="151">街景</text><text x="817" y="264">冻结背景 / a37 / 道路 / 环境</text></g><g fill="none" stroke="#36596d" stroke-width="2.3" marker-end="url(#arr)"><path d="M229 63L292 105"/><path d="M229 193L292 164"/><path d="M527 133L587 133"/><path d="M787 133L848 133"/><path d="M1048 133L1093 133"/><path d="M904 231L933 179"/></g></svg>'''


def before_after(pub):
    root=pub.art/'scale/surface_lod_v1/evaluation49'
    specs=[('old_native_delete480.png','Old formal'),('final_native_delete480.png','C2 + matte'),('surface_LOD_delete480.png','C2 + matte + LOD (research)')]
    if not (root/'receipt.json').is_file():return ''
    paths=[root/'raw113'/name for name,_ in specs]
    if not all(p.is_file() for p in paths):return ''
    cropbox=(180,120,360,320);panels=[];sources=[]
    fp=Path('C:/Windows/Fonts/arial.ttf');font=ImageFont.truetype(str(fp),18) if fp.is_file() else ImageFont.load_default()
    for p,(_,label) in zip(paths,specs):
        with Image.open(p) as im:
            assert im.size==(480,320)
            c=im.convert('RGB').crop(cropbox).resize((360,400),Image.Resampling.NEAREST)
        tile=Image.new('RGB',(360,442),(239,245,249));tile.paste(c,(0,42))
        ImageDraw.Draw(tile).text((10,10),label,font=font,fill=(18,43,60));panels.append(tile)
        sources.append(dict(path=str(p),sha256=sha(p),original=pub.copy(p,'images/originals')))
    out=Image.new('RGB',(1080,442))
    for i,p in enumerate(panels):out.paste(p,(360*i,0))
    dest=pub.out/'before_after.png';out.save(dest)
    provenance=dict(raw=113,viewport=[480,320],edit_state='delete a37',same_crop_xyxy=list(cropbox),
        display_nearest_scale=2,source_images=sources,background_retouched=False,shadow_ROI_included=[281,301,336,320],
        LOD_is_research_control_not_adoption=True)
    (pub.out/'before_after_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return pub.figure(dest,'先看同一480街景：旧formal / 新C2+matte / 固定S/F的LOD研究对照。三列同raw113、同相机、同裁剪，显示放大2倍；下方旧暗团保留，无背景修补。')


def add_lod(pub):
    root=pub.art/'scale/surface_lod_v1/evaluation49'
    rp=root/'receipt.json';sp=root/'summary.json'
    if not(rp.is_file() and sp.is_file()):return
    rec=pub.load(rp,'LOD固定49实际receipt');s=pub.load(sp,'LOD固定49实际summary')
    rows=[]
    for arm,label,base in [('LOD_vs_old_native','旧formal native',True),('LOD_vs_final_native','C2+matte native',True),('LOD_vs_final_area','C2+matte area（诊断）',True),('LOD_vs_old_native','C2+matte LOD（研究）',False)]:
        a=s[arm]['all49'];pre='baseline_' if base else ''
        rgb=a['solo.a6.'+pre+'rgb_error'];dep=a['solo.a6.'+pre+'measured_depth_error'];alpha=a['scene.keep.a6.'+pre+'visible_coverage']
        rows.append((label,fmt(rgb['count']),fmt(rgb['mean']),fmt(dep['count']),fmt(dep['mean']),fmt(alpha['mean'])))
    body='<p>此处是唯一新增的视口fallback：固定C2表面S与最终材质F，根据相机投影面积选连续UV采样间距，再做固定premultRGBA mip过滤；峰值opacity仍为.99×coverage，未叠加area稀释系数。它更平滑，部分细线也更软，不能称观感全面变锐。无新拟合、无参数网格。</p>'
    body+=table(['同一480策略','固定RGB像素','solo RGB误差','固定深度样本','LiDAR z误差(m)','joint visible α均值'],rows)
    body+='<p>LOD的49帧RGB/深度优于原native，alpha接近opaque但仍有覆盖代价；area的RGB较低伴随更明显透明度损失，未自动采用。数值均为内部开发对照，不能验证隐藏形状。</p>'
    preservation=[]
    for arm,label in [('LOD_vs_old_native','LOD vs旧native'),('LOD_vs_final_native','LOD vs新native')]:
        a=s[arm]['all49']
        for state,region,desc in [('keep','a37','keep a37'),('keep','road_visible','keep road'),('keep','outside','keep outside'),('delete','outside','delete outside')]:
            r=a['scene.'+state+'.preservation.'+region+'.parent_rgb_change']
            depth=a.get(state+'_fixed_depth_preservation.'+('road' if region=='road_visible' else region)+'.expected_depth_parent_change_m',{})
            preservation.append((label,desc,fmt(r['above_002'])+' / '+fmt(r['count']),fmt(r['max']),fmt(depth.get('mean')),fmt(depth.get('max'))))
    body+=table(['比较','原固定域','RGB差>.02 / N','RGB差max','depth差mean(m)','depth差max(m)'],preservation)
    body+='<p class="small">LOD与新native之间仅采样/过滤不同，外域expected-depth仍可能有较大max变化；冻结背景参数不等于其所有成像精确不变。</p>'
    coverage=pub.art/'scale/surface_lod_v1/coverage49'
    if (coverage/'receipt.json').is_file():pub.load(coverage/'receipt.json','LOD49精确coverage receipt')
    if (coverage/'summary.json').is_file():
        cv=pub.load(coverage/'summary.json','LOD49精确coverage summary')
        cr=[]
        for method,label in [('old_native','旧native'),('final_native','新native'),('surface_LOD','LOD研究')]:
            for state,domain,desc in [('solo','visible','solo原visible'),('joint_keep','visible','joint原visible'),('joint_keep','extra_eroded_visible','joint原额外erode1内域'),('joint_keep','reviewed','joint完整reviewed域')]:
                c=cv['all49'][method][state][domain]
                cr.append((label,desc,fmt(c['count']),fmt(c['mean']),fmt(c['min']),fmt(c['alpha_lt09']),fmt(c['alpha_lt05']),fmt(c['zero_alpha_le1e6'])))
        body+='<h3>49帧精确覆盖代价：没有零覆盖，不等于没有低alpha像素</h3>'+table(['资产','固定域','N','alpha mean','alpha min','alpha<.9','alpha<.5','alpha≤1e-6'],cr)
        n=cv['all49']['surface_LOD']['joint_keep']['visible'];i=cv['all49']['surface_LOD']['joint_keep']['extra_eroded_visible']
        body+='<p class="notice">LOD joint原visible中alpha&lt;.9为 '+fmt(n['alpha_lt09'])+'，比新native新增 '+fmt(n['LOD_new_lt09_vs_final_native'])+'；额外erode1内域仍新增 '+fmt(i['LOD_new_lt09_vs_final_native'])+'个，最小alpha '+fmt(i['min'])+'。该内域点在raw91，不在图示三pose。不能把3pose无孔外推成全49零孔，也不据此宣布全面采用；本轮不补洞、不调参。</p>'
    else:body+='<p class="notice">49帧孔计数另行核对；此处不宣称49帧零孔。</p>'
    body+='<p><strong>下载与复放：</strong><a href="surface_LOD_replay_bundle.zip">小型统一LOD replay bundle（约5.25MB）</a> · <a href="replay_bundle/manifest.json">S/F及已有环境依赖manifest</a> · <a href="replay_bundle/REPLAY_COMMAND.txt">实际命令</a>。裸canonical NPZ默认仍走普通密采样，不能单独复现LOD截图；需wrapper、posterior内参考相机/表面和canonical内最终材质。keep/delete共享同一S/F，同相机采样行精确一致。</p>'
    pub.load('scale/lod_material_SH_contract.json','LOD实际SH材质合同')
    pub.load('scale/lod_receipt.remote.json','LOD三pose与HR identity实际receipt')
    if (pub.art/'scale/lod_replay_smoke_receipt.json').is_file():pub.load('scale/lod_replay_smoke_receipt.json','LOD统一命令真实重放smoke')
    body+='<p>材质容器声明SH3，但本次45,314行全部非DC系数实际exact0；LOD未丢弃非零视角项。若将来非DC非零，当前wrapper会拒绝该输入。S是raw106参考UV构建，只验证该方向附近表面恢复与材质接入；49个同轨迹视角不证明封闭360°车辆或隐藏形状真值。保留2,778条祖先行也不等于整车已验证。</p>'
    pub.add('lod','唯一LOD视口对照：保持材质与表面，改采样频率',body)
    hr=pub.art/'scale/lod_preview/raw113_canonicalHR_fair.png'
    if hr.is_file():
        with Image.open(hr) as im:
            w=im.width//3;detail=im.crop((2*w,46,im.width,im.height))
        target=pub.out/'hr_street_detail.png';detail.save(target)
        pub.copy(hr,'images/originals')
        pub.add('hr-street','单独查看同相机的原生1920车体局部',pub.figure(target,'raw113，1920实际街景delete局部：同一C2+matte canonical资产，来自已完成的HR native joint输出；单独展示车体，不和旧480混算数值。')+'<p class="small">冻结背景native在HR仍有原有线状欠覆盖；这张图只用于车体诊断，不表示全场景HR质量优于480。</p>')


def link_previous_delivery(out):
    delivery=out.parent/'delivery';changes=[]
    banner='<aside id="current-a6-diagnosis-link" style="padding:16px 22px;background:#e9f5ff;border:2px solid #357da3;color:#12364a;font:16px/1.6 system-ui,sans-serif"><strong>本页保留前一轮快照。</strong> <a style="color:#064f7d;font-weight:700" href="../diagnosis/index.html">查看本轮修复与阴影归因（新街景对照、LOD研究分支和当前局限）</a></aside>'
    for name in ('index.html','viewer_fixed.html'):
        p=delivery/name
        if not p.is_file():continue
        text=p.read_bytes().decode('utf-8');before=sha(p)
        if 'id="current-a6-diagnosis-link"' in text:
            updated=re.sub(r'<aside id="current-a6-diagnosis-link".*?</aside>',lambda m:banner,text,count=1,flags=re.S)
        else:
            backup=WORKSPACE/'work/a6_artifact_diagnosis/report/prior_delivery_headers'/name
            backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.is_file():shutil.copy2(p,backup)
            updated=re.sub(r'(<body\b[^>]*>)',lambda m:m.group(1)+banner,text,count=1,flags=re.I)
            if updated==text:
                updated=re.sub(r'(<main\b[^>]*>)',lambda m:banner+m.group(1),text,count=1,flags=re.I)
            if updated==text:
                raise ValueError(f'No body/main navigation insertion point: {p}')
        if updated.count('id="current-a6-diagnosis-link"')!=1 or 'href="../diagnosis/index.html"' not in updated:
            raise ValueError(f'Previous report navigation missing or duplicated: {p}')
        if updated!=text:p.write_bytes(updated.encode('utf-8'))
        backup=WORKSPACE/'work/a6_artifact_diagnosis/report/prior_delivery_headers'/name
        actual=p.read_bytes()
        assert re.sub(rb'<aside id="current-a6-diagnosis-link".*?</aside>',b'',actual,count=1,flags=re.S)==backup.read_bytes()
        assert (p.parent/'../diagnosis/index.html').resolve().is_file()
        changes.append(dict(path=str(p),before_sha256=sha(backup),after_sha256=sha(p),only_banner_insertion_or_update=True,
            banner_count=1,diagnosis_href_in_banner_count=1,diagnosis_target_exists=True,strip_banner_byte_exact_to_original_backup=True))
    return changes


def find_final(pub,manifest_path):
    manifest={};origin=None
    if manifest_path and manifest_path.is_file():
        manifest=read(manifest_path);origin=str(manifest_path)
        pub.evidence_file(manifest_path,'固定三帧选图manifest')
    root=pub.art/'surface_fit/combined_v1/observed_fit'
    receipt=pub.local(manifest['receipt']) if manifest.get('receipt') else next((x for x in (root/'receipt.json',root/'formal/receipt.json') if x.is_file()),None)
    asset=pub.local(manifest['asset']) if manifest.get('asset') else next((x for x in (root/'complete_a6.npz',root/'formal/complete_a6.npz') if x.is_file()),None)
    # This explicit nine-file contract cannot change when more bundle files
    # arrive. The real parent directory supplies raw, never a basename-only
    # hardcoded caption or a recursive inventory/sort.
    expected=[root/'evaluation'/f'raw{raw}'/name for raw in FINAL_RAW_IDS for name in FINAL_FIGURE_NAMES]
    selected=[pub.local(x['path']) for x in manifest['figures']] if manifest.get('figures') else expected
    assert [p.resolve() for p in selected]==[p.resolve() for p in expected], 'Final figures must use the explicit three-view, three-file contract'
    if manifest.get('require_all_figures'):
        assert all(p.is_file() for p in selected),'A required pinned final figure is absent; retain previous publication'
    figures=[];bindings=[]
    for p in selected:
        if not p.is_file():continue
        match=re.fullmatch(r'raw(\d+)',p.parent.name)
        assert match and int(match.group(1)) in FINAL_RAW_IDS
        raw=int(match.group(1));description,columns=FINAL_FIGURE_DESCRIPTIONS[p.name]
        caption=f'raw{raw}，{description}'
        figures.append((p,caption))
        bindings.append(dict(raw=raw,path=str(p.resolve()),sha256=sha(p),caption=caption,panel_columns=columns))
    state=dict(manifest=origin,root=str(root),receipt=str(receipt) if receipt else None,asset=str(asset) if asset else None,
        available_figures=[str(x) for x,_ in figures],figure_selection='explicit_106_113_130_three_files_each',
        figure_bindings=bindings,section_emitted=False)
    if not receipt or not receipt.is_file() or not figures:return state
    j=pub.load(receipt,'联合修复正式receipt');state['section_emitted']=True
    body='<p>以下内容来自已取回的实际receipt与渲染图。该分支把候选matte、观测约束连续表面和40个BUILD的颜色拟合接回同一个持久3D资产；此处展示实际结果与审查入口，尚不据此声明采用、全域成像不变或达到用户参考图。</p>'
    scalars=[]
    selected=('status','colour_fit_steps','neural_training_steps','geometry_updates_during_colour_fit','unknown','seconds','PVD_weight_updates','BUILD_RGB_MAE')
    for k in selected:
        if k in j and isinstance(j[k],(str,bool,int,float)):scalars.append((esc(k),fmt(j[k])))
    if dig(j,'asset','rows') is not None:scalars.append(('asset.rows',fmt(j['asset']['rows'])))
    if scalars:body+=table(['正式每资产颜色拟合：receipt原字段','实际值'],scalars)
    smoke=root/'smoke/receipt.json'
    if smoke.is_file():
        sm=pub.load(smoke,'联合修复颜色smoke receipt')
        if sm.get('colour_fit_steps') is not None:
            body+='<p class="small">额外真实smoke '+fmt(sm['colour_fit_steps'])+'颜色步，'+fmt(sm.get('seconds'),3)+'秒；正式步数与smoke分列。优化的是每资产颜色，不是预训练神经权重。</p>'
    reviewpath=root.parent/'review.json'
    if reviewpath.is_file():
        review=pub.load(reviewpath,'联合修复正式审查与几何尾部风险')
        tail=dig(review,'tail_risk','tail_abs_gt06')
        if tail:
            body+='<h3>连续不等于几何已确认：高alpha内域的大位移</h3><p>正式审查记录了显著尾部风险，不能解释成只清理低alpha边缘。它们需要观测深度与遮挡约束的后续质量审阅，不由平均RGB小改善自动豁免。</p>'
            body+=table(['正式几何尾部审查','实际值'],[
                ('chart深度位移绝对值>0.6m',fmt(tail['count'])+' / '+fmt(dig(review,'tail_risk','delta_z_m','count'))+'（'+fmt(100*tail['fraction_of_chart'],3)+'%）'),
                ('该尾部prior alpha中位数 / alpha≤.9点数',fmt(tail['prior_alpha']['median'])+' / '+fmt(tail['prior_alpha_le09'])),
                ('该尾部到matte边界距离中位数',fmt(tail['matte_boundary_distance_HRpx']['median'],3)+' HRpx'),
                ('全chart最大前移 / 最大后移',fmt(-dig(review,'tail_risk','delta_z_m','min'))+' / '+fmt(dig(review,'tail_risk','delta_z_m','max'))+' m'),
                ('最大训练锚点残差',fmt(dig(review,'tail_risk','accepted_anchor_residual_m','max'))+' m')])
            body+=pub.figure(root.parent/'depth_posterior.png','正式连续表面与位移尾部审查；不是独立几何真值')
    evaluations=[]
    if manifest.get('evaluation'):evaluations=[pub.local(manifest['evaluation'])]
    else:evaluations=[x for x in (root/'evaluation/receipt.json',root/'evaluation/summary.json') if x.is_file()]
    evreceipts=[]
    for e in evaluations:
        if e.is_file():
            ev=pub.load(e,'联合修复实际评价 '+e.name)
            evreceipts.append(ev)
            if ev.get('frames') is not None:
                body+='<p class="small">实际固定'+fmt(ev['frames'])+'帧多分支前向验收，'+fmt(ev.get('seconds'),3)+'秒。PASS表示接线与规定检查完成，不是采用决定。</p>'
    hrpath=root/'evaluation/HR_summary.json'
    if hrpath.is_file():
        hr=pub.load(hrpath,'联合修复HR固定域summary');records=[]
        for domain,label in (('all49','all49混合开发域'),('HELD','HELD9内部开发域')):
            h=hr.get(domain,{})
            for method,name in (('old','旧rough-depth formal'),('final','新C2+matte formal')):
                rgb=h.get(method+'.premultiplied_RGB_error');straight=h.get(method+'.straight_RGB_error');alpha=h.get(method+'.alpha')
                if rgb and straight and alpha:
                    records.append((label,name,fmt(rgb['count']),fmt(rgb['mean']),fmt(straight['mean']),fmt(alpha['mean'])))
            old=h.get('old.premultiplied_RGB_error');new=h.get('final.premultiplied_RGB_error')
            if old and new:assert old['count']==new['count'],'HR comparison must have the same frozen domain'
        if records:body+='<h3>固定HR可见RGB域：保留真实分母</h3><p>内部开发数值，未缩小新alpha分母。initial仅有三视角记录，未混入完整49帧对比。</p>'+table(['域','资产','像素数','premul RGB误差','straight RGB误差','alpha均值'],records)
    sumpath=root/'evaluation/summary.json'
    if sumpath.is_file():
        sums=pub.load(sumpath,'联合修复native480与保留域summary')
        metrics=[];preservation=[]
        for arm,label in (('native_vs_old','native480 原算子'),('area_vs_old_area','480 相同面积控制')):
            a=dig(sums,arm,'all49')
            if not a:continue
            get=lambda k:a.get(k,{})
            rgb=get('solo.a6.rgb_error');oldrgb=get('solo.a6.baseline_rgb_error')
            depth=get('solo.a6.measured_depth_error');olddepth=get('solo.a6.baseline_measured_depth_error')
            if rgb and oldrgb and depth and olddepth:
                assert rgb['count']==oldrgb['count'] and depth['count']==olddepth['count']
                metrics.append((label,fmt(rgb['count']),fmt(oldrgb['mean']),fmt(rgb['mean']),fmt(depth['count']),fmt(olddepth['mean']),fmt(depth['mean']),fmt(get('scene.keep.a6.visible_coverage').get('mean'))))
            for scene_state,region,desc in [('keep','outside','keep outside'),('delete','outside','delete outside'),('keep','a37','keep a37'),('keep','road_visible','keep 可见road')]:
                rgb=get('scene.'+scene_state+'.preservation.'+region+'.parent_rgb_change')
                dep=get(scene_state+'_fixed_depth_preservation.'+('road' if region=='road_visible' else region)+'.expected_depth_parent_change_m')
                if rgb:
                    preservation.append((label,desc,fmt(rgb['above_002'])+' / '+fmt(rgb['count']),fmt(100*rgb['above_002']/rgb['count'],4)+'%',fmt(rgb['mean'],8),fmt(rgb['max']),fmt(dep.get('mean')),fmt(dep.get('max'))))
        if metrics:body+='<h3>同一480 / 同一行域：RGB与深度分别报告</h3><p>每行旧/新使用相同viewport策略和固定原始观测域。native480 RGB误差可退步；面积控制的收益与C2几何改动分别保留。新资产的joint可见alpha从native约.99883降至area约.97636，不能把变透明的收益等同几何变准。可见RGB和LiDAR不评价隐藏表面真实性。</p>'+table(['策略','RGB像素','旧RGB误差','新RGB误差','深度样本','旧LiDAR误差(m)','新LiDAR误差(m)','新joint alpha均值'],metrics)
        if preservation:body+='<h3>实际成像保留域：冻结背景参数仍可能改变遮挡</h3><p>下面比较同策略旧/新资产的49帧成像差。RGB阈值采用原receipt定义；outside expected-depth 的极端变化必须保留，不称全域精确保护。</p>'+table(['策略','原固定域','RGB差>.02 / N','比例','RGB差mean','RGB差max','depth差mean(m)','depth差max(m)'],preservation)
    if any(x.get('HELD_only_final_evaluation') is True for x in evreceipts):
        body+='<p class="notice">角色纠正：原评价receipt保留了 <code>HELD_only_final_evaluation=true</code> 历史字段。raw113等此前已用于诊断表示缺陷，本报告将HELD9统一标为内部开发诊断留出，不据该字段声称独立最终确认；原始指标与receipt不改。</p>'
    for p,caption in figures:body+=pub.figure(p,caption)
    if asset and asset.is_file():
        ah=sha(asset);matched=ah in set(strings(j))
        state.update(asset_sha256=ah,asset_hash_present_in_receipt=matched)
        body+='<p class="asset">'+pub.evidence_file(asset,'下载canonical研究资产 (.npz，默认普通密采样)')+'</p><p class="small">SHA256 <code>'+ah+'</code>；'+('与正式receipt字段一致。' if matched else '此为本地文件实算hash，正式绑定请核对下载receipt。')+' 裸NPZ不会自动启用LOD；复现LOD还需S/F posterior、renderer与依赖manifest，见 <a href="surface_LOD_replay_bundle.zip">统一replay bundle</a>。</p>'
    pub.add('combined','联合修复的实际3D结果',body)
    return state


def evaluation_roles(pub):
    """Keep role history separately; never alter scientific receipt bytes."""
    p=pub.art/'report/evaluation_roles.json'
    prior=read(p) if p.is_file() else {}
    history=list(prior.get('historical_receipt_flags',[]))
    source=pub.art/'surface_fit/combined_v1/observed_fit/evaluation/receipt.json'
    if source.is_file():
        r=read(source)
        if 'HELD_only_final_evaluation' in r:
            entry=dict(path=source.relative_to(pub.art).as_posix(),sha256=sha(source),
                HELD_only_final_evaluation=r['HELD_only_final_evaluation'],original_receipt_preserved=True)
            if entry not in history:history.append(entry)
    roles=dict(status='ROLE_HISTORY_EXPLICIT_INTERNAL_DEVELOPMENT_DIAGNOSTICS',
        BUILD='Colour fitting / observed constraints according to each original receipt.',
        HELD9='Not used for the recorded colour-fit objective; already used for representation-defect diagnosis and internal method development.',
        raw113_history=['Thin/sphere/surface representation defects were inspected.',
            'Current matte, depth-only, viewport and combined-native images are diagnostic evidence.'],
        independent_final_confirmation=False,untouched_generalization_test=False,
        hidden_surface_ground_truth=False,original_receipts_and_metrics_changed=False,
        historical_receipt_flags=history,
        role_notice='A historical final-evaluation flag does not undo prior development use. Numerical values are retained as internal comparisons.')
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(roles,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    pub.evidence_file(p,'评价角色历史（不改原receipt）')
    return roles


def build(args):
    art=args.artifacts.resolve();out=args.out.resolve()
    frame_manifest=args.frame_manifest or art/'frame_compare/native49_v1/manifest.json'
    frame_inputs=load_frame_compare(frame_manifest)
    baseline=scientific_baseline(out,art/'report/pre_slider_170c.html')
    pub=Publisher(args.artifacts.resolve(),args.out.resolve())
    hero=before_after(pub)
    findings=pub.load('vehicle/findings.json','车体来源归因')
    cpu=pub.load('vehicle/cpu_audit.json','采样 / 深度 / 颜色CPU审查')
    vehicle=pub.load('vehicle/gpu_result/vehicle/receipt.json','车体实际GPU分解')
    matte=pub.load('vehicle/matte_contract/matte_receipt.json','冻结SAM2单次matte')
    filled=pub.load('vehicle/matte_contract/filled_v2_receipt.json','filled_v2拓扑合同')
    app=pub.load('vehicle/matte_contract/application_receipt.json','matte-only v1实际对照')
    surface=pub.load('surface_fit/results/receipt.json','连续表面CPU拟合')
    shadow=pub.load('shadow/source_audit.json','原阴影来源审查')
    scale=pub.load('scale/results/receipt.json','双分辨率实际GPU对照')
    cov=pub.load('scale/covariance_cpu_review.json','legacy footprint CPU标定')
    if not all((findings,cpu,vehicle,matte,filled,app,surface,shadow,scale,cov)):
        raise RuntimeError('Core local evidence is incomplete; do not publish a partial diagnosis')
    # Exact source identity, not the aesthetically preferred SD board tile.
    teacher_sha=cpu['actual_input']['candidate_sha256']
    assert teacher_sha==matte['source_RGB_sha256']==filled['source_RGB_sha256']==app['input_teacher_sha256']
    body='<p>实际进入chart的是 <strong>hosted strongv2</strong>，SHA256 <code>'+teacher_sha+'</code>。SD的prompt2_seed3是另一条离线基线，不能拿SD候选图代替本资产的实际teacher。对齐同一raw106相机、同一1920像素映射后，prototype渲染已经出现弯曲和黑边；40-BUILD颜色拟合并非第一起因。</p>'
    body+=pub.figure('vehicle/gpu_result/vehicle/matched_teachers_106.png','同一reference106：实际strongv2 / 独立SD候选 / real+generated点色 / prototype native / formal native')
    body+=pub.figure('vehicle/source106_stages.png','reference106已出现缺陷：seed颜色→prototype native→除alpha诊断→40-BUILD点色→formal native；除alpha不能恢复已错的表面排序')
    f0=vehicle['frames'][0]
    body+=table(['已核实事项','实际证据'],[
        ('核心并非普遍premul/straight错误','核心alpha均值 '+fmt(cpu['normalization']['106_projective_surface']['mean_alpha_core'])+'；除alpha后的核心RGB平均变化 '+fmt(cpu['normalization']['106_projective_surface']['premul_vs_normalized_RGB_L1_core']['mean'])),
        ('保留prior不是三视角主因','106/113/130 retained最大alpha·T = '+ ' / '.join(fmt(x['retained_contribution_max']) for x in vehicle['frames'])),
        ('像素中心/单位往返自洽','HR回投最大误差 '+fmt(cpu['backprojection']['reprojection_HR_px_max'],8)+' px；这不验证先验深度本身正确')])
    pub.add('actual-input','1 · 先确认真正进入3D的2D输入',body)

    body='<p>旧流程按PVD对象域选点，却把候选图的黑色canvas也采成RGB，并对chart赋予0.99 opacity。因此部分外轮廓黑色变成了真正参与深度排序的不透明材质。洋红marker只改被标记来源的颜色，保留几何与opacity，显出车顶/侧面黑圈；它也会标出合法黑轮胎，<strong>不能按亮度删除所有暗色</strong>。</p>'
    body+=pub.figure('vehicle/gpu_result/vehicle/raw106/decomposition.png','原型 / 正式 / chart-only / retained-only / source-dark改洋红：来源标记包括真实暗轮胎，不是删除mask')
    body+=table(['raw106来源诊断','实际值'],[
        ('alpha>0.98且近黑输出像素（含合法暗材质）',fmt(f0['opaque_dark_output_pixels'])),
        ('其中source-dark alpha·T >0.9的像素',fmt(f0['source_dark_opaque_dark_pixels_mass_gt09'])),
        ('generated chart近黑采样行',fmt(cpu['sampling']['generated_sample_dark_rows'])),
        ('bilinear跨外canvas tap的行（独立边界采样项）',fmt(cpu['sampling']['generated_rows_with_bilinear_background_mask_tap']))])
    body+='<p>修复合同用<strong>同一strongv2的一次冻结SAM2分割</strong>区分前景与canvas，真实known优先。纹理采样先分别双线性过滤 RGB×alpha 和 alpha，再仅对有效alpha反预乘。黑轮胎属于前景；SAM logits不被当成物理opacity。filled_v2只填封闭孔，再并入known；不膨胀、不做半径搜索、不增加SAM调用。</p>'
    body+=pub.figure('vehicle/matte_contract/matte_review_panel.png','候选特定matte审查：真实known优先，保留暗色前景，排除PVD域内候选canvas')
    body+=pub.figure('vehicle/matte_contract/filled_v2_review_panel.png','filled_v2固定拓扑修订：仅填封闭孔，保持与外部相通的车底空隙')
    body+=table(['合同 / 独立对照','实际值'],[
        ('SAM调用 / neural权重更新', '1 / '+fmt(matte['model_training_updates'])),
        ('v1保护的dark foreground像素',fmt(matte['dark_foreground_pixels_protected'])),
        ('filled_v2封闭孔填充像素 / 新增SAM调用',fmt(filled['closed_raw_SAM_holes_filled_pixels'])+' / '+fmt(filled['SAM_predict_calls_added'])),
        ('filled_v2允许chart行 / 原chart行',fmt(filled['effective_allowed_chart_rows'])+' / '+fmt(filled['chart_rows']))])
    body+='<p>下面是 <strong>matte-only v1</strong> 的因果对照，几何XYZ/depth/sigma未变；不是最终filled_v2组合结果，也不是总质量分数。保护点的参数逐字节不变，遮挡变化仍会改变实际成像。</p>'
    body+=table(['raw','原近黑不透明像素','matte后（仍含轮胎）','known RGB mean Δ','known RGB max Δ'],[(str(x['raw']),fmt(x['before_opaque_dark']),fmt(x['matte_opaque_dark']),fmt(x['known_HR_RGB_delta']['mean'],8),fmt(x['known_HR_RGB_delta']['max'])) for x in app['frames']])
    body+='<p class="small">例如raw113的known RGB最大变化 '+fmt(app['frames'][1]['known_HR_RGB_delta']['max'])+'，alpha最大变化 '+fmt(app['frames'][1]['known_alpha_delta_maxabs'])+'；不能把“观测点参数保护”改称“所有观测图像精确不变”。</p>'
    pub.add('matte','2 · 黑canvas进入不透明材质；轮胎需要保留',body)

    body='<p>alpha归一的expected depth是层混合统计量，旧流程直接把逐像素均值当作连接表面。车体内相邻深度存在大跳变，渲染排序随之把直线纹理扭曲。保持raw106投影XY、角尺寸sigma、点数、opacity与颜色，仅改深度的平面oracle使线条变直；该平面只用于定位，<strong>不是物理修复或360°几何</strong>。40个BUILD的SH0拟合还会为粗糙排序写入颜色补偿，产生后续可见颗粒。</p>'
    body+=pub.figure('vehicle/gpu_result/vehicle/raw106/depth_oracle.png','reference106深度 / 排序oracle：比较prototype及formal颜色；平面是非物理诊断控制')
    body+=table(['粗糙深度 / 颜色证据','实际值'],[
        ('内域相邻深度差p99',fmt(cpu['depth_surface']['interior_adjacent_depth_difference_m']['p99'])+' m'),
        ('内域相邻差>10cm的对数',fmt(cpu['depth_surface']['interior_adjacent_pairs_gt10cm'])),
        ('surfel局部sigma中位数',fmt(cpu['depth_surface']['sigma_local_m']['median'])+' m'),
        ('40-BUILD后新增近黑real-seed / gen-seed行',fmt(cpu['observed_fit']['group_changes']['real_seed']['newly_dark_after_observed_fit'])+' / '+fmt(cpu['observed_fit']['group_changes']['generated_seed']['newly_dark_after_observed_fit']))])
    body+='<p>当前连续表面对照沿reference106射线拟合C2 B-spline深度，由BUILD可见LiDAR锚定，并保留先验与曲率项；没有自由调整pose或XY，也不后训练PVD。只有关联明确的观测进入拟合。下表是<strong>训练观测拟合</strong>，不能称泛化：平均误差下降，但最坏值变大，需保留审查。</p>'
    err=surface['measured_training_camera_z_abs_error_m']
    body+=table(['同一训练关联域camera-z绝对误差 (m)','原深度','连续表面'],[(x,fmt(err['prior'][x]),fmt(err['posterior'][x])) for x in ('count','mean','median','p99','max')])
    body+=pub.figure('surface_fit/previews/113/depth_only_comparison.png','raw113实际3D预览：原型 / matte支持原型 / 仅连续深度 / 仅matte颜色初始化 / 深度+matte初始化；尚未套用最终颜色拟合')
    body+='<p class="small">本图及raw113参与了表示缺陷分析。原始receipt中的HELD/不参与拟合标记保留；它们在本报告仅是内部开发诊断留出，不是严格未触碰的最终确认集。</p>'
    pub.add('depth','3 · 粗糙expected depth被当作表面；颜色拟合补偿其排序',body)

    body='<p>旧图库把1920黑底单体放大图与480全街景直接对照，同时混入遮挡与分辨率。此处只用同一资产、同一相机、同一A行，对106/113/130分别做480/1920 solo与joint；比较 <strong>native480 与 1920实际渲染的4×4面积下采样</strong>。这是跨分辨率自一致误差，绝非隐藏RGB真值。</p>'
    body+='<p class="small">本表使用固定旧rough-depth formal资产 '+esc(scale['asset_sha256'][:12])+'…；新C2+matte资产0b1f69…的频率表现单列于LOD对照，保留各自真实分母。</p>'
    scale_rows=[]
    for v in scale['views']:
        b=v['arms']['LR_baseline']['vs_area_downsample_HR'];c=v['arms']['LR_area_compensated']['vs_area_downsample_HR']
        scale_rows.append((str(v['raw']),fmt(b['solo']['object']['rgb_MAE']),fmt(c['solo']['object']['rgb_MAE']),fmt(b['joint']['object']['rgb_MAE']),fmt(c['joint']['object']['rgb_MAE']),fmt(v['compensation']['chart_factor']['mean'])))
    body+=table(['raw','solo原480 MAE','solo面积控制','joint原480 MAE','joint面积控制','chart系数均值'],scale_rows)
    body+=pub.figure('scale/results/raw106/base_vs_comp_fair480.png','raw106同480公平对照：原solo / 固定面积控制 / D4(HR solo) / 原joint / 面积控制joint')
    body+='<p>active源使用固定 <code>+0.3I</code> 的像素协方差滤波，未随采样面积归一。source以0.6HRpx标定的小球，到480的理想角尺寸约0.15px，当前最小footprint会使密集surfel相互覆盖得更强。此外legacy <code>neg_fov=True</code> 的footprint投影语义与新采样的物理fx标定不一致。</p>'
    body+=table(['raw','理想HR σ小/大','legacyHR未滤波 σ小/大','native480滤波后 σ小/大'],[(str(x['raw']),fmt(x['physical_pinhole_HR_unfiltered_sigma_minor']['median'],4)+' / '+fmt(x['physical_pinhole_HR_unfiltered_sigma_major']['median'],4),fmt(x['active_legacy_HR_unfiltered_sigma_minor']['median'],4)+' / '+fmt(x['active_legacy_HR_unfiltered_sigma_major']['median'],4),fmt(x['active_legacy_LR_filtered_sigma_minor']['median'],4)+' / '+fmt(x['active_legacy_LR_filtered_sigma_major']['median'],4)) for x in cov['views']])
    body+='<p>唯一面积控制为chart渲染行副本乘 <code>k=f_R/f_HR</code>，<code>f=√(det G / det(G+0.3I))</code>；HR严格为1。原资产opacity未写回，非chart行、全场景camera、背景与共享.so均未改。控制受 <a href="https://niujinshuchong.github.io/mip-splatting/">Mip-Splatting</a> 的面积归一化启发，未实现其完整方法。它降低三视角RGB自一致差异，但106 joint alpha误差略升；不能单独证明几何正确。canonical HR保留，480控制仅为viewport适配对照。</p>'
    pub.add('scale','4 · 480像素核与采样频率进一步放大外观差异',body)

    geometry_candidates=[]
    for base in (pub.art/'shadow',pub.art/'shadow/results'):
        if base.is_dir():geometry_candidates.extend(base.glob('conditional_geometry*.json'))
    geometry_candidates.sort(key=lambda p:('v2' in p.name,p.stat().st_mtime_ns),reverse=True)
    gp=geometry_candidates[0] if geometry_candidates else None
    geometry=pub.load(gp,'最新same-sort条件几何') if gp else None
    body='<p>原阴影在本轮<strong>只排查来源，未修改背景或道路</strong>。同相机同排序的分组RGB/alpha·T与中性颜色控制表明，两块暗色的直接载体不同。环境项是剩余射线的加性背景，不能把它解释为乘在车体或路面上的照明。</p>'
    body+=table(['固定暗域','直接载体 / alpha·T','保持opacity的中性颜色控制 Δ亮度指数'],[(str(x['raw'])+'（'+fmt(x['dark_core_pixels'])+'px）','static '+fmt(dig(x,'alpha_T','legacy_static'))+'；road '+fmt(dig(x,'alpha_T','road')),'static '+fmt(x['neutral_static_luma_delta'])+'；road '+fmt(x['neutral_road_luma_delta'])) for x in shadow['ROI_summary'] if x['dark_core_pixels']])
    body+='<p>raw105近路static暗色占位，不能仅凭图像位置把它叫当前车尾或车牌残留。条件中心深度/邻近道路切平面证据区分了这块与现存a37；这仍不追溯历史高斯的出生身份或证明真实物理阴影。raw113则以road输入RGB旧暗色为主；<strong>历史1K只训练static opacity gate，road RGB/geometry/opacity冻结</strong>，不能称那次1K训练新学出路面阴影。后来的fine颜色场另列。</p>'
    geo_rows=[]
    if geometry:
        core=dig(geometry,'fixed_domains','same_dark_core')
        if core:
            for label,path in [
                ('105 delete static camera-z (m)',('states','delete','groups','legacy_static','pooled_conditional_camera_z_m')),
                ('105 delete road camera-z (m)',('states','delete','groups','road','pooled_conditional_camera_z_m')),
                ('105 static相对已有路面中心高度 (m)',('states','delete','pooled_static_height_above_road_m')),
                ('105 keep a37 camera-z (m)',('states','keep','groups','a37','pooled_conditional_camera_z_m')),
                ('105同域a37关联LiDAR z均值 (m)',('same_frame_a37_associated_LiDAR_camera_z_m','mean'))]:
                value=dig(core,*path)
                if value is not None:geo_rows.append((label,fmt(value)))
        body+='<p class="small">条件几何来源：'+esc(gp.relative_to(pub.art).as_posix())+'；原JSON完整保留。</p>'
    if geo_rows:body+=table(['同域几何证据','实际值'],geo_rows)
    body+=pub.figure('shadow/shadow_source_controls.png','阴影来源独立对照：固定ROI，原10K删除a37 / road1K关fine / 开fine / static中性 / road中性；仅诊断，不是部署修复')
    body+='<p class="small">delete时a37 packed与active行均为0，物理移除与原masked删除输出一致；这些暗块仍可留在static/road。raw130同屏幕坐标没有同样暗核，它不是同一世界点的对应真值。无事实同相机“删除后干净道路”RGB，不能从中性颜色实验推断真实无阴影外观。</p>'
    pub.add('shadow','5 · 原阴影是独立问题：105近路static，113旧road RGB',body)

    manifest=args.final_manifest or pub.art/'report/final_result_manifest.json'
    final=find_final(pub,manifest)
    add_lod(pub)
    roles=evaluation_roles(pub)
    pub.add('architecture','数据流与当前边界',architecture()+'<p>2D图仅提供材质proposal。冻结PVD给出先验；连续表面改变的是每资产几何表示，40-BUILD拟合改变的是每资产颜色参数。单一a6-local资产沿轨迹进入真实native排序渲染，未使用屏幕覆盖。当前3pose及HELD9均已参与内部开发诊断，不构成独立最终确认或360°泛化证明。</p>')
    for rel,label in [
        ('scale/relative_area_asset.py','相对面积Asset wrapper'),('scale/scale_probe.py','双分辨率实际探针'),
        ('scale/surface_lod_asset.py','固定S/F频率采样Asset wrapper'),('scale/replay_surface_lod.py','LOD统一重放入口'),
        ('report/package_lod_replay.py','小型replay bundle打包器'),
        ('vehicle/gpu_decompose.py','车体GPU来源分解'),('vehicle/matte_contract/build_filled_v2_contract.py','fixed filled_v2合同'),
        ('vehicle/matte_contract/build_matte_texture_samples.py','premult/alpha纹理采样'),
        ('surface_fit/fit_surface.py','观测锚定连续表面'),('surface_fit/combine_rgba.py','matte与连续表面接合'),
        ('surface_fit/fit_observed.py','40-BUILD每资产颜色拟合'),
        ('shadow/diagnose_shadow_layers.py','原阴影分层控制'),('shadow/conditional_geometry.py','条件几何计算'),
        ('review/scale_forward_ledger.json','scale计算ledger'),('shadow/qa_receipt_v2.json','阴影QA receipt'),
        ('report/publish_diagnosis.py','本报告发布器源码')]:pub.evidence_file(rel,label)
    body='<p><strong>本轮PVD及所有预训练神经权重更新为0。</strong>SAM为冻结推理；颜色迭代是每资产参数拟合，连续深度是观测约束表面求解，均不等于neural weight training。原始图、JSON和代码分别保存，正文仅列支持结论的字段。</p>'
    body+=table(['工作','实际已记账','解释'],[
        ('车体分解',fmt(vehicle['seconds'],3)+' s','实际GPU forward；无优化'),
        ('matte SAM推理',fmt(matte['seconds'],3)+' s / 1 call','冻结SAM2.1；后续filled_v2无新增predict'),
        ('matte-only v1',fmt(app['seconds'],3)+' s','独立opacity/边界取样因果对照'),
        ('C2表面',fmt(surface['seconds'],3)+' s / '+fmt(surface['contract']['IRLS_solves'])+' IRLS solves','CPU求解；非神经权重训练'),
        ('scale',fmt(scale['render_calls'])+' forwards / '+fmt(scale['seconds'],3)+' s','计时含加载、CPU和保存；不冒充CUDA-only计时'),
        ('原阴影分层',fmt(shadow['actual_GPU_seconds'],3)+' s','本轮诊断，背景参数未更新')])
    extra=[]
    cl=pub.load('surface_fit/combined_v1/compute_ledger.json','C2+matte完整计算ledger')
    if cl:extra.append(('每资产颜色update计算',fmt(cl['method_formal_colour_steps'])+'正式 + '+fmt(cl['smoke_colour_steps'])+' smoke + '+fmt(sum(cl['disposable_exact_resume_nextstep_computations'].values()))+' disposable恢复核查 = '+fmt(cl['total_colour_update_computations']),'导出的正式资产仍在第1000步；神经权重更新0'))
    lp=pub.load('scale/lod_receipt.remote.json','LOD三pose计算receipt')
    if lp:extra.append(('LOD三pose',fmt(lp['render_calls'])+' forwards / '+fmt(lp['seconds'],3)+' s','含reference106 HR identity与既有旧资产113对照'))
    le=pub.load('scale/surface_lod_v1/evaluation49/receipt.json','LOD49计算receipt')
    if le:extra.append(('LOD固定49',fmt(le['frames'])+'帧 / '+fmt(le['seconds'],3)+' s','四策略原分母，零拟合；只4raw保存RGB，其余45保留metrics'))
    lc=pub.load('scale/surface_lod_v1/coverage49/receipt.json','LOD覆盖读出计算receipt')
    if lc:extra.append(('遗漏覆盖计数的必要读出',fmt(lc['additional_raster_forwards'])+' forwards / '+fmt(lc['seconds'],3)+' s','固定方法、固定mask，无补洞/调参'))
    lr=pub.load('scale/lod_replay_smoke_receipt.json','统一replay smoke计算receipt')
    if lr:extra.append(('统一入口重放smoke',fmt(lr['native_raster_calls'])+' forwards / '+fmt(lr['seconds'],3)+' s','keep/delete A行exact；街景PNG与49输出SHA exact'))
    if extra:body+=table(['额外开销 / 完整预算','实际记录','说明'],extra)
    body+='<div class="downloads">'+''.join('<p><a href="'+esc(x['href'])+'">'+esc(x['label'])+'</a><span class="small"> · '+esc(x['href'])+'</span></p>' for x in pub.evidence)+'</div>'
    pub.add('evidence','计算边界与下载',body)
    parent=pub.art/'report/parent_conclusion.json'
    if parent.is_file():
        conclusion=read(parent)
        if conclusion.get('approved') is True and conclusion.get('text'):
            pub.add('parent-review','最终审阅说明','<p>'+esc(conclusion['text'])+'</p>');pub.evidence_file(parent,'父审阅结论')
    toc='<nav>'+''.join('<a href="#'+a+'">'+esc(t)+'</a>' for a,t,_ in pub.sections)+'</nav>'
    sections=''.join('<section id="'+a+'"><h2>'+esc(t)+'</h2>'+b+'</section>' for a,t,b in pub.sections)
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>车体插回伪影：定位与修复对照</title><style>:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:#172c3c;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:36px 34px 80px}h1{font-size:32px;line-height:1.3;margin:8px 0 18px}h2{font-size:23px;line-height:1.4;margin:0 0 18px}p{max-width:1200px}section{margin-top:32px;padding:26px;background:white;border:1px solid #d5e0e7;border-radius:12px;scroll-margin-top:16px}nav{display:flex;flex-wrap:wrap;gap:10px 18px;margin:26px 0}nav a{font-size:14px}a{color:#0e5b83;text-decoration-thickness:1px;text-underline-offset:3px}figure{margin:24px 0}figcaption{font-size:14px;font-weight:600;color:#345368;margin-bottom:8px}img{display:block;width:100%;height:auto;border:1px solid #cfdae2;background:#f3f7fa}svg{width:100%;height:auto;background:#f4f8fb;border:1px solid #d5e0e7;border-radius:8px}.table{max-width:100%;overflow:auto;margin:20px 0}table{border-collapse:collapse;min-width:600px;font-size:14px}th,td{padding:10px 14px;border:1px solid #d1dde5;text-align:left;vertical-align:top}th{background:#edf4f8}tbody tr:nth-child(even){background:#f8fafc}.lead{font-size:18px;max-width:1100px}.notice{border-left:4px solid #b77520;background:#fff3df;padding:16px 20px;max-width:1200px}.small{font-size:13px;color:#526a7a}code{font:12px/1.6 ui-monospace,monospace;overflow-wrap:anywhere;background:#edf3f7;padding:2px 4px}.asset{font-size:19px;font-weight:700}.downloads{columns:2;column-gap:34px}.downloads p{break-inside:avoid;margin:9px 0}.downloads span{display:block;overflow-wrap:anywhere}@media(max-width:800px){main{padding:20px 12px}section{padding:18px 12px}h1{font-size:27px}.downloads{columns:1}}</style></head><body><main><div class="small">scene008 / a6 · 冻结PVD与预训练权重 · 实际native3D证据</div><h1>车体插回伪影：定位与修复对照</h1><p class="lead">缺陷不只来自“放回场景”。同一strongv2候选在reference106进入3D时已经发生黑canvas不透明化与深度排序弯曲；观测颜色拟合补偿粗糙表面，480像素核又放大采样差异。这些环节已有独立控制，原阴影则需要另行归因。</p><p class="notice">本页区分2D候选、实际3D渲染和诊断控制。研究修复只有实际文件齐备后才展示；不自动声明达到参考图、全域图像不变或正式采用。HELD仅说明拟合角色，相关帧已用于表示缺陷分析，属于内部开发诊断。</p>'''+toc+sections+'</main></body></html>'
    assert pub.uri_bytes<=MAX_IMAGE_URI_BYTES
    lead='<p class="lead">先看同分辨率街景。已查清2D→3D转换损失，并接入候选前景matte、连续表面与固定频率采样；黑环和保险杠粗波纹减轻。LOD更平滑，细线也略软。研究候选仍有覆盖代价、大深度位移及局部成像变化，未满足全面采用标准；背景暗团本轮只归因、未修补。</p>'+hero+'<p class="notice"><strong>当前研究候选：</strong>C2+matte canonical资产，另含LOD视口研究对照。area是有透明度损失的诊断分支，未自动采用。新增表面由raw106参考UV构建，49同轨迹视角不证明封闭360°车辆或隐藏形状真值；保留2,778条PVD祖先行也不等于整车已验证。</p>'
    page=re.sub(r'<p class="lead">.*?</p>',lambda m:lead,page,count=1,flags=re.S)
    frame_module,frame_receipt=publish_frame_compare(frame_inputs,pub.out,Path(__file__).with_name('frame_compare_controller.js'))
    page=inject_preserving_baseline(page,frame_module,baseline)
    blob=page.encode('utf-8');assert len(blob)<=MAX_HTML_BYTES,(len(blob),pub.uri_bytes)
    (pub.out/'index.html').write_bytes(blob)
    (pub.out/'architecture.svg').write_text(architecture(),encoding='utf-8')
    pub.copy(Path(__file__),'code')
    receipt=dict(status='READY_OFFLINE_REPORT_WITH_LOCAL_FRAME_DATASET',created_UTC=datetime.now(timezone.utc).isoformat(),
        artifacts=str(pub.art),output=str(pub.out),HTML_bytes=len(blob),embedded_image_URI_bytes=pub.uri_bytes,
        embedded_images=len(pub.images),JavaScript_required=dict(slider_interaction=True,initial_frame=False,static_fallback=False,scientific_report=False),external_images=0,
        delivery_dependencies='Original21 scientific images embedded; frame PNGs require the adjacent frames directory. Keep it when moving this report.',
        frame_compare=frame_receipt,scientific_baseline_SHA256='170ccef3351164a1ce8660a07d2f906f8e522fd406c9b0956de4b1eae56a0191',
        final_hook=final,evaluation_roles=roles,latest_conditional_geometry=str(gp) if gp else None,
        teacher_sha256=teacher_sha,evidence=pub.evidence,images=pub.images,
        PVD_neural_weight_updates=0,pretrained_neural_weight_updates=0,
        same_camera_internal_diagnostics=True,independent_final_confirmation_claim=False,
        automatic_adoption=False,parent_final_conclusion_separate=True)
    (pub.out/'publication_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    qa=verify(pub.out);qa['old_report_links']=link_previous_delivery(pub.out)
    (pub.out/'publication_qa.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    manifest={str(x.relative_to(pub.out)).replace('\\','/'):sha(x) for x in sorted(pub.out.rglob('*')) if x.is_file() and x.name!='sha256.json' and 'scale' not in x.relative_to(pub.out).parts[:1]}
    (pub.out/'sha256.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'],index=str(pub.out/'index.html'),HTML_bytes=len(blob),images=len(pub.images),final_section=final['section_emitted'],QA=qa['status']),ensure_ascii=False))


def verify(out):
    class Parse(HTMLParser):
        def __init__(self):
            super().__init__();self.sources=[];self.local=[];self.scripts=[];self.hidden=[];self.sections=[];self.frame_sources=[]
        def handle_starttag(self,t,a):
            attrs=dict(a)
            if t=='section':self.sections.append(attrs.get('id'))
            in_frames='frame-compare' in self.sections
            if t=='script':self.scripts.append(dict(id=attrs.get('id'),src=attrs.get('src'),in_frames=in_frames))
            for k,v in a:
                if k=='hidden':self.hidden.append(dict(id=attrs.get('id'),in_frames=in_frames))
                if k in ('href','src') and v and not(v.startswith('data:') or '://' in v or v.startswith('#')):
                    self.local.append(unquote(urlsplit(v).path))
                if t=='img' and k=='src':
                    if in_frames:self.frame_sources.append(v)
                    else:self.sources.append(v)
        def handle_endtag(self,t):
            if t=='section' and self.sections:self.sections.pop()
    p=out/'index.html';h=Parse();content=p.read_text(encoding='utf-8');h.feed(content)
    for x in h.sources:
        assert x.startswith('data:image/')
        im=Image.open(io.BytesIO(base64.b64decode(x.split(',',1)[1])));im.verify()
    assert all((out/x).is_file() for x in h.local)
    assert {x['id'] for x in h.scripts}=={'fc-data','fc-controller'} and len(h.scripts)==2
    assert all(x['in_frames'] and x['src'] is None for x in h.scripts)
    assert all(x['in_frames'] and x['id']=='fc-overlay' for x in h.hidden) and len(h.hidden)==1
    assert '未取回' not in content
    fc=read(out/'frame_compare_manifest.json');initial=fc['frames']['113']
    expected_initial=[initial['reference']['path'] if c['id']=='reference' else initial['delete'][c['id']]['path'] for c in fc['columns']]
    assert h.frame_sources==expected_initial
    assert all(x.startswith('frames/') and not x.startswith(('data:','http:','https:')) for x in h.frame_sources)
    stripped=re.sub(re.escape(FC_BEGIN)+r'.*?'+re.escape(FC_END),'',content,count=1,flags=re.S).encode('utf-8')
    assert hashlib.sha256(stripped).hexdigest()=='170ccef3351164a1ce8660a07d2f906f8e522fd406c9b0956de4b1eae56a0191'
    assert len(h.sources)==21
    receipt=read(out/'publication_receipt.json');final=receipt['final_hook']
    binding_checks=[]
    for b in final.get('figure_bindings',[]):
        source=Path(b['path'])
        assert source.parent.name==f"raw{b['raw']}"
        assert b['caption'].startswith(f"raw{b['raw']}，") and esc(b['caption']) in content
        assert sha(source)==b['sha256']
        records=[x for x in receipt['images'] if x['source']==str(source.resolve())]
        assert len(records)==1 and records[0]['caption']==b['caption'] and records[0]['source_sha256']==b['sha256']
        binding_checks.append(dict(raw=b['raw'],source=str(source),sha256=b['sha256'],caption=b['caption'],panel_columns=b['panel_columns']))
    if final.get('section_emitted'):
        assert [(x['raw'],Path(x['source']).name) for x in binding_checks]==[(raw,name) for raw in FINAL_RAW_IDS for name in FINAL_FIGURE_NAMES]
    return dict(status='PASS_STATIC_SCIENCE_PLUS_SCOPED_LOCAL_FRAME_UI',HTML_sha256=sha(p),
        image_data_URIs=len(h.sources),decoded_image_QA=True,local_links=len(h.local),
        local_link_QA=True,scripts=h.scripts,hidden_elements=h.hidden,HTML_bytes=p.stat().st_size,
        frame_initial_relative_sources=h.frame_sources,frame_initial_raw=113,frame_initial_state='delete',
        frame_manifest_complete=True,scientific_baseline_byte_exact_after_section_strip=True,
        explicit_final_figure_bindings=binding_checks,figure_binding_QA=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts',type=Path,default=WORKSPACE/'work/a6_artifact_diagnosis')
    ap.add_argument('--out',type=Path,default=WORKSPACE/'outputs/scene008_vehicle_appearance/diagnosis')
    ap.add_argument('--final-manifest',type=Path)
    ap.add_argument('--frame-manifest',type=Path)
    build(ap.parse_args())
