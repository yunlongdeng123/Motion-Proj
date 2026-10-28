"""用户指定蓝框内的局部白底诊断。仅空间贡献门控，不产生整颗Gaussian归属标签。"""
from pathlib import Path
import json, re, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
SOURCE = HERE.parent / 'recall_upperbound/recall_ceiling_r1'
OUT = WORKSPACE / 'outputs/V43_Ownership_20261027/gpu_v43/bounded_white_v1'
RAWS = (98, 113, 114)
# 用户第二张截图蓝色外边界，在480×320原像素坐标中的近似描摹；边界像素计入域内。
BLUE113 = np.array([[265,166],[265,157],[291,137],[319,134],[341,139],
                    [417,139],[422,147],[421,212],[382,244],[265,244]], dtype=float)
# 用户红框所示车底范围；只作为本次人工指定查询，不是自动阴影GT。
CONTACT113 = np.array([[276,203],[357,203],[357,236],[276,236]], dtype=float)
BODY113 = np.array([292,176,352,223], dtype=float)

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def bounds(mask):
    y,x=np.where(mask); return np.array([x.min(),y.min(),x.max()+1,y.max()+1])

def transport(points, box):
    return (points-BODY113[:2])/(BODY113[2:]-BODY113[:2])*(box[2:]-box[:2])+box[:2]

def raster(points):
    im=Image.new('1',(480,320)); ImageDraw.Draw(im).polygon([tuple(p) for p in np.rint(points).astype(int)],fill=1)
    return np.array(im,dtype=bool)

def display(rgb):
    return np.round(np.clip(rgb.transpose(1,2,0),0,1)*255).astype('uint8')

def gate(state,mask):
    # W=(1-M)*(G+T*B+T*E)+M*1。域内同时关闭Gaussian与Env贡献，白色为unknown占位。
    # 此处不是整行删除，因此跨界Gaussian的域外贡献严格保留；global owner一律不写回。
    rgb=state['native_rgb'].copy(); rgb[:,mask]=1.
    return rgb

def build():
    OUT.mkdir(parents=True,exist_ok=True); records=[]; contact=Image.new('RGB',(1440,3*352),(12,24,37))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for row,raw in enumerate(RAWS):
        src=SOURCE/f'{raw:03d}'; dest=OUT/f'{raw:03d}'; dest.mkdir(exist_ok=True)
        state=np.load(src/'existing/maps.npz'); before=np.load(src/'actor_omitted/maps.npz')
        native=display(state['native_rgb']); source_png=np.array(Image.open(src/'existing/native.png'))
        assert np.array_equal(native,source_png)
        body=np.load(src/'domains.npz')['maps'][2]>0; box=bounds(body)
        polygon=transport(BLUE113,box); allowed=raster(polygon)
        effect=raster(transport(CONTACT113,box))&allowed
        # existing 已有剥离产生的显示变化仅决定诊断白底支持；非检测器输入或训练标签。
        changed=np.max(np.abs(state['native_rgb']-before['native_rgb']),axis=0)>1/255
        clipped=(state['native_rgb']<0).any(0)|(state['native_rgb']>1).any(0)
        exposed=changed&clipped&allowed
        # 包含明确车身和人工车底查询；未知理由分列，避免将环境texel伪装成Gaussian。
        # 当前验收采用显式局部包络，避免仅按截断RGB阈值漏掉未饱和的彩色尾迹。
        # 保留蓝框上部正常天空；从车身最高点向上8源像素到蓝框底部整带暂缓显示。
        local_top=int(box[1]-8*(box[3]-box[1])/(BODY113[3]-BODY113[1]))
        mask=allowed & (np.arange(320)[:,None]>=local_top)
        compact=gate(state,mask); full=gate(state,allowed)
        arr=display(compact); full_arr=display(full)
        assert not np.any(mask&~allowed)
        assert np.array_equal(compact[:,~allowed],state['native_rgb'][:,~allowed])
        assert np.array_equal(arr[~allowed],native[~allowed])
        assert np.all(arr[mask]==255) and np.all(full_arr[allowed]==255)
        assert np.all(mask[effect])
        Image.fromarray(native).save(dest/'existing.png'); Image.fromarray(arr).save(dest/'white.png')
        Image.fromarray(full_arr).save(dest/'whole_boundary_white.png')
        Image.fromarray((mask*255).astype('uint8')).save(dest/'unknown_mask.png')
        Image.fromarray((allowed*255).astype('uint8')).save(dest/'allowed_mask.png')
        Image.fromarray(np.abs(arr.astype(int)-native.astype(int)).astype('uint8')).save(dest/'difference.png')
        overlay=Image.fromarray(native); draw=ImageDraw.Draw(overlay)
        draw.line([tuple(p) for p in polygon]+[tuple(polygon[0])],fill='#12baf5',width=2)
        red=transport(CONTACT113,box); draw.line([tuple(p) for p in red]+[tuple(red[0])],fill='#ff5964',width=2)
        overlay.save(dest/'scope.png')
        shutil.copy2(src/'original.png',dest/'original.png')
        np.savez_compressed(dest/'diagnostic_gate.npz',allowed=allowed,unknown_display_gate=mask,
                            user_contact_query=effect,existing_exposed_support=exposed,
                            output_rgb=compact,source_native_rgb=state['native_rgb'],polygon=polygon)
        dark=((native*.0+[.2126,.7152,.0722])*native).sum(2)/255<.2
        rec=dict(raw=raw,source_stage='existing',boundary_polygon=polygon.tolist(),body_bounds=box.tolist(),
            boundary_source='user blue screenshot traced at raw113; body-box affine transport for raw98/114; query only',
            allowed_pixels=int(allowed.sum()),unknown_display_pixels=int(mask.sum()),effect_query_pixels=int(effect.sum()),
            dark_effect_pixels_before=int((dark&effect).sum()),dark_effect_pixels_after=0,
            outside_boundary_changed_pixels=int(np.any(arr[~allowed]!=native[~allowed],axis=1).sum()),
            outside_boundary_float_max_delta=float(np.abs(compact[:,~allowed]-state['native_rgb'][:,~allowed]).max()),
            mask_leakage_pixels=int((mask&~allowed).sum()),unknown_rgb_min=int(arr[mask].min()),unknown_rgb_max=int(arr[mask].max()),
            display_artifact_query='manual local envelope: body top minus 8 scaled source pixels down to user boundary bottom, strictly clipped to blue polygon',
            local_top_y=local_top,
            global_gaussian_owner_modified=False,gaussian_unknown_labels_written=0,environment_parameters_modified=False,
            camera_local_contribution_gate=True,inpainting=False,automatic_detector=False,training_updates=0,human_verdict=None)
        write(dest/'receipt.json',rec);records.append(rec)
        for col,(name,pic) in enumerate([('existing 原合成',Image.fromarray(native)),('蓝框上限 / 红框车底',overlay),('限定区域 unknown 白底',Image.fromarray(arr))]):
            x,y=col*480,row*352;ImageDraw.Draw(contact).text((x+9,y+7),f'raw{raw} · {name}',font=font,fill='white');contact.paste(pic,(x,y+32))
    contact.save(OUT/'comparison.png')
    manifest=dict(schema='bounded-white-v1',records=records,semantic_recall_not_measured=True,
        purpose='用户指定局部剥离显示验收，非自动unknown归属或干净补景',source_run=str(SOURCE),
        display_equation='W = (1-M) * existing_native_RGB + M * [1,1,1]',
        global_Gaussian_labels_changed=False,environment_texels_relabelled=False)
    write(OUT/'manifest.json',manifest)
    arch='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 140"><rect width="1200" height="140" fill="#f3f7fa"/>
    <g font-family="Microsoft YaHei,sans-serif" fill="#233b50" text-anchor="middle"><rect x="12" y="15" width="330" height="110" rx="8" fill="#dfeaf3"/><text x="177" y="45">冻结 existing 分量</text><text x="177" y="78">高斯 G → 透射 T → 原环境 E</text><text x="177" y="104">同帧 / 同相机</text><path d="M350 70H425" stroke="#3d6785" stroke-width="3"/><polygon points="425,63 437,70 425,77"/>
    <rect x="445" y="15" width="310" height="110" rx="8" fill="#e9e0f2"/><text x="600" y="45">用户指定局部查询</text><text x="600" y="78">蓝框硬上限 → 车身 / 车底 / 已暴露域</text><text x="600" y="104">未知支撑 M，框外强制 0</text><path d="M763 70H835" stroke="#3d6785" stroke-width="3"/><polygon points="835,63 847,70 835,77"/>
    <rect x="855" y="15" width="333" height="110" rx="8" fill="#deeee5"/><text x="1020" y="45">空间门控与验收</text><text x="1020" y="78">域内白底 → 域外 existing 原值</text><text x="1020" y="104">零越界；不写回全局高斯身份</text></g></svg>'''
    (OUT/'architecture.svg').write_text(arch,encoding='utf-8')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>V43 蓝框限定白底验收</title><style>body{font:16px/1.6 system-ui;background:#0d1723;color:#d9e8f5;margin:24px}a{color:#7ac8ff}button,select{font:inherit;padding:8px;background:#1c3144;color:white;border:1px solid #6c9bb8;border-radius:5px}.row{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%;display:block}figure{margin:0}h1{font-size:26px}.notice{padding:12px;background:#193246}.small{font-size:14px;color:#b3c8d8}#receipt{white-space:pre-wrap}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #36536b}@media(max-width:800px){.row{grid-template-columns:1fr}}</style>
    <h1>蓝框限定：existing → 局部 unknown 白底</h1><p class="notice">按用户最新边界：蓝框外禁止新增未知显示。框内车底黑影与选定彩色噪声均显示白色占位；环境项同时关闭。框外与 existing 逐像素相同。<b>这是局部贡献门控诊断，不是自动归属或补景结果。</b></p>
    <p><a href="../../../GS_Scene_Diagnostics_20261027/index.html">回共享入口</a> · <a href="../recall_ceiling_v1/index.html">此前扩域实验（历史对照）</a> · <a href="comparison.png">三帧完整对照</a> · <a href="manifest.json">数据与边界凭据</a></p>
    <label>帧 <select id="raw"><option>113</option><option>98</option><option>114</option></select></label> <label>白底范围 <select id="mode"><option value="white">车身到车底局部带（含彩噪）</option><option value="whole_boundary_white">整个蓝框上限（范围对照）</option></select></label>
    <p id="status"></p><div class="row"><figure><h2>existing 原合成</h2><img id="existing"><figcaption>只以第二列 existing 为基线。</figcaption></figure><figure><h2>限定区域白底诊断</h2><img id="output"><figcaption>白色表示暂缓归属的局部显示空洞。不是恢复出的白色路面。</figcaption></figure></div>
    <details><summary>边界、unknown 显示支撑与差分</summary><div class="row"><figure><h3>蓝色允许域；红色车底查询</h3><img id="scope"></figure><figure><h3>本帧 unknown 显示支撑</h3><img id="mask"></figure></div><p>蓝框由用户截图近似描摹；另两帧按原车身框平移缩放，未验证跨视角三维归属。没有把整颗跨界高斯全局打成 unknown，也没有把环境 texel 伪装成高斯。</p><a id="float">下载浮点输出与局部门控</a><pre id="receipt"></pre></details>
    <h2>实现范围</h2><img src="architecture.svg"><p>默认未知显示支撑采用显式局部包络：车身最高点向上 8 个源像素起，向下到蓝框底部，严格裁在蓝框内。这样将未饱和的彩色尾迹也包含进去；蓝框上部天空保留。本版不按颜色阈值假装检测准确，也不将环境纹理当成高斯身份。显示公式 W=(1−M)×existing RGB+M×白色。这是用户指定查询的区域验收，不反哺检测器或训练标签；本版不证明召回率或泛化性。界外若原来已有噪声，按边界要求原样保留。</p>
    <script>const data=MANIFEST;function update(){const n=+document.getElementById('raw').value,p=String(n).padStart(3,'0'),mode=document.getElementById('mode').value,r=data.records.find(x=>x.raw===n);for(const [id,f] of Object.entries({existing:'existing.png',output:mode+'.png',scope:'scope.png',mask:mode==='white'?'unknown_mask.png':'allowed_mask.png'}))document.getElementById(id).src=p+'/'+f;document.getElementById('status').textContent='raw'+n+' · 蓝框外变化 0 像素 · 框外浮点差 0 · 车底查询区黑色残留 0 · 此结果为显示门控';document.getElementById('receipt').textContent=JSON.stringify(r,null,2);document.getElementById('float').href=p+'/diagnostic_gate.npz'}document.getElementById('raw').onchange=update;document.getElementById('mode').onchange=update;update()</script></html>'''.replace('MANIFEST',json.dumps(manifest,ensure_ascii=False))
    (OUT/'index.html').write_text(html,encoding='utf-8')
    # 同一共享入口优先展示当前收紧版本；历史结果保留，但不作为当前验收。
    for path,url in [(WORKSPACE/'outputs/GS_Scene_Diagnostics_20261027/index.html','../V43_Ownership_20261027/gpu_v43/bounded_white_v1/index.html'),
        (WORKSPACE/'outputs/GS_Scene_Diagnostics_20261027/14705303724557273004_3105_000_3125_000/index.html','../../V43_Ownership_20261027/gpu_v43/bounded_white_v1/index.html'),
        (OUT.parent/'recall_ceiling_v1/index.html','../bounded_white_v1/index.html')]:
        text=path.read_text(encoding='utf-8'); key='bounded-white-link'; start=f'<!-- {key}-start -->';end=f'<!-- {key}-end -->'
        addition=f'{start}<div style="padding:15px;background:#19374b;color:#edf7ff"><strong>当前验收：蓝框外不动、框内黑影与彩噪白底显示。</strong> <a style="color:#80d6ff" href="{url}">打开三帧限定区域验收 →</a> 先前扩域结果保留为历史对照。</div>{end}'
        if start in text:text=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda m:addition,text,flags=re.S)
        else:text=re.sub(r'(<body[^>]*>)',lambda m:m.group(1)+addition,text,count=1)
        path.write_text(text,encoding='utf-8')
    print(json.dumps(records,ensure_ascii=False,indent=2))

if __name__=='__main__':build()
