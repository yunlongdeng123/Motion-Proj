"""Chinese scientific report generated from actual measured outputs."""
from pathlib import Path
import json,csv,shutil,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/waymo_val029_experiment'
AS=OUT/'assets'
FONT=Path('C:/Windows/Fonts/msyh.ttc')
def font(n):return ImageFont.truetype(str(FONT),n)
def frame(raw,stage,name='delete.png'):return Image.open(AS/'frames'/f'{raw:03d}'/stage/name).convert('RGB')
def label(draw,xy,text,n=24,fill='#142c3a'):draw.text(xy,text,font=font(n),fill=fill)
def compare(raw,name,columns):
    W=2040;col=500;margin=10;canvas=Image.new('RGB',(W,786),'#f2f5f7');d=ImageDraw.Draw(canvas)
    for i,(title,pic) in enumerate(columns):
        x=10+i*510;label(d,(x,14),title,22);canvas.paste(pic.resize((500,333),Image.Resampling.BICUBIC),(x,55));crop=pic.crop((280,125,450,250)).resize((500,368),Image.Resampling.NEAREST);canvas.paste(crop,(x,404));d.rectangle((x+280*500/480,55+125*333/320,x+450*500/480,55+250*333/320),outline='#eab143',width=3)
    canvas.save(OUT/name)
def main():
    OUT.mkdir(exist_ok=True);m=json.loads((OUT/'metrics.json').read_text(encoding='utf8'));audit=json.loads((OUT/'reveal_audit.json').read_text(encoding='utf8'))
    compare(155,'query155_comparison.png',[('原始相机观测 · 155',frame(155,'final_all_eval','observed.png')),('仅删车辆来源分支',frame(155,'formal10k_eval')),('跨帧归属后删除',frame(155,'semantic_all_eval')),('表面与道路组合后删除',frame(155,'final_all_eval'))])
    compare(190,'negative190_comparison.png',[('原始相机观测 · 190',frame(190,'final_all_eval','observed.png')),('基础重建 · 保留',frame(190,'formal10k_eval','keep.png')),('完整组合 · 保留',frame(190,'final_all_eval','keep.png')),('完整组合 · 删除',frame(190,'final_all_eval'))])
    # Original photographs, actual resolved masks, and measured correspondences.
    v=json.loads((ROOT/'outputs/waymo_edit_case_selection/verified_correspondences.json').read_text(encoding='utf8'));canvas=Image.new('RGB',(1400,600),'#f2f5f7');d=ImageDraw.Draw(canvas)
    for j,raw in enumerate([155,190]):
        pic=Image.open(AS/'frames'/f'{raw:03d}/source_full.png').resize((960,640),Image.Resampling.LANCZOS).convert('RGB');pts=np.array([p['query_uv' if j==0 else 'reference_uv'] for p in v['pairs']])
        from scipy.ndimage import binary_erosion
        masks=np.array(Image.open(ROOT/'work/val029_gpu/masks_pilot_resolved_r2'/f'{raw:03d}/labels.png').resize((960,640),Image.Resampling.NEAREST));pixels=np.array(pic)
        for actor,colour in [(70,[255,153,38]),(25,[0,222,255]),(20,[61,230,144])]:
            mask=masks==actor;edge=mask&~binary_erosion(mask,iterations=2);pixels[edge]=colour
        pic=Image.fromarray(pixels);draw=ImageDraw.Draw(pic)
        for x,y in pts:draw.ellipse((x-2,y-2,x+2,y+2),fill='#00e5ff',outline='#102a38')
        crop=pic.crop((560,250,910,500)).resize((680,486),Image.Resampling.LANCZOS);canvas.paste(crop,(j*700+10,75));label(d,(j*700+10,8),'遮挡时刻 · FRONT / 155' if j==0 else '另一真实观测 · FRONT / 190',27);label(d,(j*700+10,43),'橙色 删除红车　青色 SUV 表面　绿色 保持灰车',20)
    label(d,(15,568),'青点为同一 SUV 表面的 67 个实测对应；跨时刻照片提供稀疏证据，不是删除图像的逐像素真值。',20);canvas.save(OUT/'measured_correspondence.png')
    shutil.copy2(ROOT/'work/val029_gpu/mask_qa_resolved.png',OUT/'segmentation_four_frames.png')
    # Actual per-frame errors, with all positive and negative frames retained.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fp=FontProperties(fname=str(FONT));rows=list(csv.DictReader((AS/'evidence/analysis/per_frame_regions.csv').open(encoding='utf8')));fig,axes=plt.subplots(1,2,figsize=(12,3.5),layout='constrained')
    for ax,region,title in zip(axes,['revealed_white_suv','preserved_gray'],['后方白色 SUV · 62 个可见时刻','右侧灰车 · 77 个可见时刻']):
        chosen=[r for r in rows if r['stage']=='final' and r['region']==region and int(r['pixels'])>0];x=[int(r['raw']) for r in chosen];ax.plot(x,[float(r['native_keep_gt_mae']) for r in chosen],color='#64748b',label='基础重建');ax.plot(x,[float(r['keep_gt_mae']) for r in chosen],color='#ce5b35',label='完整组合');ax.set_title(title,fontproperties=fp);ax.set_xlabel('原始帧号',fontproperties=fp);ax.set_ylabel('可见轮廓内 RGB 绝对误差',fontproperties=fp);ax.grid(alpha=.2);ax.legend(prop=fp,frameon=False)
    fig.savefig(OUT/'visible_vehicle_errors.png',dpi=180);fig.savefig(OUT/'visible_vehicle_errors.svg');plt.close(fig)
    # Simple architecture components, following the user's diagram preference.
    boxes=[(20,80,180,'真实驾驶录像','图像 · 标定 · 测距 · 轨迹'),(240,80,170,'观测与重建','158 帧构建 · 40 帧留出'),(450,80,170,'跨帧对象归属','静态和动态共同统计'),(660,20,190,'车辆表面恢复','对象坐标 · 新 SUV 资产'),(660,150,190,'道路表面恢复','世界坐标 · 已测 / 推断'),(900,80,190,'统一深度渲染','同一持久场景 · 重算遮挡'),(1140,80,160,'成对编辑结果','保留 / 删除 · 质量验收')]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1320 260"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#54717e"/></marker></defs><rect width="1320" height="260" fill="#f2f5f7"/>']
    for x,y,w,t,s in boxes:svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="85" rx="10" fill="white" stroke="#a9bec8"/><text x="{x+w/2}" y="{y+33}" text-anchor="middle" font-family="Microsoft Yahei,sans-serif" font-size="20" fill="#142e3b">{t}</text><text x="{x+w/2}" y="{y+60}" text-anchor="middle" font-family="Microsoft Yahei,sans-serif" font-size="13" fill="#58707b">{s}</text>')
    for path in ['M200 122H230','M410 122H440','M620 122H640V62H650','M620 122H640V192H650','M850 62H875V122H890','M850 192H875V122H890','M1090 122H1130']:svg.append(f'<path d="{path}" fill="none" stroke="#54717e" stroke-width="2.2" marker-end="url(#a)"/>')
    svg.append('</svg>');(OUT/'architecture.svg').write_text(''.join(svg),encoding='utf8')
    def weighted(region,field='keep_gt_mae'):return m['stages']['final']['regions'][region][field]['pixel_weighted']
    values=[]
    for role,region in [('后方白色 SUV','revealed_white_suv'),('右侧灰车','preserved_gray'),('全部其他车辆','all_other_vehicles')]:
        r=m['stages']['final']['regions'][region];values.append(f'<tr><td>{role}</td><td>{r["frames"]} / {r["pixels"]:,}</td><td>{r["native_keep_gt_mae"]["pixel_weighted"]:.5f}</td><td>{r["keep_gt_mae"]["pixel_weighted"]:.5f}</td><td>{r["frames_improved_vs_native"]} 改善 / {r["frames_worse_vs_native"]} 上升</td></tr>')
    roadrows=[]
    for stage,name in [('native','仅删来源分支'),('semantic','跨帧归属删除'),('asset','再加入 SUV 表面'),('final','再加入道路与交接门')]:
        r=m['road_proxy'][stage];a=m['sparse_reveal_anchors'][stage];roadrows.append(f'<tr><td>{name}</td><td>{r["real_domain"]["mae"]:.5f}</td><td>{r["inferred_domain"]["mae"]:.5f}</td><td>{a["render_depth_mae_m"]:.3f} m</td></tr>')
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>独立驾驶街段的车辆删除实验</title><style>
    :root{{--ink:#17303d;--muted:#526975;--line:#d7e1e6;--accent:#b64e2d}}*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f8;color:var(--ink);font:17px/1.85 system-ui,"Microsoft Yahei",sans-serif}}main{{max-width:1180px;margin:auto;padding:50px 36px 90px;background:white}}h1{{font-size:38px;line-height:1.4;max-width:950px}}h2{{font-size:25px;margin-top:48px}}h3{{font-size:20px}}p{{max-width:970px}}.eyebrow,.muted,figcaption{{color:var(--muted)}}.eyebrow{{letter-spacing:2px;font-size:13px}}.lead{{font-size:21px}}.verdict{{border-left:5px solid var(--accent);background:#fff4ed;padding:18px 24px}}.chips{{display:flex;flex-wrap:wrap;gap:10px}}.chips span{{background:#edf4f7;border:1px solid var(--line);padding:5px 12px;border-radius:5px;font-size:14px}}figure{{margin:28px 0}}img{{width:100%;height:auto;border:1px solid var(--line);border-radius:6px}}figcaption{{font-size:14px;line-height:1.7;margin-top:8px}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{text-align:left;padding:12px;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#edf4f7}}.scroll{{overflow-x:auto}}video{{width:100%;background:#111827;border-radius:8px}}a{{color:#126b8b}}code{{font-size:14px;overflow-wrap:anywhere}}details{{padding:14px 18px;background:#f5f8fa;margin:15px 0}}footer{{margin-top:45px;border-top:1px solid var(--line);font-size:14px;color:var(--muted)}}@media(max-width:700px){{main{{padding:25px 16px}}h1{{font-size:28px}}body{{font-size:16px}}td,th{{padding:8px}}}}
    </style><main><div class="eyebrow">独立街段实验 · 2026-11-08 · WAYMO 验证集</div><h1>删除路口右侧红车后，<br>后方 SUV 能否正确显露？</h1>
    <p class="lead">一段新录像已完成分割核验、真实 GPU smoke、10,000 步重建、车辆与道路资产构建，以及完整保留／删除实验。车辆曲面与道路预处理有适配差异，因此本轮不能称为完全冻结方法的迁移验证。</p><div class="chips"><span>198 个原始时刻</span><span>158 构建 + 40 连续留出</span><span>前相机 · 480 × 320 渲染</span><span>未使用 subagent</span></div>
    <p class="verdict"><strong>本次自动适配版本未通过质量验收。</strong>删除目标的直接贡献归零、同一场景可重复编辑等机械检查通过，但白色 SUV 的总体可见外观误差上升，灰车 77 个可见时刻中有 72 个退化。已恢复的 SUV 表面放回整场景后仍受到其他表示干扰，不能把目标消失等同于正确显露。</p>
    <h2>一段新录像如何变成可编辑场景</h2><p>输入相机图像、相机标定、激光测距和对象轨迹。首先建立带时间的三维场景，再用构建时刻的可见贡献，把静态和动态高斯共同分配给实例。道路在世界坐标恢复，车辆在各自对象坐标恢复。编辑时复用同一套资产，只移除目标实例，并通过一次统一深度排序重新计算遮挡。</p><figure><img src="architecture.svg" alt="输入到场景恢复与统一编辑渲染的组件图"><figcaption>本轮执行的是当前候选系统的新场景实例化。新 SUV 的参考曲面和道路图表采用自动适配器，见下文差异说明。</figcaption></figure>
    <h2>实验中是哪几辆车</h2><p>待删除的是棕榈树路口右侧等待的红色轿车。它遮住后方白色 SUV 的一部分车身；右侧灰车作为保持对象。查询时刻为原始第 155 帧，第 190 帧提供同一 SUV 表面的真实观测。三辆车只是本次验收角色，分割和跨帧归属规则对所有可用对象一致执行。</p><figure><img src="measured_correspondence.png" alt="两次原始观测及同一表面的实测对应"><figcaption>同一 SUV 局部坐标内的 67 个激光测量锚点。它们来自构建集，可检查表面与遮挡关系，不构成独立隐藏区域真值。</figcaption></figure>
    <h2>四帧分割与 smoke</h2><p>第 130、155、160、190 帧均做过视觉核查。仅用二维框的第一版会把红车收入 SUV 掩码；改用同帧实测点提示，并按三维框射线入深度解决掩码重叠后，三辆车的可见区域得到分离。参考帧的 67 个锚点均落在 SUV 掩码内，查询帧有 65 个落在红车遮挡轮廓内。这里没有人工逐像素分割真值，也未代填人工质量 verdict。</p><figure><img src="segmentation_four_frames.png" alt="四帧分割核验"><figcaption>分割核验通过后才导出全部 158 个构建时刻的监督。框、点提示和射线重叠解析仍有标定近似与轮廓误差。</figcaption></figure>
    <p>真实 GPU smoke 持续 <strong>603.67 秒</strong>，完成 5,914 步并保存完整检查点。随后完成评价、优化器／随机数／相机采样／运行状态的精确恢复，再执行 2 步进入后续阶段。共 <strong>5,916 次工程更新</strong>，与正式 10,000 步分别记账。峰值保留显存约 5.69 GiB。道路交接门另在第 32 步保存并精确恢复，累计正式 1,000 步。</p>
    <h2>从来源删除到完整组合</h2><figure><img src="query155_comparison.png" alt="查询帧各阶段删除对照"><figcaption>第 155 帧，同一相机。上排为全图，下排为同一固定区域放大。仅删除来源分支时红车基本仍在；跨帧归属处理静态与动态表示后，红车明显消失。完整组合仍需接受显露与保持检查。</figcaption></figure>
    <p>跨 158 帧的可见贡献多数规则识别出红车的 813 个静态高斯和 46 个动态高斯，阈值为严格大于 0.5。未分配动态高斯保留原来源身份。SUV 原生表示退出后，接入本场景的 PVD 先验、实测深度约束的 C² 曲面、固定几何材质拟合与 SurfaceLOD 采样；道路使用新街段实测点和跨时刻真实颜色。保留与删除共用资产，未按屏幕掩码拼接结果图。</p>
    <div class="scroll"><table><thead><tr><th>累计阶段</th><th>实测道路代理误差 ↓</th><th>推断道路代理误差 ↓</th><th>67 锚点混合渲染深度误差 ↓</th></tr></thead><tbody>{''.join(roadrows)}</tbody></table></div><p class="muted">道路两域分别为 2,436 和 4,185 个跨帧像素样本，均使用相同分母。颜色代理来自构建观测或邻近插值，不是独立删除真值。深度是统一光栅的 alpha 加权读出；最后一列仅针对第 155 帧的稀疏锚点。</p>
    <h2>恢复了资产，为什么仍没有正确显露</h2><p>67 个实测锚点到 SUV 资产最近表面的平均距离为 <strong>2.30 厘米</strong>。SUV 单独渲染时，这些位置平均不透明度为 0.993，深度误差为 <strong>4.56 厘米</strong>；放回完整场景后，深度误差为 <strong>3.36 米</strong>，SUV 自身贡献约 47.7%，其他已有表示贡献约 52.3%。该只读控制表明，当前主要限制涉及场景表示交接和遮挡竞争。它没有把其他贡献进一步唯一归因为某一种高斯来源，也没有证明资产在所有视角完整正确。</p>
    <h2>保留对象的完整分母</h2><div class="scroll"><table><thead><tr><th>原观测可见区域</th><th>帧数 / 像素样本</th><th>基础重建 MAE ↓</th><th>完整组合 MAE ↓</th><th>逐帧变化</th></tr></thead><tbody>{''.join(values)}</tbody></table></div><p>这里比较的是<strong>保留状态对原相机图像</strong>的 RGB 误差，按像素加权。SUV 有 41/62 帧改善，但后段更大的车身区域退化，使总体误差从 0.07534 升至 0.09447。灰车从 0.04979 升至 0.06002，72/77 帧上升。只报改善帧数或整图平均值会掩盖这些问题。</p><figure><img src="visible_vehicle_errors.png" alt="全部可见时刻的车辆误差曲线"><figcaption>全部有可见掩码的时刻，没有删去负例。两条曲线均对原始相机观测评价。</figcaption></figure><figure><img src="negative190_comparison.png" alt="后段负例的保留删除对照"><figcaption>第 190 帧的负例。完整组合中 SUV、灰车及目标车保留外观均有退化；原始图像和基础重建共同保留，便于判断代价。</figcaption></figure>
    <h2>留出、道路支持与通过的检查</h2><p>原始 79–118 帧连续 40 个时刻的 RGB 和测距未进入重建、分割监督、跨帧融合或资产拟合。拟合完成后只解码留出 RGB 做评价，没有解码留出激光。相机沿用构建集归一化；在 6 个构建相机上核对变换矩阵，最大差为 0。留出整图 MAE 为基础重建 0.11659、完整组合 0.11159，明显高于构建集。该窗口没有白色 SUV 的官方前向可见帧，因此不能据此宣称 SUV 显露在独立时刻通过验证。</p><p>道路图表为 2 厘米网格，44,247 个单元中有 7,823 个具有至少三个一致真实观测，18,468 个标记为最多 0.6 米邻域内的沥青推断；17,956 个保持未知。用于几何的 4,965 个道路测量点中有 4,381 个平面内点，拟合残差约 3.72 厘米。推断不是测量，也不说明车底隐藏路面已恢复正确。</p>
    <ul><li>158 构建时刻、40 留出时刻均完成成对保留／删除和保留复放；复放结果逐值相同。</li><li>目标实例的直接贡献归零。最终构建帧分组贡献和总 alpha 的最大差约 1.73 × 10⁻⁶。</li><li>基础场景参数未在只读评价和交接门优化中改写。</li><li>目标掩码为空的 82 帧中，51 帧仍有删除引起的像素变化，最大为 2,348 像素；这是外延与归属稳定性的警示，不能当作正确编辑。</li></ul>
    <h2>完整视频</h2><p>从第 0 到 197 帧按原始 10 Hz 连续排列，留出段已按原时刻插回。视频内依次为原始观测、基础重建、完整组合保留、完整组合删除。</p><video controls preload="metadata" src="assets/media/full198_comparison.mp4"></video><p>下方固定局部放大覆盖 119–197 帧。放大采用最近邻，便于观察原渲染像素。</p><video controls preload="metadata" src="assets/media/event119_197_crops.mp4"></video>
    <h2>实现边界、账本与复现</h2><p>这次完成了新街段的端到端实例化，<strong>尚不是严格冻结全部预处理代码的迁移成功</strong>。原街段的人工参考曲面准备改为最大可见面积参考视图和自动深度关联；道路准备改为自动局部稳健测量平面。保留了原生监督、SGCM、NPS、采样规则与 30K 学习率时钟，正式在 fresh 10K 停止；完整动态 alpha MSE 系数为 100，整图 RGB 与静态 RGB 系数均为 1。本轮没有重新拟合原街段资产，也没有复制原街段颜色系数。MOG-2 的条件生成道路场、严格不透明求交与反事实可见性训练损失仍未实现。</p>
    <p>计算分别记账为原生正式 10,000 步、工程 smoke／恢复 5,916 步、冻结 PVD 先验 1,000 次扩散推断、曲面 5 轮 IRLS、固定几何材质 1,000 次 FISTA、道路交接门 32 + 968 = 1,000 步。PVD 扩散步和材质求解步不是神经训练更新。原生初始化、环境入口、dtype 和只读加载的失败现场均保留；零更新失败不计入训练步数。材质设计矩阵复用前次采集，62 个构建观测的来源由点库与日志记录，最终求解收据的 frames 数组为空属于缓存分支记录缺项。</p>
    <p>下轮可研究如何让旧表示在恢复表面接入时正确退出，同时约束未编辑车辆保持。本轮没有追加针对某一帧的修补，也没有据此证明某个新训练目标有效。</p>
    <details><summary>原始身份与恢复入口</summary><p>街段 <code>12820461091157089924_5202_916_5222_916</code>；目标红车 <code>lvxXvuSB3kueTyP2ignh7w</code>，SUV <code>DwjZr_bYUSZOr-XDGregnA</code>，灰车 <code>AyRpKRmelCT4A709Lo8pvA</code>。原生正整数身份分别为 70 / 25 / 20，仅用于数据绑定。</p><p>远端实验目录 <code>/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val029_v44_20261108_r1</code>。完整恢复包包括 fresh10K 参数、Adam／运行状态、新资产、原生引擎／API、CPU 观测库与实例掩码，压缩后约 1.31 GB；原始 TFRecord 的 SHA 与路径在输入收据中，原文件未重复打包。全部 1,424 份渲染浮点层远端保留并登记 SHA，四个关键帧的原始浮点层已另取回。</p></details>
    <p><a href="metrics.json">完整汇总指标</a> · <a href="assets/evidence/analysis/per_frame_regions.csv">逐帧区域表</a> · <a href="reveal_audit.json">显露只读控制</a> · <a href="run_manifest.json">执行与恢复清单</a> · <a href="assets/evidence/source/continue_edit_pipeline.py">执行队列源码</a></p>
    <footer>Task / run · WAYMO-VAL029-V44-TRANSFER-20261108 / waymo_val029_v44_20261108_r1<br>本轮质量结论为未通过；人工 verdict 保持 null。failure_ledger_refs = [V82-F06]，failure_ledger_delta = updated:V82-F06。训练结束后的取回、提交与电源回执以执行清单为准。</footer></main></html>'''
    (OUT/'index.html').write_text(document,encoding='utf8')
    manifest=dict(task_id='WAYMO-VAL029-V44-TRANSFER-20261108',run_id='waymo_val029_v44_20261108_r1',seed=0,formal_native_updates=10000,engineering_native_updates=5916,road_gate_updates=1000,material_FISTA_steps=1000,C2_IRLS_rounds=5,PVD_frozen_diffusion_steps=1000,BUILD_raws=list(range(79))+list(range(119,198)),HELD_raws=list(range(79,119)),source_camera='FRONT',train_render_wh=[480,320],source_wh=[1920,1280],LR_horizon=30000,formal_stop=10000,quality_accepted=False,human_verdict=None,strict_frozen_preprocessor_transfer=False,failure_ledger_refs=['V82-F06'],failure_ledger_delta='updated:V82-F06',GPU='NVIDIA RTX4080 SUPER /32760MiB reported',native_smoke_seconds=603.670078791678,native_formal_seconds=1128.2306792717427,remote_root='/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val029_v44_20261108_r1',closeout='pending verification, Git push and authorized automatic GPU shutdown')
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('REPORT_BUILT',OUT/'index.html')
if __name__=='__main__':main()
