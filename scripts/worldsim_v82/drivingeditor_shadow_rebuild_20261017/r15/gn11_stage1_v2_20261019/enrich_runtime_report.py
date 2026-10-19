"""Attach actual frozen-road evidence and live snapshot to the candidate report."""
import datetime,hashlib,html,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def enrich(text,W,O):
    meta=json.loads((W/'actual_input_bundle.json').read_text(encoding='utf-8'))
    runtime=json.loads((W/'runtime_progress.json').read_text(encoding='utf-8'))
    if not meta['SGCM_plane_coordinate_roundtrip_verified']:raise RuntimeError('real road binding unverified')
    gate=runtime['gpu_gate']
    if gate['status']!='PASSED_ACTUAL_CUDA_FORWARD_BACKWARD_ZERO_UPDATES':raise RuntimeError('actual GPU receipt missing')
    (O/'actual_gpu_gate_receipt.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
    status=runtime['status'];step=int(status.get('iteration',0));phase=status.get('status','unknown')
    started=datetime.datetime.fromisoformat(status['started_utc']).astimezone(datetime.timezone(datetime.timedelta(hours=8)))
    header=f'新版第一阶段已启动：{step:,}/30,000；状态 {html.escape(phase)}。'
    if phase=='complete':header='第一阶段30K训练完成；独立评估和视频的实际完成范围见下方，质量未通过验收。'
    if 'failed' in phase:header='当前运行出现失败，已保留证据；没有自动重启或重放。'
    a='<strong>本地候选代码与 CPU 检查完成；尚未开始新版训练。</strong><br>\nAutoDL 待用户开机。真实 LiDAR / 相机输入绑定、CUDA 零更新前后向检查和完整 30K 训练均待执行。下方是输入与供体预览，不是修复后的渲染结果。'
    b=f'<strong>{header}</strong><br>历史启动：{started:%Y-%m-%d %H:%M:%S} 新加坡时间；历史训练PID {status["pid"]}。真实49帧输入和CUDA零更新检查已通过，固定相机跨时间静态RGB最大差为0。<br>采样UTC：{html.escape(runtime["collected_utc"])}；设备状态：{html.escape(runtime["gpu"])}。页面中的道路图是查询来源证据，训练效果尚未验收。'
    if a not in text:raise RuntimeError('report phase anchor changed')
    text=text.replace(a,b)
    text=text.replace('当前没有得到真实 raw89 查询曝光比例。','真实查询结果见下方分身份表；仅表示当前严格检查确认的观测，不是隐藏道路真值或最终重建质量。')
    text=text.replace('世界道路投影、真实曝光与未曝光分类待服务器输入绑定后显示，当前不把车身轮廓伪装成已恢复的道路。','真实世界道路投影、融合来源与未知分类已完成，见下方实际查询图。这里仍保留原始定位预览，便于核对输入角色。')
    text=text.replace('以上数量<strong>尚未</strong>通过真实道路几何与全部遮挡物核验，不是实际供体用量。服务器准备阶段还需实测路面支撑；无支撑时停止准备，不能凭颜色认定沥青。','上表是本地颜色候选数。真实道路几何与全部遮挡检查已另完成，实际保留数量见下方供体核查；二者分母不同。')
    text=text.replace('已验证与开机后执行','验证与执行进度')
    text=text.replace('开机后先只读绑定原 A 五身份的真实有效配置、输入、mask、坐标和全 49 帧，生成逐点证据与全幅预览；失败则保留证据并定位。','已完成真实A49输入、mask、相机和坐标绑定，生成逐点道路证据。相机模型锚帧raw99、SGCM道路锚帧raw89分别绑定，地面平面往返验证通过；全部49相机记录的内参与查询标定差为0。')
    text=text.replace('隔离部署候选；核对显存、磁盘及 GPU 锁，执行真实 CUDA 前向 / 反向零更新检查。相机固定、时间改变时静态 RGB 必须保持不变，模型梯度必须有限。','已隔离部署并通过真实CUDA前向 / 反向零更新检查：静态固定相机时间-0.48与0.48的RGB最大差0；静态 / 动态 / 环境梯度组6 / 9 / 1均有限，静态时间参数无梯度，三个优化器实际更新计数均0。')
    text=text.replace('通过后按已授权相同初始化 fresh 30K 执行。','正在按已授权相同初始化 fresh 30K 执行。')
    if phase=='complete':
        text=text.replace('正在按已授权相同初始化 fresh 30K 执行。','主实验已按授权完成相同初始化 fresh 30K；消融状态另见四组比较页。')
    text=text.replace('旧 automation-2 保持暂停，旧队列不恢复；本版尚未启动训练，human_verdict=null。','automation-2已更新为本版每30分钟跟进；仅阶段变化、失败、需操作或完成时通知。旧队列不恢复，human_verdict=null。')
    text=text.replace('仅文件、源码与本地 PNG 检查；未宣称浏览器交互验收或训练效果通过。','已完成文件、源码、真实输入和CUDA零更新检查；未宣称浏览器交互验收或训练效果通过。')
    with np.load(W/'actual_query_evidence.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(W.parent/'identity_completion/metadata/native_masks_for_review/0008089.npz',allow_pickle=False) as z:masks=z['masks']
    px=data['pixels_raw89'];bad=data['invalid_geometry_pixels_raw89'];pro=data['provenance'];primary=data['primary_observed']
    rows=[]
    for i,actor in enumerate([6,37]):
        selected=masks[px[:,1],px[:,0],i];invalid=int(masks[bad[:,1],bad[:,0],i].sum())
        rows.append(dict(actor=actor,total_body_query_pixels=int(masks[:,:,i].sum()),
            late131_137_confirmed=int((selected&primary).sum()),all49_confirmed=int((selected&(pro==1)).sum()),
            prior=int((selected&(pro==2)).sum()),unknown=int((selected&(pro==0)).sum())+invalid,invalid_geometry=invalid))
    summary=dict(status='ACTUAL_FROZEN_QUERY_CLASSIFICATION_NOT_MODEL_QUALITY',query_raw=89,
        road_anchor=meta['coordinate_anchor_frame'],model_anchor=meta['model_anchor_frame'],
        total_query_body_pixels=meta['query_body_pixels_total'],valid_geometry=meta['query_pixels'],actors=rows,
        denominator='raw89 body-mask image-ray samples; not uniform road area',
        actual_observed_visibility_accepted_not_hidden_ground_truth=True,
        evidence_sha256=hashlib.sha256((W/'actual_query_evidence.npz').read_bytes()).hexdigest(),
        source_bundle_sha256=gate['bundle_sha256'],training_targets_changed=False)
    (O/'actual_road_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    shutil.copyfile(W/'actual_query_evidence.npz',O/'actual_road_query_evidence.npz')
    gallery=O/'actual_query_projection';gallery.mkdir(exist_ok=True)
    for raw in [89,*range(131,138)]:
        index=raw-89;fid=f'0008{raw:03d}'
        original=W.parent/'baseline_engineering_20261018/GN11_stage1_full49'/fid/'gt.png'
        im=Image.open(original).convert('RGB');assert im.size==(480,320);draw=ImageDraw.Draw(im)
        uv=data['source_uv'][index];inside=data['gate_in_view'][index]
        color_index=pro.copy()
        if raw!=89:
            color_index[pro==1]=3
            color_index[(pro==1)&data['contributors'][index]]=1
        for color in [0,2,3,1]:
            for q in np.flatnonzero(inside&(color_index==color)):
                x,y=np.floor(uv[q]+.5).astype(int);draw.point((int(x),int(y)),fill={0:'#ff5261',1:'#32df88',2:'#d786ff',3:'#38acff'}[color])
        if raw==89:
            for x,y in bad:draw.point((int(x),int(y)),fill='#ff5261')
        im.save(gallery/f'{fid}.png')
    table=''.join(f'<tr><td>a{r["actor"]}</td><td>{r["total_body_query_pixels"]:,}</td><td>{r["late131_137_confirmed"]:,}</td><td>{r["all49_confirmed"]:,}</td><td>{r["prior"]:,}</td><td>{r["unknown"]:,}</td></tr>' for r in rows)
    buttons=''.join(f'<button data-query-frame="actual_query_projection/0008{r:03d}.png" aria-pressed="{str(r==89).lower()}">raw{r}</button>' for r in [89,*range(131,138)])
    donor=json.loads((O/'actual_donor_provenance.json').read_text(encoding='utf-8')) if (O/'actual_donor_provenance.json').exists() else None
    donor_html=''
    if donor:
        donor_html=f'<h3>供体实际来源核查</h3><p>深度与颜色检查候选：左{donor["depth_and_color_candidates"][0]:,} / 右{donor["depth_and_color_candidates"][1]:,}；再经高度场支撑核验，保留左{donor["retained_donor_samples"][0]:,} / 右{donor["retained_donor_samples"][1]:,}个供体样本。5,119个先验query使用了{donor["unique_used_donor_samples"]:,}个不同供体样本。独立CPU重建的供体索引、类别与RGB全部匹配冻结目标，RGB最大差{donor["reconstructed_RGB_max_abs"]}，没有修改训练输入。</p><figure><img src="actual_donor_support.png" alt="raw89完整画面，青色是经实际几何支撑检查保留的供体像素"><figcaption>青色为实际几何合格供体；取色仍使用未画框原图。先验来源于不同世界位置，不计入真实恢复。</figcaption></figure>'
    nodes=[]
    for path in sorted(O.glob('checkpoint_*_audit.json')):
        audit=json.loads(path.read_text(encoding='utf-8'))
        n=int(audit['step'])
        if (audit['status']!='PASSED_CPU_NODE_DURABILITY' or
            audit['sampler_steps']!=n or n not in range(5000,30001,5000) or
            not all(audit[k] for k in ['SGCM_present','RNG_present','native49_present']) or
            audit['GPU_restore_performed'] or audit['training_updates']!=0):
            raise RuntimeError('invalid named checkpoint receipt '+path.name)
        names=[f'chkpnt{n}.pth',f'dynamic_chkpnt{n}.pth',f'env_light_chkpnt{n}.pth',f'training_state{n}.pth']
        sizes=' / '.join(f'{audit["files"][name]["bytes"]:,}' for name in names)
        nodes.append(f'<tr><td>{n:,}</td><td>{sizes}</td><td>sampler {n:,} / RNG / SGCM / 原生49齐全</td><td><a href="{path.name}">CPU核验凭据</a></td></tr>')
    if nodes:
        donor_html+='<h3>实际完整保存节点</h3><table><thead><tr><th>步数</th><th>静态 / 动态 / 环境 / 训练状态（B）</th><th>续训状态与原生评估</th><th>证据</th></tr></thead><tbody>'+''.join(nodes)+'</tbody></table><p>每个列出的节点均有独立CPU只读凭据，模型与Adam张量有限。没有做节点GPU恢复或增加优化更新，不代表图像质量通过。</p>'
    if (O/'initialization_protocol_binding.json').exists():
        donor_html+='<p>同输入与seed协议进一步核对：实际前100步选帧、时间扰动、动态学习率和静态XYZ学习率与原版逐项一致；静态时间参数按本版方法关闭。旧版没有逐位初始GPU参数快照，因此不把协议一致夸大成已完成旧初始张量逐位对照。<a href="initialization_protocol_binding.json">协议核对凭据</a>。</p>'
    section=f'''<h2>已完成的真实世界道路查询</h2><p>原始query共15,869个车身像素射线样本，其中15,625个有有效道路交点。模型坐标锚帧raw99与地面拟合坐标锚帧raw89是不同角色，均已绑定核验。</p>
<table><thead><tr><th>车辆</th><th>query样本</th><th>raw131–137确认</th><th>全49帧确认</th><th>供体先验</th><th>未知</th></tr></thead><tbody>{table}</tbody></table>
<p>晚段确认包含在全49确认内，不能相加；全49确认、先验和未知三项合计为原始query数。分母是像素射线样本，不能当均匀世界面积。a6未获确认，说明当前严格条件没有足够证据，不能据此断言物理上始终未曝光。先验与未知均不计入真实恢复率。</p>
<div role="group" aria-label="实际query世界投影">{buttons}</div><figure><img id="queryFrame" src="actual_query_projection/0008089.png" alt="raw89 query来源分类的完整画面"><figcaption id="queryCaption">raw89：绿=在其他帧确认的同世界位置真实观测；紫=异位置供体先验；红=未知（含244个无有效道路交点样本）。这是数据来源标注，非模型训练结果。</figcaption></figure>
<p>raw131–137：绿=该帧实际参与融合的确认观测；蓝=同位置在其他帧确认、该帧未参与融合；紫=供体先验；红=未知。投影落在遮挡物上只表示query位置，不能算该帧曝光。</p>{donor_html}
<p><a href="actual_road_summary.json">分身份统计与分母</a> · <a href="actual_road_query_evidence.npz">完整世界点、来源UV与可见性证据NPZ（约10MB，无模型）</a> · <a href="actual_gpu_gate_receipt.json">真实CUDA零更新凭据</a> · <a href="runtime_progress.json">当前运行采样</a> · <a href="actual_donor_provenance.json">冻结供体来源核查</a> · <a href="calibration_binding_receipt.json">全49相机内参核对</a></p>'''
    text=text.replace('<h2>参考工作与实际迁移范围</h2>',section+'<h2>参考工作与实际迁移范围</h2>')
    js='''<script>document.querySelectorAll('button[data-query-frame]').forEach(b=>b.addEventListener('click',()=>{document.getElementById('queryFrame').src=b.dataset.queryFrame;document.getElementById('queryCaption').textContent=b.textContent==='raw89'?'raw89：绿=其他帧确认的同世界位置观测；紫=供体先验；红=未知。非训练结果。':'后续帧：绿=本帧参与融合；蓝=其他帧确认；紫=供体先验；红=未知。投影在遮挡物上不算曝光。';document.querySelectorAll('button[data-query-frame]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));}));</script>'''
    if (O/'ablations/index.html').exists():
        main_phase='主实验训练已完成，全部评估与质量复核尚待交付' if phase=='complete' else '主实验继续'
        text=text.replace('<h2>模块与数据流</h2>',f'<p><a href="ablations/index.html">查看用户新增的三组30K消融与实际排队状态</a>：仅归属、归属＋真实曝光、归属＋供体先验；最多两路并发，{main_phase}。</p><h2>模块与数据流</h2>')
    return text.replace('</html>',js+'</html>').replace('>配置</a>','>冻结方法配置</a>')
