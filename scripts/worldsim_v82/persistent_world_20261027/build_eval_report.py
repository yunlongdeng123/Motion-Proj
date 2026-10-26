"""本地阶段二报告：严格区分准备、执行、质量验证，不把规格数当成绩。"""
from pathlib import Path
from html import escape
from html.parser import HTMLParser
import json
import shutil
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/4D_Editing_Evaluation_20261027'
DATA = OUT / 'delivery'
manifest = json.loads((DATA / 'manifest.json').read_text())
summary = json.loads((DATA / 'summary.json').read_text())
ADMISSION=OUT/'admission'
admission_section=''
scale_section=''
if (ADMISSION/'benchmark_scale_ledger.json').exists():
    scale=json.loads((ADMISSION/'benchmark_scale_ledger.json').read_text())
    family_labels={'ego_speed':'自车变速','ego_lane_change':'自车变道','ego_direction':'自车转向','vehicle_speed':'他车变速','vehicle_lane_change':'他车变道','vehicle_direction':'他车转向','vehicle_insertion':'跨场景车辆插入','vehicle_removal':'指定车辆删除'}
    family_rows=''.join(f'<tr><td>{family_labels[name]}</td><td>{d["materialized"]}</td><td>{d["input_screened"]}</td><td>{d["reserved_input_screened"]}</td><td>{d["completed"]}</td></tr>' for name,d in scale['families'].items())
    expansion=scale.get('expansion') or {};done=len(expansion.get('completed',[]))
    scale_section=f'''<section><h2>实际扩展其他case：八类任务，至少109个有效编辑</h2><img src="admission/benchmark_expansion_architecture.svg"><p>HorizonForge完整论文评测109编辑；展示页32编辑、128视频，不能混成32独立场景。本轮执行目标128有效编辑，长期240；正式完成不足109就继续补齐，不能以输入包、帧数、种子或多个方法凑数。</p><p><strong>30个额外片段已通过CPU三路提取：本快照完成{done}/30个新增真实传感器包。</strong> 加上复用输入共{scale['materialized_contexts']}段已落盘八类请求包，{scale['input_screened_cases']}个请求通过输入筛查；正式准入与新八类编辑完成仍为0。每包有49帧真实RGB/LiDAR/标定/位姿和31时刻米制请求，新窗口重新质检。开发优先，23封存只准备输入，不参与方法选择。</p><table><tr><th>任务</th><th>实际输入请求包</th><th>输入筛查通过</th><th>其中封存输入通过</th><th>实际完成</th></tr>{family_rows}</table><p>跨场景插入的资产库与朝向/尺度适配、自车轨迹渲染、操作空间支撑仍须核验；这里不是效果报告。新分割已排队，合格目标关联后接每场景一次smoke＋30K。<a href="admission/benchmark_scale_ledger.json">分阶段规模账本</a> · <a href="admission/BENCHMARK_SCALE.md">规模与准入合同</a> · 快照时间{escape(scale['updated_at_utc'])}</p></section>'''
    for path in sorted((ADMISSION/'benchmark_inputs').glob('*/summary.json')):
        d=json.loads(path.read_text());key=path.parent.name
        if key in ['14705303724557273004_3105_000_3125_000','4295449061847708198_3769_000_3789_000','473735159277431842_630_095_650_095']:continue
        scale_section+=f'<details><summary>其他实际输入：{key} · {d["input_screened_cases"]}/8输入通过 · {escape(d["split"])}</summary><img loading="lazy" src="admission/benchmark_inputs/{key}/case_inputs.jpg"><p>上图是任务输入，不是编辑效果。<a href="admission/benchmark_inputs/{key}/summary.json">真实传感器与任务状态</a> · <a href="admission/benchmark_inputs/{key}/input_quality.json">新窗口质检</a></p></details>'
    if (ADMISSION/'benchmark_adapter_cpu_audit_v1.json').exists():
        d=json.loads((ADMISSION/'benchmark_adapter_cpu_audit_v1.json').read_text())
        scale_section+=f'<section><h2>八类执行适配器：真实相机曲线与不同场景资产</h2><img src="admission/benchmark_execution_architecture.svg"><p>输入相机到原生相机的49帧最大差{d["camera_original49_max_abs"]:.3g}；跨场景资产按米制尺寸转换均值、协方差与SH方向，位置/协方差往返检查通过。首例31帧按源车真实尺寸复查框碰撞{d["actual_source_dimensions_collision_frames"]}帧。底部只按目标框粗对齐，尚未证明道路支撑、光照适配或新视角正确。</p><p>这些是CPU工程核对，不能代替GPU入口。<a href="admission/benchmark_adapter_cpu_audit_v1.json">逐帧变换审计</a></p>'
        for path in sorted((ADMISSION/'benchmark_execution').glob('*/status.json')):
            item=json.loads(path.read_text());relative=str(path.parent.relative_to(OUT)).replace('\\','/')
            scale_section+=f'<p>{path.parent.name}：{escape(item["phase"])}；原生诊断输出族{len(item["completed_native_families"])}（旧父的ego_speed是全局播放对照，须另查固定世界时钟控制），固定后端{len(item["completed_fixed_backend_families"])}，正式准入仍0。<a href="{relative}/status.json">真实执行状态</a></p>'
            delivery=path.parent/'cpu_delivery'
            if (delivery/'summary.json').exists():
                done=json.loads((delivery/'summary.json').read_text());scale_section+=f'<img src="{relative}/cpu_delivery/comparison.jpg">'
                for family in done['families']:
                    name=family['family'];scale_section+=f'<details><summary>{escape(name)} · 原生31帧</summary><video controls preload="metadata" src="{relative}/cpu_delivery/{name}/native.mp4"></video></details>'
                    if family['fixed_backend_executed']:scale_section+=f'<details><summary>{escape(name)} · 固定后端</summary><video controls preload="metadata" src="{relative}/cpu_delivery/{name}/generated.mp4"></video></details>'
        scale_section+='</section>'
    if (ADMISSION/'benchmark_asset_diversity_audit_v1.json').exists():
        asset=json.loads((ADMISSION/'benchmark_asset_diversity_audit_v1.json').read_text())
        scale_section+=f'<p>插车输入的资产多样性仍不足：{asset["input_eligible_insertion_requests"]}个请求只有{asset["distinct_source_contexts"]}个源场景。不能以换位置的次数代替任意车辆外观泛化。需在合格新资产形成后冻结新版资产库；旧请求保留，版本不重复计数。<a href="admission/benchmark_asset_diversity_audit_v1.json">资产来源分布</a> · <a href="admission/TIME_AND_SPACE_CONTRACT.md">时钟、空间与准入边界</a></p>'
    if (ADMISSION/'ego_speed_clock_CPU_audit.json').exists():
        audit=json.loads((ADMISSION/'ego_speed_clock_CPU_audit.json').read_text())
        scale_section+=f'<section><h2>控制语义：自车变速与整体播放变速分开</h2><p>此前八类入口同时重定时相机与世界，这只能作为整体播放对照。真正的自车单独变速保留输出世界时刻与其他车辆时刻，只改变相机路径查询。原对照完整保留，两种输出不增加案例数。CPU核对{audit["rows_checked"]}条输入时钟；raw{audit["nonanchor_raw"]}的其他车辆实测轨迹在两种时钟下相差{audit["nonanchor_other_actor_position_difference_m"]:.2f}m，说明混用时钟会改变研究问题。这是输入轨迹差，不是渲染质量分数。</p><a href="admission/ego_speed_clock_CPU_audit.json">CPU时钟审计</a>'
        for path in sorted((ADMISSION/'ego_speed_clock_control').glob('*/status.json')):
            d=json.loads(path.read_text());relative=str(path.parent.relative_to(OUT)).replace('\\','/')
            scale_section+=f'<p>真实GPU对照状态：{escape(d["phase"])}；<a href="{relative}/status.json">运行收据</a>。</p>'
            if (path.parent/'summary.json').exists():scale_section+=f'<img src="{relative}/comparison.jpg"><p><a href="{relative}/summary.json">31帧时钟分离与实际浮点差异</a>，差异不等于质量提升。</p>'
        scale_section+='</section>'
    if (ADMISSION/'benchmark_request_space_audit_v1.json').exists():
        audit=json.loads((ADMISSION/'benchmark_request_space_audit_v1.json').read_text())
        rows=''.join(f'<tr><td>{escape(name)}</td><td>{d["cases"]}</td><td>{d["cases_with_conflict"]}</td><td>{d["conflict_frames"]}/{d["audited_frames"]}</td></tr>' for name,d in audit['by_family'].items())
        scale_section+=f'<section><h2>33段输入的请求空间复核</h2><table><tr><th>任务族</th><th>输入合格请求</th><th>有标注框冲突请求</th><th>冲突帧/核对帧</th></tr>{rows}</table><p>车辆使用三维标注框与真实源资产尺寸；自车只检查相机中心是否进入车辆框，不是自车车身碰撞或可行驶空间认证。删车不需新位置碰撞检查。未读取编辑输出、未改变输入筛查资格，也未用框过滤训练点。冲突单列压力层，不能算作②失败；无框冲突不等于道路几何支撑。</p><a href="admission/benchmark_request_space_audit_v1.json">逐帧审计与完整分母</a></section>'
if (ADMISSION/'registry.json').exists():
    registry=json.loads((ADMISSION/'registry.json').read_text())
    shutil.copy2(Path(__file__).parent/'native_training_architecture.svg',ADMISSION/'native_training_architecture.svg')
    training_rows=[]
    for path in sorted((ADMISSION/'native_training').glob('*/status.json')):
        item=json.loads(path.read_text());context=path.parent.name;assets=ADMISSION/'motion_inputs'/context
        build=json.loads((assets/'actor_build_returned.json').read_text())
        links=' · '.join(f'<a href="admission/motion_inputs/{context}/actor_review_{i}.jpg">49帧核验 {i+1}/3</a>' for i in range(3))
        gate=path.parent/'smoke_gate.json';gate_text=''
        if gate.exists():
            g=json.loads(gate.read_text());extra=g['fresh_extra_updates']+g['restore_extra_updates']
            gate_text=f'<br><small>smoke通过：{g["real_seconds"]:.2f}s，额外{extra}更新（含评价/保存）</small><br><a href="admission/native_training/{context}/smoke_restore/restore_verification.json">完整恢复核验</a>'
        training_rows.append(f'<tr><td>{context}<br><small>{links}</small></td><td>{build["dynamic_real_points"]:,}真实动态点<br>600,000静态初始点</td><td>{escape(item["phase"])}<br><small>30K一次；每5K完整四件</small>{gate_text}</td></tr>')
    quota=json.loads((ADMISSION/'quota_state.json').read_text())
    admission_section=f'''<section><h2>最新：训练与多操作诊断交错推进</h2><p><strong>每个场景完成30K，立即优先执行该场景多种编辑及固定后端，再训练下一场景。</strong> 后续训练在GPU锁之外等待前一场景首轮诊断，CPU质检、准备与失败分析异步；不能用不断准备和训练代替失败定位。首轮结果按①实例/轨迹与资产、②背景污染/缺失、③后端改坏分别归因，推动阶段三选择。</p><p>不等所有候选完成质检。场景数据与目标就绪后进入原生几何守卫和真实GPU smoke；训练后再检查原生资产、背景与操作空间，决定哪些编辑适合正式评测。训练准入与完整评测准入分开，后者目前仍为0。</p>
<img src="admission/native_training_architecture.svg"><table><tr><th>新开发场景</th><th>真实初始化</th><th>本地证据快照</th></tr>{''.join(training_rows)}</table>
<p>原V41-B/GN11的方法、损失、学习率、采样和增密预算不变，正式从零训练30K；5K/10K/15K/20K/25K/30K保存静态、动态、环境、完整训练状态和评价。真实smoke包括fresh≥300秒与严格恢复≥300秒，开销单列。每个新场景当前只有一个经过核验的动态目标；不能据此声称任意多对象归属已解决。</p>
<p>仅本场景30K成功且磁盘紧张时清其中间checkpoint；先核对最终可恢复四件与49帧评价，保留精确文件/字节/散列删除收据。bug自主修复和必要重放，旧现场保留。周额度快照剩余{quota['remaining_percent']}%；未触发20%停止线。</p></section>
<section><h2>输入准入已补齐，原“最大框”窗口存在偏差</h2><p>36段自动原始检查：34通过、2复核，均未被包装成正式完成案例。v2增加目标边框接触检查后，旧最大框窗口只有1个普通运动候选、0个变速候选。按操作重新选窗可提出33个普通运动候选、21个运动目标变速候选；这些仍需窗口复核、分割、资产和几何检查。</p>
<p>三段已准备同一49帧的真实全分辨率RGB、五LiDAR双回波与TOP运动补偿、相机和位姿；自动DEVA全部完成，完整可见目标均逐帧检查。GT框只提出身份，不生成掩码或过滤训练点。原始预测边界和未决状态保留。</p>
<p><a href="admission/registry.json">输入检查明细</a> · <a href="admission/operation_window_proposals.json">逐操作窗口提案</a> · <a href="admission/ADMISSION_POLICY.json">冻结门限</a></p></section>'''
    withdrawal_rows=[]
    for name in ['scene0017','scene0029']:
        d=json.loads((ADMISSION/'withdrawal'/name/'summary.json').read_text());m=d['methods']
        withdrawal_rows.append(f'<tr><td>{name}</td><td>{d["frame_summed_generation_mask_pixels"]:,}</td><td>{d["frame_summed_reference_pixels"]:,} / 核心{d["frame_summed_core_reference_pixels"]:,}</td><td>{m["input_plane_neighbor_color"]["core"]["MAE"]:.5f}</td><td>{m["fixed_propainter"]["core"]["MAE"]:.5f}</td></tr>')
    admission_section+=f'''<section><h2>道路撤源组件：保留完整分母，单列内部核心</h2><p>两路段49帧的全部关联RGB先遮挡、世界洞LiDAR先移除，不输入旧高斯/潜特征。4×2.4m洞与0.3m保护带；最近实测LiDAR可见性和车辆边缘排除定义稀疏参考。核心为距边界≥0.75m。输入平面与邻色只读保留输入，固定ProPainter仍是二维基线。</p>
<table><tr><th>场景</th><th>撤源mask累计像素</th><th>稀疏参考累计像素</th><th>确定性核心MAE</th><th>固定ProPainter核心MAE</th></tr>{''.join(withdrawal_rows)}</table>
<p>这是累计帧像素，非唯一世界面积或完整恢复率。场景0017的核心MAE改善但PSNR略降（28.55→28.17dB）；不能说所有指标变好。整个基座没有撤源重训，组件结果不证明高斯三维背景正确。稀疏标签之外仍是未知。</p>'''
    for name in ['scene0017','scene0029']:
        admission_section+=f'<details><summary>{name}：组件比较与参考域</summary><img loading="lazy" src="admission/withdrawal/{name}/comparison.jpg"><img loading="lazy" src="admission/withdrawal/{name}/reference_layers.png"><p><a href="admission/withdrawal/{name}/summary.json">分层指标、来源和分母</a></p></details>'
    admission_section+='</section>'
    motion=ADMISSION/'motion_requests/status.json'
    if motion.exists():
        request_rows=[]
        for item in json.loads(motion.read_text())['scenes']:
            context=item['context'];collisions='；'.join(f'{k}: {v}/31' for k,v in item['collisions'].items() if v) or '未见标注框碰撞；空间支撑仍待检查'
            status_file=ADMISSION/'motion_execution'/context/'status.json'
            executed=json.loads(status_file.read_text()) if status_file.exists() else {}
            execution=f'原生变体 {executed.get("completed_native_cases",0)}；后端 {executed.get("completed_backend_cases",0)}<br><small>{escape(executed.get("phase","not_dispatched"))}</small>'
            request_rows.append(f'<tr><td>{context}</td><td>{item["single_request_specs"]}单请求变体 + 1组合协议<br><strong>{execution}</strong></td><td>{collisions}</td><td><a href="admission/motion_requests/{context}/summary.json">物理时刻、轨迹和限制</a></td></tr>')
        admission_section+=f'''<section><h2>接续运动与相机测试：请求先做物理一致性检查</h2><img src="admission/motion_diagnostics_architecture.svg"><p>共同中央31个相机时刻；0.75/1.25倍速的查询都在原49帧支持内，无时间截断或外推。新增平移、复用插入、±15°转向、0.5m相机侧移和3°旋转入口；同一个30K checkpoint复用，不训练每case。队列等待同场景完整30K后先做真实GPU单帧原生一致性检查，再完整执行。首场景6次连续原生修订已执行；后端连续对照仍待完成。</p><table><tr><th>场景</th><th>规格数≠结果数</th><th>输入标注框碰撞压力</th><th>来源</th></tr>{''.join(request_rows)}</table><p>克隆插入也必须检查与保留原车的碰撞。冲突请求保留为压力层，不当正常编辑质量失败，也不推断旧场景所有拖影均由碰撞造成。新位置道路支撑尚未证明；标注框仅用于请求条件与诊断，未过滤训练点。相机未知mask未定义时保留原生诊断，不伪称视频补完。</p></section>'''
        for picture in sorted((ADMISSION/'motion_execution').glob('*/*/native.png')):
            case_dir=picture.parent;relative=str(case_dir.relative_to(OUT)).replace('\\','/')
            generated=case_dir/'generated.png';backend=(f'<img src="{relative}/generated.png">' if generated.exists() else '<p>固定后端结果尚未完成或该操作仅做原生诊断。</p>')
            admission_section+=f'<section><h2>新场景实际输出：{escape(case_dir.parent.name)} / {escape(case_dir.name)}</h2><p>左为原生，右为固定后端（若已完成）；仅供失败定位，无反事实RGB真值，质量仍待评审。</p><div class="grid"><img src="{relative}/native.png">{backend}</div></section>'
            for filename,label in [('native_h264.mp4','生成前原生3D'),('generated_h264.mp4','固定ProPainter')]:
                if (case_dir/filename).exists():admission_section+=f'<details><summary>{escape(case_dir.name)}：{label}31帧视频</summary><video controls preload="metadata" src="{relative}/{filename}"></video></details>'
        for path in sorted((ADMISSION/'motion_execution').glob('*/continuous/summary.json')):
            d=json.loads(path.read_text());relative=str(path.parent.relative_to(OUT)).replace('\\','/')
            panels=''.join(f'<article><p>{escape(x["revision"])}</p><img src="{relative}/{x["revision"]}.png"></article>' for x in d['revisions'])
            admission_section+=f'<section><h2>实际连续请求原生基线：{path.parent.parent.name}</h2><p>{d["revision_count"]}次修订，每次31帧，始终操作同一克隆实例。原车删除后保持删除；移动和转向不重新插入原车。相机回访与同状态普通缓存差{d["revisit_same_revision_original_camera_max_difference"]}；这是工程控制，不能证明学习到的持久记忆收益。连续请求固定后端仍待执行。</p><div class="grid">{panels}</div><a href="{relative}/summary.json">版本状态与缓存控制</a></section>'
        if list((ADMISSION/'motion_execution').glob('*/continuous/summary.json')):
            admission_section=admission_section.replace('首场景6次连续原生修订已执行；后端连续对照仍待完成。','首场景连续原生6次修订×31帧已实际运行；连续视频后端仍待执行。')
        for path in sorted((ADMISSION/'motion_execution').glob('*/continuous/fixed_backend/cpu_delivery_summary.json')):
            d=json.loads(path.read_text());relative=path.parent.relative_to(OUT).as_posix()
            pairs=''.join(f'<tr><td>{escape(name)}</td><td>{row["frame_pixel_observations"]:,}</td><td>{row["native_float_MAE_pixel_weighted"]:.3g}</td><td>{row["generated_png_MAE_pixel_weighted"]:.5f}</td></tr>' for name,row in d['pairwise_domain_control']['per_revision'].items())
            clips=''.join(f'<details><summary>{name}：31帧，{"普通缓存复用" if name in ["delete_original","revisit_original_camera"] else "本轮实际推理"}</summary><video controls preload="metadata" src="{relative}/{name}.mp4"></video></details>' for name in ['delete_original','insert_reused','move_inserted','turn_inserted','revisit_original_camera'] if (path.parent/(name+'.mp4')).exists())
            admission_section+=f'<section><h2>连续请求固定后端对照已完成：覆盖分母必须同时看</h2><p>3次实际ProPainter推理＋2次完全相同输入的普通缓存复用，不是5次新推理。五版本共同暴露域共{d["common_frame_pixel_observations"]:,}个累计帧像素，每帧{d["common_pixels_min"]}–{d["common_pixels_max"]}，中位数{d["common_pixels_median"]:.0f}；这是很小的局部域，不是整片道路面积。原生浮点变化MAE {d["native_float_change_MAE_macro"]:.3g}，生成PNG变化MAE {d["generated_png_change_MAE_macro"]:.5f}。两者精度不同，不能作倍率结论；也没有反事实RGB真值、相同输入重复推理方差或学习记忆收益证明。</p><img src="{relative}/comparison.jpg"><p>上排原生，中排固定后端，下排仅共同域内相对删除版本的差异×5；黄线标域。逐对比较可扩大暴露域，但各行分母不同，不能互当质量排名。</p><table><tr><th>对删除版本</th><th>逐对累计帧像素</th><th>原生浮点变化MAE</th><th>生成PNG变化MAE</th></tr>{pairs}</table><a href="{relative}/cpu_delivery_summary.json">完整域、逐帧分母和精度说明</a>{clips}</section>'
            admission_section=admission_section.replace('连续请求固定后端仍待执行。','连续请求固定后端已完成，详见覆盖分母审计。').replace('连续视频后端仍待执行。','连续视频后端3次实际推理和2次普通缓存已完成。')
        sh=ADMISSION/'sh_operator_control/cpu_control_summary.json'
        if sh.exists():
            d=json.loads(sh.read_text());rows=''.join(f'<tr><td>{r["case"]}</td><td>{r["actor_domain_rgb_change_MAE_macro"]:.5f}</td><td>{max(r["unchanged_geometry_channels_max_difference"].values())}</td></tr>' for r in d['records'])
            admission_section+=f'<section><h2>转向算子控制：SH随物体旋转</h2><img src="admission/sh_operator_control/comparison.jpg"><table><tr><th>操作</th><th>动态支撑域RGB变化MAE</th><th>alpha／动态权重／深度和最大差</th></tr>{rows}</table><p>原生无编辑适配差0，主副本未变。3变体×31帧真实GPU完成；只改变外观方向约定，几何通道逐像素不变。此处是算子控制，不是重新打光，也没有转向反事实RGB真值。原世界系SH基线仍保留。</p><a href="admission/sh_operator_control/cpu_control_summary.json">GPU输出CPU核对</a></section>'
        for path in sorted((ADMISSION/'motion_execution').glob('*/diagnostics/summary.json')):
            d=json.loads(path.read_text());relative=str(path.parent.relative_to(OUT)).replace('\\','/')
            rows=''.join(f'<tr><td>{escape(x["case"])}</td><td>{x["generation_pixels_mean"]:.1f}</td><td>{x["preservation_MAE_macro"] if x["preservation_MAE_macro"] is not None else "新视角无对应真值"}</td><td>{x["backend"]["outside_mask_max_byte_difference"] if x["backend"] else "未调用"}</td></tr>' for x in d['cases'])
            admission_section+=f'<section><h2>场景{path.parent.parent.name}多操作证据：生成前后分开</h2><img src="{relative}/comparison.jpg"><table><tr><th>操作</th><th>平均修补像素</th><th>保留域对原生图MAE</th><th>后端mask外最大字节差</th></tr>{rows}</table><p>保留域排除原车、新车投影与边缘；这些指标只检查编辑局部性与后端负担，不是删车真值、几何成功率或新视角PSNR。原生适配差{d["adapter_native_max_difference"]}，主副本未变。<a href="{relative}/summary.json">逐帧指标</a></p></section>'
    for path in sorted((ADMISSION/'motion_execution').glob('*/diagnostics/summary.json')):
        d=json.loads(path.read_text());branch=d.get('branch_contributions')
        if branch:
            admission_section+=f'<p>首场景DEVA车身内缩2像素区域：动态合成权重均值{branch["body_core_dynamic_weight_macro"]:.4f}，静态{branch["body_core_static_weight_macro"]:.4f}，静态占优像素比例{branch["body_core_static_dominant_fraction_macro"]:.2%}。这是分支贡献诊断，不能当污染率；玻璃、透明度和掩码不确定性也会留下背景贡献。需结合独立动态层、静态层和变换控制定位失真。</p>'
    audit=ADMISSION/'native_actor_source_audit.json'
    if audit.exists():
        audit_rows=''.join(f'<tr><td>{x["context"]}</td><td>{x["total_dynamic_points"]:,}</td><td>{100*x["inside_annotated_box_fraction"]["0.0"]:.2f}%</td><td>{x["world_replay_max_abs_m"]:.1f}m</td></tr>' for x in json.loads(audit.read_text()))
        admission_section+=f'<section><h2>初始化来源核对</h2><table><tr><th>场景</th><th>真实动态点</th><th>未扩张标注框内比例</th><th>逐源点世界回放差</th></tr>{audit_rows}</table><p>源索引、世界坐标和模型时间逐点对齐；框一致性不证明语义纯度，不把3D框作为训练过滤器。<a href="admission/native_actor_source_audit.json">审计详情</a></p></section>'
    for path in sorted((ADMISSION/'native_training').glob('*/checkpoint*/event.json')):
        node=path.parent;event=json.loads(path.read_text());picture=node/'contact.jpg'
        canvas=Image.new('RGB',(1920,350),(240,244,248))
        from PIL import ImageDraw
        for i,name in enumerate(['gt','composite','static','dynamic']):
            canvas.paste(Image.open(node/(name+'.png')).convert('RGB'),(i*480,30));ImageDraw.Draw(canvas).text((i*480+8,8),name+f' / {event["iteration"]}',fill=(20,40,60))
        canvas.save(picture,quality=94)
        relative=str(node.relative_to(OUT)).replace('\\','/')
        stage='正式30K完成' if event['iteration']==30000 else '训练中间快照'
        interpretation='六档完整节点和30000条有限更新已核对，资产与编辑质量仍待检查。' if event['iteration']==30000 else '尚未达到30K，不提前接受编辑质量或判定最终失败。'
        admission_section+=f'<section><h2>{stage}：{path.parent.parent.name} · {event["iteration"]:,}步</h2><img loading="lazy" src="{relative}/contact.jpg"><p>原生保存的中间相机；依次GT、完整、静态、动态。{interpretation}<a href="{relative}/paper_metrics.json">原生49帧区域指标</a></p></section>'
for name in ['scene0017', 'scene0029', 'scene0008_repair10k']:
    video = DATA / name / 'delete_video.mp4'
    result = video.with_name('delete_h264.mp4')
    if video.exists() and not result.exists():
        subprocess.run([shutil.which('ffmpeg'), '-v', 'error', '-y', '-i', str(video), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-an', str(result)], check=True)

failures = [
    dict(scene='scene0008', operation='删除指定车辆 / 移动 / 复用插入', observation='中心静态残影、近处条带；部分动态行归属未决。',
         status='已有贡献追踪与图像证据；跨场景对应机制仍需验证', cause='静态污染已在该场景核实；不能推广所有黑色区域',
         evidence='../Persistent_World_Autoresearch_20261027/operations_raw102.jpg'),
    dict(scene='scene0017', operation='整动态分支删除 / 移动 / 复用插入', observation='删除原生图较干净，固定ProPainter在车原位产生暗色模糊区域；移动与复用车辆有明显蓝色拖影。',
         status='中间帧图像已检查，49帧保留供时序复核', cause='生成改坏与资产表达/编辑算子分别待定位；不直接归为缺道路几何',
         evidence='delivery/scene0017/contact.jpg'),
    dict(scene='scene0029', operation='整动态分支删除 / 移动 / 复用插入', observation='车辆消失后，原位置仍留下大块暗色轮廓；固定视频后端未去掉。插入资产存在重叠和拖影。',
         status='中间帧图像已检查，49帧保留供时序复核', cause='背景中烘焙阴影/污染是候选解释；需要贡献与真实显露对照，尚未确认几何错误',
         evidence='delivery/scene0029/contact.jpg'),
]
if (ADMISSION/'motion_execution/14705303724557273004_3105_000_3125_000/diagnostics/summary.json').exists():
    failures.append(dict(scene='14705303724557273004_3105_000_3125_000',operation='单目标删除、平移、转向、变速、克隆和相机变化',observation='raw109删除后保留SUV形状静态残影；ProPainter变成暗块；平移/转向资产失真，相机侧移暴露其他车辆和背景失真。',status='实际11变体31帧及6固定视频已完成，6次连续原生修订已执行；同时间raw109图像已检查',cause='②静态残留可见；①资产/算子与静态遮挡尚需分离；③暗块无反事实GT不能直接认定改坏正确道路。SH物体系旋转CPU控制通过、GPU控制待执行。',evidence='admission/motion_execution/14705303724557273004_3105_000_3125_000/diagnostics/comparison.jpg'))
hold_audit=ADMISSION/'motion_inputs/13506499849906169066_120_000_140_000/association_candidate_audit.json'
if hold_audit.exists():
    failures.append(dict(scene='13506499849906169066_120_000_140_000',operation='输入实例关联准入，未开始训练或编辑',observation='DEVA65候选保留点仅1.70%处于目标框扩0.1m内，与视觉邻车吻合；DEVA63有7帧主组件守卫失败。',status='HOLD保留，49帧候选已CPU审计；mask/点/守卫均未改',cause='①输入实例关联/源点支撑问题，不能归为②背景修复失败；框只用于诊断。',evidence='admission/motion_inputs/13506499849906169066_120_000_140_000/association_candidate_audit.json'))
night_hold=ADMISSION/'motion_inputs/3417928259332148981_7018_550_7038_550/association_review_hold.json'
if night_hold.exists():
    failures.append(dict(scene='3417928259332148981_7018_550_7038_550',operation='夜间输入实例关联准入',observation='49帧contact逐帧检查，候选ID13/2/95切换，多帧仅车顶/尾灯及碎片，前景银车遮挡。',status='HOLD，无selection_reviewed、不构建训练资产；保留困难层',cause='①输入/关联问题；没有运行编辑，不能计为②失败。',evidence='admission/motion_inputs/3417928259332148981_7018_550_7038_550/association_review_hold.json'))
if (ADMISSION/'sh_operator_control/cpu_control_summary.json').exists():
    for row in failures:row['cause']=row['cause'].replace('SH物体系旋转CPU控制通过、GPU控制待执行。','SH物体系旋转真实GPU3变体31帧控制完成，几何通道差0；仅小幅颜色变化，raw109重影仍在，单改SH约定不足。')
for path in sorted((ADMISSION/'motion_execution').glob('*/continuous/fixed_backend/cpu_delivery_summary.json')):
    failures.append(dict(scene=path.parents[2].name,operation='连续编辑固定后端一致性诊断',observation='五版本共同暴露域1569累计帧像素；原生浮点变化MAE3.07e-7，生成PNG变化0.01005。逐对暴露域分母更大且不同。',status='3次新推理、2次精确输入普通缓存，图像和31帧视频已检查/保留',cause='③局部跨请求变化证据；共同域每帧2–180像素很小，无反事实GT及相同输入重推方差，不能证明全背景质量或记忆创新。',evidence=path.relative_to(OUT).as_posix()))
(OUT / 'failure_matrix.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')
labels = {'previous_method_development': '原方法开发', 'previous_reconstruction_development': '历史重建开发',
          'new_diagnostic_development': '新增诊断开发', 'heldout_reserved': '封存：未跑编辑评测'}
gallery = ''.join(f'<article><img loading="lazy" src="delivery/gallery/{escape(s["context"])}.jpg"><p>{escape(s["context"])}</p><small>{labels[s["split"]]} · {escape(s["location"])} · {escape(s["time_of_day"])}<br>{s["frames"]}原始帧 / 49帧PNG；像素掩码与道路GT待补齐</small></article>' for s in manifest['scenes'])
failure_rows = ''.join(f'<tr><td>{x["scene"]}</td><td>{x["observation"]}</td><td>{x["cause"]}</td><td>{x["status"]}</td></tr>' for x in failures)
body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>4D 编辑 · 评测驱动研究</title>
<style>body{{background:#f3f7fa;color:#203c51;font:16px/1.7 "Microsoft YaHei",sans-serif;margin:0}}main{{max-width:1400px;margin:auto;padding:25px}}section{{background:white;border:1px solid #d8e3eb;border-radius:12px;margin:22px 0;padding:22px}}h1{{font-size:29px}}h2{{font-size:22px}}a{{color:#126c9a}}img,video{{max-width:100%}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;border-bottom:1px solid #dce5ed;padding:12px;vertical-align:top}}.note{{background:#fff1d5;padding:16px;border-left:4px solid #b68c37}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}article{{border:1px solid #dce5ed;padding:10px;border-radius:8px}}article p{{font-size:12px;overflow-wrap:anywhere}}small{{font-size:13px}}video{{width:100%;background:#152536}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}main{{padding:12px}}}}</style>
<main><h1>4D 编辑评测：先发现重复失败，再改进背景修复</h1><p>2026-10-27 · 阶段二早期 · 质量待用户评审 · 持续自主推进</p>
<p><a href="../Persistent_World_Autoresearch_20261027/index.html">原开发场景证据</a> · <a href="delivery/EVALUATION_PROTOCOL.md">冻结评测合同</a> · <a href="delivery/manifest.json">场景与案例规格</a> · <a href="failure_matrix.json">失败矩阵</a></p>
{scale_section}
{admission_section}
<section><h2>已经做了什么，哪些仍未完成</h2><table><tr><th>层次</th><th>实际进展</th><th>不能据此宣称</th></tr>
<tr><td>数据准备</td><td>36个真实context逐记录双CRC核验；1764帧RGB和原始轨迹标注。原方法开发1段、历史重建开发2段、新开发10段、封存23段。</td><td>尚未全部成为高斯编辑案例；context之间物理路段重叠仍待审计。</td></tr>
<tr><td>任务规格</td><td>六类任务，共216份待执行规格，先划场景再选目标/窗口。</td><td>不是216项已完成实验；严格撤源、核心真值与多请求复用尚未完成。</td></tr>
<tr><td>实际运行</td><td>新场景首个30K后11原生变体×31帧、6固定视频、6次连续原生修订完成。历史两个开发路段另有12条49帧操作序列，保留其协议区别。</td><td>整动态分支操作不能冒充指定实例归属已解决；不同历史基座不能直接算方法增益。</td></tr>
<tr><td>阶段三</td><td>候选几何smoke正常保存，共3985额外更新；正式10K未启动，已撤下。视频后端保持ProPainter。</td><td>不提前选择核心算法，也不切换后端掩盖失败。</td></tr></table>
<p class="note">阶段一闭环已经运行，当前重点是阶段二。持久世界状态仍是框架目标；必须证明三维背景在换视角、反复编辑时正确，并减少③修补需求，才能形成算法贡献。</p>
<img src="delivery/evaluation_architecture.svg" alt="多场景评测、固定编辑和生成后端、三维与视频分开检查、失败驱动算法迭代"></section>
<section><h2>新增路段暴露的问题并不相同</h2><table><tr><th>场景</th><th>实际观察</th><th>候选原因与归因边界</th><th>证据状态</th></tr>{failure_rows}</table>
<p>这说明不能把所有失败统一写成“缺道路几何”。生成覆盖、背景中的对象相关暗色外观、资产换视角表达、实例归属，需要分别量化。下面是固定中间帧，完整49帧序列已保留。</p></section>'''
for name, title in [('scene0017', '路口白色轿车'), ('scene0029', '路口红色轿车')]:
    body += f'<section><h2>{title} · {name}</h2><p>上排：原生完整、删除、删除＋固定ProPainter、移动。下排：复用插入、相机侧移、重复删除。所有操作均未训练新修复算法。</p><img src="delivery/{name}/contact.jpg"><video controls preload="metadata" src="delivery/{name}/delete_h264.mp4"></video><p><a href="delivery/{name}/summary.json">逐帧原生记录</a> · <a href="delivery/{name}/video_summary.json">视频改动/时间/显存记录</a></p></section>'
body += f'''<section><h2>数据清单与接续任务</h2><p>下面只展示输入RGB，封存场景没有运行编辑效果评测。原始三维标签、像素语义、可信道路参考、原生checkpoint的就绪状态分别记录；框不作为像素掩码。</p><div class="grid">{gallery}</div>
<ol><li>补齐新增开发路段的像素掩码、米制几何与高斯输入；封存场景保持方法选择隔离。</li><li>严格撤源先移除全部关联输入再编码/初始化，分别报告边界、过渡、内部核心；做不到的只标module-only。</li><li>按完整目标与参考分母评价污染残留、保留对象、跨视角/请求一致性和视频修补负担；无标签中心仍为未知。</li><li>依据跨场景反复失败再设计②，固定①③消融；保留简单锚点、确定性补全和普通缓存强对照。</li></ol></section>
<section><h2>资源与质量状态</h2><p>同一心跳每30分钟继续。每次先查Codex周额度；剩余严格低于20%时快速保存收尾、取回必要结果并关闭AutoDL，再暂停心跳。在此之前继续自主研究，不因阶段结束或负结果关机。</p><p>额度快照见本页最新流水区，以后续工具读取为准。质量仍为quality_accepted=false、human_verdict=null。重复渲染最大差0仅为工程检查，不是世界记忆收益。</p><p>本页做了图片和文件链接校验，没有声称浏览器交互QA。<a href="delivery/summary.json">执行与指标清单</a></p></section></main></html>'''
(OUT / 'index.html').write_text(body, encoding='utf-8')
class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.paths=[]
    def handle_starttag(self, tag, attrs):
        self.paths += [v for k,v in attrs if k in ['src','href'] and v and not v.startswith(('http:','https:','#'))]
parser=Links();parser.feed(body)
broken=[p for p in parser.paths if not (OUT / p).exists()]
for file in [*DATA.rglob('*.jpg'),*ADMISSION.rglob('*.jpg'),*ADMISSION.rglob('*.png')]:
    with Image.open(file) as im:im.verify()
assert not broken,broken
result=dict(local_links=len(parser.paths),broken=broken,actual_independent_heldout_results=0,
            raw_contexts=36,prepared_pngs=1764,historical_operation_sequences=12,new_native_variants=11,new_fixed_backend_cases=6,continuous_native_revisions=6,new_eight_family_completed=0,all_six_families_complete=False,
            browser_render_qa=False,quality_accepted=False,human_verdict=None)
result['completed_native_training_contexts']=len(list((ADMISSION/'native_training').glob('*/formal30k_agent_acceptance.json')))
result['continuous_fixed_backend_fresh_calls']=sum(json.loads(p.read_text())['fresh_backend_calls'] for p in (ADMISSION/'motion_execution').glob('*/continuous/fixed_backend/cpu_delivery_summary.json'))
result['SH_object_frame_GPU_control_complete']=(ADMISSION/'sh_operator_control/cpu_control_summary.json').exists()
(OUT/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
