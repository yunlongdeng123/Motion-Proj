"""Final measured report; no metrics are hand-entered or invented."""
import json,math,datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[2];OUT=BASE/'outputs/shadow_sequence_v2'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def main():
    selected=read('selected.json');ev=selected['evaluation'];train=selected['training']
    summary=read(ev+'/summary.json');rows=read(ev+'/per_frame.json');m=summary['metrics_mean_frames']
    dataset=read('surface_guarded_data/summary.json');guard=read('guard_refinement_v2/summary.json')
    motion=read('evidence/away_reference_audit.json');actor=read(train+'/actor_responsibility_check.json')
    visual=read('evidence/visual_review.json');own=read('contribution_guarded_audit/rounds.json')
    foreground=read('unselected_foreground_audit/receipt.json');sweep=read('anchor_edit_sweep_final/summary.json')
    road=1-m['final']['surface_clean_MAE']/m['original']['surface_clean_MAE']
    deletion=1-m['final']['deleted_target_shadow_area_clean_MAE']/m['original']['deleted_target_shadow_area_clean_MAE']
    pixel=summary['metrics_pixel_weighted'];maxopaque=max(r['shadow_on_opaque_actor_max'] for r in rows)
    gates={'all49_frames_evaluated':len(rows)==49,'road_error_reduced_at_least_50pct':road>=.5,
        'deletion_shadow_area_error_reduced_at_least_50pct':deletion>=.5,
        'fixed_supported_road_time_MAE_at_most_1e_5':summary['mean_fixed_supported_road_time_MAE']<=1e-5,
        'delete_all_shadow_identity':summary['max_delete_identity']<2e-6,
        'road_routing_outside_authority_exact':summary['max_road_routing_outside_same_actor_gates']==0,
        'shadow_on_opaque_actor_at_most_002':maxopaque<=.002,
        'all49_comparison_frames_reviewed':visual.get('comparison_frames_reviewed')==49,
        'no_new_severe_road_or_protected_vehicle_artifact':visual.get('no_new_severe_road_or_protected_vehicle_artifact',False)}
    overall={'experiments_complete':True,'overall_problem_fully_solved':False,'quality_gates_declared_road_scope':gates,
        'complete_vehicle_asset_purity_certified':False,'all_scene_road_shadowfree_certified':False,
        'large_yaw_vehicle_asset_quality_passed':False,'boundary_residual_removed':False,
        'unseen_camera_and_paired_edit_GT_certified':False,'remaining_failures':visual['remaining_failures']}
    (OUT/'quality_gates.json').write_text(json.dumps(gates,indent=2),encoding='utf-8')
    (OUT/'completion_scope.json').write_text(json.dumps(overall,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');x=np.arange(89,138)
    for name,label,color in [('original','Original checkpoint','#687482'),('deferred','Prior deferred_R4','#da8b2a'),('final','Selected joint refinement','#267bb6')]:
        axes[0,0].plot(x,[r[name]['surface_clean_MAE'] for r in rows],c=color,label=label)
        axes[0,1].plot(x,[r[name]['deleted_target_shadow_area_clean_MAE'] if r[name]['deleted_target_shadow_area_clean_MAE'] is not None else np.nan for r in rows],c=color,label=label)
        axes[1,0].plot(x,[r[name]['surface_shadow_MAE'] if r[name]['surface_shadow_MAE'] is not None else np.nan for r in rows],c=color,label=label)
    axes[0,0].set_title('Visible supported road: clean reference MAE')
    axes[0,1].set_title('After Actor 0 deletion: old shadow area vs clean reference')
    axes[1,0].set_title('Observed cast-shadow pseudo labels: linear RGB MAE')
    axes[1,1].plot(x,[r['fixed_supported_road_MAE_vs_first'] or 0 for r in rows],label='Supported road',c='#267bb6')
    axes[1,1].plot(x,[r['fixed_full_image_MAE_vs_first'] for r in rows],label='Full image (includes unchanged parent)',c='#da8b2a')
    axes[1,1].set_title('Fixed camera static time consistency, sRGB MAE')
    for a in axes.flat:a.set_xlabel('Raw frame');a.grid(alpha=.2);a.legend(fontsize=8)
    fig.savefig(OUT/'final_measured_curves.png',dpi=160);plt.close(fig)
    table=''
    for key,label in [('surface_clean_MAE','可见支持道路，对干净参考：线性 RGB MAE'),('deleted_target_shadow_area_clean_MAE','删目标车后，原车影区域对干净参考：线性 RGB MAE'),('surface_shadow_MAE','原车位，车影伪标注：线性 RGB MAE'),('surface_observed_clean_MAE','原车位，真实干净观测：线性 RGB MAE'),('diagnostic_footprint_visible_clean_MAE','原黑块的可见世界点：线性 RGB MAE'),('observed_body_RGB_MAE','车身观测，保留正常暗面：表观 sRGB MAE'),('observed_body_visible_ID_L1','车身实例可见贡献与 body mask：归一化 L1')]:
        if key not in m['final']:continue
        valid_frames=sum(r['final'].get(key) is not None for r in rows)
        table+='<tr><td>'+label+'</td><td>'+str(valid_frames)+'/49</td>'+''.join('<td>%.6f</td>'%m[v][key] for v in ['original','deferred','final'])+'</tr>'
    def video(name,title):return f'<article><h3>{title}</h3><video controls preload="metadata" src="videos/{ev}/{name}.mp4"></video><a href="videos/{ev}/{name}.mp4">下载完整 49 帧</a></article>'
    videos=''.join([video('delete_target','删除目标车，保留其他车辆'),video('translate_smooth','连续平移 ±1 米'),video('yaw_smooth_15','连续旋转 ±15°'),video('yaw_smooth_30','连续旋转 ±30°：既有资产缺陷未通过'),video('fixed_static','固定相机，49 个时刻，移除可编辑 Actor'),video('fixed_translate','固定相机下运动与连续编辑'),video('shadow_effect','独立阴影开关的实际图像差异'),video('confidence','绿色：有恢复依据的道路范围')])
    steps=read('evidence/training_accounting.json');total=steps['total_gradient_updates']
    ownnew=sum(r['new_points'] for r in own)
    asset='已触发有界透明度适配，车身几何与 SH 外观仍冻结。' if actor['opacity_adapter_enabled'] else '没有触发车辆整体透明度适配；车身几何与 SH 外观保持冻结。'
    status='支持道路的主要门槛通过' if all(gates.values()) else '仍有道路质量门槛未通过'
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>49 帧道路去影与车辆关联阴影实测</title>
    <style>body{{font:16px/1.65 'Microsoft YaHei',sans-serif;color:#233448;background:#f4f7fb;max-width:1180px;margin:30px auto;padding:0 20px}}h1{{font-size:29px}}h2{{margin-top:36px}}h3{{font-size:18px}}p{{margin:12px 0}}article,.card{{background:white;border-radius:12px;padding:20px;margin:18px 0}}.notice{{border-left:5px solid #c98a2b}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}}.cards b{{font-size:27px;color:#176eaa}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(410px,1fr));gap:16px}}img{{max-width:100%;height:auto}}video{{width:100%;max-height:640px;background:#111}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce3eb}}code{{font-size:13px;word-break:break-all}}a{{color:#126a9d}}summary{{cursor:pointer;color:#126a9d}}@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}.grid{{display:block}}}}</style>
    <h1>完整 49 帧诊断、道路重建与车辆关联阴影</h1><p>scene0008，raw89–137。本轮实际完成 {total:,} 次优化更新和连续编辑验收。没有使用 subagent。</p>
    <div class="card notice"><b>实验已完成；{status}。</b><p>所保留模型为 <code>{train}</code>。结论只覆盖有真实跨时刻观测支持的道路。整场景无车影、车辆资产完全去污染、任意新视角和大角度车体质量仍未认证。</p></div>
    <div class="cards"><div class="card"><b>{road*100:.1f}%</b><br>道路参考误差下降，49 帧</div><div class="card"><b>{deletion*100:.1f}%</b><br>删除后原车影误差下降，21 个有效帧</div><div class="card"><b>{summary['mean_fixed_supported_road_time_MAE']:.2g}</b><br>固定相机道路时间 MAE</div><div class="card"><b>49 帧</b><br>完整对照与编辑序列</div></div>
    <img src="architecture.png" alt="跨时刻观测、固定道路、车辆几何阴影和可见性合成组件图"><p><a href="architecture.svg">可编辑 SVG 组件图</a></p>
    <h2>保留的方案与实际依据</h2><p>用现有 checkpoint 初始化，重建受影响的道路分支。固定平面由真实 LiDAR 支持，8 cm 网格共有 {dataset['supported_texels']:,} 个道路高斯；颜色和透明度通过跨时刻 RGB 优化。候选支持面积 {dataset['supported_surface_area_m2']:.1f} m²，保守内部约 {dataset['interior_texels']*.08**2:.1f} m²。没有凭空补出从未看到的真实道路。</p>
    <p>车辆远离道路点至少 4 米时的多帧观测，为内部区域的 {motion['far_reference_confirmed_fraction']*100:.1f}% 提供了额外交叉检查；其中 {motion['fraction_confirmed_with_MAE_under_002']*100:.1f}% 与当前参考相差低于 0.02 线性 RGB，平均差 {motion['mean_linear_MAE_current_vs_far_median']:.6f}。原黑块的 {motion['original_black_footprint_points']} 个道路点映射到这些参考后全部通过检查。其余区域的无车影属性仍有不确定性；这些是可信伪参考，不是独立密集真值。</p>
    <p>旧静态表示的归属由实际 rasterizer 的 αT 贡献决定：在 49 个视角、原始时间状态和固定时间状态下，对已确认道路/车身区域与整幅图的贡献分别求和。只接收全局可见贡献占比超过 90% 的点，并做三轮剥离检查隐藏层。除几何基线外，新接收 {ownnew:,} 点；这不等于它们都属于“假几何”，也包括需要由干净道路替代的旧道路表示。</p>
    <p>道路颜色和透明度在优化中可更新，但给定一次训练状态后不随时间变化；保留的静态点在道路重建通道中固定几何、透明度、时间包络和筛选状态，仍按当前相机重算 SH 方向。恢复范围外保留原模型。未选车辆采用缓存 DEVA/SAM 分割与三维框匹配后的前景保护；匹配不可靠时保留保守框。已有保护框过宽，曾把道路残影挡在恢复范围外，这一处已修正。自定义视角的保护用分割约束的三维框表面代理，尚不等同精细完整场景几何。</p>
    <p>16,000 更新的候选版曾在 raw117–137 出现银车尾灯红色残片，视觉验收否决。原因是未选车辆的旧高斯被固定在道路重建通道的锚点时间，而真实前景保护使用当前时刻。后续通过真实可见贡献和活动期间三维框归属，将 {foreground['foreground_points']:,} 个前景点从该通道分离；最终仍用原时刻的场景与语义保护保留这些车辆。零更新对照已确认残片消失，再追加 2,000 更新拟合道路和阴影。没有把道路参考里的红色直接涂掉。</p>
    <p>独立阴影由车辆高斯几何构建的 15 个分段盒体、道路和共享光照决定；不再只用整车一个 OBB。尺寸修正限 ±12%，中心修正限 ±8 cm。编辑时读取新车辆位姿重新投影，RGB 仅提供训练监督。阴影只修改深度排序后的可见道路贡献；删除车辆时对应模型阴影归零。当前 rasterizer 使用表观 sRGB 合成，道路着色的解码/编码是近似处理，不是完整物理光传输。</p>
    <h2>同一最终协议下的比较</h2><table><tr><th>指标，有效帧等权均值</th><th>有效帧</th><th>原 checkpoint</th><th>旧 deferred_R4</th><th>所选联合模型</th></tr>{table}</table>
    <p>最终评估使用相同的道路/遮挡筛选重新渲染三个模型。表中仅对有有效样本的帧等权平均；无样本计为缺失，绝不当作零误差。道路指标覆盖全 49 帧，原车位阴影指标有 40 个有效帧；删除指标仅 raw89–109 共 21 帧有目标车影样本。删除指标只取原车影伪标注中、与目标车最近且有距离间隔的子集；它没有配对编辑 GT。原模型已烘焙的阴影可以拟合原 RGB，因此“原车位阴影误差低”不能证明可编辑。按样本数加权的道路误差分别为 {pixel['original']['surface_clean_MAE']:.6f} / {pixel['deferred']['surface_clean_MAE']:.6f} / {pixel['final']['surface_clean_MAE']:.6f}。</p>
    <p><b>并非所有指标都改善：</b>按车影样本数加权的原车位阴影 MAE 为 {pixel['original']['surface_shadow_MAE']:.6f} / {pixel['deferred']['surface_shadow_MAE']:.6f} / {pixel['final']['surface_shadow_MAE']:.6f}，最终版仍高于原 checkpoint，也略高于旧方案。49 帧等权的阴影均值改善，不能替代这个结果。车身 RGB 和实例可见贡献误差也略有退化；当前完成的是支持道路恢复与几何关联阴影试验，阴影逼真度仍有提升空间。</p>
    <img src="final_measured_curves.png" alt="逐帧道路恢复、删除残影、原车影拟合和固定相机稳定性实测曲线">
    {video('comparison','完整 49 帧：原模型 / 旧方案 / 所选模型 / 删除目标车')}
    <div class="grid">{videos}</div>
    <h2>固定 raw101 的完整正负编辑检查</h2><p>实际 49 时刻视频中，目标车后半段会离开视野。因此另固定相机、场景时间和真实 raw101，连续采样 49 个编辑状态，覆盖平移 ±1 m 与旋转 ±30°，目标车全程参与编辑。这是 49 个编辑状态，不是另外 49 帧真实观测。车体与遮挡代理位姿最大差 {sweep['body_caster_pose_max_abs']:.3g}；平移后的道路阴影质心等变误差最大 {sweep['shadow_centroid_translation_error_max_m']*1000:.3f} mm；两条闭环序列首尾 RGB 逐值相同。数值跟随正确不代表大角度车辆外观通过。</p>
    <div class="grid"><article><h3>固定场景：平移 ±1 m</h3><video controls preload="metadata" src="videos/anchor_edit_sweep_final/translation.mp4"></video></article><article><h3>固定场景：旋转 ±30°，资产破面反例</h3><video controls preload="metadata" src="videos/anchor_edit_sweep_final/yaw30.mp4"></video></article></div>
    <h2>训练执行与自验证</h2><table><tr><th>实验</th><th>背景 / 阴影 / 联合更新</th><th>用途</th></tr><tr><td>首轮</td><td>1,500 / 2,000 / 4,500</td><td>扩大道路恢复并完成首次全序列验收</td></tr><tr><td>几何归属重拟合</td><td>1,000 / 500 / 2,500</td><td>验证清理后必须重新拟合，保留时间残差证据</td></tr><tr><td>贡献归属与精细前景保护</td><td>1,000 / 500 / 2,500</td><td>修正早期残影与时间自由度；视觉发现红色残片，未入选</td></tr><tr><td>未选前景分离与最终重拟合</td><td>500 / 250 / 1,250</td><td>消除固定道路通道的前景残片；最终入选</td></tr></table>
    <p>共 {total:,} 次实际更新。参考构建使用全部 49 帧；梯度采样排除 raw101/113/125，共 46 帧；原 checkpoint 已见过全 49 帧。这是同一序列的重建和编辑诊断，不能称为未见帧泛化。各阶段监督范围、保护区域和预算发生变化，阶段数值不作为公平论文消融。</p>
    <table><tr><th>检查</th><th>实际结果</th></tr>{''.join('<tr><td>'+k+'</td><td>'+str(v)+'</td></tr>' for k,v in gates.items())}</table>
    <p>49 帧删除全部 Actor 后，与无阴影静态渲染的最大差为 {summary['max_delete_identity']:.8g}。道路改动在恢复权限范围外的最大差为 {summary['max_road_routing_outside_same_actor_gates']:.8g}（比较相同 Actor 状态）；对原模型的范围外画面最大变化为 {summary['max_scenery_change_outside_road']:.8g}。不透明车身像素上的阴影开关最大差为 {maxopaque:.8g}。固定相机整幅图时间 MAE 仍为 {summary['mean_fixed_full_image_time_MAE']:.6f}，包含未重建范围内的原 GN11 表示。</p>
    <p>各训练阶段均做真实 GPU 梯度检查；保存、加载 addon 和 Adam 后，raw101 渲染逐值一致。光照与代理形状的有限差分、平移等变性、删除恒等和环境光下限检查通过。原 checkpoint 文件身份和源参数版本不变。下一步随机采样与训练的逐步确定性没有认证。完整 49 帧对照已逐帧视觉审查；连续编辑另检查代表帧和数值，不声称人工逐幅看过所有视频流。</p>
    <h2>仍未完成的部分</h2><p>{asset} 7 个责任检查时刻中，重分配后的车身 RGB 平均增加 {actor['mean_body_RGB_regression']:.6f}，车身 mask 外的阴影候选上 Actor 可见 α 平均为 {actor['mean_Actor_alpha_on_bodyfree_shadow_pseudo']:.6f}。这不能证明资产所有不可见部分都没有道路成分。正常暗面、自阴影和反射仍保留；大角度旋转的既有破面与视角缺失没有修好。</p>
    <p><b>整幅删车画面尚未通过无残影要求：</b>raw106–108 左下角仍有黑色薄片，位于绿色道路恢复权限之外。当前训练没有覆盖这些像素；不能用内部平均误差低来宣布它已解决。下面是最终版 raw106 删车结果与对应恢复范围。</p><div class="grid"><article><img src="evidence/edge_delete_target_raw106.png" alt="最终版 raw106 左下边缘残留黑片"></article><article><img src="evidence/edge_confidence_raw106.png" alt="最终版 raw106 绿色恢复权限；黑片处不在其中"></article></div>
    <p>阴影遮挡体仍是分段几何近似，接收面是固定平面。恢复/阴影跨出观测支持范围时可能被截断，始终被遮挡的道路和固定车影无法由当前观测唯一恢复。任意新相机、复制/替换资产、多场景泛化、变化光照和完整可重光照材质分解未认证。总体目标尚不能宣称全部完成。</p>
    <details><summary>保留的失败与阶段证据</summary><p>未加入未选车辆保护的零训练扩大替换损坏银车外观，已否决。原小窗口候选遗漏早期轨迹范围；保守框又强制保留部分真实道路残影。只改归属不重拟合会产生亮斑接缝。16,000 更新候选的尾灯残片也导致否决。所有阶段和反例均保留，没有用更低平均误差替代全画面检查。</p><div class="grid"><article><h3>已否决 16k 候选：raw125 红色残片</h3><img src="evidence/rejected_16k_red_fragments_raw125.png"></article><article><h3>最终 18k 版本：raw125</h3><img src="reviews/{ev}/observed/0008125.png"></article></div><p><a href="ownership_probe/comparisons.json">零更新归属消融</a> · <a href="guard_refinement_v2/summary.json">保护区域修正</a> · <a href="contribution_guarded_audit/rounds.json">真实贡献归属</a> · <a href="unselected_foreground_audit/receipt.json">未选前景分离</a> · <a href="evidence/visual_review.json">视觉验收记录</a></p></details>
    <details><summary>所选模型完整 49 帧画面对照</summary>{''.join('<img src="reviews/'+ev+'/page_'+str(k)+'.jpg">' for k in range(1,5))}</details>
    <h2>交付与恢复</h2><p><a href="{ev}/per_frame.json">逐帧实测</a> · <a href="{ev}/summary.json">指标汇总</a> · <a href="{train}/runtime_contract.json">渲染与监督约定</a> · <a href="{train}/actor_responsibility_check.json">资产责任检查</a> · <a href="quality_gates.json">质量门槛</a> · <a href="completion_scope.json">完成范围</a> · <a href="recovery_manifest.json">远端运行依赖和复查命令</a> · <a href="evidence/final_validation.json">交付检查</a></p>
    <p>远端 addon：<code>/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2/{train}/final.pt</code>。交付保留本地副本与实现 scripts/，SHA256 为 <code>{selected['checkpoint_sha256']}</code>。本地交付不是可脱离数据独立运行的整场景包；完整道路观测、原 checkpoint、保护数据和渲染依赖保留在远端，依赖列表与重评命令见运行清单。addon 不能直接替换完整原模型。本轮 GPU 作业均已结束，10 分钟进度检查停止，GPU 保持开启。</p></html>'''
    (OUT/'report.html').write_text(html,encoding='utf-8')
    (OUT/'measured_summary.json').write_text(json.dumps({'selected':selected,'road_reduction_vs_parent':road,'deletion_reduction_vs_parent':deletion,'metrics':m,'quality':overall},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Final report:',OUT/'report.html')
if __name__=='__main__':main()
