"""将实际GPU收据、隔离评分与逐车登记生成独立报告；不写共享平台。"""
from collections import Counter
from html import escape as esc
import argparse
import json
from pathlib import Path
import re

OUT=Path(__file__).resolve().parents[3]/'outputs/V43_R4_GPU_20261031'
GPU=OUT/'gpu_runs'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def fmt(value):
    return '不可计算' if value is None else f'{value:.6f}' if isinstance(value,float) else str(value)

def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def rel(path):
    return path.relative_to(OUT).as_posix()

def label(scene):
    if scene=='sim_m0_original10k':return 'procedural_world_dataset_r2_vehicle0 · 原 M0 10K'
    if scene=='sim_natural_hybrid8_10k':return 'procedural_world_dataset_r2_vehicle0 · hybrid8 10K'
    if scene.startswith('real_first_scene'):return 'Waymo 14705303724557273004_3105_000_3125_000 · 原生 30K'
    return scene

def domain_label(path):
    text=str(path).lower()
    if 'purity' in text:return '合法可见贡献 ≥95% 的细分域'
    if 'maskfix' in text:return '修正掩码后的原候选域'
    return '原候选域（真实掩码未受影响）'

def main():
    registry=read(OUT/'actor_registry.json');queue=read(OUT/'actor_preparation_queue.json');assets=read(OUT/'existing_bound_assets.json')
    manifests=[];invalid_manifests=[]
    for path in GPU.rglob('gpu_manifest.json'):
        data=read(path)
        if 'enumerate' in str(path.parent):
            is_invalid_sim = data['scene_id'].startswith('sim_') and '/v43_decision_r4_cpu_20261031/run_r1/' in data.get('cpu_frozen_scene_dir','')
            (invalid_manifests if is_invalid_sim else manifests).append((path,data))
    if not manifests:raise ValueError('没有实际完整枚举收据')
    # r2修正real摘要语义；同一个domain只用最新有效评分，原评分仍保留在包内。
    score_candidates={}
    for path in GPU.rglob('scores.json'):
        data=read(path)
        if 'actions' not in data:continue
        if data['scene_id'].startswith('sim_') and '/v43_decision_r4_cpu_20261031/run_r1/' in data.get('cpu_frozen_scene_dir',''):continue
        key=(data['scene_id'],data.get('candidate_hash'),data.get('shared_route_context_hash'))
        score_candidates.setdefault(key,[]).append((path,data))
    scores=[]
    for paths in score_candidates.values():
        def revision(item):
            match=re.search(r'isolated_scores_r(\d+)',str(item[0]));return int(match.group(1)) if match else 0
        scores.append(max(paths,key=revision))
    manifest_rows=[]
    for path,m in sorted(manifests,key=lambda x:str(x[0])):
        restore=path.parent/'complete_restore_replay_receipt.json'
        manifest_rows.append([esc(label(m['scene_id'])),esc(domain_label(path)),
                              f"{len(m['evaluated_anchor_frames'])} / {m['frame_total_denominator']}",
                              m['action_count'],len(m['rows']),fmt(m['allkeep_native_none_parity_max_abs']),fmt(m.get('wall_seconds'))])
    score_rows=[];action_tables=[];summary_scores=[];operation_rows=[]
    for path,s in sorted(scores,key=lambda x:str(x[0])):
        actions=s['actions'];baseline=next(a for a in actions if a['action_id']=='h000')
        best=s.get('best_verified_shared_residual');base=baseline.get('joint_target_loss')
        domain=domain_label(path)
        score_rows.append([esc(label(s['scene_id'])),esc(domain),s['evaluated_route_count'],
                           fmt(s.get('development_safe_action_count')),fmt(base),fmt(best),
                           fmt(base-best) if base is not None and best is not None else '不可计算',
                           esc(str(s.get('best_safe_shared_action_id') or '不可计算'))])
        if s.get('best_safe_shared_action_id'):
            selected=next(a for a in actions if a['action_id']==s['best_safe_shared_action_id'])
            for q in ('delete','shift','rotate'):
                left=baseline['per_operation_target_losses'][q];right=selected['per_operation_target_losses'][q]
                operation_rows.append([esc(label(s['scene_id'])),selected['action_id'],q,fmt(left),fmt(right),fmt(right-left)])
        rows=[]
        for a in actions:
            obs=a['observations'];none=max([o.get('none_raw_rgb_change_mae',0) or 0 for o in obs])
            keep=max([o.get('keep_worsened_fraction',0) or 0 for o in obs])
            rows.append([a['action_id'],fmt(a.get('joint_target_loss')),
                         '通过' if a.get('development_constraint_pass') is True else '未通过' if a.get('development_constraint_pass') is False else '未知',
                         f'{keep*100:.3f}%' if a.get('reference_available') else '不可计算',fmt(none),fmt(a.get('finite_decision_gap'))])
        action_tables.append('<details><summary>'+esc(label(s['scene_id'])+' · '+domain)+'</summary>'+table(['共享方案','联合目标误差','开发约束','保留域恶化比例（最大）','none 变化（最大）','有限域差'],rows)+'</details>')
        summary_scores.append(dict(scene_id=s['scene_id'],domain=domain,source_scores=rel(path),reference_available=baseline.get('reference_available'),baseline_joint_target_loss=base,
                                   best_safe_shared_action_id=s.get('best_safe_shared_action_id'),best_verified_shared_residual=best,
                                   baseline_to_best_safe_improvement=(base-best) if base is not None and best is not None else None,
                                   development_safe_action_count=s.get('development_safe_action_count'),evaluated_route_count=s['evaluated_route_count']))
    bycontext={}
    for row in registry['rows']:bycontext.setdefault(row['context_id'],[]).append(row)
    hf_contexts={read(path)['context_id']:read(path)['scene_number'] for path in (OUT/'actor_screen_horizonforge').glob('scene_*/summary.json')}
    if len(hf_contexts)!=11:raise ValueError('HorizonForge来源场景应完整11个')
    actor_rows=[]
    for context,rows in sorted(bycontext.items()):
        is_hf=context in hf_contexts;display=('HorizonForge '+hf_contexts[context]+' · ') if is_hf else ''
        first=next(r for r in queue['rows'] if r['context_id']==context and r['preparation_priority_within_scene']==1)
        actor_rows.append([esc(display+'Waymo '+context),len(rows),rows[0]['raw_frames_denominator'],
                           esc(first['track_id']),f"{first['recommended_window_first']}–{first['recommended_window_last']}",first['recommended_window_front_frames'],
                           '需准备掩码、重建与候选' if is_hf else '已有模型；按逐车映射准备'])
    gallery=[]
    for path,m in sorted(manifests,key=lambda x:str(x[0])):
        frame=m['evaluated_anchor_frames'][len(m['evaluated_anchor_frames'])//2]
        matched=next((s for _,s in scores if s.get('shared_route_context_hash')==m.get('shared_route_context_hash')),None)
        best_id=matched.get('best_safe_shared_action_id') if matched else None
        ids=['h000',best_id or m['allrebind_action_id']]
        images=[]
        for action in dict.fromkeys(ids):
            for operation in ('none','delete','shift','rotate'):
                image=path.parent/'routes'/action/operation/(frame+'_final.png')
                if image.is_file():images.append('<figure><img loading="lazy" src="'+esc(rel(image))+'" alt="'+esc(action+' '+operation)+'"><figcaption>'+esc(action+' · '+operation+' · '+frame)+'</figcaption></figure>')
        gallery.append('<details><summary>'+esc(label(m['scene_id'])+' · '+domain_label(path))+'</summary><div class="images">'+''.join(images)+'</div></details>')
    main_calls=sum(len(m['rows']) for _,m in manifests)
    summary=dict(task_id='WS-V43-R4-GPU-SMALL-20261031',actor_scenes=len(bycontext),vehicle_tracks=len(registry['rows']),preparation_candidates=len(queue['rows']),
                 downloaded_missing_full_raw_files=10,downloaded_missing_raw_bytes=9843128477,horizonforge_crc_context_checked_records=2181,
                 native_enumerated_domains=len(manifests),main_enumeration_route_edit_outputs=main_calls,score_domains=summary_scores,
                 input_mask_invalid_old_simulation_domains=len(invalid_manifests),input_mask_invalid_old_simulation_outputs=sum(len(m['rows']) for _,m in invalid_manifests),
                 formal_eligibility=None,task_completion_threshold=None,learned_generalization_result=None,training_updates=0,human_verdict=None,
                 shared_platform='visual_diagnostics_and_actor_browser_only',browser_interactive_QA=False,
                 failure_ledger_refs=['V82-F01','V82-F07','V82-F08'],failure_ledger_delta=['V82-F07','V82-F08'])
    (OUT/'report_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V43 R4 小规模验证</title><style>
    *{box-sizing:border-box}body{max-width:1200px;margin:30px auto;padding:0 24px 60px;color:#203347;background:#f5f7fa;font:16px/1.75 "Microsoft YaHei",sans-serif}h1{font-size:29px;line-height:1.4}h2{font-size:22px;margin:32px 0 14px}p{max-width:1100px}a{color:#146382}section{background:white;padding:18px 22px;margin:20px 0;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dce4ec;padding:9px;text-align:left;vertical-align:top;overflow-wrap:anywhere}th{background:#edf3f7}.scroll{overflow-x:auto}details{margin:15px 0;border:1px solid #d6e1e9;border-radius:7px;padding:12px}summary{cursor:pointer;overflow-wrap:anywhere}.images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}figure{margin:0}figure img{width:100%;display:block}figcaption{font-size:13px;color:#526477}img.diagram{width:100%;height:auto}code{overflow-wrap:anywhere}.note{color:#536b80;font-size:14px}@media(max-width:700px){body{padding:0 12px}.images{grid-template-columns:1fr}section{padding:14px}}
    </style><h1>V43 R4 小规模验证</h1>'''
    html+='<p>本轮完成逐车盘点、原生 GPU 共享绑定和隔离评价。<b>修正后的两套仿真候选均存在满足暂定开发约束的更好共享绑定。</b>M0 联合目标误差 0.185311→0.184845，hybrid8 为 0.157236→0.151483；这验证了当前有限域内存在可比较的选择，没有证明学生学习或跨世界泛化。</p><p>GPU 复查修正了前一轮仿真掩码的 alpha 通道读取错误：候选从错误的 392,468／490,823 行变为 4,445／30,024 行，真实场景仍为 2,703 行。旧仿真输出保留为工程错误记录，不纳入下表研究比较。32 帧原始掩码、96 张实际训练用灰度掩码及训练读取器已分别核对，原重建训练不受本次通道错误影响。</p>'
    html+='<p><a href="actor_candidates/index.html">浏览车辆候选</a> · <a href="actor_registry.csv">全部车辆登记</a> · <a href="actor_preparation_queue.csv">45 条准备候选</a> · <a href="report_summary.json">数值摘要</a></p>'
    html+='<section><h2>输入、共享决策与评价</h2><img class="diagram" src="architecture.svg" alt="冻结表示和合法输入进入共享绑定，经原生编辑器后，由隔离评价分别输出决策差异与能力残余"><p>同一份保留／重新绑定方案先构成场景，再执行 none、删除、平移或旋转。原始几何、颜色、透明度、高斯数量及环境不训练；独立编辑参考只进入评分端。暂缓仍是查询层行为，本轮没有已训练策略。</p></section>'
    html+='<section><h2>每个场景先登记车辆</h2><p>四个工作台场景与 11 个 HorizonForge 场景，共 <b>1,228 条车辆轨迹</b>，包括未进入前相机的车辆。新增取回十个完整片段，共 9,843,128,477 字节；11 个 HorizonForge 全片的 2,181 条记录逐条核对 CRC 和 context。该部分未重建 11 个新场景。</p><p>初筛只读原始标注、相机与框内 LiDAR 标注计数。后者表示车辆在 FRONT 可见时的三维框内点数，不能当成前相机车身深度像素。框不是车身掩码，重叠不是物理遮挡真值，12 方位扇区不是隐藏表面完整性。每场准备队列保留三辆；所有正式资格仍为未知。</p>'
    html+=table(['来源与完整原始编号','车辆总数','全片帧数','首条准备候选 track','建议原始帧窗口','窗口 FRONT 帧','工程准备'],actor_rows)
    html+='<p class="note">上表的首条候选用于安排准备，不能称为最佳可编辑车辆。工作台展示的全序列前六张预览与 49 帧准备队列采用不同排序。全部 45 条及未入选车辆均保留在 CSV。</p></section>'
    html+='<section><h2>原生接线验收</h2><p>原三个配置各登记 8 个共享方案。仅在现有独立编辑参考的 0／8／15 三锚评分；真实场景取原 49 帧的首／中／末锚。原有 16／16／49 帧总分母保留，其余帧未测，不补零。M0 和 hybrid8 来自同一物理世界。</p>'
    html+=table(['配置','输入域','已测／登记帧','共享方案','输出数','原生 none 最大差','枚举墙钟秒'],manifest_rows)
    html+='<p>每组实际覆盖保存、退出后恢复完整原模型组件／运行状态／随机状态、FP32 重放及下一阶段枚举入口。冻结模型与优化器状态通过原始完整检查点引用保留；推理没有创建新优化器。修复后的 CPU 合同测试共 15 项通过，包含不透明 RGBA／RGB／灰度／LA 的实际 PNG 回环及逐帧掩码分母核对。本轮没有长训练，未用 CPU 检查替代原生验证。hybrid 的加载器归一化副作用及精确恢复另留工程收据，后续改用私有输入副本。</p></section>'
    html+='<section><h2>联合操作比较与能力边界</h2><p>联合目标对各非空旧位／新车体区域先归一，再等权合并帧和操作。保留域与 none 单独约束：MAE 增量上限 0.005；保留域像素误差增量超过 0.02 的比例上限 1%；none 原始 RGB 变化 MAE 上限 0.005。它们是评分前声明的开发条件，没有被称为部署安全阈值。</p>'
    html+=table(['配置','评分域','已评方案','满足开发约束方案','全部保留目标误差','同域最佳残余','全部保留→最佳改善','同域最佳共享方案'],score_rows)
    html+='<p>这些最优方案 ID 来自隔离真值的后验完整枚举，角色是有限域评价参照；部署决策器尚未训练。学生泛化、风险—覆盖和任务完成率本轮均不作数值结论。</p>'
    html+='<p>正确候选的 M0 有 2／8 份方案满足开发约束，hybrid8 为 4／8；最优共享方案分别为 h002、h006。三档约束敏感性 0.001／0.005／0.01 下，最优方案与通过数均未变。原定 95% 贡献细分没有追加正式枚举；两个早期探针均基于无效的旧候选全集，仅保留为工程诊断，不进入研究比较。</p>'
    if (OUT/'independent_review/maskfix_coarse_pareto.png').is_file():
        html+='<figure><img style="max-width:100%" src="independent_review/maskfix_coarse_pareto.png" alt="正确掩码下联合目标与保留损伤对照"><figcaption>正确候选域：同一份共享方案同时评价全部操作；两个重建配置来自同一个程序化世界。</figcaption></figure>'
    html+='<h3>同一共享方案在各操作上的变化</h3>'+table(['配置','共享方案','操作','全部保留','联合最优共享方案','误差变化（正数为恶化）'],operation_rows)
    html+='<p>hybrid8 的联合最优 h006 在旋转上的目标误差从 0.160598 升至 0.161511（恶化约 0.000913）。联合改善不代表每种操作都改善；这里没有按操作另挑绑定后拼成成功结果。旧错误掩码域的数值只作工程历史，不用于 D/C 判断。</p>'
    if (OUT/'independent_review/m0_pareto_counterexample.png').is_file():html+='<details><summary>历史工程错误域反例（不计研究比较）</summary><figure><img style="max-width:100%" src="independent_review/m0_pareto_counterexample.png" alt="错误候选域的历史诊断图"><figcaption>输入掩码无效的旧 M0 域；仅留存工程诊断，不能推断正确候选域的收益或局限。</figcaption></figure></details>'
    html+=''.join(action_tables)
    html+='<p>真实场景仅核对实际 none 保持与工程有效性；缺少完整独立编辑真值，所以反事实安全候选数、决策差距和最佳编辑残余均不可计算。即使 none 保持通过，也不能说删除／旋转已安全。完成阈值未校准，所有表示能力分层仍为 C?；没有把拒判当作完成任务。</p></section>'
    references=''.join('<figure><img src="independent_reference_samples/'+q+'/shadow_on/rgb/000008.png" alt="仅评价的独立仿真 '+q+' 参考"><figcaption>独立仿真参考 · '+q+' · 000008（仅评分端）</figcaption></figure>' for q in ('none','delete','shift','rotate'))
    html+='<section><h2>实际影像</h2><p>先列独立仿真中帧参考，再展开原始保留与联合最优共享绑定。真实场景另列全部重新绑定作对照，不给它最优或成功标签。</p><div class="images">'+references+'</div><p>代理检查实际中帧输出后，可见删除仍有车体残留、旋转仍有重影；有限域最佳不等于完成编辑。当前输出不记作人工验收通过，也不据此断言所有更细路由或其他观测都不可行。</p>'+''.join(gallery)+'</section>'
    html+='<section><h2>本轮边界与下一入口</h2><p>本轮是原生小域与车辆准备验证：三配置仍只有两个已有物理祖先，完整编辑参考仅来自一个程序化世界。来源组粗细、合法可见贡献、观测缺失和表示残余分别登记。没有启动回归、偏好或策略训练，也没有完成跨世界可学习验证。后续先按逐车准备队列补齐合法掩码、背景显露和逐车候选，再锁定独立祖先及校准分区。</p><p>共享平台保持场景导航、候选浏览和折叠影像对照；本报告、阶段说明和实验日志独立交付。静态引用、脚本和选择逻辑已检查；浏览器工具此前阻止 file://，因此没有声称完成真实浏览器交互验收。人工验收字段保留 null。</p></section>'
    (OUT/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps(dict(report=str(OUT/'report.html'),main_calls=main_calls,domains=len(manifests),score_domains=len(scores)),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    OUT=args.output.resolve();GPU=OUT/'gpu_runs';main()
