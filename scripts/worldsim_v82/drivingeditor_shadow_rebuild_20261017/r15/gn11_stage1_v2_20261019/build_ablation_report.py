"""呈现实际消融状态与比较合同，不以训练进度替代视觉结论。"""
from pathlib import Path
import datetime,html,json,shutil,zipfile
W=Path(__file__).resolve().parent
O=W.parent.parent/'outputs/gn11_stage1_v2_20261019/ablations'

def main():
    runtime=json.loads((W/'ablation_runtime.json').read_text(encoding='utf-8'))
    plan=json.loads((W/'ablations/plan.json').read_text(encoding='utf-8'))
    O.mkdir(parents=True,exist_ok=True)
    for src,name in [(W/'ablations/plan.json','plan.json'),(W/'ablations/cpu_contract_receipt.json','cpu_contract_receipt.json'),
                     (W/'road_ablations_source.zip','source.zip')]:shutil.copyfile(src,O/name)
    contract=dict(task=plan['task'],matched_steps=list(range(5000,30001,5000)),iterations_each=30000,
        arms=['full','real_only','prior_only','ownership_only'],actors=[6,37,5,26,18],seed=0,
        full_frame_wh=[480,320],frames=list(range(89,138)),second_stage_updates=0,
        shared_ownership_fix=True,frozen_provenance=True,all_views_are_training_views=True,
        old_native_baseline='original A first-stage30K; same input/seed protocol, old initial GPU parameter snapshot unavailable',
        old_native_missing_matched_nodes='use only actually existing nodes; never fabricate or replay missing baseline steps',
        regions=['OBSERVED_same_world_position','PRIOR_different_position','UNKNOWN'],
        quality_checks=['static vehicle shapes','dynamic a6/a37 retention','same-world observed road','prior texture and seams','unknown areas','shadows'],
        parameter_updates_in_final_evaluation=0,human_verdict=None,quality_accepted=False)
    (O/'comparison_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    rows=[]
    names=dict(full='完整版',real_only='归属＋跨帧真实观测',prior_only='归属＋左右供体先验',ownership_only='仅归属修复')
    for name in contract['arms']:
        if name=='full':
            status=runtime['main'];weights=dict(observed_rgb=1.,prior_rgb=.1,observed_depth=.1,prior_depth=.01)
            gate='既有CUDA零更新准入通过';pid=status['pid']
        else:
            arm=runtime['arms'][name];status=arm['status'] or {}
            weights=arm['manifest']['road_weights'];gate='CUDA零更新通过' if arm['gate'] else ('准入中' if arm['gate_status'] else '排队，未执行准入')
            pid=(arm['train_launch'] or {}).get('pid','—')
        phase=html.escape(status.get('status','queued'));step=status.get('iteration',0)
        rows.append(f'<tr><td>{names[name]}</td><td>{weights["observed_rgb"]} / {weights["prior_rgb"]}</td><td>{weights["observed_depth"]} / {weights["prior_depth"]}</td><td>{step:,}/30,000 · {phase}<br>PID {pid}</td><td>{gate}</td></tr>')
    svg='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 350"><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10" fill="none" stroke="#9cb9cf"/></marker></defs><rect width="1000" height="350" fill="#132131"/><g fill="#223c50" stroke="#7695ad"><rect x="20" y="118" width="170" height="115" rx="10"/><rect x="230" y="118" width="170" height="115" rx="10"/><rect x="450" y="30" width="235" height="90" rx="10"/><rect x="450" y="200" width="235" height="90" rx="10"/><rect x="735" y="100" width="240" height="150" rx="10"/></g><g fill="none" stroke="#9cb9cf" stroke-width="2" marker-end="url(#a)"><path d="M190 175H230"/><path d="M400 150H425V75H450"/><path d="M400 200H425V245H450"/><path d="M685 75H710V145H735"/><path d="M685 245H710V205H735"/></g><g text-anchor="middle" fill="#eef6fc" font-family="Microsoft YaHei,sans-serif" font-size="18"><text x="105" y="150">同五身份 / 49帧</text><text x="105" y="181">同初始化协议</text><text x="105" y="212">seed0 / 各30K</text><text x="315" y="151">共同归属修复</text><text x="315" y="183">严格静态</text><text x="315" y="215">路由 / 体积排除</text><text x="568" y="63">同世界位置真实曝光</text><text x="568" y="96">开 / 关</text><text x="568" y="233">异位置供体先验</text><text x="568" y="266">开 / 关</text><text x="855" y="135">4组完整比较</text><text x="855" y="170">最多2路并发</text><text x="855" y="205">相同步数 / 全幅49</text><text x="855" y="237">来源区域分别复核</text><text x="500" y="327" font-size="16">道路来源为2×2开关；归属修复单独组是两种道路目标都关闭的对照。</text></g></svg>'''
    (O/'architecture.svg').write_text(svg,encoding='utf-8')
    free=runtime['disk_free_bytes']/1e9;reserve=runtime['disk_required_estimate_bytes']/1e9
    node_rows=[]
    for path in sorted(O.glob('*_checkpoint_*_audit.json')):
        name,node=path.name.split('_checkpoint_',1);step=int(node.split('_',1)[0])
        receipt=json.loads(path.read_text(encoding='utf-8'))
        if (name not in runtime['arms'] or receipt['status']!='PASSED_CPU_NODE_DURABILITY' or receipt['step']!=step or
            receipt['sampler_steps']!=step or receipt['GPU_restore_performed'] or receipt['training_updates']!=0 or
            step not in runtime['arms'][name]['audited_steps']):raise RuntimeError('invalid arm checkpoint receipt '+path.name)
        node_rows.append(f'<tr><td>{names[name]}</td><td>{step:,}</td><td>模型/Adam有限，RNG/SGCM/sampler和原生49齐全</td><td><a href="{path.name}">独立CPU凭据</a></td></tr>')
    node_html='<h2>已实际核验的消融节点</h2><table><tbody>'+''.join(node_rows)+'</tbody></table><p>只读CPU检查不含节点GPU恢复、不增加优化更新，也不代表视觉质量通过。完整版节点另见主报告；不能将主实验凭据冒充消融凭据。</p>' if node_rows else ''
    main_phase='完整版已完成30K训练' if runtime['main']['status']=='complete' else '主实验继续运行'
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GN11 v2 · 三组30K消融</title><style>body{{background:#0c1621;color:#e4edf5;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;margin:0}}main{{max-width:1120px;margin:auto;padding:30px 20px}}a{{color:#83d0ff}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:12px;border-bottom:1px solid #40566a;text-align:left}}img{{width:100%;height:auto}}.state{{padding:16px;background:#22354a;border-radius:10px}}h2{{margin-top:36px}}code{{color:#b0e4d8}}</style><main><h1>GN11 v2：三组30K消融</h1>
<p class="state">真实采样UTC：{html.escape(runtime['collected_utc'])}。控制器PID{runtime['registry']['controller_pid']}，并发上限为两路，原主实验和gate均计入上限。完整效果对照尚未完成，human_verdict=null。</p>
<p>用户已授权各组相同五身份、完整raw89–137、相同输入初始化协议和seed0，fresh第一阶段30K、第二阶段0。{main_phase}；新组独立目录、独立优化器、每5K保存，不加载旧训练权重、不自动重放。</p>
<h2>比较结构</h2><img src="architecture.svg" alt="共同输入和归属修复、真实曝光和供体开关、四组全幅对照的组件箭头图">
<table><thead><tr><th>实验</th><th>真实 / 先验 RGB权重</th><th>真实 / 先验深度权重</th><th>实际进度</th><th>工程准入</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>两种来源沿用完整版原权重，关闭时权重为0；没有重新归一化、改mask、改目标颜色、扩供体或混入窗外帧。三组源码与完整版的区别仅在道路权重配置和并行准入包装；静态表示、车身监督路由、体积排除、动态完整mask alpha MSE100保持一致。</p>
<h2>如何判断哪项有效、哪项回退</h2><p>仅归属组是道路目标全关闭的对照；真实观测组对比它，检查真实观测带来的变化；供体组对比它，检查先验带来的变化。完整版分别对比两组，检查两种来源共同使用时的增益或冲突。再与原A原生第一阶段实际存在的相同步数节点比较归属修复组合的效果；缺失的旧节点不补造、不重放。旧版无初始GPU张量快照，不能把输入/seed协议一致说成旧初始参数已逐位对照。</p>
<p>冻结query分为7,787个同世界位置确认样本、5,119个异位置供体先验样本、2,963个未知样本；它们是raw89车身像素射线样本，不是均匀道路面积。仅跨帧组只监督OBSERVED，仅供体组只监督PRIOR；未将已有真实曝光区域改成供体区域。覆盖不同，分别检查真实曝光、先验和未知区域，同时保留完整480×320画面。</p>
<p>30K后输出各组49帧full / dynamic / static / delete_all / alpha零更新评估，并实际观看：静态车形、a6/a37动态保持、同世界位置道路、先验纹理/接缝、未知区域和阴影。全49是训练重建视图，3D框是额外归属和遮挡约束，不能称独立测试或干净背景真值。当前训练loss、准入通过和节点完整性均不替代视觉结论。</p>
{node_html}<h2>资源和证据</h2><p>每个新worker设置10GiB Torch分配上限；并发训练上限为2，gate也占槽。磁盘本轮空闲{free:.3f}GB，保留全部剩余保存节点的估算需求{reserve:.3f}GB（增密结束前按配置点数加10%，结束后按实际点数，加共享8GB余量；不是严格点数上界）。保存空间不足时保留内存等待，不删除历史资产。</p>
<p><a href="runtime.json">实际进程、GPU、存储与各组准入状态</a> · <a href="plan.json">冻结消融计划</a> · <a href="comparison_contract.json">完整比较合同</a> · <a href="cpu_contract_receipt.json">CPU来源梯度与队列检查</a> · <a href="source.zip">隔离源码与许可证</a></p>
<p><a href="../index.html">当前完整版报告与道路来源证据</a> · <a href="../../baseline_engineering_20261018/index.html">已完成原A/B基线</a></p><p>全部四组训练、评估、下载、实际质量对照和Git交付完成后，再检查所有worker与待执行控制器并按授权关机。文件和源码检查不等同浏览器UI验收。</p></main></html>'''
    (O/'index.html').write_text(doc,encoding='utf-8')
    protocols=[]
    for name in ['real_only','prior_only','ownership_only']:
        path=O/(name+'_protocol_receipt.json')
        if path.exists():protocols.append(f'<a href="{path.name}">{names[name]}</a>')
    if protocols:
        doc=doc.replace('<h2>资源和证据</h2>','<p>已实际核对以下消融的前100步，选帧、时间扰动和全部优化器学习率与当前完整版逐项一致：'+' · '.join(protocols)+'。这不是旧初始GPU参数逐位比较。</p><h2>资源和证据</h2>')
        (O/'index.html').write_text(doc,encoding='utf-8')
    print(json.dumps(dict(report=str(O/'index.html'),arms=contract['arms'],quality_accepted=False)))

if __name__=='__main__':main()
