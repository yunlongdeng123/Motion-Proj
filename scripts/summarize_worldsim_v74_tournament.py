"""固定标准的配对日志汇总；只读已有评价，不更新方法参数。"""
import argparse,json,itertools
from pathlib import Path
from collections import defaultdict
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--evaluations',type=Path,nargs='+',required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
KEYS=['hit_rate','early_rate','miss_rate','positive_surface_recall_02','any_correct_intersection','mean_free_intrusion_m','positive_surface_mean_m']
CONTROLS={'WEX':['A0_soft','A1_fixed','A2_greedy','A3_milp'],'RIF':['B0_sampled','B1_exact','B2_regular'],'DCS':['C0_fixed','C1_residual','C2_local_network','C3_no_demand']}
summaries={ds:{} for ds in ['nuscenes','av2']};rows=[]
for folder in args.evaluations:
    summary=json.loads((folder/'summary.json').read_text())
    for ds,methods in summary['datasets'].items():summaries[ds].update(methods)
    rows+=json.loads((folder/'per_actor.json').read_text())

def pair(ds,method,control):
    mm=summaries[ds][method]['ready']['metrics_query'];cc=summaries[ds][control]['ready']['metrics_query'];result={}
    for k in KEYS:
        left=mm[k]['per_log'];right=cc[k]['per_log'];logs=sorted(set(left)&set(right));delta=np.array([left[l]-right[l] for l in logs])
        # 五日志仅3125种有放回重采样，精确枚举经验bootstrap，避免选种子。
        means=[float(delta[list(ids)].mean()) for ids in itertools.product(range(len(delta)),repeat=len(delta))] if len(delta) else []
        result[k]={'paired_logs':logs,'per_log_delta':dict(zip(logs,delta.tolist())),'delta':float(delta.mean()) if len(delta) else None,
                   'empirical_paired_log_bootstrap_95':np.quantile(means,[.025,.975]).tolist() if means else None,
                   'candidate_mean':mm[k]['mean'],'control_mean':cc[k]['mean']}
    d={k:v['delta'] for k,v in result.items()};complete=all(v is not None for v in d.values())
    protection=complete and d['miss_rate']<=.01 and d['positive_surface_recall_02']>=-.01 and d['any_correct_intersection']>=-.01 and d['positive_surface_mean_m']<=.01
    growth=complete and d['hit_rate']>=.03 and d['early_rate']<=.005 and d['mean_free_intrusion_m']<=.005
    physical=complete and d['early_rate']<=-.03 and d['hit_rate']>=-.005 and d['mean_free_intrusion_m']<=-max(.01,.1*cc['mean_free_intrusion_m']['mean'])
    counts={'growth':sum(v>0 for v in result['hit_rate']['per_log_delta'].values()),'physical':sum(v<0 for v in result['early_rate']['per_log_delta'].values())}
    required=4 if ds=='nuscenes' else None
    direction=(growth and counts['growth']>=4) or (physical and counts['physical']>=4) if ds=='nuscenes' else None
    contract=summaries[ds][method]['ready'];outputs_ok=contract['missing_method_output']==0 and contract['budget_exceeded']==0 and contract['empty_surfaces']==0
    return {'metrics':result,'support_protection':protection,'growth_path':growth,'physical_path':physical,'direction_counts':counts,'required_main_logs':required,
            'complete_output_budget':outputs_ok,'main_screen_pass':bool(protection and (growth or physical) and direction and outputs_ok) if ds=='nuscenes' else None,
            'boundary':'predeclared screening numbers, not human verdict or statistical proof; AV2 separately reported with no invented 4/3 rule'}

comparisons={}
for ds in summaries:
    comparisons[ds]={}
    for method,controls in CONTROLS.items():
        if method not in summaries[ds]:continue
        comparisons[ds][method]={c:pair(ds,method,c) for c in controls if c in summaries[ds]}

costs={}
for ds in summaries:
    costs[ds]={}
    for method in summaries[ds]:
        subset=[r for r in rows if r['dataset']==ds and r['method']==method and not r.get('contract_missing_input')]
        reconstruction=[r.get('reconstruction') or {} for r in subset]
        times=[r['wall_seconds'] for r in reconstruction if r.get('wall_seconds') is not None]
        faces=[r['metrics_query']['faces'] for r in subset if r.get('metrics_query')]
        costs[ds][method]={'ready_objects':len(subset),'reconstruction_seconds_median':float(np.median(times)) if times else None,
            'reconstruction_seconds_sum':sum(times) if times else None,'time_defined_objects':len(times),'faces_median':float(np.median(faces)) if faces else None,'faces_max':max(faces) if faces else None,
            'cost_boundary':'timed per-asset reconstruction only; historical reuse has no measured reconstruction time; FIT training and external setup separately reported'}
data={'protocol':'43 ready +23 missing BUILD; log equal metrics, dataset specific FIT and exposed DEV; no independent FINAL claims',
      'sources':[str(p) for p in args.evaluations],'comparisons':comparisons,'costs':costs,'summary':summaries,'human_verdict':None}
(args.output/'paired_log_comparisons.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
lines=['# V74 固定表面真实留出比较','', '配置冻结后统一读取，两个数据集各自 FIT 与 DEV。43 ready 对象和23缺BUILD合同对象均保留；没有自有QUERY返回时指标未定义，不填零。以下是预定筛选标准计算，人工verdict留空。','',
'```mermaid','flowchart LR','  F[FIT数据] --> P[各自定标或提议网络]','  B[对象BUILD观测] --> A[WEX共享域交换]','  B --> R[RIF区间有限元]','  B --> C[DCS需求生片]','  P --> A','  P --> R','  P --> C','  A --> M[固定三角表面]','  R --> M','  C --> M','  M --> Q[真实首次求交]','  H[留出QUERY真值] --> E[配对日志评价]','  Q --> E','```','']
for ds,methods in summaries.items():
    lines+=['## '+ds,'','日志等权均值。命中/提前/缺失/召回单位%，侵入和距离单位m。','',
            '| 方法 | hit | early | miss | recall | free | distance | 中位面数 | 中位构建秒 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method,s in methods.items():
        q=s['ready']['metrics_query'];values=[]
        for k in ['hit_rate','early_rate','miss_rate','positive_surface_recall_02','mean_free_intrusion_m','positive_surface_mean_m']:
            val=q[k]['mean'];values.append('未定义' if val is None else f'{val*(100 if k.endswith("rate") or k.endswith("02") else 1):.3f}')
        cost=costs[ds][method];tm=cost['reconstruction_seconds_median'];lines.append('| '+method+' | '+' | '.join(values)+f' | {cost["faces_median"]} | '+(f'{tm:.3f}' if tm is not None else '历史复用')+' |')
    lines+=['','| 候选−控制 | Δhit(pp) | Δearly(pp) | Δmiss(pp) | Δrecall(pp) | Δfree(m) | 支撑保护 | 主筛选路径 |','|---|---:|---:|---:|---:|---:|---|---|']
    for method,controls in comparisons[ds].items():
        for control,result in controls.items():
            vals=[]
            for k in ['hit_rate','early_rate','miss_rate','positive_surface_recall_02','mean_free_intrusion_m']:
                val=result['metrics'][k]['delta'];vals.append('未定义' if val is None else f'{val*(1 if k=="mean_free_intrusion_m" else 100):+.3f}')
            lines.append('| '+method+' − '+control+' | '+' | '.join(vals)+f' | {result["support_protection"]} | {result["main_screen_pass"]} |')
    lines+=['']
lines+=['小样本区间按日志配对有放回bootstrap枚举。全对象合同分母、逐日志原值、全部七指标区间、缺输入与面预算情况保存在 paired_log_comparisons.json；区间跨零不表示已证明等效。现代外部基线与历史训练来源、原生面预算须单列，不能据此伪造同预算论文排名。','']
(args.output/'REAL_RESULTS.md').write_text('\n'.join(lines))
for ds,comp in comparisons.items():
    print(ds,{m:{c:r['main_screen_pass'] for c,r in refs.items()} for m,refs in comp.items()})
