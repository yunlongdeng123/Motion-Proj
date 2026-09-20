"""展示固定额外seed的完整结果，包括反向结果和失败终态。"""
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(fig,path):
    for ext in ['png','svg','pdf']:fig.savefig(path.with_suffix('.'+ext),dpi=180,facecolor='white')
    plt.close(fig)
    p=path.with_suffix('.svg');p.write_text('\n'.join(s.rstrip() for s in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')


def build(src,output):
    out=output/'shape-confirmation';out.mkdir(exist_ok=True)
    d=json.loads((src/'comparison.json').read_text());assert d['status']=='complete'
    shutil.copy2(src/'comparison.json',out/'comparison.json')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors=['#6b7f94','#ac6b26'];fig,axes=plt.subplots(2,2,figsize=(13,8.6),gridspec_kw={'width_ratios':[1,1.55]})
    for i,c in enumerate(d['cases']):
        ax=axes[i,0];pairs=c['pairs']
        if not pairs:continue
        for j,p in enumerate(pairs):
            val=p['progress_extent_minus_class_m'];ax.bar(j,val,width=.5,color=colors[j]);ax.text(j,val,f'{val:+.3f}',ha='center',va='bottom' if val>=0 else 'top')
        ax.set_xticks(range(len(pairs)),[f'Seed {p["seed"]}' for p in pairs]);ax.set_ylabel('Visible extent − fixed class (m)')
        ax.set_title(c['log_id'][:8]+' | Executed progress',loc='left');ax.margins(y=.3)
        ax=axes[i,1];keys=['mean_error_extent_minus_class_mps2','underbraking_extent_minus_class_mps2','overbraking_extent_minus_class_mps2']
        x=np.arange(3)
        for j,p in enumerate(pairs):
            vals=[p[k] for k in keys];bars=ax.bar(x+(j-.5)*.32,vals,.32,color=colors[j],label=f'Seed {p["seed"]}')
            for b,v in zip(bars,vals):ax.annotate(f'{v:+.2f}',(b.get_x()+b.get_width()/2,v),xytext=(0,4 if v>=0 else -4),textcoords='offset points',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
        ax.set_xticks(x,['Mean absolute error','Max underbraking','Max overbraking'])
        ax.set_ylabel('Visible extent − fixed class (m/s²)');ax.set_title('Action-error tradeoffs',loc='left');ax.margins(y=.3)
    for ax in axes.flat:ax.axhline(0,color='#738899',lw=.8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.legend(*axes[0,1].get_legend_handles_labels(),loc='upper right',ncol=2,frameon=False)
    fig.suptitle('One fixed additional seed: state export and driving feedback',x=.03,ha='left',fontweight='bold',fontsize=17)
    fig.text(.03,.025,'Primary rule frozen before seed 43: progress difference > 1 m AND higher mean error AND higher underbraking in 02678d04.\nTwo exposed logs; 117 frames per run. GT baselines precede paired arms. No horizon, threshold or seed search.',fontsize=10,color='#526878')
    fig.tight_layout(rect=(0,.09,1,.93));save(fig,out/'replication-summary')
    fig,axes=plt.subplots(2,2,figsize=(13,8.4))
    for i,c in enumerate(d['cases']):
        for j,seed in enumerate([42,43]):
            ax=axes[i,j]
            for k,arm in enumerate(['dvgt_class_prior','dvgt_visible_extent']):
                series=next((s for s in c['series'] if s['seed']==seed and s['arm']==arm),None)
                if series:ax.plot(series['time_s'],series['action_error_mps2'],'o-',ms=3,color=['#237a9b','#d26b3f'][k],label=['Fixed class','Visible extent'][k])
            ax.axhline(0,color='#738899',lw=.8);ax.grid(alpha=.15);ax.set(xlabel='Time (s)',ylabel='Applied − own-ego reference (m/s²)',title=f'{c["log_id"][:8]} | Seed {seed}',xlim=(0,3.867))
        lo=min(a.get_ylim()[0] for a in axes[i]);hi=max(a.get_ylim()[1] for a in axes[i])
        for a in axes[i]:a.set_ylim(lo,hi)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
    fig.text(.5,.02,'All 15 policy observations; positive error means less braking than the fixed physical reference at that run\'s own ego state.',ha='center',fontsize=10,color='#526878')
    fig.tight_layout(rect=(0,.055,1,.95));save(fig,out/'all-action-errors')
    decision={'primary_reproduced':'主候选通过这一次seed复核；尚不等于独立日志确认或完整恢复。',
              'primary_not_reproduced_close_candidate':'主候选未通过事前复核条件；关闭这个具体候选，不追加seed救结果。',
              'not_evaluable_baseline_or_engineering_stop':'队列在基线或工程环节停止，配对规则未完成评价，不能宣称科学复现。'}[d['decision']]
    rows=[];videos=[]
    labels={'gt_clean':'GT条件','dvgt_class_prior':'固定类别尺寸','dvgt_visible_extent':'可见范围拟合'}
    for c in d['cases']:
        for r in c['rows']:
            rows.append(f'<tr><td>{c["log_id"][:8]}</td><td>{r["seed"]}</td><td>{labels[r["arm"]]}</td><td>{r["progress_m"]:.3f}</td><td>{r["mean_abs_action_error_at_own_ego_mps2"]:.3f}</td><td>{r["median_abs_action_error_at_own_ego_mps2"]:.3f}</td><td>{r["max_underbraking_mps2"]:.3f}</td><td>{r["max_overbraking_mps2"]:.3f}</td></tr>')
        name=f'{c["log_id"][:8]}-two-seed.mp4'
        if (src/name).exists():
            shutil.copy2(src/name,out/name)
            videos.append(f'<h3>{c["log_id"][:8]}：上排seed42，下排seed43；左列类别先验，右列可见拟合</h3><video controls preload="metadata" src="shape-confirmation/{name}"></video>')
    return f'''<section id="shape-confirmation"><h2>一次额外seed复核：状态导出候选是否重现？</h2>
    <div class="card"><strong>{decision}</strong><p>seed43在运行前固定。先生成两个GT条件基线，沿用原门控，再运行原样的两个适配器；没有重新拟合输入、增加时长或调整策略。队列终态：{d['queue_status']}，新生成{d['new_generated_frames']}帧，逐帧回放{d['new_replay_frames']}帧。</p></div>
    <img src="shape-confirmation/replication-summary.png" alt="两个seed的行进差与全部动作误差取舍">
    <p>差值统一为可见拟合减固定类别先验。动作误差比较各分支自身ego状态下的物理参考，不等于“行进越少越安全”。第一个场景为事前指定主候选，第二个场景完整保留；两者均为已曝光开发日志。</p>
    <img src="shape-confirmation/all-action-errors.png" alt="两场景两seed全部15次动作误差">
    {''.join(videos)}
    <h3>GT与所有配对分支的绝对指标</h3><table><tr><th>日志</th><th>seed</th><th>输入</th><th>行进 m</th><th>平均误差 m/s²</th><th>中位误差 m/s²</th><th>最大欠制动 m/s²</th><th>最大额外制动 m/s²</th></tr>{''.join(rows)}</table>
    <p>第二任务的小效应需要保留：seed42／43下配对行进差为0.301／0.257米，平均动作误差差为0.123／0.119 m/s²。固定类别尺寸在两个seed都通过原绝对门控，可见拟合则都因动作误差中位数超过0.5未通过；最大额外制动相同。这个局部现象可由普通先验恢复，不支持为了它开发复杂新方法，也不替代主候选的失败判定。</p>
    <p>复核指标与判定规则在读取seed43结果前冻结；seed42是发现结果，不能倒称预注册确认。只有两个seed，也未隔离尺寸与中心、生成器与感知器的贡献，不支持普遍性结论。原GT形状额外信息对照仍在前一节保留。</p>
    <p><a href="shape-confirmation/replication-summary.pdf">复核图PDF</a> · <a href="shape-confirmation/replication-summary.svg">SVG</a> · <a href="shape-confirmation/comparison.json">完整曲线、指标与机器判定</a></p></section>'''
