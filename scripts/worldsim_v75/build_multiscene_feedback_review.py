"""汇集两个已完成制动任务，配对展示生成反馈和普通直接状态控制。"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=['gt_clean','dvgt_metric','dvgt_lidar_scaled','reference_lidar']
NAMES={'gt_clean':'GT条件','dvgt_metric':'DVGT','dvgt_lidar_scaled':'全局尺度','reference_lidar':'目标LiDAR'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--case1',type=Path,required=True);p.add_argument('--control1',type=Path,required=True)
    p.add_argument('--case2',type=Path,required=True);p.add_argument('--architecture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--temporal-audit',type=Path)
    p.add_argument('--moving-screen',type=Path)
    p.add_argument('--shape-feedback',type=Path)
    p.add_argument('--shape-audit',type=Path)
    p.add_argument('--shape-confirmation',type=Path)
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    cases=[]
    for label,src,ctrl in [('02678d04',a.case1,a.control1),('24642607',a.case2,a.case2/'state_control')]:
        data=json.loads((src/'comparison.json').read_text());state=json.loads((ctrl/'result.json').read_text())
        assert data['status']=='complete' and state['status']=='complete'
        cases.append({'label':label,'rgb':data,'state':state})
        folder=out/label;folder.mkdir(exist_ok=True)
        shutil.copy2(src/'comparison.json',folder/'rgb-comparison.json');shutil.copy2(ctrl/'result.json',folder/'state-comparison.json')
        shutil.copy2(src/'actual-policy-inputs.jpg',folder/'actual-policy-inputs.jpg')
    shutil.copy2(a.architecture,out/'architecture.svg')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12,8.1))
    for i,case in enumerate(cases):
        data=case['state']['cases'][1:];x=np.arange(3)
        for column,keys in enumerate([['direct_progress_change_vs_gt_m','rgb_progress_change_vs_gt_m'],
                                      ['direct_mean_abs_action_change_vs_gt_mps2','rgb_mean_abs_action_change_vs_gt_mps2']]):
            ax=axes[i,column]
            for j,key in enumerate(keys):
                v=[r[key] for r in data]
                bars=ax.bar(x+(j-.5)*.34,v,width=.34,color=['#386986','#d96c4b'][j],label=['Direct state + IDM','Generated RGB + IDM'][j])
                for b,value in zip(bars,v):ax.annotate(f'{value:+.2f}' if column==0 else f'{value:.3f}',(b.get_x()+b.get_width()/2,value),xytext=(0,4 if value>=0 else -4),textcoords='offset points',ha='center',va='bottom' if value>=0 else 'top',fontsize=9)
            ax.axhline(0,color='#77909d',lw=.7);ax.set_xticks(x,['DVGT','Global scale','Target LiDAR'])
            ax.set_title(f'{case["label"]} | '+('Executed progress' if column==0 else 'Applied action response'),loc='left')
            ax.set_ylabel('Change from interface-specific GT (m)' if column==0 else 'Mean absolute change from GT (m/s²)')
            ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True);ax.margins(y=.25)
    for col in range(2):
        lims=[ax.get_ylim() for ax in axes[:,col]];limits=(min(t[0] for t in lims),max(t[1] for t in lims))
        for ax in axes[:,col]:ax.set_ylim(limits)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
    fig.text(.5,.02,'Two exposed development logs • 117 frames / 15 decisions per arm • extra-information direct-state diagnostic • no seed or horizon sweep',ha='center',fontsize=9,color='#567085')
    fig.tight_layout(rect=(0,.055,1,.95))
    for ext in ['png','svg','pdf']:fig.savefig(out/f'two-task-feedback.{ext}',dpi=180)
    plt.close(fig)
    table=[];sections=[]
    for case in cases:
        for r in case['state']['cases']:
            rgb=next(x for x in case['rgb']['cases'] if x['arm']==r['arm'])
            table.append(f'<tr><td>{case["label"]}</td><td>{NAMES[r["arm"]]}</td><td>{r["initial_geometry_residual_m"]:.3f}</td><td>{r["rgb_progress_change_vs_gt_m"]:+.3f}</td><td>{r["direct_progress_change_vs_gt_m"]:+.3f}</td><td>{r["rgb_mean_abs_action_change_vs_gt_mps2"]:.3f}</td><td>{rgb["final_target_reference_clearance_m"]:.3f}</td></tr>')
        sections.append(f'<h2>{case["label"]}：四组实际反馈视频</h2><video controls preload="metadata" src="{case["label"]}/four-arm-feedback.mp4"></video><details><summary>实际策略输入与前车选择</summary><img src="{case["label"]}/actual-policy-inputs.jpg"></details><p><a href="{case["label"]}/rgb-comparison.json">RGB结果</a> · <a href="{case["label"]}/state-comparison.json">普通状态控制</a></p>')
    (out/'comparison.json').write_text(json.dumps({'status':'complete','cases':cases,'human_verdict':None,'scope':'two exposed development logs, not independent confirmation'},indent=2)+'\n',encoding='utf-8')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7.5 两个任务的生成闭环</title><style>body{max-width:1140px;margin:36px auto;padding:0 24px;background:#f4f7fa;color:#203c50;font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif}h1{font-size:36px;line-height:1.3}h2{margin-top:36px;font-size:24px}img,video{width:100%;border-radius:10px;background:white}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;border-bottom:1px solid #ccdbe5;text-align:left}.card{background:white;border:1px solid #ccdbe5;border-radius:12px;padding:24px;margin:24px 0}.tag{color:#688395;font-size:14px}a{color:#087899}details{margin:22px 0}</style><p class="tag">V7.5 · 2026-09-20 · 两个已曝光开发日志 · 实际反馈与普通强控制</p>
    <h1>重建误差怎样传到<br>生成状态和实际制动？</h1><p>相同任务内比较GT条件、自然DVGT读出、普通全局尺度、额外目标LiDAR参照。生成RGB驱动策略，再由动作更新ego和下一段生成；直接状态控制作为额外信息诊断。</p><img src="architecture.svg" alt="两条控制接口及反馈路径">
    <div class="card"><strong>同一报告保留两条任务、四组输入和普通解释。</strong><p>每组117帧、15次决策，seed42固定。先通过真实RGB策略与GT生成反馈基线，再读取重建误差；不按残余大小决定保留。图中进度和动作变化均相对本接口的GT条件分支，不把两个接口之差直接称为世界模型误差。</p></div>
    <img src="two-task-feedback.png" alt="两个制动任务的生成RGB和直接状态控制对比">
    <table><tr><th>日志</th><th>条件</th><th>中心残余 m</th><th>RGB进度变化 m</th><th>状态控制变化 m</th><th>RGB动作差 m/s²</th><th>RGB末端目标间距 m</th></tr>'''+''.join(table)+'''</table><p>更多制动或更少进度不自动代表改善；末端间距相对于固定真实目标足迹。只覆盖约3.87秒，不声称完成停车或长期安全。</p>'''+''.join(sections)+'''
    <h2>来源和信息边界</h2><p>02678d04来自原48窗口后的接近任务开发。24642607来自另一次冻结6日志×3起点的有限窗口：18→2个记录减速→1个可见前车任务；该日志曾用于其他研究，因此不称独立确认。两者真实输入和GT生成基线均通过，未采用未通过恢复条件的IoU关联变体。</p><p>DVGT官方前向只接7张截止前RGB。距离读出额外使用已知标定、GT尺寸和朝向，属于受控距离诊断，不能冒称原生端到端场景重建排名。全局尺度用背景LiDAR；目标参照用三个截止前扫描与GT目标掩码，无GT运动平移补偿。原4点/5点拒绝记录保留，有限补证后分别42/55点。</p><p>固定路线、地形、其他演员轨迹和非反应式交通均有明确边界。没有跨SOTA普遍性证据；任何主方法仍需实际恢复和独立确认。</p><p><a href="two-task-feedback.pdf">PDF图</a> · <a href="two-task-feedback.svg">SVG图</a> · <a href="comparison.json">两任务完整结果</a> · <a href="../V75_State_Control_Review/index.html">前一任务的普通关联控制</a></p></html>'''
    second=cases[1]['state']['cases'];raw=second[1];anchor=second[3]
    finding=f'<div class="card"><strong>保留第二个任务中的好案例，降低单目标中心修复主张的优先级。</strong><p>24642607的DVGT残余{raw["initial_geometry_residual_m"]:.2f}米，只对应{abs(raw["rgb_progress_change_vs_gt_m"]):.2f}米进度差。目标LiDAR将几何残余降至{anchor["initial_geometry_residual_m"]:.2f}米，平均动作差{raw["rgb_mean_abs_action_change_vs_gt_mps2"]:.3f}→{anchor["rgb_mean_abs_action_change_vs_gt_mps2"]:.3f} m/s²，但进度差变为{anchor["rgb_progress_change_vs_gt_m"]:+.2f}米。两任务均未得到动作与执行一致恢复的严重badcase，不继续扩大本例平移、尺度、关联或时长来维持主张。</p></div>'
    html=html.replace('<img src="two-task-feedback.png"',finding+'<img src="two-task-feedback.png"')
    if a.temporal_audit:
        from build_temporal_audit_review import build
        html=html.replace('</html>',build(a.temporal_audit,out)+'</html>')
    if a.moving_screen:
        from build_moving_following_review import build
        html=html.replace('</html>',build(a.moving_screen,out)+'</html>')
    if a.shape_feedback:
        if not a.shape_audit:p.error('--shape-feedback requires --shape-audit')
        from build_shape_feedback_review import build
        html=html.replace('<img src="architecture.svg"', '<div class="card"><strong>最新：相近近端距离，仍可能产生不同制动。</strong><p>两种普通形状导出，输入近端间距差小于4毫米。实际生成反馈中，两任务的配对行进差为2.757／0.301米；普通类别先验部分缓解动作偏差，尚未一致恢复。4段468帧已完成。</p><a href="#shape-feedback">查看实际视频、完整曲线和输入边界 →</a></div><img src="architecture.svg"')
        html=html.replace('</html>',build(a.shape_feedback,a.shape_audit,out)+'</html>')
    if a.shape_confirmation:
        from build_shape_confirmation_review import build
        confirmation=json.loads((a.shape_confirmation/'comparison.json').read_text())
        if confirmation['decision']=='primary_not_reproduced_close_candidate':
            html=html.replace('最新：相近近端距离，仍可能产生不同制动。','最新：一次额外seed未复现主候选，按事前规则关闭。')
            html=html.replace('两种普通形状导出，输入近端间距差小于4毫米。实际生成反馈中，两任务的配对行进差为2.757／0.301米；普通类别先验部分缓解动作偏差，尚未一致恢复。4段468帧已完成。',
                              '发现阶段的主场景配对行进差为+2.757米；固定seed43后变为−0.324米，平均误差和欠制动差也反向。两例GT基线通过，全部预定分支保留，不追加seed维护这个候选。')
        html=html.replace('新证据：相近的近端距离，不保证相同的生成闭环','发现阶段（seed42）：相近距离与生成反馈的候选差异')
        html=html.replace('<img src="architecture.svg"','<p><a href="#shape-confirmation">最新：查看唯一额外seed的复核结果 →</a></p><img src="architecture.svg"')
        html=html.replace('</html>',build(a.shape_confirmation,out)+'</html>')
    (out/'index.html').write_text(html,encoding='utf-8',newline='\n')
    for path in out.glob('*.svg'):path.write_text('\n'.join(s.rstrip() for s in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    print(out/'index.html')


if __name__=='__main__':main()
