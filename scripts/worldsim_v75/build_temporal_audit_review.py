"""在双任务闭环审阅页内呈现有限时间状态审计，不扩建另一套状态文档。"""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build(audit, output):
    audit, output = Path(audit), Path(output)
    folder = output/'temporal-audit'; folder.mkdir(parents=True, exist_ok=True)
    data = json.loads((audit/'result.json').read_text(encoding='utf-8'))
    assert data['status'] == 'complete' and data['new_model_calls'] == 0
    shutil.copy2(audit/'result.json', folder/'result.json')
    shutil.copy2(audit/'protocol.json', folder/'protocol.json')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(11.8,7.4))
    table=[];images=[]
    for i,case in enumerate(data['cases']):
        name=case['log_id']; short=name[:8]; t=np.array(case['future_times_s'])
        xyz=np.array(case['reference_future_centers']); v=np.array(case['reference_past_velocity_mps'])
        ax=axes[0,i]
        ax.plot(t,np.linalg.norm(xyz[:,:2]-xyz[0,:2],axis=1),color='#376b84',label='Recorded reference displacement')
        ax.plot(t,np.linalg.norm(v[:2])*t,color='#dc8540',ls='--',label='Past-only reference CV displacement')
        ax.set(title=f'{short} | target motion',xlabel='Time after start (s)',ylabel='Displacement from initial position (m)',ylim=(0,.5))
        ax.grid(alpha=.18);ax.legend(fontsize=8,loc='upper left')
        ax=axes[1,i]; keys=['target_static','target_past_cv']
        values=[case['controls'][k]['progress_change_vs_recorded_m'] for k in keys]
        bars=ax.bar(range(2),values,color=['#376b84','#dc8540'],width=.5)
        ax.axhline(0,color='#82949d',lw=.8)
        for bar,val in zip(bars,values):
            ax.annotate(f'{val:+.4f} m',(bar.get_x()+bar.get_width()/2,val),xytext=(0,5 if val>=0 else -5),textcoords='offset points',ha='center',va='bottom' if val>=0 else 'top')
        ax.set(xticks=range(2),xticklabels=['Hold initial pose','Past reference CV'],ylabel='Executed progress change (m)',ylim=(-.12,.15),title='Direct state + IDM: change from recorded trajectory')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        controls=case['controls']; readout=case['readout']
        table.append(f'<tr><td>{short}</td><td>{case["reference_past_speed_xy_mps"]:.3f}</td><td>{case["reference_future_max_displacement_xy_m"]:.3f}</td><td>{controls["target_static"]["progress_change_vs_recorded_m"]:+.4f}</td><td>{controls["target_past_cv"]["progress_change_vs_recorded_m"]:+.4f}</td><td>{readout["matched_count"]}/21</td></tr>')
        shutil.copy2(audit/name/'past-input-review.jpg',folder/f'{short}-past.jpg')
        images.append(f'<details><summary>{short}：过去真实RGB、参考投影与可用观测</summary><img src="temporal-audit/{short}-past.jpg" alt="真实过去输入与对应目标"></details>')
    fig.suptitle('Both braking tasks approach a nearly stationary target',x=.055,ha='left',fontsize=16)
    fig.text(.5,.015,'117 frames / 15 decisions • shared initial pose and RGB action • extra-information CPU control • no new video generation',ha='center',fontsize=9,color='#567085')
    fig.tight_layout(rect=(0,.05,1,.94))
    for ext in ['png','svg','pdf']:fig.savefig(folder/f'temporal-controls.{ext}',dpi=170)
    plt.close(fig)
    svg='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 240"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#426578"/></marker></defs><rect width="1040" height="240" rx="12" fill="#fff"/><g font-family="sans-serif" text-anchor="middle" fill="#203c50" font-size="17"><rect x="20" y="30" width="195" height="65" rx="8" fill="#e8f2f5"/><text x="117" y="57">过去真实 RGB</text><text x="117" y="81" font-size="14">已保存观测的可用性审计</text><rect x="20" y="135" width="195" height="65" rx="8" fill="#fff1df"/><text x="117" y="162">过去标注＋共同初始姿态</text><text x="117" y="186" font-size="14">额外信息控制</text><rect x="275" y="135" width="170" height="65" rx="8" fill="#fff1df"/><text x="360" y="162">静止 / 匀速轨迹</text><text x="360" y="186" font-size="14">不读取未来速度</text><rect x="505" y="135" width="160" height="65" rx="8" fill="#e8f2f5"/><text x="585" y="162">当前条件状态</text><text x="585" y="186" font-size="14">共享其他演员</text><rect x="725" y="135" width="130" height="65" rx="8" fill="#e8f2f5"/><text x="790" y="174">固定 IDM</text><rect x="910" y="135" width="110" height="65" rx="8" fill="#e8f2f5"/><text x="965" y="174">ego 执行</text><text x="635" y="54" font-size="18">本轮只做观测审计与普通状态闭环</text><text x="635" y="84" font-size="15">没有新的 DVGT 或世界模型推理；不将跟踪速度冒称模型输出</text></g><g stroke="#426578" stroke-width="2" fill="none" marker-end="url(#a)"><path d="M215 167 H270"/><path d="M445 167 H500"/><path d="M665 167 H720"/><path d="M855 167 H905"/><path d="M965 200 V222 H585 V202"/></g></svg>'''
    (folder/'architecture.svg').write_text(svg+'\n',encoding='utf-8')
    for path in folder.glob('*.svg'):
        path.write_text('\n'.join(s.rstrip() for s in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    return '''<h2>时间状态审计：这两个任务适合检验运动误差吗？</h2><div class="card"><strong>两例近静止前车没有提供有意义的运动修复空间；关闭在这两例上追加速度干预。</strong><p>旧四组条件逐帧核验：目标中心差是恒定平移，尺寸、朝向和其他演员均相同，目标继续使用记录的未来轨迹。因此已有DVGT结果是位置干预，不是自然运动重建结果。</p></div><img src="temporal-audit/architecture.svg" alt="过去观测审计和静止匀速直接状态控制架构"><img src="temporal-audit/temporal-controls.png" alt="真实目标接近静止与普通时间控制的微小进度影响"><table><tr><th>日志</th><th>过去参考速度 m/s</th><th>未来最大参考位移 m</th><th>静止控制进度差 m</th><th>过去CV进度差 m</th><th>可用对应观测</th></tr>'''+''.join(table)+'''</table><p>此处速度来自参考标注的固定一秒最小二乘拟合，微小位移可能包含标注误差。静止/匀速都共享GT起点、尺寸和朝向；其他演员仍用记录轨迹，属于额外信息诊断。六条117帧状态反馈中，两条精确复现旧基线，另外四条为新控制；均无参考重叠。CPU耗时约6秒，未初始化CUDA。</p><p>02678d04的0/21是已保存策略管线中的可用观测数，包含检测、类别/分数与地面接触过滤，不等于原始检测器召回率为零。24642607有21/21对应、同一track ID，但接触点拟合速度0.523 m/s，参考约0.044 m/s，前后半段速度向量差2.362 m/s。视角变化的接触点不是刚体中心；该波动不能冒称DVGT运动失效，也不据此人为给目标加速。</p>'''+''.join(images)+'''<p>已核查的<a href="https://github.com/wzzheng/DVGT/blob/51cf3f6d11fdff8bc7e2bbe1a88f71665ccb2236/dvgt/models/architectures/dvgt1.py">官方DVGT-1接口</a>输出时序点图与ego位姿，没有对象速度、身份或未来轨迹输出；<a href="https://github.com/NVIDIA/flashdreams/blob/bc711d6f95693d73693e6b5fac75749cc6fcf1d7/integrations_v2/omnidreams/impl/grpc/protos/video_model.proto">OmniDreams接口</a>接收外部逐时刻actor轨迹。两任务均有截止前3时刻×7视角的原始图像清单，但未运行新的时序DVGT，不能由文件存在声称运动可观测性已通过。</p><p><a href="temporal-audit/temporal-controls.pdf">时间控制PDF图</a> · <a href="temporal-audit/temporal-controls.svg">SVG图</a> · <a href="temporal-audit/protocol.json">冻结协议</a> · <a href="temporal-audit/result.json">完整结果</a></p>'''
