"""把任务层零准入和完整分母写进现有审阅页，不包装成模型负结果。"""
from collections import Counter
import html
import json
from pathlib import Path
import shutil

REASONS={'ego_below_fixed_task_speed':'ego起始速度不足2 m/s',
         'no_route_leader':'40m范围/路线内无前车',
         'target_not_persistent_full_task':'前车持续性不足',
         'no_forward_moving_leader':'持续前车接近静止',
         'moving_following_geometry_qualified':'任务几何合格'}


def build(source, output):
    source,output=Path(source),Path(output)
    data=json.loads((source/'result.json').read_text(encoding='utf-8'))
    protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    assert data['status']=='complete' and data['window_count']==18
    assert data['qualified_log_count']==0 and not data['selected']
    folder=output/'moving-following-screen';folder.mkdir(parents=True,exist_ok=True)
    for name in ['result.json','protocol.json']:shutil.copy2(source/name,folder/name)
    counts=Counter(r['reason'] for r in data['rows'])
    assert dict(counts)=={'ego_below_fixed_task_speed':3,'no_route_leader':13,'target_not_persistent_full_task':1,'no_forward_moving_leader':1}
    boxes=[(20,'固定6日志 × 3起点','18 个窗口'),(260,'ego达到跟车速度','15 个窗口'),
           (500,'路线内出现前车','2 个窗口'),(740,'持续且真实运动','0 个合格任务')]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 990 185"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#47697b"/></marker></defs><rect width="990" height="185" rx="12" fill="#fff"/>']
    for i,(x,title,value) in enumerate(boxes):
        svg.append(f'<rect x="{x}" y="25" width="210" height="82" rx="9" fill="{"#fff0dc" if i==3 else "#e5f0f4"}"/><text x="{x+105}" y="55" text-anchor="middle" font-family="sans-serif" font-size="17" fill="#224255">{title}</text><text x="{x+105}" y="84" text-anchor="middle" font-family="sans-serif" font-size="18" fill="#224255">{value}</text>')
        if i<3:svg.append(f'<path d="M{x+210} 66 H{x+236}" stroke="#47697b" stroke-width="2" marker-end="url(#arrow)"/>')
    svg.append('<text x="495" y="147" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#47697b">输入：真实RGB时间戳、标定、参考轨迹 → 任务资格 → 输出：完整分母；未进入模型推理</text></svg>')
    (folder/'architecture.svg').write_text(''.join(svg)+'\n',encoding='utf-8',newline='\n')
    table=[]
    for row in data['rows']:
        speed=row.get('past_speed_xy_mps')
        table.append(f'<tr><td>{row["log_id"][:8]}</td><td>+{row["offset_seconds"]:.1f}s</td><td>{REASONS.get(row["reason"],html.escape(row["reason"]))}</td><td>{speed:.3f}</td></tr>' if speed is not None else
                     f'<tr><td>{row["log_id"][:8]}</td><td>+{row["offset_seconds"]:.1f}s</td><td>{REASONS.get(row["reason"],html.escape(row["reason"]))}</td><td>未进入此检查</td></tr>')
    return '''<h2>真实运动跟车窗口：18个起点，0个任务准入</h2><div class="card"><strong>当前有限数据没有提供合格运动任务，不是模型运动重建的负结果。</strong><p>复用之前固定的6日志和+2.5/+4.5/+6.5秒起点；协议先于运动和模型误差检查保存。新问题要求向前运动的目标会改变普通跟车动作，不再要求日志ego已经减速。旧制动筛查计数保持不变。</p></div><img src="moving-following-screen/architecture.svg" alt="真实运动跟车任务资格流程和完整排除数量"><p>3个窗口ego速度不足；13个在规定40m范围和路线内没有前车；1个前车只出现于2/5个测量时刻；剩下1个就是近静止任务24642607，其过去参考速度0.044m/s、前两秒位移0.059m。没有目标走到视觉可观测性和动作相关性检查，不能把这两项未执行记为失败。</p><details><summary>查看18个窗口的全部排除记录</summary><table><tr><th>日志</th><th>起点</th><th>首个排除原因</th><th>过去目标速度 m/s</th></tr>'''+''.join(table)+'''</table></details><p>本任务只做CPU资格筛查，用时124.36秒；0次新检测、重建、生成或训练。关闭该窗口，不改变阈值、补选来源或给目标增加合成运动。它没有证明运动误差不重要，也没有产生新的科学failure卡。</p><p><a href="moving-following-screen/protocol.json">冻结协议</a> · <a href="moving-following-screen/result.json">完整分母与结果</a></p>'''
