"""User-requested wrap-up only: evidence-based brief, no experiment launch."""
from pathlib import Path
import json,statistics,html,re

B=Path(__file__).resolve().parents[2];O=B/'outputs/V43_Ownership_20261027/gpu_v43';R=O/'target_unknown_v2';A=O/'target_unknown_delivery_v1';C=O/'target_conditional_delivery_v1';W=Path(__file__).parent
SCENE='14705303724557273004_3105_000_3125_000';S=B/'outputs/GS_Scene_Diagnostics_20261027'/SCENE

def diagram():
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1360" height="940" viewBox="0 0 1360 940"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#68879b"/></marker></defs><rect width="1360" height="940" fill="#f6fafc"/>']
    def txt(x,y,lines,size=21):
        for i,t in enumerate(lines):out.append(f'<text x="{x}" y="{y+i*(size+10)}" font-size="{size}" text-anchor="middle" font-family="Microsoft YaHei,sans-serif" fill="#234657">{html.escape(t)}</text>')
    def box(x,y,w,h,lines,fill='#e7eff7',size=21):
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="#adc3d1" stroke-width="1.5"/>');txt(x+w/2,y+32,lines,size)
    def arrow(x,y,x2,y2):out.append(f'<path d="M{x} {y}L{x2} {y2}" fill="none" stroke="#68879b" stroke-width="2" marker-end="url(#a)"/>')
    txt(680,45,['V43 阶段简报：身份、结构与显露背景'],30)
    box(30,85,1300,95,['输入：原49帧高斯与目标跟踪；199帧RGB / LiDAR / 相机证据','原V1无编辑直出保留；四字段分存：语义、实例、运动、证据'],size=22)
    for x,title,fill in [(30,'① 证据与目标专属unknown','#fff1d9'),(475,'② 联合分解与背景补层','#e0f2e9'),(920,'③ 原生渲染与分项验收','#fff1d9')]:box(x,225,410,505,[],fill);txt(x+205,266,[title],22)
    box(50,297,370,96,['alpha×T追踪目标相关贡献','保留全量行ID与原始分支']);arrow(235,393,235,423)
    box(50,423,370,110,['实测背景与前景分别取证','缺测、冲突、混合支撑单列','static批次 ≠ 世界静止'],size=20);arrow(235,533,235,563)
    box(50,563,370,128,['新增审计：原static含运动','下一关：同一世界表面支持','不把深度相符直接当身份'],size=20)
    box(495,297,370,96,['目标相关保留率＋DC适配','可选：按时刻贡献历史门控']);arrow(680,393,680,423)
    box(495,423,370,110,['世界静止的实测背景补层','可选：受限表面插值','可选：低权重视频伪监督'],size=20);arrow(680,533,680,563)
    box(495,563,370,128,['联合学习与固定剥离对照','保护邻近背景、远处、其他车','学习权重尚不是身份概率'],size=20)
    box(940,297,370,96,['同一HTML帧轴：RGB＋3D','实际保留率、几何与证据']);arrow(1125,393,1125,423)
    box(940,423,370,110,['逐项验收：残影 / 背景 / 结构','新相机压力测试另存','生成教师不充当实测GT'],size=20);arrow(1125,533,1125,563)
    box(940,563,370,128,['目前：结构改善，残影未净','三候选供下一轮验证','本轮封存，不新增实验'],size=20)
    arrow(235,180,235,225);arrow(440,478,475,478);arrow(885,478,920,478);arrow(1125,730,1125,777)
    box(30,777,1300,116,['独立车辆编辑支线：Actor局部刚体资产 → 内部 / 邻近外域 / 其余外域opacity监督 → 转向覆盖与边缘检查','保留原V1；不采用固定密度×2作默认。后续冻结规则跨场景，再验收联合优化是否优于强流水线。'],size=21)
    out.append('</svg>');(R/'wrapup_architecture.svg').write_text(''.join(out),encoding='utf8')

def main():
    diagram();stats=[]
    for name,root in [('joint199',A),('joint_dense199',A),('split199',A),('conditional_joint',C),('conditional_split',C)]:
        d=json.loads((root/name/'summary.json').read_text());rows=[x for x in d['metrics'] if x['split']=='calibration_holdout'];r=next(x for x in d['metrics'] if x['raw']==113)
        stats.append(dict(name=name,step=d['step'],protection_changed_mean=statistics.mean(x['protected_pixels_changed_gt1byte'] for x in rows),raw113_protection_changed=r['protected_pixels_changed_gt1byte'],background_seeds=d['measured_background_seeds']))
    rank=['conditional_joint','joint_dense199','joint199'];lookup={x['name']:x for x in stats};motion=json.loads((R/'static_motion_audit/summary.json').read_text())
    result=dict(status='user_requested_wrapup_research_held',top3_research_candidates=rank,quality_accepted=False,ghost_free_RGB=False,background_protection_passed=False,independent_generalization_verified=False,formal_updates=50000,pilot_updates=3000,extra_smoke_updates=980,total_updates=53980,statistics=stats,static_branch_motion_audit=motion,checkpoint_downloaded=False)
    (R/'wrapup_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    summary=json.loads((R/'summary.json').read_text());summary.update(result);summary['experiments']=stats;(R/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    top=[
      ['1','按时刻软归属＋联合优化＋视频先验','conditional_joint','按时刻贡献历史控制目标相关保留率/DC；联合学习背景补层，低权重ProPainter伪监督','保护区改动最少；仍有原位暗残影。时间门控与教师同时改变，收益未拆开归因。'],
      ['2','受限插值背景＋联合优化','joint_dense199','目标专属unknown保留率/DC与背景联合优化；背景由重复实测表面内插值加密','无视频先验；相对第3项收益有限，增加补层数量，仍有灰雾与暗残留。'],
      ['3','直接实测背景＋联合优化','joint199','只用重复实测点建补层，优化目标相关行的保留率/DC与补层','最简单、无视频先验，适合保留作后续可迁移性基准；仍有残影和域外变化。']]
    changes=[
      ['目标专属alpha×T归因','算法','从原生可见贡献找目标相关行，未把全局unknown全删；2D语义、实例、运动、证据分别存','依赖掩码和原生渲染器辅助通道；换case可复用，物理身份正确性未证明'],
      ['软保留率＋DC＋补层联合优化','算法','联合调整候选行opacity保留率/DC与新背景层，其他原参数冻结；保留率不是身份概率','逐场景优化，尚不是可直接迁移的unknown分类器'],
      ['按时刻贡献历史门控','算法','原49帧目标贡献比例，邻近±1帧中位平滑；时间变化的保留率和DC幅度','不输入case ID，但依赖源视角；新相机沿用同一时刻门控，未证明视角无关'],
      ['分区域背景保护','算法＋损失权重','邻近外域、其他车辆、其余外域分别归一化；总保护权重2','代码不绑定本case；保护参考仍含原模型错误，不能代表干净背景GT'],
      ['199帧实测背景记忆','证据构建＋规则参数','排除预测前景与标注物体；10cm体素、≥3帧且跨度≥0.5秒；直接补层692颗','199帧是背景证据，原生查询仍49帧；当前5raw-frame跨度依赖10Hz，泛化前须改实际时间戳'],
      ['受限道路表面插值','几何构建＋规则参数','最长边1m、法线竖直分量>.95、只在实测包围三角形内；补层6080颗','非外推；插值位置/颜色不是新增实测，不能保证未见背景正确'],
      ['ProPainter低权重教师','外部先验＋权重参数','沿用旧固定结果，31帧可用；权重.25，仅原修补mask且排除真实背景监督像素','最终交付仍为原生高斯渲染；训练依赖生成伪监督，不能当真实身份或背景GT'],
      ['逐帧硬剥离纯度阈值','诊断规则／调参','当前帧目标贡献比例≥.75或.95时整行抑制；固定剥离组使用.75','减少域外伤害但亮色环境泄漏、残影仍明显，未采纳'],
      ['Actor局部刚体与有界结构','车辆表示改动','实测目标点初始化，局部结构约束，随统一刚体轨迹编辑','3D形状与转向明显改善；表示、容量及损失同时变化，不能只归因身份规则'],
      ['车身内／近外／远外opacity监督','算法＋损失权重','内.05、近外.025、其余外.025；带宽clip(round(.1√车身像素数),2,12)','同起点冻结几何与颜色；核心alpha .9301→.9396，边缘.5841→.5489；跨场景待验'],
      ['固定密度×2、共享MLP、环境clamp','诊断对照','密度×2核心更深但边缘也更厚；MLP无额外收益；clamp仅定位环境泄漏','均未设默认，也未作为已验证的归属修复']]
    todo=[
      ['P0','重做运动一致的背景证据','分开原static批次与真实静止；跟踪同一世界表面，不能把同一行的多帧深度支持当静止证明。'],
      ['P0','混合高斯支撑拆分','区分车辆几何、背景几何、车辆外观污染；验证拆分能去残影且不切掉背景。'],
      ['P0','拆开时间门控与视频教师','固定数据、步数与初始化，完成有／无门控×有／无教师对照，不能把当前组合收益归给某一组件。'],
      ['P1','环境回填的有效性','在匹配覆盖率下比较原环境与受限回填；同时验收颜色、曝光、几何，不能用clamp掩盖缺陷。'],
      ['P1','更强的先分解后补景基线','使用相同背景证据、教师和计算预算；现有硬剥离较弱，尚不能证明联合优化普遍更优。'],
      ['P1','冻结规则跨场景验证','先第二场景；停止逐case调参，核验残影、其他车辆与道路保护。第三场景继续暂缓。'],
      ['P1','新相机／时间／编辑幅度解耦','补匹配原模型对照与可见真值；当前±0.5m仅压力图。固定幅度跨时间、同时间多幅度分别比较。'],
      ['P2','可学习unknown归属','先人工验收可靠正负样本和混合支撑，再造数据训练；另设校准和独立场景测试。'],
      ['P2','车底与阴影','车体身份与路面光照效应分开；按此前约定暂缓。']]
    def mdtable(head,rows):return '|'+ '|'.join(head)+'|\n|'+ '|'.join(['---']*len(head))+'|\n'+''.join('|'+ '|'.join(str(x) for x in row)+'|\n' for row in rows)+'\n'
    md='''# V43 阶段简报｜2026-10-28

本轮按要求收尾，不再新增实验。**车辆局部结构和转向已改善；目标专属unknown仍未同时做到“RGB无车影、正确背景不破坏”。** 以下Top 3是继续研究的候选排序，全部尚未验收通过。保留原V1无编辑主通路、旧模型、坏例和对照。

![组件图](wrapup_architecture.svg)

## Top 3：按残影与背景保护的综合观察排序

'''
    md+=mdtable(['顺位','候选','实验ID','改了什么','当前判断'],top)
    md+='排序参考raw98/113/114同帧目视及背景影响，**不能理解为残影已去净或第1项所有维度最好**。原位暗残影三项均在；第1项减少域外改动，第2与第3的差异较小。按时刻固定硬剥离与逐帧75%控制仍暴露亮色环境伪影，保留作失败对照。\n\n'
    md+=mdtable(['候选／对照','16帧保护区改变像素/帧','raw113保护区改变像素','新增背景高斯'],[[s['name'],f"{s['protection_changed_mean']:.0f}",s['raw113_protection_changed'],s['background_seeds']] for s in [lookup[n] for n in rank+['conditional_split','split199']]])
    md+='改变像素：原车身外扩8px之外，三通道平均绝对变化>1/255；仅是改动范围，**不是背景正确率**。33帧优化、16帧校准检查；基础重建和199帧记忆已见这些数据，视频教师也见其31帧，不能称独立泛化留出。raw113属于训练帧。\n\n'
    md+='## 算法、证据与调参分列\n\n'+mdtable(['改动','性质','实际实现／参数','可迁移性与限制'],changes)
    md+='''## 本轮最关键的新诊断

原代码的static批次允许逐高斯SHM运动和时间opacity门控，并不等于世界静止背景。27,642个目标相关候选中，25,413个在时间门控有效的采样时刻位置跨度>10cm；此前7,491个“背景深度支持”行中，6,569个也超过10cm。这里的有效仅指时间门控>.05，不等于相机中实际可见。页面中world_static描述已更正；旧试验标签不追溯篡改，蓝色改称“背景深度支持（未核验静止）”。

候选全视图贡献约12.85%落在目标车身域；其余不能自动当正确背景，但整行删除会牵连大量域外成像。原环境存在带符号和超范围颜色，删除opacity后通过(1-alpha)×env暴露；raw113整行剥离上界图中约80.7%的超亮饱和像素由环境项主导。因此白洞不能全部归因于道路几何缺失。

这些发现要求下一轮把“分支名”“世界运动”“几何归属”“车辆外观污染”分开。当前保留率只是优化旋钮，不是学成的实例归属概率，也没有把unknown分类问题解决。

## 暂存实验与验证清单（本轮不执行）

'''+mdtable(['优先级','待做项','验收目的'],todo)
    md+='''## 执行、复现与收尾

本轮unknown实验为5个独立10K分支，共50,000正式更新；pilot额外3,000，smoke与恢复额外980，合计53,980。不是一个模型顺序训练50K。早期三组与最后两组分别保留，同起点的正式分支不从smoke续跑。最后两组均完成100更新smoke、保存/评价、完整状态恢复再10更新及下一入口，原生RGB恢复差0。此账不包含前一批车辆结构与opacity试验。

正式权重保留AutoDL，15个正式checkpoint已计算SHA；两批交付包均验证归档及逐文件SHA，没有下载模型。第一场景HTML已接入新增按时刻联合/固定剥离与75%/95%诊断，左右RGB与3D使用同一时刻的实际参数。19类语义编码保留：0 unknown；1 road，2 sidewalk，3 building，4 wall，5 fence，6 pole，7 traffic light，8 traffic sign，9 vegetation，10 terrain，11 sky，12 person，13 rider，14 car，15 truck，16 bus，17 train，18 motorcycle，19 bicycle。V2证据颜色独立：橙=目标相关未分清，紫=混合，蓝=背景深度支持未核验静止，灰=此目标之外，青=新增背景补层（语义unknown）。

新相机±0.5m图只作压力测试；无GT及匹配原模型对照，残影和周围车辆重边仍可见，不支持新视角泛化结论。页面经真实Canvas运行与本地路径/图像校验；浏览器file协议访问受工具限制，未声称完成浏览器交互验收。

已暂停新实验入口，第二未追加、第三继续暂缓，旧队列和心跳未恢复。Git提交、取回验证和最终资源状态另见收尾回执。failure_ledger_refs=[V82-F01]；failure_ledger_delta=none。
'''
    (R/'BRIEF.md').write_text(md,encoding='utf8');(W/'V43_WRAPUP.md').write_text(md,encoding='utf8')
    def table(head,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    base='../../../GS_Scene_Diagnostics_20261027/'+SCENE+'/index.html#workspace?group=v43_unknown_v2_'
    nav=' '.join(f'<a href="{base}{n}&amp;raw=113">Top {i+1} · RGB＋3D</a>' for i,n in enumerate(rank))
    body='<h1>V43 阶段简报</h1><p>2026-10-28 · 用户要求收尾 · 新实验已暂停</p><div class="callout"><b>结构和转向改善；干净剥离尚未成功。</b> 三候选均有暗残影，背景保护仍待验证。Top 3是下一轮研究优先级。</div><p>'+nav+'</p><img src="wrapup_architecture.svg" alt="架构组件与数据流">'
    body+='<h2>Top 3 候选</h2>'+table(['顺位','候选','实验ID','算法变化','当前判断'],top)
    body+='<p>同一验收目标：减少车辆残影并保护背景。raw98/113/114目视与保护区改动共同参考；不是成功方法排名。下表数值仅衡量改动范围，不能当背景正确率。所有新增归属训练仅第一场景，尚无独立泛化证据。</p>'+table(['实验','16帧保护区变化/帧','raw113变化','背景高斯'],[[s['name'],f"{s['protection_changed_mean']:.0f}",s['raw113_protection_changed'],s['background_seeds']] for s in [lookup[n] for n in rank+['conditional_split','split199']]])
    body+='<h2>同帧原生RGB</h2><p>上排：原删动态基线、直接联合、插值联合；下排：按时刻联合＋教师、按时刻固定剥离＋教师、逐帧75%剥离。三张图均保留失败现象。</p>'
    for raw in [98,113,114]:body+=f'<details {"open" if raw==113 else ""}><summary>raw{raw} 六组对照</summary><img src="wrap_compare_raw{raw}.png" alt="raw{raw}原生渲染对照"></details>'
    body+='<h2>哪些是算法，哪些只是数据或调参</h2>'+table(['改动','性质','实现／参数','可迁移性与限制'],changes)
    body+='<h2>关键纠正：static批次不等于世界静止</h2><p>27,642个目标相关候选中，25,413个时间门控有效时的位置跨度超过10cm。旧7,491个背景深度支持行中，6,569个也超过10cm。时间门控有效不等于相机实际可见。页面已更正描述；不能再把多帧深度相符直接当可靠静止背景。</p><p>目标相关行只有约12.85%的全视图贡献落在车身域；整行删除牵连域外。环境回填还存在超范围颜色，raw113硬删除上界图约80.7%的饱和像素由环境项主导。这两项是后续算法设计依据，尚不是修复成果。</p>'
    body+='<h2>暂存实验／验证清单</h2>'+table(['优先级','待做项','目的'],todo)
    body+='<h2>已保存与限制</h2><p>本轮unknown：5×10K正式＋3K pilot＋980 smoke/恢复＝53,980更新，分别保留；第二未追加、第三暂缓。15个正式checkpoint在远端留存并校验SHA，两个本地交付包逐文件核验。199帧是背景证据，原生查询49帧；33/16划分只是校准检查，非独立泛化测试。</p><p>按时刻联合最终图是原生高斯重渲染；其训练用了低权重ProPainter生成教师。门控与教师贡献尚未拆开。±0.5m新相机仅压力图；保留率不是身份概率。原V1默认直出保留，密度×2与环境clamp未设默认。</p><p><a href="BRIEF.md">完整简报与语义map</a> · <a href="wrapup_summary.json">统计</a> · <a href="static_motion_audit/summary.json">运动审计</a> · <a href="viewer_validation.json">Canvas验证</a> · <a href="remote_closeout.json">远端保存回执</a> · <a href="closeout_final.json">最终收尾回执</a></p>'
    css='body{max-width:1420px;margin:35px auto;padding:0 24px;font:16px/1.65 "Microsoft YaHei",sans-serif;background:#f4f7fb;color:#213d50}h1,h2{color:#163e53}h2{margin-top:32px}a{color:#1d63a5;margin-right:18px}img{width:100%;height:auto;margin:10px 0}.callout{background:#fff0d7;border-left:5px solid #d79932;padding:16px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}th,td{padding:11px;border:1px solid #cbd7e2;text-align:left;vertical-align:top}th{background:#e5eef6}summary{cursor:pointer;font-weight:bold}details{margin:15px 0}'
    (R/'brief.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V43阶段简报</title><style>'+css+'</style><body>'+body+'</body></html>',encoding='utf8')
    # Preserve the earlier stage report, but qualify its scope and corrected branch definition.
    p=R/'REPORT.md';old=p.read_text(encoding='utf8');old=re.sub(r'^> 最新收尾.*?\n\n','',old,flags=re.S) if old.startswith('> 最新收尾') else old
    lead='> 最新收尾：[Top 3与延期实验简报](BRIEF.md)。下文为前三组阶段记录；后来两组、最新总账和运动审计以简报为准。原static几何参数冻结不代表世界静止：原分支自带SHM运动；“背景支持”仅指背景深度支持，尚未核验静止。\n\n'
    p.write_text(lead+old,encoding='utf8');(W/'TARGET_UNKNOWN_V2.md').write_text(lead+old,encoding='utf8')
    for p,href in [(S/'index.html','../../V43_Ownership_20261027/gpu_v43/target_unknown_v2/brief.html'),(B/'outputs/GS_Scene_Diagnostics_20261027/index.html','../V43_Ownership_20261027/gpu_v43/target_unknown_v2/brief.html'),(B/'outputs/V43_Ownership_20261027/index.html','gpu_v43/target_unknown_v2/brief.html')]:
        s=p.read_text(encoding='utf8');s=re.sub(r'<!-- v43-wrapup-start -->.*?<!-- v43-wrapup-end -->','',s,flags=re.S);panel=f'<!-- v43-wrapup-start --><section class="panel notice" id="v43Wrapup"><h2>V43 阶段收尾：Top 3 候选与后续验证清单</h2><p><a href="{href}">打开简报：算法／调参／可迁移性／同帧RGB →</a></p><p>新实验已暂停；三候选仍有残影。重要纠正：原static批次含时变运动，深度支持不等于可靠静止背景。</p></section><!-- v43-wrapup-end -->';s=re.sub(r'(<main\b[^>]*>)',lambda m:m.group(1)+panel,s,count=1);p.write_text(s,encoding='utf8')
    print(json.dumps(dict(brief=str(R/'brief.html'),top3=rank,experiments_held=True)))

if __name__=='__main__':main()
