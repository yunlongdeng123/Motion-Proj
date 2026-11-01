"""机制证据单独进入报告；轻量交付保留实际核验样本，不改共享平台为报告入口。"""
from pathlib import Path
import json,shutil,tarfile,hashlib,html
from PIL import Image
R=Path(__file__).resolve().parent;D=R/'delivery'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    target=D/'mechanisms';shutil.copytree(R/'mechanisms_real',target)
    sim=target/'simulation';sim.mkdir()
    for p in (R/'sim_factor_controls').rglob('*.png'):
        q=sim/p.relative_to(R/'sim_factor_controls');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for n in ['summary.json','pre_control_contract.json']:shutil.copy2(R/'sim_factor_controls'/n,sim/n)
    for case in read(target/'summary.json')['cases']:
        key=case['case'];source=R/'mechanisms_gpu'/key;dest=target/key/'gpu';dest.mkdir()
        shutil.copy2(source/'technical_and_component_checks.json',dest/'technical_and_component_checks.json')
        for p in source.glob('*/*.png'):
            if p.parent.name=='observed':continue
            q=dest/p.relative_to(source);q.parent.mkdir(exist_ok=True);shutil.copy2(p,q)
    ledger=read(target/'mechanism_ledger.json');summary=read(target/'summary.json');simulation=read(sim/'summary.json')
    labels={'partially_checked':'已做局部检查，因果隔离未完成',
        'projective_conflict_checked_physical_cause_unresolved':'跨域贡献已测，多物理表面未证',
        'not_causally_tested':'因果检查未完成',
        'components_numerically_separated_semantics_unresolved':'分量已拆开，语义来源未定',
        'view_support_checked_surface_completeness_unresolved':'视向支持已测，新面完整性未证',
        'technical_execution_checked_binding_correctness_in_scope':'运动执行已核对，身份正确性仍在本轮范围'}
    stats=''.join('<tr>'+''.join('<td>'+str(x[k])+'</td>' for k in ['case','measured_depth_pixels','depth_disagreement_pixels','appearance_confounded_pixels','sparse_historical_ground_witness_pixels','dual_body_outside_rows'])+'</tr>' for x in summary['cases'])
    sections=[]
    for case in summary['cases']:
        key=case['case'];rows=[r for r in ledger['rows'] if r['case']==key]
        table=''.join('<tr><td>'+str(r['category'])+'</td><td>'+labels[r['check_state']]+'</td><td>'+r['checks_actually_executed']+'</td><td>'+r['excluded_from_ownership_judgement']+'</td></tr>' for r in rows)
        evidence=read(target/key/'evidence.json');frame=evidence['delete_components'][0]['frame'];routes=[r['route'] for r in evidence['delete_components'] if r['frame']==frame]
        route=routes[-1]
        images=''.join('<figure><img src="'+key+'/'+frame+'_'+route+'_delete_'+k+'.png" alt="'+label+'"><figcaption>'+label+'</figcaption></figure>' for k,label in [('final','删除图'),('C_G','三维点贡献 C_G'),('fill','环境填充 fill'),('E','环境辐射 E')])
        sections.append('<section><h2>'+key.replace('waymo-train-','Waymo · scene-').replace('_',' / ')+'</h2><table><tr><th>类</th><th>当前状态</th><th>实际检查</th><th>禁止归属判断的范围与边界</th></tr>'+table+'</table><p><a href="'+key+'/evidence.json">逐帧观测与分量数值</a> · <a href="'+key+'/gpu/technical_and_component_checks.json">实际行变换与原生分量检查</a> · 每帧区域掩码位于 '+key+'/regions/。</p><div class="grid">'+images+'</div><p>上述为首锚、固定全绑定控制的原生分量展示；幅度非车形语义标签。C_G/fill为预乘贡献，不能与独立材质颜色混淆。</p></section>')
    source='<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="220"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#466780"/></marker></defs><g fill="#eef4f8" stroke="#7696ae"><rect x="10" y="60" width="220" height="100" rx="7"/><rect x="285" y="60" width="250" height="100" rx="7"/><rect x="590" y="60" width="230" height="100" rx="7"/><rect x="875" y="60" width="215" height="100" rx="7"/></g><g font-family="Microsoft YaHei,sans-serif" text-anchor="middle" font-size="19" fill="#203c50"><text x="120" y="96">真实观测与冻结表示</text><text x="120" y="130">独立单因素控制</text><text x="410" y="96">深度、颜色、历史射线</text><text x="410" y="130">点 / 环境 / 运动分量</text><text x="705" y="96">40项逐例证据</text><text x="705" y="130">已核对 / 部分 / 未测</text><text x="982" y="96">可评分区域门禁</text><text x="982" y="130">本轮训练标签为0</text></g><g stroke="#466780" stroke-width="2" marker-end="url(#a)"><path d="M232 110H280"/><path d="M537 110H585"/><path d="M822 110H870"/></g></svg>'
    (target/'architecture.svg').write_text(source,encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五例机制核对与评分禁用域</title><style>body{font:16px/1.75 system-ui,sans-serif;max-width:1250px;margin:30px auto;padding:0 20px;color:#253d50}h1{font-size:28px}h2{font-size:23px;margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #d5e0e7;text-align:left;vertical-align:top}img{max-width:100%}a{color:#185c91}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.grid figure{margin:0}.grid img{width:100%}.note{padding:16px;background:#fff7e7;border-left:4px solid #d1a15c}@media(max-width:700px){.grid{grid-template-columns:1fr 1fr}}</style><a href="../report.html">← 五例实验说明</a><h1>机制核对：已执行的检查、未决项与评分禁用域</h1><p class="note">5例×8类，共40条记录，不等于40项隔离完成。真实数据的因果根源仍有未决项。当前所有归属训练标签区域均为禁用，未生成偏好好坏答案；原始任务和失败分母保留。</p><img src="architecture.svg" alt="机制检查组件与评分门禁"><h2>实际观测量</h2><p>像素数为49帧累计观测，不是独立样本数。平均深度不是首交面；0.30米与10/255仅作描述分组，不是资格或归属正确性阈值。跨域行计数只证明同一行有车身及其外域贡献，不证明兼顾两物理表面。</p><table><tr><th>车辆</th><th>目标实测深度像素</th><th>平均深度不一致代理</th><th>深度近似一致但RGB有误差</th><th>稀疏历史地面见证</th><th>跨域贡献行</th></tr>'''+stats+'''</table><p>历史见证实际执行了跨时地面实测点、重复观测及注册车辆盒的射线遮挡检查，仍未核实稠密路面、非车辆遮挡、光照与辐射稳定性。无见证只表示在本次49帧证据中尚未核实，不能说整个世界从未观察。</p><h2>受控仿真核验的范围</h2><p>在已有单个程序化世界的真实原生GPU上，新增28份控制输出：原绑定基线，以及纯绑定错误、位置偏移、SH常量颜色、仅环境辐射、时间接错、前向区域点群移除六组副本。原检查点不改。颜色控制保持alpha/深度不变；环境控制保持C_G/alpha/深度不变；纯绑定错误的none与基线一致。9项约束不变性检查通过。</p><p class="note">这是部分E1控制，不是全因素评价规则的完成验收，也不是自动根因分类器的准确率。形状/尺度的全部变化、独立多物理表面高斯、物理阴影/反射和完整未见背景控制仍未完成。点群移除只检验表示容量混淆，不证明完整真实车头。未使用独立编辑GT评分，不能冒称完整奖励误判率验证。</p><p><a href="simulation/summary.json">实际单因素字段差异与门禁记录</a> · <a href="simulation/pre_control_contract.json">计算前控制合同</a></p><h2>逐例、逐类记录</h2>'''+''.join(sections)+'''<h2>生成好坏答案之前的硬边界</h2><p>当前ownership_training_label_allowed逐帧均为全零。观测深度、原图一致性与保持域变化可以继续作为证据，但不升级为独立删除/移动/旋转GT。后续只有证据和验证足够的有效域才可生成配对答案，未知与干扰区保留在总体分母及未完成任务中。技术接线错误与“点跟错车”明确分开；后者仍是V43本轮需要承担的核心问题。</p><p><a href="mechanism_ledger.json">40项机器可读记录与标签门禁</a> · <a href="summary.json">实测汇总</a></p></html>'''
    (target/'index.html').write_text(page,encoding='utf-8')
    main_report=(D/'report.html').read_text(encoding='utf-8')
    main_report=main_report.replace('<h2>R4阶段边界</h2>','<h2>八类机制与训练标签门禁</h2><p>本轮又实际补充了原始深度/RGB、稀疏历史地面射线、点与环境分量，以及样本行的跨时跨操作变换核对。5例×8类40项记录逐条注明已执行、部分检查或尚未因果验证；没有把unknown当完成。独立E1新增28份原生单因素控制和9项不变性核验，但多表面、阴影反射与完整未知背景等验证仍缺。所有归属训练标签区域保持禁用，未生成好坏答案。</p><p><a href="mechanisms/index.html">逐例机制证据、实际核查范围与区域禁用规则</a></p><h2>R4阶段边界</h2>')
    main_report=main_report.replace('两域结果分开比较','各候选域结果分开比较')
    (D/'report.html').write_text(main_report,encoding='utf-8')
    s=read(D/'summary.json');s.update(mechanism_records=40,mechanism_isolation_complete=False,mechanism_control_native_outputs=28,
        mechanism_control_invariance_checks_passed=simulation['all_invariance_checks_passed'],training_preference_pairs_generated=0,
        ownership_training_label_pixels=0,failure_ledger_refs=['V82-F09','V82-F10','V82-F11','V82-F12'])
    write(D/'summary.json',s)
    for name in ['mechanism_gpu.py','mechanisms_real.py','sim_factor_controls.py','controller_mechanisms.py','enrich_mechanism_delivery.py']:
        shutil.copy2(R/name,D/'source'/name)
    # 轻量包：视频与JPEG浏览副本，保留中锚none/delete两端点FP32。全部原PNG/FP32仍在远端。
    light=R/'delivery_portable';light.mkdir(exist_ok=False)
    for p in D.rglob('*'):
        if not p.is_file():continue
        relative=p.relative_to(D)
        if relative.parts[0] in ['video_frames','fp32_controls']:continue
        if relative.parts[:2]==('review','assets') and p.suffix=='.png':continue
        q=light/relative;q.parent.mkdir(parents=True,exist_ok=True);os_link(p,q)
    text=(D/'review/data.js').read_text(encoding='utf-8');data=json.loads(text[len('window.FIVE_DATA='):-1])
    for scene in data:
        for domain in scene['domains'].values():
            for item in domain['images']:
                p=D/'review'/item['path'];new=Path(item['path']).with_suffix('.jpg');q=light/'review'/new;q.parent.mkdir(parents=True,exist_ok=True)
                Image.open(p).convert('RGB').save(q,quality=95,subsampling=0);item['path']=new.as_posix()
        middle=scene['frames'][24]['frame_id']
        for p in (D/'fp32_controls'/scene['case']).glob('*/**/*.npz'):
            if p.stem!=middle or p.parent.name not in ['none','delete']:continue
            q=light/p.relative_to(D);q.parent.mkdir(parents=True,exist_ok=True);os_link(p,q)
    # data.js已硬链接，必须以新文件替换，不能写穿完整原始交付。
    tmp=light/'review/data_portable.tmp';tmp.write_text('window.FIVE_DATA='+json.dumps(data,ensure_ascii=False)+';',encoding='utf-8');tmp.replace(light/'review/data.js')
    write(light/'portable_scope.json',dict(browser_frames='JPEG quality95, no chroma subsampling; lossy browsing only',
        videos='native-source H264 browsing copies, with optional annotation-position boxes',
        FP32='fixed middle anchor; none/delete; original/all-bound full-source endpoints,20 files',
        complete_remote_native_outputs=str(R),metrics_from_original_FP32=True,not_all_native_arrays_retrieved=True))
    with tarfile.open(R/'delivery_portable.tar.gz','w:gz') as t:t.add(light,arcname='delivery')
    p=R/'delivery_portable.tar.gz';h=hashlib.sha256(p.read_bytes()).hexdigest()
    write(R/'portable_archive_receipt.json',dict(path=str(p),bytes=p.stat().st_size,sha256=h))
    print(json.dumps(dict(status='COMPLETE',bytes=p.stat().st_size,sha256=h)))
def os_link(source,target):
    import os
    os.link(source,target)
if __name__=='__main__':main()
