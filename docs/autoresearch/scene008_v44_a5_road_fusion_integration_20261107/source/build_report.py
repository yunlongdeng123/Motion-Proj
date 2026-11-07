"""Build only from actual retrieved native outputs and independent CPU evidence."""
import csv
import hashlib
import html
import json
from datetime import datetime,timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote,unquote,urlsplit
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def h(value):return html.escape(str(value),quote=True)
def number(value):return '无支持' if value is None else f'{value:.7f}'


class Builder:
    def __init__(self):self.inputs=set()
    def read(self,path):
        p=ROOT/path
        if not p.is_file():return None
        self.inputs.add(p)
        return json.loads(p.read_text(encoding='utf-8-sig'),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
    def link(self,path,label):
        return '<a href="'+quote(path,safe='/._-')+'">'+h(label)+'</a>' if (ROOT/path).is_file() else '<span class="pending">'+h(label)+'（待取回）</span>'
    def image(self,path,label):
        return '<a href="'+path+'"><img src="'+path+'" alt="'+h(label)+'"></a>' if (ROOT/path).is_file() else '<p class="pending">'+h(label)+'：实际图像待取回。</p>'
    def metrics(self,review,label):
        if not review:return '<p class="pending">'+h(label)+'尚无独立CPU实际结果。</p>'
        gates=review['numeric_gates'];passed=sum(g['passed'] is True for g in gates);failed=[g for g in gates if g['passed'] is False];unsupported=sum(g['passed'] is None for g in gates)
        seam_rows=''.join('<tr><td>'+h(r['variant'])+'</td><td>'+h(r['state'].upper())+'</td><td>'+h(r['edge'])+'</td><td>'+str(r['bins'])+'</td><td>'+number(r['baseline'])+'</td><td>'+number(r['candidate'])+'</td><td class="'+('worse' if r['delta'] and r['delta']>1e-6 else '')+'">'+number(r['delta'])+'</td></tr>' for r in review['pooled_world_seams'])
        result='<p>'+h(label)+'：'+str(passed)+'项通过 / '+str(len(failed))+'项失败 / '+str(unsupported)+'项空域无支持。unsupported不计作通过。</p><table><thead><tr><th>候选</th><th>状态</th><th>固定世界边界</th><th>配对bin</th><th>latest</th><th>候选</th><th>Δ残差跳变</th></tr></thead><tbody>'+seam_rows+'</tbody></table>'
        records=review['records'];by={(r['raw'],r['state'],r['variant']):r for r in records}
        primary=[]
        for state in ['keep','delete']:
            baseline=by.get((128,state,'baseline'))
            if not baseline:continue
            bs=baseline['primary128_seams']['near_x']['mean'];bm=baseline['primary128_RGB']['mean']
            for variant in review['variants_evaluated']:
                if variant=='baseline':continue
                row=by[(128,state,variant)];cs=row['primary128_seams']['near_x']['mean'];cm=row['primary128_RGB']['mean']
                primary.append('<tr><td>'+variant+'</td><td>'+state.upper()+'</td><td>'+number(bm)+' → '+number(cm)+'</td><td>'+number(bs)+' → '+number(cs)+'</td><td>'+f'{(cs-bs)/bs:+.2%}'+'</td></tr>')
        result+='<h3>原raw128蓝框独立保留</h3><table><thead><tr><th>候选</th><th>状态</th><th>原1385道路像素 RGB MAE</th><th>原6bin near残差jump</th><th>near相对变化</th></tr></thead><tbody>'+''.join(primary)+'</tbody></table><p>主蓝框两侧35/33像素，与worldpatch128两侧36/33分别计算，不拼接。RGB与残差jump以原观测的0–1编码为单位；评分保留原生float，不另加上界截断，PNG仅作显示。</p>'
        worst=[]
        for name in ['a5_RGB','a5_upper_RGB','a6_RGB','a6_depth_m','a37_RGB','protected_nonroad_RGB','original_road_RGB','nonroad_guard:alpha_pixel_max_abs','nonroad_guard:depth_pixel_max_abs_m']:
            for variant in review['variants_evaluated']:
                if variant=='baseline':continue
                selected=[g for g in gates if g['name']==variant+':'+name and g['value'] is not None]
                if selected:
                    g=max(selected,key=lambda x:x['value']);worst.append('<tr><td>'+variant+'</td><td>'+h(name)+'</td><td>'+str(g.get('raw',''))+'/'+h(g.get('state','').upper())+'</td><td>'+number(g['value'])+'</td><td>'+number(g['threshold'])+'</td><td>'+('PASS' if g['passed'] else 'FAIL')+'</td></tr>')
        result+='<h3>每帧原保护门：最坏值也保留</h3><table><thead><tr><th>候选</th><th>原门</th><th>raw/状态</th><th>最坏变化</th><th>原上限</th><th>结果</th></tr></thead><tbody>'+''.join(worst)+'</tbody></table><p>RGB列是对原GT的MAE变化；a6_depth_m为测深MAE变化（m）。nonroad_guard两列是候选对latest的逐像素最大差（alpha或m），不是对GT的精度，也不据MAE通过宣称像素完全一致。</p>'
        result+='<details><summary>全部失败门（不删帧、不改阈值）</summary><table><thead><tr><th>名称</th><th>raw</th><th>状态</th><th>值</th><th>上限</th></tr></thead><tbody>'+''.join('<tr><td>'+h(g['name'])+'</td><td>'+str(g.get('raw',''))+'</td><td>'+h(g.get('state',''))+'</td><td>'+number(g['value'])+'</td><td>'+number(g['threshold'])+'</td></tr>' for g in failed)+'</tbody></table></details>'
        return result
    def wheel(self,review):
        if not review:return ''
        rows=[]
        for row in review['records']+review['diagnostic_records']:
            if row['state']!='delete' or 'fixed_wheel_rays_vs_latest' not in row:continue
            value=row['fixed_wheel_rays_vs_latest'];increment=row.get('fixed_wheel_rays_vs_AA')
            rows.append('<tr><td>'+str(row['raw'])+'</td><td>'+row['variant']+'</td><td>'+str(value['ray_count'])+'</td><td>'+number(value['max_assetW_abs_delta'])+'</td><td>'+str(value['chart_first_new_drop_gt001'])+'</td><td>'+number(increment['max_assetW_abs_delta'] if increment else None)+'</td></tr>')
        return '<table><thead><tr><th>raw</th><th>候选</th><th>固定ray</th><th>vs latest最大|WΔ|</th><th>chart-first新增下降&gt;.01</th><th>C vs AA最大|WΔ|</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    def build(self):
        config=self.read('evidence/report_evidence_index.json') or {}
        contract=self.read('contract/independent_evaluation_contract_v1.json')
        source=self.read('evidence/source_road_review.json')
        initial_path=config.get('initial_metrics','evidence/review/cpu_independent_pilot4_with_wheel_historical_v2.json')
        initial=self.read(initial_path)
        d_path=config.get('D_metrics','evidence/review/cpu_independent_D4_fixed_support_v2.json')
        d=self.read(d_path)
        method=self.read('contract/fixed_build_support_amendment_v2.json')
        d_scope=self.read('evidence/D4_reporting_scope.json')
        e_method=self.read('contract/existing_surface_appearance_feasibility_v1.json')
        e_fit_method=self.read('contract/existing_surface_fusion_bake_pilot_v1.json')
        e_qualification=self.read('evidence/review/independent_E_qualification_cpu.json')
        e_path=config.get('E_metrics','evidence/review/cpu_independent_E4_existing_surface.json')
        e=self.read(e_path)
        f_method=self.read('contract/bounded_appearance_constraint_feasibility_v1.json')
        f_fit_method=self.read('contract/bounded_surface_fusion_bake_pilot_v1.json')
        f_path=config.get('F_metrics','evidence/review/cpu_independent_F4_bounded_surface.json');f=self.read(f_path)
        leakage_path='evidence/review/identity_supervision_audit_F400_v1.json';leakage=self.read(leakage_path)
        fit_context=self.read('evidence/review/F_source_fit_context.json')
        g_method=self.read('contract/joint_seam_edit_safe_feasibility_v1.json')
        geometry_scope=self.read('evidence/report_scope_geometry_takeover.json')
        silver_path='evidence/review/silver_support_group_audit_v1.json';silver=self.read(silver_path)
        g_receipt=self.read('evidence/G4_joint_seam_edit_operator/receipt.json')
        g_ledger=self.read('evidence/G4_joint_seam_edit_operator/execution_ledger_appendix.json')
        ledger_path=config.get('ledger','evidence/road_execution_ledger_v1.json');ledger=self.read(ledger_path)
        root_visual_path=config.get('root_visual','evidence/root_pilot_visual_review.json');root_visual=self.read(root_visual_path)
        root_quick_path=config.get('root_quick','evidence/review/root_quick_pilot.json');root_quick=self.read(root_quick_path)
        if not initial:raise FileNotFoundError(initial_path)
        assert initial['contract_sha256']==sha(ROOT/'contract/independent_evaluation_contract_v1.json')
        state='INITIAL_AA_C_FAILED_POSTHOC_D_PENDING_NO_ALL49'
        lead='AA/C直接接入未守住预封无回退门，未放行49，也未更新中央viewer。C局部色差减轻，但原128近端接缝更差，a6与道路外保护域出现回退。'
        if d:
            failed=len(d['failed_gates']);extras=d.get('additional_root_wheel71_release_gates',[])
            if failed or any(g['passed'] is False for g in extras):
                state='REGISTERED_PILOT_FAILURES_RETAINED_NO_ALL49_RELEASE';lead+=' 后验D也有实际失败，按同门保留原结果。'
            else:
                state='POSTHOC_D_SUPPORTED_NUMERIC_GATES_PASSED_ROOT_REVIEW_SEPARATE';lead+=' D的有支持数值门已通过；结构图检、完整71ray和明确扩49授权另列，不自动发布。'
        else:lead+=' D是另立合同的固定新增行控制，目前等待实际资格/成像，不能预判成功。'
        if e_fit_method and not e:
            state='E_APPEARANCE_FIT_PENDING_ACTUAL_RESULTS_PREVIOUS_FAILURES_RETAINED'
            lead+=' E原有表面材质回写已登记固定预算，真实生命周期与正式结果尚待取回，不预判成功。'
        elif e:
            method_fail=any(g['passed'] is False for g in e.get('E_registered_global_geometry_invariants',[]))
            if e['failed_gates'] or method_fail:
                state='E_REGISTERED_PILOT_FAILED_NO_ALL49_RELEASE';lead+=' E保持住所有其他无回退门与全图几何不变量，但原128主near改善不足5%及1e-4，四项改善门失败，不放行49。' if not method_fail and all('primary_near_' in g['name'] for g in e['failed_gates']) else ' E原门或几何不变量仍有失败，不放行49。'
            else:
                state='E_SUPPORTED_NUMERIC_GATES_PASSED_AWAIT_ROOT_VISUAL_RELEASE';lead+=' E有支持数值门通过，仍需实际原图审阅与明确49授权。'
        if f_method:
            state='F_BOUNDED_APPEARANCE_FEASIBILITY_PENDING_PREVIOUS_FAILURES_RETAINED'
            lead+=' 现另立F仅做分域有界颜色自由度的资格/容量检查；原门不放宽，没有新的拟合结果。'
        if f_fit_method:
            state='F_BOUNDED_APPEARANCE_FIT_PENDING_ACTUAL_RESULTS_PREVIOUS_FAILURES_RETAINED'
            lead+=' F资格后已登记一次fresh400有界外观拟合，真实生命周期和正式效果待取回；不把operator容量写成已实现收益。'
        if f:
            state='F_REGISTERED_PILOT_FAILED_AND_DELETE_ARTIFACT_REVIEW_NO_ALL49'
            lead='AA/C、D、E的原始失败全部保留。F实际fresh400完成，守住原车体/道路外均值门与几何不变量，但横向接缝回退、原128 near仅改善4.45%/3.15%，未达到预封5%门；DELETE还出现新增暗横条，不能据原道路均值通过称删除效果保持。未放行49，中央viewer保持最新基线。'
        if g_method:
            state='G_JOINT_SEAM_DELETE_SAFE_FEASIBILITY_PENDING_NO_ALL49'
            lead+=' 现只登记G联合接缝可达性与删除足迹保护诊断，未有新拟合或修缝结果；原4560阈值不变。'
        if geometry_scope:
            state='FULL_ROAD_SUPPORT_OCCLUSION_INTEGRATION_PENDING_NO_ALL49'
            lead='目前尚未完整采用原C道路算法。C完整支撑能盖住静态银色补片，却同时遮挡真实C2轮胎；颜色回写E/F没有解决该几何问题，F还新增DELETE暗条，均不采纳。当前主线转为完整道路支撑接入与遮挡控制，去银片、保车轮分别验；新接入尚无通过结果，49未放行，最新v44＋a5默认保持。'
        initial_figures=''.join('<h3>'+label+'</h3>'+self.image('figures/initial_pilot4_'+s+'_four_frames.png','最新基线/AA/C正式四帧'+s) for s,label in [('delete','仅DELETE a37'),('keep','完整KEEP')])
        evidence=[('contract/independent_evaluation_contract_v1.json','候选前冻结的4560数值合同'),(initial_path,'独立正式四帧＋130 CPU审查'),
          (root_quick_path,'root独立spot复核（不替代正式评价）'),('evidence/pilot4_images_retrieval_verification.json','实际三variant36PNG逐SHA取回'),
          ('evidence/review/road_adapter_cpu_audit.json','道路资产/ABI CPU核对'),('evidence/review/material_boundary_cpu_audit.json','固定材质边界source配对'),
          ('evidence/source_road_review.json','原道路交付162JSON/1137PNG读取核验'),('source/evaluate_saved_road_integration.py','独立CPU评价源码'),
          ('source/fixed_domains.py','纯NumPy固定mask定义（无环境副作用）'),('source/make_pilot_boards.py','实际PNG图版构建器'),
          ('source/build_report.py','报告构建器'),('comparison.csv','原域逐帧RGB/深度CSV'),('report_data.json','报告数据及输入manifest'),
          (ledger_path,'实际计算账本'),(root_visual_path,'root实际原图审阅记录')]
        evidence.extend([('contract/existing_surface_fusion_bake_pilot_v1.json','E固定400paired预算/完整生命周期合同'),
            ('evidence/review/independent_E_qualification_cpu.json','E真实资格/可写覆盖独立CPU复算'),
            ('evidence/review/baseline_appearance_nonnegative_cpu_audit.json','E非负/HDR接口CPU fixture（非GPU验收）'),
            ('model/road_appearance_package.tar.gz','道路外观重现包：资格/δRGB/绑定/源码，不含原Gaussweights'),
            ('evidence/review/independent_E_formal_cpu.json','E实际100/200/400 checkpoint/导出/不变量独立CPU审计')])
        evidence.append(('contract/bounded_appearance_constraint_feasibility_v1.json','F后验有界颜色资格合同（尚非拟合结果）'))
        evidence.append(('contract/bounded_surface_fusion_bake_pilot_v1.json','F独立fresh400有界外观拟合合同'))
        evidence.extend([(f_path,'F独立CPU正式四帧评价（含全部失败门）'),
            ('evidence/review/independent_F_formal_cpu.json','F实际预算/保存/几何不变量独立CPU复算'),
            ('evidence/F_formal400_r1/returned.json','F实际400材质更新与808F/800B收据'),
            ('model/F_inference_package_v1.tar.gz','F失败候选重现元数据包：材质/矩阵/API，另依赖原模型'),
            ('evidence/review/F_inference_package_local_verification.json','F元数据包本机逐文件核验'),
            (leakage_path,'F监督域与暗条逐像素CPU审计'),
            ('evidence/review/F_source_fit_context.json','原拟合mask与阴影上下文图'),
            ('source/audit_supervision_identity_leakage.py','只读暗化top20/监督域审计源码'),
            ('source/audit_F_fit_context.py','原fit边界上下文图源码'),
            ('contract/joint_seam_edit_safe_feasibility_v1.json','G后验联合可达性/追加删除足迹保护合同'),
            ('source/registered_seam_pairs.py','纯NumPy原53接缝pair接口（无CUDA环境写入）'),
            ('evidence/review/registered_seam_pairs_cpu_receipt.json','原53pair每bin像素数/mask SHA独立核对'),
            ('evidence/road_execution_through_F400_v1_retrieval_verification.json','through F完整状态/源码/收据逐SHA取回')])
        evidence.extend([('evidence/report_scope_geometry_takeover.json','用户纠正后的完整支撑/遮挡主线'),
            (silver_path,'缓存128/137银片与C2逐组alpha_T审计'),
            ('source/audit_silver_support_groups.py','银片来源审计源码（不加载模型）'),
            ('evidence/review/saved_raw130_component_takeover_cpu.json','原130固定35轮胎ray组件接管核对'),
            ('evidence/G4_joint_seam_edit_operator/receipt.json','G实际22F/22B操作子读取收据（不含拟合）'),
            ('evidence/G4_joint_seam_edit_operator/signed_residual_J_supplement.json','G signed residual与已取消后续说明'),
            ('evidence/G4_joint_seam_edit_operator/execution_ledger_appendix.json','G实际追加账本')])
        d_section='<p class="pending">D实际成像与独立评价尚未完整取回；当前不填值。</p>'
        if d:
            d_section=self.metrics(d,'D后验固定行控制')+'<p>'+self.link(d_path,'D独立原始评价')+'</p>'+self.wheel(d)
            if d.get('additional_root_wheel71_release_gates'):
                d_section+='<pre>'+h(json.dumps(d['additional_root_wheel71_release_gates'],ensure_ascii=False,indent=2))+'</pre>'
            d_section+=self.image('figures/D4_v2_delete_four_frames.png','latest/C/D后验四帧DELETE原图')
            if d_scope:
                d_section+='<p>D仅保留62/40,392个新增行，保护工程在这些BUILD视图有用，但没有带来接缝改善。D首关已经失败，130额外2次forward已取消；上方130空值表示未执行，不是自动待跑任务。</p>'
        e_section='<p class="pending">E实际拟合和原生评价尚待取回，未填效果值。</p>'
        if e:
            e_section=self.metrics(e,'E原有表面回写')+'<p>'+self.link(e_path,'E独立CPU完整评价')+'</p>'
            invariants=e.get('E_registered_global_geometry_invariants',[])
            e_section+='<p>方法另要求全图alpha/depth/原7groupW与baseline一致：'+str(sum(g['passed'] for g in invariants))+'/'+str(len(invariants))+'个实际检查通过。仅指这些几何读出，RGB仍按原门独立评价。</p>'
            e_section+=self.image('figures/E4_delete_four_frames.png','E原表面回写四帧DELETE')+self.image('figures/E4_raw128_delete_original_blue_box.png','E主128近缝同原框显示')
            e_section+='<details><summary>E完整KEEP四帧</summary>'+self.image('figures/E4_keep_four_frames.png','E完整KEEP四帧')+'</details>'
            ranges=[dict(raw=r['raw'],state=r['state'],variant=r['variant'],**r['native_RGB_range']) for r in e['records']]
            e_section+='<details><summary>实际原生RGB范围（未额外截断）</summary><pre>'+h(json.dumps(ranges,ensure_ascii=False,indent=2))+'</pre></details>'
        f_section='<p class="pending">F实际400结果尚未取回，不填效果值。</p>'
        if f:
            f_section=self.metrics(f,'F有界原表面外观回写')+'<p>'+self.link(f_path,'F独立CPU完整评价')+'</p>'
            invariants=f.get('F_registered_global_geometry_invariants',[])
            f_section+='<p>全图alpha/depth/原7groupW：'+str(sum(g['passed'] for g in invariants))+'/'+str(len(invariants))+'个实际检查通过。原有保护域的RGB均值界通过，不意味着每像素不变或删除后足迹无伪影。</p>'
            f_section+=self.image('figures/F4_delete_four_frames.png','latest/E/F实际四帧DELETE')+self.image('figures/F4_raw128_delete_original_blue_box.png','F原128蓝框接缝')
            f_section+='<details><summary>F实际KEEP四帧</summary>'+self.image('figures/F4_keep_four_frames.png','F实际KEEP四帧')+'</details>'
        leakage_section='<p class="pending">尚无保存数组的独立监督域审计。</p>'
        if leakage:
            leakage_section='<p>四帧实际fit_mask与actor_labels&gt;0、full37(label5)、visible37的交集全部为0。目标文件在原车体域虽然保留Observed RGB，这些像素并未进入拟合loss，不能认定直接用车尾/牌照像素监督DELETE。fit仍包含部分原观测阴影/暗路面，原观测不是删除后的隐藏道路真值。</p>'
            leakage_section+='<p>新暗条跨入原a37投影域：raw94有638个原车体像素的编码亮度变化&lt;−0.05，最大RGB移动0.2431；例如(246,282)为label5/visible37、fit=False、road_edit=True，KEEP37W≈0.999897，而DELETE37W=0。所有geometry/group alpha与语义a5贡献保持一致，变化来自本次static/oldroad颜色场。该证据表明跨域颜色耦合，不能把横条仅称为合法阴影；尚未唯一追溯到某个源阴影。</p>'
            leakage_section+='<p>原保护域排除了road_edit，KEEP可见a37又遮住其后表面；两者的均值通过不能覆盖DELETE暴露表面的颜色。这里的新增暗化阈值和裁剪是后验诊断，不更改原门。</p>'
            leakage_section+=self.image('evidence/review/identity_supervision_audit_F400_v1/raw094_delete_identity_crop.png','raw94 Observed/目标/latest DELETE/F DELETE，紫色为原full37边界')
            leakage_section+=self.image('evidence/review/identity_supervision_audit_F400_v1/raw128_source_fit_context.png','raw128原fit青边/原full37紫边与F暗条；显示裁剪不是优化ROI')
            leakage_section+='<details><summary>全部4帧top20暗化位置与源文件SHA</summary><p>'+self.link(leakage_path,'完整逐像素诊断JSON')+' · '+self.link('evidence/review/F_source_fit_context.json','原fit上下文JSON')+'</p></details>'
        silver_section='<p class="pending">银片逐组缓存审计尚待取回。</p>'
        if silver:
            silver_section='<p>raw137银片诊断矩形[207,207,274,227]的DELETE中，baseline贡献均值native_static=0.82410、oldroad=0.03286、C2=0；C插入后newroad=0.92407、static=0.04294，确有新支撑覆盖静态补片。该137矩形是实际看图后的定位，仅作组件诊断。</p><p>raw128指定矩形[210,202,302,242]同时含补片和真实C2：baseline DELETE的C2贡献均值0.04367，C后降为0.02589；C2主导像素174→94。原static均值0.37629→0.03216、newroad=0.40778。不能把整个矩形视为残片一并覆盖或退役。alpha_T归属不是RGB分解，也不是新的像素语义真值。</p>'
            silver_section+=self.image('evidence/review/silver_support_group_audit_v1/raw128_delete_silver_groups.png','raw128银片/static与真实C2分开；C去片同时侵入轮胎贡献')
            silver_section+=self.image('evidence/review/silver_support_group_audit_v1/raw137_delete_silver_groups.png','raw137银片由native_static占主，C新路接管')
            silver_section+='<p>'+self.link(silver_path,'两个ROI两状态完整贡献/源SHA')+' · '+self.link('evidence/review/saved_raw130_component_takeover_cpu.json','独立原130轮胎ray组件证据')+'</p>'
        architecture='architecture_road_support.svg' if geometry_scope else 'architecture_bounded_surface.svg' if f_fit_method else 'architecture_existing_surface.svg' if e_fit_method else 'architecture.svg'
        style='body{margin:0;background:#f3f7f9;color:#193b4a;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1400px;margin:auto;padding:28px}section{background:white;border:1px solid #dce7ec;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:29px}h2{font-size:22px}img{display:block;max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #dce6eb;text-align:left}th{background:#edf4f7}a{color:#006b83}.worse{color:#ad3636}.pending{color:#946322}.notice{background:#fff4de;border-left:4px solid #c99435;padding:14px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}footer{font-size:13px;color:#607783}'
        page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene008：道路融合接入的无回退验证</title><style>{style}</style></head><body><main>
<h1>Scene008：道路融合接入的无回退验证</h1><p class="notice">{lead}</p><p>基线是已交付的最新SemanticActor5V44Integration，包含固定49BUILD语义sidecar。先比较latest、加AA支撑、加同支撑C材质；不把C对AA的收益替代对最新基线的守护。本轮首关raw94/105/128/137，raw130轮胎独立诊断。</p>
<section><h2>组件与数据流</h2><img src="{architecture}" alt="冻结最新v44加a5，完整道路支撑与原生深度排序，去银片及真实C2轮胎分别验证"></section>
<section><h2>当前关键：银片与真实轮胎不是同一组件</h2>{silver_section}</section>
<section><h2>实际四帧：道路与车体同时保留视野</h2><p>94/105直接粘贴完整480×320；128用原蓝框(165,188)–(208,234)，137用原世界mask包围盒外扩6像素。同一行三列同crop；局部NEAREST放大仅用于显示，不是新算法图或评价域。源PNG不改。</p>{initial_figures}</section>
<section><h2>正式固定域评价：AA/C对最新基线</h2>{self.metrics(initial,'AA/C首关')}<p>raw94世界near/lateral均无配对，只检查a5早段保护。其余三帧合计near24bin/lateral23bin；far无支持。a5在本轮只有94/105可见，不能把原全49的121128/25906像素称成本轮实际分母。</p></section>
<section><h2>原轮胎射线：颜色收益不能覆盖贡献损失</h2>{self.wheel(initial)}<p>128的36条属于正式四帧诊断，130的35条单独计账，不混入pilot接缝pooled。W是原native C2的alpha贡献，既有chart-first/ground-first是诊断类别，不是隐藏轮胎真值或不透明首交。C与AA的W一致，不能据此称C相对latest保住轮胎。</p>{self.image('figures/initial_pilot4_wheel130_separate.png','raw130三variant独立轮胎诊断')}</section>
<section><h2>D：后验固定新增行控制，原评价门不变</h2><p>观察到首轮回退后另立METHOD V2。仅在4个BUILD开发视图累计新增道路child的inside/guard贡献，选inside&gt;0且outside==0；固定行mask跨query与KEEP/DELETE共用。guard额外包含已冻结baseline既有C2、语义a5以及KEEP a37的正贡献像素，保护GT可见mask以外的既有模型尾部。无RGB挑域、无阈值搜索、无参数/材质更新；四帧因此不是HELD。</p><p>QUERY只读取固定三维行mask；原v44输入依赖保持。原4560评价域与阈值没有放宽，另增加71条原轮胎ray无新增W损失&gt;1e-6的root release要求。校准时outside=0不自动保证过滤后或其它视图也为0，必须重新成像。</p><p>{self.link('contract/fixed_build_support_amendment_v2.json','后验方法V2合同')} · {self.link('contract/fixed_build_support_amendment_v1.json','未执行V1历史合同')}</p>{d_section}</section>
<section><h2>E：复用融合目标，在原有表面回写外观</h2><p>原native_static与oldroad共有34,376个合格行（15,262＋19,114），不增加或退役几何。资格在4个BUILD视图中完成，不是拟合成功。原128主68像素中这些安全行可写贡献仅约0.18363/67.54≈0.272%，且oldroad为0；整体道路可写覆盖不能冒充主接缝覆盖，也不单凭比例断言不可能改善。</p><p>一次固定预算验证：同一行δRGB zero-init，投影到[-1,1]，Adam .01、400paired steps；原C完整合成目标和原legal mask，4帧现在都属于BUILD。先真实CUDA δ0恒等、smoke4＋disposable2＋完整恢复2及保存/评价，再fresh zero开始正式。实际训练账按收据，不以合同预算冒充已执行。</p><p>QUERY在原SH/fine求色后施加同一持久δ并clamp_min(0)，没有上界clamp；原v44输入依赖保持，目标/GT/mask不参与当帧选行或调色。原Gaussian/SH/geometry/identity为0更新，新材质优化有实际更新，二者分开。仅旧400步数相同不构成公平方法比较：原C是BUILD40/6159标量，E是BUILD4/103128标量。</p>{e_section}</section>
<section><h2>F：分域有界原表面颜色，原门不放宽</h2><p>E的零spill支撑在主缝太少，因此另立后验资格合同。保持同一原场景与原4560评价门，测每个BUILD帧/状态、每个原保护域的非负native贡献矩阵A。若以wᵣ=maxₘ Aₘᵣ/(3Nₘ·5e−7)，限制Σᵣwᵣ‖δᵣ‖₁≤1，则固定T与ReLU的1-Lipschitz性给出每个已校准保护域RGB MAE回退≤5e−7，保留原1e−6门的一半浮点裕量。</p><p>这是均值界，不是像素max或额外49相机保证；空域N=0仍为unsupported。资格得到203,932个inside-positive原有行，不意味着每列可独立无限使用。F实际fresh zero、同4源目标/合法mask与Adam.01完成400paired更新，δ先box cap1再weighted L1预算投影；不是续训E或调整E的cap救结果。真实预算约束通过，质量仍有失败。</p><p>w0与LP只表示operator capacity，不是已实现误差改善。QUERY仍是固定三维行共享δ、原7group与相同几何；不能继承E额外all-model-alpha像素的joint RGB exact性质。</p>{f_section}</section>
<section><h2>F新增暗条：监督输入与删除足迹独立核对</h2>{leakage_section}</section>
<section><h2>G历史诊断：后续合成扰动与颜色求解已取消</h2><p>针对F的失败读取原world near/lateral及原128独立蓝框53个pair的signed Jacobian，并补充DELETE原full37(label5)或baseline KEEP37W&gt;0.5的足迹颜色域。原primary35/33像素与world128的36/33分开，raw94无接缝pair；未使用暗条裁剪作为优化ROI。</p><p>实际22F/22dummyB读取已在远端完成，0 Adam/0原模型更新；计划8F合成扰动与CPU颜色solve在执行前取消。该操作子证据不是成功拟合或新候选图，主线已回到完整道路支撑和遮挡。</p><p>{self.link('contract/joint_seam_edit_safe_feasibility_v1.json','G历史诊断合同')} · {self.link('evidence/review/registered_seam_pairs_cpu_receipt.json','原53 pair逐bin数量与mask SHA')} · {self.link('evidence/G4_joint_seam_edit_operator/receipt.json','实际操作子收据')}</p></section>
<section><h2>原道路报告：研究候选与本次整合分开</h2><p>原C基于同40,392行n2＋relativeMip支撑、400paired steps的持久共享材质；2053父RGB可拟合，8045父行保留初值。C使用已有真实RGB固定10px边缘锚定，收益不是纯Poisson或隐藏纹理恢复。原C对AA的横向均值改善，原128 near和全49 near却回退；此前就没有全场景采用。</p><p>本次增加支撑相对latest会改变遮挡/alpha/期望深度；不能把旧C-vs-AA的几何读出一致移植成新整合也无损。旧HELD9目标含query O仅供构造诊断，原native材质不以query O调色；同场景曝光限制保留。</p><details><summary>原交付独立审阅与完整49联系图（不是本轮新结果）</summary><p>{self.link('evidence/source_road_review.json','原manifest/review/资产SHA及像素验证')} · {self.link('evidence/source_image_review.json','实际联系图查看记录')}</p>{self.image('figures/source_C_all49_delete_road_contact.png','旧道路C全49 DELETE对AA研究结果')}{self.image('figures/source_C_all49_keep_road_contact.png','旧道路C全49 KEEP对AA研究结果')}</details></section>
<section><h2>数值门、来源和执行</h2><p>每帧a5 body/upper、a6 RGB、KEEP a37及protected_nonroad RGB MAE变化≤1e-6；a6测深MAE变化≤1e-5m。道路外固定guard alpha/depth最大差≤1e-6/1e-5m。near/lateral等bin双状态pooled不回退，原128 near须两状态各至少5%相对且1e-4绝对改善。旧宽松门仅历史参考，不替代本轮标准。</p><p>所有域来自原观测/冻结registry，不按候选alpha改分母。所有空域/无支持保留null，RGB均值不代表逐像素无损。Observed中a37仍在，没有删除后隐藏道路真值；human_verdict=null。</p><ul>{''.join('<li>'+self.link(path,label)+'</li>' for path,label in evidence)}</ul><details><summary>实际账本与root原图记录</summary><pre>{h(json.dumps({'through_F_ledger':ledger,'G_appendix':g_ledger,'root_visual':root_visual},ensure_ascii=False,indent=2))}</pre></details></section>
<footer>状态：{state}。构建于{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}。本页不启动GPU、不发布中央viewer；GPU保持开机。</footer></main></body></html>'''
        with (ROOT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            fields=['case','raw','state','variant','diagnostic_only','domain','pixels','baseline_MAE','candidate_MAE','delta_MAE'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for name,review in [('initial',initial),('D',d),('E',e),('F',f)]:
                if not review:continue
                rows=review['records']+review['diagnostic_records'];base={(r['raw'],r['state']):r for r in rows if r['variant']=='baseline'}
                for row in rows:
                    if row['variant']=='baseline':continue
                    for key,value in row['metrics'].items():
                        before=base[(row['raw'],row['state'])]['metrics'][key]['mean'];after=value['mean']
                        writer.writerow(dict(case=name,raw=row['raw'],state=row['state'],variant=row['variant'],diagnostic_only=row['diagnostic_only'],domain=key,pixels=value['pixels'],baseline_MAE=before,candidate_MAE=after,delta_MAE=after-before if after is not None else None))
        (ROOT/'index.html').write_text(page,encoding='utf-8');(ROOT/'report_data.json').write_text('{}\n',encoding='utf-8')
        class Parser(HTMLParser):
            def __init__(self):super().__init__();self.links=[]
            def handle_starttag(self,t,a):self.links.extend(v for k,v in a if k in ('href','src') and v)
        parser=Parser();parser.feed(page);checked=[];images=[]
        for value in parser.links:
            u=urlsplit(value)
            if u.scheme or u.netloc or not u.path:continue
            p=ROOT/unquote(u.path);assert p.is_file(),value;checked.append(value)
            if p.suffix=='.png':
                with Image.open(p) as im:im.verify()
                with Image.open(p) as im:images.append(dict(path=unquote(u.path),size=list(im.size)))
        self.inputs.update([ROOT/'source/build_report.py',ROOT/architecture])
        data=dict(status=state,generated_utc=datetime.now(timezone.utc).isoformat(),initial_review=initial,D_review=d,
            frozen_evaluation_contract=contract,posterior_method_contract=method,D_reporting_scope=d_scope,
            E_feasibility_method_contract=e_method,E_fit_method_contract=e_fit_method,E_qualification=e_qualification,E_review=e,F_feasibility_contract=f_method,F_fit_method_contract=f_fit_method,F_review=f,
            F_supervision_identity_audit=leakage,F_fit_context=fit_context,G_feasibility_contract=g_method,G_receipt=g_receipt,G_ledger_appendix=g_ledger,
            current_geometry_scope=geometry_scope,silver_group_audit=silver,source_review=source,
            root_spot_check=root_quick,root_visual=root_visual,execution_ledger=ledger,full49_released=False,
            central_viewer_modified=False,quality_accepted=False,human_verdict=None,
            input_manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(self.inputs)],
            qa=dict(status='PASS',checked_relative_links=len(checked),missing=[],images=images))
        (ROOT/'report_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(dict(status=state,links=len(checked),images=len(images))))


if __name__=='__main__':Builder().build()
