"""保留旧回归原因审计，追加真实R2 49帧执行与验收凭据；不生成渲染。"""
import json, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
CTX = '14705303724557273004_3105_000_3125_000'
SCENE = ROOT / 'outputs/GS_Scene_Diagnostics_20261027' / CTX
R2 = ROOT / 'work/v43_unknown_detection_20261028/bounded_white_v1/bounded_native_r2'
OUT = ROOT / 'outputs/V43_Ownership_20261027/gpu_v43/r2_recovery_20261029'
WORK = Path(__file__).resolve().parent

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SCENE/'bounded_unknown_manifest.json').read_text(encoding='utf8'))
    legacy_text = (SCENE/'target_detection_data.js').read_text(encoding='utf8')
    legacy = json.JSONDecoder().raw_decode(legacy_text.split('=', 1)[1])[0]
    completed = sorted(int(k) for k,v in legacy['frames'].items() if v.get('evidence_path'))
    missing = sorted(int(k) for k,v in legacy['frames'].items() if not v.get('evidence_path'))
    assert completed == [98,113,114] and len(missing) == 46
    assert len(manifest['frames']) == 49
    records = []
    sheet = Image.new('RGB', (960,1056), '#0c1825')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for i,raw in enumerate(completed):
        old = np.load(R2/f'{raw:03d}/evidence.npz', allow_pickle=False)
        f = manifest['frames'][str(raw)]
        receipt = json.loads((SCENE/f['receipt_path']).read_text(encoding='utf8'))
        new = np.load(SCENE/f['evidence_path'], allow_pickle=False)
        old_final = np.union1d(old['existing_removed_ids'],old['new_unknown_ids'])
        new_final = new['final_removed_ids']
        rec = dict(raw=raw, r2_initial=len(old['existing_removed_ids']),
                   r2_added=len(old['new_unknown_ids']), r2_final=len(old_final),
                   empty_final=len(new_final), r2_missing_in_empty=len(np.setdiff1d(old_final,new_final)),
                   empty_not_in_r2=len(np.setdiff1d(new_final,old_final)),
                   inherited_seed_outside_current_eligible=len(np.setdiff1d(old['existing_removed_ids'],old['eligible_ids'])),
                   r2_query_pixels=int(old['query'].sum()), empty_query_pixels=int(new['query'].sum()),
                   same_blue=np.array_equal(old['allowed'],new['allowed']),
                   same_display=np.array_equal(old['display_scope'],new['display_scope']),
                   r2_contact_alpha=float(old['alpha'][new['contact']].mean()),
                   empty_contact_alpha=receipt['contact_alpha_mean'])
        assert rec['same_blue'] and rec['same_display']
        records.append(rec)
        for c,(label,path) in enumerate([
                ('真实 R2 · 已认可视觉基线',R2/f'{raw:03d}/white.png'),
                ('空集合49帧臂 · 非R2，未采纳',SCENE/f['images']['white'])]):
            draw.text((c*480+7,i*352+5),f'raw{raw} | {label}',font=font,fill='white')
            with Image.open(path) as im:
                assert im.size == (480,320)
                sheet.paste(im.convert('RGB'),(c*480,i*352+32))
    sheet.save(OUT/'r2_vs_empty.png')
    audit = dict(task='WS-V43-R2-RECOVERY-20261029',kind='CPU evidence audit; no new GPU rendering',
                 legacy_panel=dict(exported=completed,not_exported=missing,total=49),
                 empty_arm_complete=49, r2_complete=3, r2_all49_complete=False,
                 records=records, training_updates=0,
                 labels_are_semantic_GT=False, quality_of_empty_arm='user_rejected_as_R2_successor')
    # The original CPU-only audit records the failure at that time. Never
    # rewrite it to imply the old all49 arm was accepted or GPU-validated.
    if not (OUT/'audit.json').exists():
        (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    delivery=WORK/'r2_delivery'
    native=json.loads((delivery/'r2_inherited49/summary.json').read_text(encoding='utf8'))
    numeric=json.loads((delivery/'local_audit.json').read_text(encoding='utf8'))
    stats=json.loads((WORK/'r2_quality_stats.json').read_text(encoding='utf8'))
    visual=json.loads((WORK/'ai_visual_review.json').read_text(encoding='utf8'))
    contact=json.loads((WORK/'cpu_delivery/contact49/summary.json').read_text(encoding='utf8'))
    gate_names=['anchor_replay_validation','contact_anchor_validation','seed_anchor_validation','seed_full49_validation','extend_anchor_validation']
    gates={k:json.loads((delivery/(k+'.json')).read_text(encoding='utf8')) for k in gate_names}
    assert native['status']=='complete' and native['frames']==49 and numeric['passed']
    assert all(g['passed'] for g in gates.values())
    anchor_checks=[x for x in numeric['frames'] if x['raw'] in [98,113,114]]
    assert all(x['checks']['anchor_oldR2_floats_exact'] and x['checks']['anchor_oldR2_IDs_exact'] for x in anchor_checks)
    completion=dict(task='WS-V43-R2-RECOVERY-20261029',context=CTX,raws=list(range(77,126)),
        phase='R2_inherited49_complete_pending_user_visual_acceptance',
        original_R2_3_anchor_exact=True,all_5_gates_passed=True,remote_full_float_audit_passed=True,
        local_full49_float_present=False,PNG_checked=441,training_updates=0,
        GPU_render49_wall_seconds=native['wall_seconds'],CPU_contact49_wall_seconds=contact['wall_seconds'],
        same_frame_existing_preserved=True,query='body/contact/existing_float_out_of_range',
        new_outside_center_rows=0,range_stats=stats['ranges'],human_verdict=None,
        AI_visual_review_is_not_human_acceptance=True,
        scene2_3_status='old non-R2 source-only ablations retained; not rerun in this recovery')
    write_json=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
    write_json(OUT/'completion.json',completion)
    attachments={**{k+'.json':delivery/(k+'.json') for k in gate_names},
        'local_audit.json':delivery/'local_audit.json','r2_quality_stats.json':WORK/'r2_quality_stats.json',
        'ai_visual_review.json':WORK/'ai_visual_review.json',
        'r2_inherited_canvas_qa.json':WORK/'r2_inherited_canvas_qa.json'}
    for name,source in attachments.items():shutil.copy2(source,OUT/name)
    for name in ['r2_inherited49_white_crops.png','r2_inherited49_suspect_fullframes.png']:
        shutil.copy2(delivery/name,OUT/name)
    rows=''.join(f"<tr><td>raw{r['raw']}</td><td>{r['r2_initial']:,}</td><td>{r['r2_final']:,}</td><td>{r['empty_final']:,}</td><td>{r['r2_missing_in_empty']:,}</td><td>{r['r2_query_pixels']:,} → {r['empty_query_pixels']:,}</td></tr>" for r in records)
    page='''<!doctype html><meta charset="utf-8"><title>R2继承回归核对</title>
<style>body{font:16px/1.65 system-ui;max-width:1180px;margin:28px auto;padding:20px;background:#0c1825;color:#daeafa}a{color:#8ed7ff}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #49627a}img,svg{max-width:100%}.notice{padding:16px;background:#243c51;border-left:4px solid #e7b967}</style>
<h1>R2继承回归核对 · 2026-10-29</h1>
<p class="notice"><b>真实R2继承版已完成场景一49帧（raw77–125）。</b>先复现原3锚点，再恢复同帧existing集合与原查询，全49帧完成GPU剥离及独立CPU浮点验收。raw98／113／114的行ID、查询、G、alpha和局部白底与原R2逐值一致，浮点差0。用户指出的旧pending与继承断开问题保留在下面的原因审计；当前结果不再采用空集合替代R2。</p>
<p><a href="../../../GS_Scene_Diagnostics_20261027/14705303724557273004_3105_000_3125_000/index.html#workspace?group=v43_r2_inherited49&amp;raw=113&amp;channel=rgb">打开共享工作台：真实R2继承49帧</a> · <a href="../bounded_white_v1/index.html">原三帧R2报告（原文件保留）</a> · <a href="completion.json">本轮完成凭据</a> · <a href="audit.json">旧CPU原因审计（保留当时状态）</a></p>
<h2>本轮执行与检查</h2><table><tr><th>项目</th><th>实际结果</th><th>说明</th></tr><tr><td>覆盖与依赖</td><td>49／49帧；5个验证门通过</td><td>原接触证据、原photo分类existing种子；锚点先验，再扩到其余46帧</td></tr><tr><td>原R2回归</td><td>3锚点ID和查询相同；G／alpha／白底误差0</td><td>保留原初始化、彩噪查询、蓝框与框外existing合成</td></tr><tr><td>完整浮点验收</td><td>49帧通过；441张PNG逐一解码与编码核对</td><td>远端完整NPZ验收，本地仅保存完整ID、图像和紧凑证据</td></tr><tr><td>蓝框与合成</td><td>新增框外中心0；范围外像素差0；白底公式误差0</td><td>旧existing集合自带框外中心11–554行，单独记录，不混称为新增框内unknown</td></tr><tr><td>候选及收敛</td><td>最终每帧6,662–26,166行；2–3次求值收敛</td><td>仍用原上限8次求值、alpha×T阈值1e-7</td></tr><tr><td>开销</td><td>GPU 49帧79.84秒；CPU接触证据96.32秒；训练0</td><td>这是两项阶段实测耗时，不包含回归与其余接线开销</td></tr><tr><td>场景二／三</td><td>本轮未重做</td><td>共享平台保留此前非R2消融，不称其为真实R2继承结果</td></tr></table>
<h2>恢复的继承链</h2>
<svg viewBox="0 0 1180 220" role="img" aria-label="原检测证据生成existing集合，R2逐帧归因，严格回归后发布"><defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0 0L7 3L0 6" fill="#93c1dc"/></marker></defs><g fill="#1c3c50" stroke="#93c1dc"><rect x="5" y="25" width="335" height="165" rx="8"/><rect x="385" y="25" width="380" height="165" rx="8"/><rect x="810" y="25" width="365" height="165" rx="8"/></g><g fill="#e0eff8" text-anchor="middle" font-size="17"><text x="172" y="57">① 同帧原始证据</text><text x="172" y="92">车体／阴影／接触区证据</text><text x="172" y="124">原分类规则 → existing行ID</text><text x="172" y="158">保留冻结检查点与来源</text><text x="575" y="57">② 继承真实R2</text><text x="575" y="92">已有集合＋车身／车底／彩噪查询</text><text x="575" y="124">蓝框内新增 → 删除 → 重算贡献</text><text x="575" y="158">G＋透射率×白底；框外existing</text><text x="993" y="57">③ 回归与完整交付</text><text x="993" y="92">98／113／114先复现</text><text x="993" y="124">通过后49帧，逐帧状态与几何</text><text x="993" y="158">不以其他算法或邻帧填空</text></g><g stroke="#93c1dc" stroke-width="3" marker-end="url(#a)"><path d="M340 105H380"/><path d="M765 105H805"/></g></svg>
<h2>三个实际改动导致继承断开</h2><table><tr><th>环节</th><th>真实R2</th><th>后续49帧臂（未采纳）</th></tr><tr><td>初始删除集合</td><td>同帧已有候选existing</td><td>空集合</td></tr><tr><td>查询</td><td>车身＋车底接触区＋existing浮点越界彩噪</td><td>只有车身＋车底</td></tr><tr><td>局部范围外基线</td><td>existing原生输出</td><td>只去Actor的输出</td></tr></table>
<p>三锚点蓝框与局部显示范围相同。下面的遗漏是原static行集合差，不能直接解释成车辆语义漏检率。R2自身仍是诊断候选算法，未成为自动语义真值。</p>
<table><tr><th>raw</th><th>R2已有候选</th><th>R2总删除</th><th>空集合臂删除</th><th>R2中有而后者漏掉</th><th>查询像素：R2→空集合臂</th></tr>ROWS</table>
<p>R2既有集合中另有381／20／17行中心位于当前蓝框外。它们来自既有基线，不是本次新增；回归必须单列此项，不能隐瞒成“所有被删行都在框内”。新增unknown继续严格受蓝框约束。二维框外保护也不等于全局三维删除无影响。</p>
<h2>同帧真实图（未调色、未补景）</h2><img src="r2_vs_empty.png" alt="三帧真实R2与空集合49帧臂逐行对比">
<h2>49帧AI视觉核查与已知边界</h2><p>AI查看了全49帧局部拼图及11帧完整图（raw77、82、84、86、87、90、93、98、101、113、114）。未发现明显跳帧或严重视觉失败；这是AI辅助核查，<b>不替代用户人工验收</b>。raw82–87、90、93–94、98、101、113–114仍可见局部浅暗边或品红／绿色细边，101和113–114较明显。不能将细纹仍在的结果描述成完全纯白。</p>
<p>接触查询区与整个query内亮度小于0.2的像素计数均为0，但接触区平均alpha仍为0.338–0.548；跨界高斯仍贡献。这个阈值只用于排查深色残留，不是语义召回率、去影成功率或逐帧稳定性的证明。白底采用真实剥离后的G＋(1−alpha)×白色，仅在限定范围显示；框外使用同帧existing。它是局部空洞诊断，不是补景或永久语义身份写回。</p>
<p><a href="r2_inherited49_white_crops.png">查看全部49帧局部拼图</a> · <a href="r2_inherited49_suspect_fullframes.png">查看重点完整帧</a> · <a href="ai_visual_review.json">AI核查记录（非人工验收）</a> · <a href="r2_quality_stats.json">逐帧残留与候选统计</a></p>
<details><summary>全部49帧局部拼图（仅用于诊断）</summary><img src="r2_inherited49_white_crops.png" alt="真实R2继承版49帧局部图"></details>
<h2>执行凭据与后续边界</h2><p><a href="anchor_replay_validation.json">原R2重放门</a> · <a href="contact_anchor_validation.json">接触区证据门</a> · <a href="seed_anchor_validation.json">原种子锚点门</a> · <a href="seed_full49_validation.json">49帧种子门</a> · <a href="extend_anchor_validation.json">继承渲染锚点门</a> · <a href="local_audit.json">远端全浮点独立验收</a> · <a href="r2_inherited_canvas_qa.json">网页Canvas接线验收</a></p>
<p>当前等待用户检查49帧效果。场景二三尚需沿本轮验证过的R2继承链补齐原始证据并单独回归，不能复用空集合臂冒充R2。所有旧图与旧审计保留，新增结果未覆盖参考。</p>'''.replace('ROWS',rows)
    (OUT/'index.html').write_text(page,encoding='utf8')
    print(json.dumps(completion,ensure_ascii=True))

if __name__ == '__main__':main()
