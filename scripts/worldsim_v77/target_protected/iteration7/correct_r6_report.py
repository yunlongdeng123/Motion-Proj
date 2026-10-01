"""保留r6原始视频/数值，同时纠正加载证据混淆。"""
from pathlib import Path
import json,shutil
REPO=Path('/root/autodl-tmp/motion_proj_v77');TASK=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');ROOT=TASK/'r7'
def main():
    correction={'date':'2026-10-01','r6_training_missing_encoder_tensors':106,'r6_inference_missing_tensors':0,
                'random_frozen_target_encoder':True,'scientific_attribution_valid':False,
                'evidence':'r6/logs/train_pilot.log:47 versus r6/logs/evaluate_pilot.log:15',
                'preserve':'all old metrics/videos/AI scores unchanged; only correct interpretation',
                'followup':'r7 restore original official SVD encoder, strict load, same data/80 tensors/320x576/160 steps'}
    (ROOT/'r6_correction.json').write_text(json.dumps(correction,ensure_ascii=False,indent=2)+'\n')
    (TASK/'r6/training_checkpoint_correction.json').write_text(json.dumps(correction,ensure_ascii=False,indent=2)+'\n')
    backup=Path('/root/autodl-tmp/codex_backups/WS-V77-TARGET-PROTECTED-20260929/r7-start')
    report=REPO/'docs/v77/TARGET_PROTECTED_FINETUNE_R6.md'
    if '## r7 排查纠正' not in report.read_text():
        dst=backup/'docs/v77/TARGET_PROTECTED_FINETUNE_R6.md';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(report,dst)
        text=report.read_text().replace('完整已训练`model.safetensors`初始化，CLIP也由此恢复，0 missing / 0 unexpected；','完整已训练`model.safetensors`初始化，CLIP由此恢复；**训练实际106个目标encoder权重缺失，原“0 missing”引用了推理日志，见r7纠正**；')
        pos=text.index('```mermaid')
        text=text[:pos]+'''## r7 排查纠正

2026-10-01查原始日志：训练`logs/train_pilot.log:47`为106 missing / 0 unexpected，全部是`first_stage_model.encoder.*`；推理`evaluate_pilot.log:15`才是0 missing。推理权重不包含训练目标encoder，train.yaml新建的encoder随机初始化并被冻结，Y因此映射到错误latent空间。上一版将推理加载当成训练加载，表述错误，现纠正。160步及输出数值/视频仍是真实执行结果，但**不能用它们否定数据或更新模块**；loss下降也不是有效训练成功证据。r7只恢复官方SVD的106目标encoder权重，严格加载后先做同数据/同80张量/同320×576/160步控制；原证据保留。

'''+text[pos:];report.write_text(text)
    html=TASK/'r6/delivery/index.html'
    notice='<aside id="r7-encoder-correction" style="background:#562b20;padding:18px;border:2px solid #ffb287"><strong>r7排查纠正：r6训练目标encoder的106个权重未加载，随机初始化后被冻结。</strong><p>之前“0 missing”属于推理加载，不能认证训练。下方旧视频与数值全部保留，但不能据此归因数据或微调模块；loss下降无效。r7已登记只修encoder的同配方控制。</p></aside>'
    if 'r7-encoder-correction' not in html.read_text():
        dst=backup/'r6_delivery_index.html';shutil.copy2(html,dst)
        text=html.read_text();end=text.index('</h1>')+len('</h1>');html.write_text(text[:end]+notice+text[end:])
    for rel in ['r6/pilot_summary.json','r6/delivery/summary.json','r6/closeout.json']:
        p=TASK/rel;s=json.loads(p.read_text());s['r7_training_encoder_correction']=correction;p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
    (REPO/'docs/RESEARCH_STATUS.md').write_text('''# 当前研究状态

2026-10-01，v77；`WS-V77-TARGET-PROTECTED-20260929/r7`已定位r6实质训练工程错误：训练日志106个first_stage encoder missing，推理0 missing；原报告引用错日志。随机被冻结目标encoder让Y监督latent错位，旧loss下降不能判有效。原r6视频/数值、坏权重与所有数据保留并注释，默认原模型不变。

已仅从官方SVD获取106个原始目标encoder权重（136.65MB byte ranges，未下载完整SVD），严格恢复。正在encoder_fixed_lowres同数据/48train4val/80张量/320×576/160步控制；不修改data/loss/seed/架构。原native_spatial3步发现错误后停止留证，扩大模块臂取消；先排工程错误，再判数据或模块。全帧审计训练保护区占画面平均0.188%，洞3.238%；覆盖不足是后续候选，不先宣称因果。

四个留出共一个scene，另两个固定训练例作容量诊断。无final/Ω/GLB/电源/自动化操作。run `/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r7`。failure_ledger_refs: [V77-F02]；failure_ledger_delta: pending。
''')
    print('r6 correction saved; old outputs retained')
if __name__=='__main__':main()
