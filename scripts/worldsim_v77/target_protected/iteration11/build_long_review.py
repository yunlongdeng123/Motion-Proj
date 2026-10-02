"""已完成的12例过程候选全帧审核；独立AI与人工评分分开。"""
from pathlib import Path
import sys,json
from collections import Counter
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(P/'iteration9'))
from temporal_factory import T,read,dump
O=T/'r16';D=O/'data_review'

def main():
 rows=read(D/'synthetic_manifest.json')['clips'];qa={c['case_id']:c for c in read(O/'independent_data_reviews.json')['cases']}
 checks=read(O/'technical_checks.json');assert checks['all_pass'] and len(rows)==len(qa)
 cases=[]
 for c in rows:
  q=qa[c['case_id']];assert q['human_verdict'] is None
  cases.append({'id':c['case_id'],'scene':c['scene'],'type':c['process_family']+' / '+c['type'],'group':c['source_split'],'review_frame':15,'frame_count':30,'contact':c['review_contact'],'videos':c['videos'],'frame_pattern':f'assets/{c["case_id"]}/{{i}}_{{role}}.jpg','note':f'{c["camera"]}；红A是合成待删轮廓，绿B/蓝C仅标活跃真实保护实例。relative speed delta={c["relative_speed_mps"]}m/s，不是绝对世界速度。窗口{c["temporal_process"]["duration_s"]:.2f}s；训练看到第四栏遮洞条件，完整X只作示意。','qa':f'独立gpt-6-sol xhigh，非fast，AI{q["assistant_score"]}／{q["decision"]}：{q["reason"]}。仅视觉审0/15/29；三十帧机器合同通过，不能据此认证视频全部时序。人工空。','metrics':{'process':c['temporal_process'],'active_protected_tokens':c['protected_instances'],'pixel_checks':c['pixel_metrics'],'human_verdict':None}})
 admitted=[c for c in rows if qa[c['case_id']]['decision']=='pass' and qa[c['case_id']]['assistant_score']==2]
 counts={s:dict(Counter(c['process_family'] for c in admitted if c['source_split']==s)) for s in ['train','validation']}
 dump(O/'coverage_closeout.json',{'cases':len(rows),'independent_AI2':len(admitted),'process_counts':counts,'sweep_train_cases':sum(c['source_split']=='train' and c['temporal_process']['sweep_over_any_B'] for c in admitted),'sweep_train_worlds':len({c['scene'] for c in admitted if c['source_split']=='train' and c['temporal_process']['sweep_over_any_B']}),'sweep_val_cases':sum(c['source_split']=='validation' and c['temporal_process']['sweep_over_any_B'] for c in admitted),'sweep_val_worlds':len({c['scene'] for c in admitted if c['source_split']=='validation' and c['temporal_process']['sweep_over_any_B']}),'training_steps':0,'human_verdict':None})
 data={'mode':'data','cases':cases,'roles':[{'key':k,'label':v} for k,v in [('gt','真实Y／恢复GT'),('input','合成X完整示意'),('labels','红A／活跃真实B与C'),('condition','模型实际遮洞条件')]],'score_roles':[{'key':'data','label':'数据质量'}]}
 template=(P/'iteration9/review_template.html').read_text().replace('max="9"','max="${(c.frame_count||10)-1}"').replace('/ 10帧','/ ${c.frame_count||10}帧').replace('i<10;i++','i<(c.frame_count||10);i++').replace('href="index.html"','href="../v77-target-protected-r14/index.html"').replace('v77-r8-','v77-r16-').replace("run:'r8'","run:'r16'").replace('v77_r8_','v77_r16_')
 arch='''<div class="box"><svg viewBox="0 0 1100 170" role="img" aria-label="相对路径遮挡数据组件"><defs><marker id="patharr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#91cfff"/></marker></defs><g fill="#233e59" stroke="#789bbd"><rect x="10" y="45" width="210" height="80" rx="8"/><rect x="280" y="45" width="240" height="80" rx="8"/><rect x="580" y="45" width="210" height="80" rx="8"/><rect x="850" y="45" width="240" height="80" rx="8"/></g><g fill="white" text-anchor="middle" font-size="18"><text x="115" y="76">真实nuScenes Y与相机</text><text x="115" y="106">旧合法虚拟A路径</text><text x="400" y="76">沿路径拟合相对速度</text><text x="400" y="106">重新检查空间与实际遮挡</text><text x="685" y="76">X／H全三十帧合同</text><text x="685" y="106">X全部影响被H擦除</text><text x="970" y="76">独立AI＋人工逐帧页</text><text x="970" y="106">覆盖检查，训练0</text></g><g stroke="#91cfff" stroke-width="2" marker-end="url(#patharr)"><path d="M220,85H275"/><path d="M520,85H575"/><path d="M790,85H845"/></g></svg></div>'''
 start=template.index('<div class="box"><svg');end=template.index('</svg></div>',start)+len('</svg></div>');template=(template[:start]+arch+template[end:]).replace('固定f5目标区域放大对照（插值放大，未增加真实细节）','0/15/29三关键帧过程对照')
 intro=f'<p><strong>约三秒真实曝光全30帧；候选{len(rows)}，独立AI2 {len(admitted)}。尚未训练。</strong></p><p>三秒监督Y始终真实，合成X影响全部擦除后再resize和VAE；旧一秒AI评分不传递。仍按原空间/mask/ego/GT距离/深度和保护85%上限检查全部30帧；AI视觉只看0/15/29，不能认证每帧视频。人工逐帧评分空。当前仅证明数据输入，不是三秒模型结果或真实任务收益。</p><p>按来源场景隔离，曾训练world只能train，其余已曝光world是DEV val；未曝光final未用。两种端点拟合加静止对照只跑一次固定来源，不看生成模型输出挑数据。</p><p><a href="../v77-target-protected-r14/index.html">r14模型控制</a> · <a href="../v77-target-protected-r15/data_review.html">旧一秒新过程输入</a></p>'
 (D/'data_review.html').write_text(template.replace('__TITLE__','v77 r16 · 三秒遮挡过程输入').replace('__INTRO__',intro).replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')))
 dump(D/'data_manifest.json',data);dump(D/'coverage_closeout.json',read(O/'coverage_closeout.json'));print('DATA_HTML',len(cases),counts)
if __name__=='__main__':main()
