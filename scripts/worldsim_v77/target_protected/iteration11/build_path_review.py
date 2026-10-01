"""已完成的12例过程候选全帧审核；独立AI与人工评分分开。"""
from pathlib import Path
import sys,json
from collections import Counter
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(P/'iteration9'))
from temporal_factory import T,read,dump
O=T/'r15';D=O/'data_review'

def main():
 rows=read(D/'synthetic_manifest.json')['clips'];qa={c['case_id']:c for c in read(O/'independent_data_reviews.json')['cases']}
 checks=read(O/'technical_checks.json');assert checks['all_pass'] and len(rows)==len(qa)==12
 cases=[]
 for c in rows:
  q=qa[c['case_id']];assert q['decision']=='pass' and q['assistant_score']==2
  cases.append({'id':c['case_id'],'scene':c['scene'],'type':c['process_family']+' / '+c['type'],'group':c['source_split'],'review_frame':5,'contact':c['review_contact'],'videos':c['videos'],'frame_pattern':f'assets/{c["case_id"]}/{{i}}_{{role}}.jpg','note':f'{c["camera"]}；红A是合成待删轮廓，绿B/蓝C仅标活跃真实保护实例。relative speed delta={c["relative_speed_mps"]}m/s，不是绝对世界速度。窗口{c["temporal_process"]["duration_s"]:.2f}s；训练看到第四栏遮洞条件，完整X只作示意。','qa':f'独立gpt-6-sol xhigh，非fast，AI2：{q["reason"]}。仅视觉审0/5/9；十帧机器合同通过，不能据此认证视频全部时序。人工空。','metrics':{'process':c['temporal_process'],'active_protected_tokens':c['protected_instances'],'pixel_checks':c['pixel_metrics'],'human_verdict':None}})
 counts={s:dict(Counter(c['process_family'] for c in rows if c['source_split']==s)) for s in ['train','validation']}
 dump(O/'coverage_closeout.json',{'cases':12,'independent_AI2':12,'process_counts':counts,'sweep_train_cases':6,'sweep_train_worlds':4,'sweep_val_cases':0,'coverage_goal':{'sweep_train':8,'sweep_val':3,'sweep_val_worlds':2},'coverage_goal_met':False,'training_steps':0,'all_inputs_preserved':True,'human_verdict':None})
 data={'mode':'data','cases':cases,'roles':[{'key':k,'label':v} for k,v in [('gt','真实Y／恢复GT'),('input','合成X完整示意'),('labels','红A／活跃真实B与C'),('condition','模型实际遮洞条件')]],'score_roles':[{'key':'data','label':'数据质量'}]}
 template=(P/'iteration9/review_template.html').read_text().replace('href="index.html"','href="../v77-target-protected-r14/index.html"').replace('v77-r8-','v77-r15-').replace("run:'r8'","run:'r15'").replace('v77_r8_','v77_r15_')
 arch='''<div class="box"><svg viewBox="0 0 1100 170" role="img" aria-label="相对路径遮挡数据组件"><defs><marker id="patharr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#91cfff"/></marker></defs><g fill="#233e59" stroke="#789bbd"><rect x="10" y="45" width="210" height="80" rx="8"/><rect x="280" y="45" width="240" height="80" rx="8"/><rect x="580" y="45" width="210" height="80" rx="8"/><rect x="850" y="45" width="240" height="80" rx="8"/></g><g fill="white" text-anchor="middle" font-size="18"><text x="115" y="76">真实nuScenes Y与相机</text><text x="115" y="106">旧合法虚拟A路径</text><text x="400" y="76">沿路径拟合相对速度</text><text x="400" y="106">重新检查空间与实际遮挡</text><text x="685" y="76">X／H全十帧合同</text><text x="685" y="106">X全部影响被H擦除</text><text x="970" y="76">独立AI＋人工逐帧页</text><text x="970" y="106">覆盖未足，训练0</text></g><g stroke="#91cfff" stroke-width="2" marker-end="url(#patharr)"><path d="M220,85H275"/><path d="M520,85H575"/><path d="M790,85H845"/></g></svg></div>'''
 start=template.index('<div class="box"><svg');end=template.index('</svg></div>',start)+len('</svg></div>');template=template[:start]+arch+template[end:]
 intro='<p><strong>12个新候选均达到独立AI2，全部保留。训练6个扫过例／4world，验证扫过仍0；未达到8train／3val／2隔离val-world，因此本轮训练0步。</strong></p><p>46来源的有限两端拟合已完成，不增加同来源网格。原物理、车形、ego、GT间距和85%保护遮挡上限未放宽。每例真实Y、全部X影响覆盖、十帧空间与掩罩合同通过。AI2只是数据输入准入，不是模型收益或人工通过；三帧视觉不能认证整段时序。</p><p>GT永远是真实RGB。灰色合成车身不会进入模型条件；错误空间关系、洞漏和异常轨迹仍不准入。人工逐帧评分始终空。</p><p><a href="../v77-target-protected-r14/index.html">r14等参数量模型对照</a> · <a href="../v77-target-protected-r10/data_review.html">旧输入保留</a></p>'
 (D/'data_review.html').write_text(template.replace('__TITLE__','v77 r15 · 相对路径遮挡候选').replace('__INTRO__',intro).replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')))
 dump(D/'data_manifest.json',data);dump(D/'coverage_closeout.json',read(O/'coverage_closeout.json'));print('DATA_HTML',len(cases),counts)
if __name__=='__main__':main()
