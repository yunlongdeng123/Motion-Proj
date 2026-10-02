"""三秒数据收口：独立质量通过不等于验证覆盖或模型收益。"""
from pathlib import Path
import sys,json,shutil
P=Path('/root/autodl-tmp/motion_proj_v77');S=P/'scripts/worldsim_v77/target_protected';sys.path.insert(0,str(S/'iteration9'))
from temporal_factory import T,read,dump
O=T/'r16';E=P/'docs/autoresearch/worldsim_v77/target_protected_20260929/r16'
REPORT="""# r16：三秒连续曝光造数与覆盖边界

WS-V77-TARGET-PROTECTED-20260929/r16，wm-3090-1001，failure_ledger_refs [V77-F02]。

```mermaid
flowchart LR
 R[45来源预选／31world] --> C[40有效来源／1200真实曝光]
 C --> S[SAM2：47轨／44连续性通过]
 S --> G[地图、深度、地面与完整GT障碍]
 G --> X[30帧连续放置／Y保持真实RGB]
 X --> Q[300帧合同＋10例独立AI2]
 Q --> V[扫过验证0／训练0]
 Q --> H[四列视频＋逐帧人工评分空]
```

从已保存合法路径出发，固定四个锚点、两个端点相对速度拟合和世界静止对照，延长到约三秒实际曝光；没有复制RGB、插帧、降低物理阈值或把生成视频当Y。45来源31world事前固定，实际40来源27world、1200帧；其余五来源因曝光次序或长窗观测不合格保留拒绝。210缺失RGB由pub真实03/07包选择抽取，不需要新增数据。原pub目录误定位的失败日志及修正代码保留。地面检查沿用已登记raw近地支持修复，支持阈值2.5m不变；38/40来源通过，地面返回不是空路证明。

官方SAM2固定输入，47对象轨完成、44技术连续性通过；三条失败保留隔离。没有用GT裁mask或按结果清理。GT尺寸、距离、地面、深度顺序、ego禁区、画面边缘、轨迹与mask连续性门槛全部保持。未知动态对象仍作为障碍，不能把未分割车当背景。

最终10候选来自三个train世界：四个显露过程(scene0708)、四个世界静止A＋运动ego(scene0719)、两个扫过保护车(scene0302)。实际重载检查全部300帧，Y精确等于真实输入、擦除后X条件精确等于擦除后Y、RGB泄漏0、ego洞像素0。独立gpt-6-sol xhigh、无fast逐例看0/15/29帧及相关空间标签，10/10 AI2；人工评分始终空，抽帧不认证整段视频时序。

扫过train仅2例1world，validation扫过0，未达登记8train/3val/2valworld覆盖目标，因此本轮训练0。数据质量合格和过程覆盖不足是两个不同结论；不把零训练归为模型失败，也不无限枚举同一来源。当前模型r18仍使用r14固定旧数据，此包不偷偷混入。

审核页本地 outputs/v77-target-protected-r16/data_review.html，原始Y／合成X／洞H／实际模型可见条件四列，30帧滑条与逐帧评分导出。40 MP4实际解码1200帧、1200图片引用、10张对照图、HTML/manifest一致与JS语法均通过；未声称已在浏览器逐段播放。所有模型和拒绝均保留，真实DELETE收益关机条件false。
"""
def main():
 technical=read(O/'technical_checks.json');quality=read(O/'independent_data_reviews.json');delivery=read(O/'delivery_validation.json');sampling=read(O/'sampling_control.json');seg=read(O/'segmentation_state.json');cover=read(O/'coverage_closeout.json')
 assert technical['all_pass'] and technical['checked_frames']==300 and len(quality['cases'])==10 and cover['independent_AI2']==10
 assert delivery['success'] and delivery['videos_checked']==40 and delivery['actual_decoded_frames']==1200
 b=O/'docs_before_closeout';b.mkdir(exist_ok=True)
 for rel in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']:
  dest=b/Path(rel).name
  if not dest.exists():shutil.copy2(P/rel,dest)
 worlds=len(set(x['scene'] for x in read(O/'source_ready.json').get('sources',[])))
 close={'stage':'data_quality_complete_coverage_incomplete_no_training','selected_sources':45,'selected_worlds':31,'actual_sources':40,'actual_worlds':27,'actual_source_RGB':1200,'SAM2_tracks':47,'SAM2_technical_pass':44,'synthetic_cases':10,'synthetic_frames_checked':300,'independent_AI2':10,'coverage':cover,'delivery':{k:v for k,v in delivery.items() if k!='videos'},'training_steps':0,'real_cross_case_benefit_demonstrated':False,'shutdown_eligible':False,'human_verdict':None,'final_used':False,'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02'}
 dump(O/'closeout.json',close);dump(E/'closeout.json',close)
 for name in ['coverage_closeout.json','independent_data_reviews.json','technical_checks.json','delivery_validation.json','sampling_control.json']:
  dump(E/name,read(O/name))
 (P/'docs/v77/TARGET_PROTECTED_LONG_DATA_R16.md').write_text(REPORT)
 (E/'review_link.md').write_text('本地审核：C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-target-protected-r16/data_review.html。人工评分空；独立AI2仅为数据质量。\n')
 idx=P/'docs/EXPERIMENTS.md';lines=idx.read_text().splitlines();lines=[('| WS-V77-TARGET-PROTECTED-20260929 / r16 | 40来源1200曝光；10独立AI2三秒候选、40视频交付；val扫过0、训练0 | [报告](v77/TARGET_PROTECTED_LONG_DATA_R16.md) |' if l.startswith('| WS-V77-TARGET-PROTECTED-20260929 / r16 |') else l) for l in lines];idx.write_text('\n'.join(lines)+'\n')
 status=P/'docs/RESEARCH_STATUS.md';text=status.read_text().replace('10候选正在完整生成/独立审核，val扫过仍0，训练0。','10候选300帧合同通过且独立AI2，40视频/1200帧交付；val扫过仍0，训练0。');status.write_text(text)
 failure=P/'docs/research_failures/entries/V77-F02.md';title='## r16：长窗数据合格但隔离验证过程仍缺额'
 if title not in failure.read_text():failure.write_text(failure.read_text()+'\n\n'+title+'\n\n固定45来源实际40/1200曝光，10候选300帧合同与独立AI2通过；4显露、4静止A运动ego、2扫过均为train，val扫过0，训练0。40视频实际解码，不能把长窗质量或AI2当模型迁移收益；不放宽道路/ego/depth/GT障碍门槛，不继续同来源无限速度搜索。[报告与图](../../v77/TARGET_PROTECTED_LONG_DATA_R16.md)。failure_ledger_delta: updated V77-F02。\n')
 print('CLOSED_R16',json.dumps(close,ensure_ascii=False))
if __name__=='__main__':main()
