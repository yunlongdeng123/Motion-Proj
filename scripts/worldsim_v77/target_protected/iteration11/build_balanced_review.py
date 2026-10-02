"""同输入六权重审核；复用旧视频，仅编码恢复区域宏平均权重。"""
from pathlib import Path
import sys,os,json
import numpy as np,cv2
from PIL import Image,ImageDraw
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(P));sys.path.insert(0,str(P/'iteration9'))
from temporal_factory import T,read,dump
from render_pairs import font,encode
O=T/'r18';OLD=T/'r14'
GROUPS={'r8_frozen_synthetic':'旧合成GT验证','new_temporal_validation':'新过程GT验证','real_DEVELOPMENT':'真实DELETE开发例'}

def main(contacts_only=False,html_only=False):
 cv2.setNumThreads(1);dest=O/'review';dest.mkdir(exist_ok=True);(dest/'effects').mkdir(exist_ok=True);(dest/'contacts').mkdir(exist_ok=True)
 plan=read(O/'evaluation_plan.json');cases=[];done=0
 oldmanifest={c['id']:c for c in read(OLD/'review/effect_manifest.json')['cases']}
 qa=read(O/'assistant_effect_reviews.json') if (O/'assistant_effect_reviews.json').exists() else {'cases':[]}
 for c in plan['cases']:
  cid=c['eval_id'];src=O/'evaluation'/cid
  if not (src/'r18_metrics.json').exists():continue
  done+=1;out=dest/'effects'/cid;out.mkdir(exist_ok=True)
  panels=[]
  for i in ([] if html_only else range(10)):
   panel={}
   for role in ['target','input','base','r7','r8','r10','r14']:
    old=OLD/'review/effects'/cid/f'{i:03}_{role}.jpg';panel[role]=np.asarray(Image.open(old).convert('RGB')).copy()
    if not contacts_only and not (out/old.name).exists():os.link(old,out/old.name)
   for role in ['r18','r18_native']:
    a=np.asarray(Image.open(src/role/f'{i:05}.png').convert('RGB'));panel[role]=a
    if not contacts_only:Image.fromarray(a).save(out/f'{i:03}_{role}.jpg',quality=96)
   panels.append(panel)
  if not html_only:
   for name,ids in [('f5',[5]),('first5',list(range(5))),('last5',list(range(5,10)))]:
    sheet=Image.new('RGB',(3200,42+len(ids)*300),(16,23,34));d=ImageDraw.Draw(sheet);d.text((8,7),f'{cid} | {c.get("receiver_scene",c.get("scene"))} | r14 uniform / r18 macro-region loss; same temporal 80 tensors | {ids}',font=font(21),fill='white')
    for row,i in enumerate(ids):
     h=np.asarray(Image.open(src/'mask'/f'{i:05}.png'))>0;yy,xx=np.where(h);box=(max(0,int(xx.min())-30),max(0,int(yy.min())-30),min(1024,int(xx.max())+31),min(576,int(yy.max())+31))
     for j,role in enumerate(['target','input','base','r7','r8','r10','r14','r18']):
      im=Image.fromarray(panels[i][role]).crop(box);scale=min(394/im.width,256/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS);d.text((j*400+6,45+row*300),f'f{i} {role}',font=font(18),fill='white');sheet.paste(im,(j*400+(400-im.width)//2,74+row*300))
    sheet.save(dest/'contacts'/f'{cid}_{name}.jpg',quality=96)
  if contacts_only:continue
  if not html_only:
   for old in (OLD/'review/effects'/cid).glob('*'):
    if old.is_file() and not (out/old.name).exists():os.link(old,out/old.name)
   for role in ['r18','r18_native']:encode(out,role)
  for role in ['target','input']+plan['arms']+[a+'_native' for a in plan['arms']]:assert (out/(role+'.mp4')).exists()
  entry=dict(oldmanifest[cid]);entry.update(videos={r:f'effects/{cid}/{r}.mp4' for r in ['target','input']+plan['arms']},native_links={a:f'effects/{cid}/{a}_native.mp4' for a in plan['arms']},frame_pattern=f'effects/{cid}/{{i}}_{{role}}.jpg',contact=f'contacts/{cid}_f5.jpg')
  entry['note']+=' r18同r14全部数据、160步与80时间张量；只改变2D损失为global/H-bg/H-protected等权宏平均。B仅训练标签，推理不增加通道，架构不变。'
  entry['qa']=next((r['observations'] for r in qa['cases'] if r['eval_id']==cid),'助手效果尚待审核，人工分数空。')
  entry['metrics']={'r18':read(src/'r18_metrics.json'),'r14':read(src/'r14_metrics.json'),'human_verdict':None};cases.append(entry)
 if contacts_only:print('CONTACTS',done);return
 assert len(cases)==19
 summary=read(O/'results_summary.json');dump(dest/'results_summary.json',summary)
 data={'mode':'effects','cases':cases,'roles':[{'key':k,'label':v} for k,v in [('target','真实GT／黄框待删目标'),('input','冻结输入与生成范围'),('base','原始DriveEditor'),('r7','有效r7'),('r8','r8覆盖版'),('r10','r10空间self attention'),('r14','r14时间self attention'),('r18','r18恢复区域损失')]],'score_roles':[{'key':a,'label':a} for a in plan['arms']]}
 template=(P/'iteration9/review_template.html').read_text().replace('v77-r8-','v77-r18-').replace("run:'r8'","run:'r18'").replace('v77_r8_','v77_r18_').replace('<a href="data_review.html">造数据逐帧审核</a>','<a href="../v77-target-protected-r10/data_review.html">保留输入逐帧审核</a>')
 arch='<div class="box"><svg viewBox="0 0 1100 190" role="img" aria-label="恢复区域损失控制组件"><defs><marker id="lossarr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#91cfff"/></marker></defs><g fill="#233e59" stroke="#789bbd"><rect x="10" y="50" width="210" height="90" rx="8"/><rect x="300" y="50" width="230" height="90" rx="8"/><rect x="610" y="50" width="210" height="90" rx="8"/><rect x="890" y="50" width="200" height="90" rx="8"/></g><g fill="white" text-anchor="middle" font-size="17"><text x="115" y="79">同50train／真实Y</text><text x="115" y="105">同洞H／B仅训练标签</text><text x="115" y="128">X先擦除合成RGB</text><text x="415" y="79">原架构／时间张量80</text><text x="415" y="105">global＋背景洞＋保护洞</text><text x="415" y="128">160步／49.57M</text><text x="715" y="80">同11GT＋8真实DEV</text><text x="715" y="107">默认CFG／seed42</text><text x="990" y="80">六权重十帧对比</text><text x="990" y="107">原生另链／人工空</text></g><g stroke="#91cfff" stroke-width="2" marker-end="url(#lossarr)"><path d="M220,95H295"/><path d="M530,95H605"/><path d="M820,95H885"/></g></svg></div>'
 start=template.index('<div class="box"><svg');end=template.index('</svg></div>',start)+len('</svg></div>');template=template[:start]+arch+template[end:]
 table='<table><tr><th>scene等权 MAE</th>'+''.join('<th>'+a+'</th>' for a in plan['arms'])+'<th>case／scene</th></tr>'
 for group,stats in summary['groups'].items():
  for metric,v in stats.items():table+='<tr><td>'+GROUPS[group]+'／'+metric+'</td>'+''.join(f'<td>{v["macro_scene"][a]:.5f}</td>' if v['macro_scene'][a] is not None else '<td>N/A</td>' for a in plan['arms'])+f'<td>{v["cases"]}／{v["scenes"]}</td></tr>'
 table+='</table>'
 decision=read(O/'decision.json')['summary'] if (O/'decision.json').exists() else '对照已运行，真实收益待审核；不能凭合成误差宣布通过。'
 intro=f'<p><strong>{decision}</strong></p><p>r18与r14同原结构、初始化、50train/25world、106目标encoder、160步、320×576和49.57M时间self attention参数；唯一改2D损失归约为每帧global/H-bg/H-protected存在项等权平均。B仅训练监督，不加网络输入。r17因未传B在87步工程停止，不用于本对照，r18从原始权重重启。旧五臂95窗逐帧核对RGB/H/Y后复用，新r18为19窗。</p>{table}<p>合成GT指标与真实任务观察分开。真实8例是曝光开发例、隐藏真值未知；final未用。一秒窗，单帧不证明时序。局部写回洞外保持来自规则，不是神经保护能力；六臂原生输出均另链。每帧人工0/1/2留空。新三秒r16数据仅质量审核，未用于本轮训练。</p><p><a href="../v77-target-protected-r14/index.html">r14模块控制</a> · <a href="../v77-target-protected-r16/data_review.html">三秒数据质量／覆盖限制</a></p>'
 html=template.replace('__TITLE__','v77 r18 · 背景与保护车区域损失控制').replace('__INTRO__',intro).replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
 (dest/'index.html').write_text(html);dump(dest/'effect_manifest.json',data);print('HTML',len(cases))

if __name__=='__main__':main('--contacts-only' in sys.argv,'--html-only' in sys.argv)
