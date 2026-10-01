"""同输入五权重审核；复用已交付旧视频，只编码新时序权重。"""
from pathlib import Path
import sys,os,json
import numpy as np,cv2
from PIL import Image,ImageDraw
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(P));sys.path.insert(0,str(P/'iteration9'))
from temporal_factory import T,read,dump
from render_pairs import font,encode
O=T/'r14';OLD=T/'r10'
GROUPS={'r8_frozen_synthetic':'旧合成GT验证','new_temporal_validation':'新过程GT验证','real_DEVELOPMENT':'真实DELETE开发例'}

def main(contacts_only=False,html_only=False):
 cv2.setNumThreads(1);dest=O/'review';dest.mkdir(exist_ok=True);(dest/'effects').mkdir(exist_ok=True);(dest/'contacts').mkdir(exist_ok=True)
 plan=read(O/'evaluation_plan.json');cases=[];done=0
 oldmanifest={c['id']:c for c in read(OLD/'review/effect_manifest.json')['cases']}
 qa=read(O/'assistant_effect_reviews.json') if (O/'assistant_effect_reviews.json').exists() else {'cases':[]}
 for c in plan['cases']:
  cid=c['eval_id'];src=O/'evaluation'/cid
  if not (src/'r14_metrics.json').exists():continue
  done+=1;out=dest/'effects'/cid;out.mkdir(exist_ok=True)
  panels=[]
  for i in ([] if html_only else range(10)):
   panel={}
   for role in ['target','input','base','r7','r8','r10']:
    old=OLD/'review/effects'/cid/f'{i:03}_{role}.jpg';panel[role]=np.asarray(Image.open(old).convert('RGB')).copy()
    if not contacts_only and not (out/old.name).exists():os.link(old,out/old.name)
   for role in ['r14','r14_native']:
    a=np.asarray(Image.open(src/role/f'{i:05}.png').convert('RGB'));panel[role]=a
    if not contacts_only:Image.fromarray(a).save(out/f'{i:03}_{role}.jpg',quality=96)
   panels.append(panel)
  if not html_only:
   for name,ids in [('f5',[5]),('first5',list(range(5))),('last5',list(range(5,10)))]:
    sheet=Image.new('RGB',(2800,42+len(ids)*300),(16,23,34));d=ImageDraw.Draw(sheet);d.text((8,7),f'{cid} | {c.get("receiver_scene",c.get("scene"))} | r10 spatial / r14 temporal equal 80 tensors | {ids}',font=font(21),fill='white')
    for row,i in enumerate(ids):
     h=np.asarray(Image.open(src/'mask'/f'{i:05}.png'))>0;yy,xx=np.where(h);box=(max(0,int(xx.min())-30),max(0,int(yy.min())-30),min(1024,int(xx.max())+31),min(576,int(yy.max())+31))
     for j,role in enumerate(['target','input','base','r7','r8','r10','r14']):
      im=Image.fromarray(panels[i][role]).crop(box);scale=min(394/im.width,256/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS);d.text((j*400+6,45+row*300),f'f{i} {role}',font=font(18),fill='white');sheet.paste(im,(j*400+(400-im.width)//2,74+row*300))
    sheet.save(dest/'contacts'/f'{cid}_{name}.jpg',quality=96)
  if contacts_only:continue
  if not html_only:
   for old in (OLD/'review/effects'/cid).glob('*'):
    if old.is_file() and not (out/old.name).exists():os.link(old,out/old.name)
   for role in ['r14','r14_native']:encode(out,role)
  for role in ['target','input']+plan['arms']+[a+'_native' for a in plan['arms']]:assert (out/(role+'.mp4')).exists()
  entry=dict(oldmanifest[cid]);entry.update(videos={r:f'effects/{cid}/{r}.mp4' for r in ['target','input']+plan['arms']},native_links={a:f'effects/{cid}/{a}_native.mp4' for a in plan['arms']},frame_pattern=f'effects/{cid}/{{i}}_{{role}}.jpg',contact=f'contacts/{cid}_f5.jpg')
  entry['note']+=' r14与r10训练集合、预算和49,574,080参数数量相同，仅空间self attention换为时间self attention；原模型初始化，默认CFG不变。'
  entry['qa']=next((r['observations'] for r in qa['cases'] if r['eval_id']==cid),'助手效果尚待审核，人工分数空。')
  entry['metrics']={'r14':read(src/'r14_metrics.json'),'r10':read(src/'r10_metrics.json'),'human_verdict':None};cases.append(entry)
 if contacts_only:print('CONTACTS',done);return
 assert len(cases)==19
 summary=read(O/'results_summary.json');dump(dest/'results_summary.json',summary)
 data={'mode':'effects','cases':cases,'roles':[{'key':k,'label':v} for k,v in [('target','真实GT／黄框待删目标'),('input','冻结输入与生成范围'),('base','原始DriveEditor'),('r7','有效r7'),('r8','r8覆盖版'),('r10','r10空间self attention'),('r14','r14时间self attention')]],'score_roles':[{'key':a,'label':a} for a in plan['arms']]}
 template=(P/'iteration9/review_template.html').read_text().replace('v77-r8-','v77-r14-').replace("run:'r8'","run:'r14'").replace('v77_r8_','v77_r14_').replace('<a href="data_review.html">造数据逐帧审核</a>','<a href="../v77-target-protected-r10/data_review.html">保留输入逐帧审核</a>')
 arch='''<div class="box"><svg viewBox="0 0 1100 190" role="img" aria-label="同数量更新模块控制"><defs><marker id="scopearr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#91cfff"/></marker></defs><g fill="#233e59" stroke="#789bbd"><rect x="10" y="60" width="190" height="70" rx="8"/><rect x="270" y="10" width="230" height="65" rx="8"/><rect x="270" y="105" width="230" height="65" rx="8"/><rect x="590" y="60" width="200" height="70" rx="8"/><rect x="870" y="60" width="220" height="70" rx="8"/></g><g fill="white" text-anchor="middle" font-size="17"><text x="105" y="87">同一50train真实Y</text><text x="105" y="112">同一遮洞条件</text><text x="385" y="37">r10：空间 self attention</text><text x="385" y="61">80张量／49.57M／160步</text><text x="385" y="132">r14：时间 self attention</text><text x="385" y="156">80张量／49.57M／160步</text><text x="690" y="88">同11GT＋8真实DEV</text><text x="690" y="112">默认CFG／seed42</text><text x="980" y="88">十帧视频／原生另链</text><text x="980" y="112">逐帧人工评分空</text></g><g stroke="#91cfff" stroke-width="2" fill="none" marker-end="url(#scopearr)"><path d="M200,95H235V43H265"/><path d="M235,95V137H265"/><path d="M500,43H545V95H585"/><path d="M500,137H545V95"/><path d="M790,95H865"/></g></svg></div>'''
 start=template.index('<div class="box"><svg');end=template.index('</svg></div>',start)+len('</svg></div>');template=template[:start]+arch+template[end:]
 table='<table><tr><th>scene等权 MAE</th>'+''.join('<th>'+a+'</th>' for a in plan['arms'])+'<th>case／scene</th></tr>'
 for group,stats in summary['groups'].items():
  for metric,v in stats.items():table+='<tr><td>'+GROUPS[group]+'／'+metric+'</td>'+''.join(f'<td>{v["macro_scene"][a]:.5f}</td>' if v['macro_scene'][a] is not None else '<td>N/A</td>' for a in plan['arms'])+f'<td>{v["cases"]}／{v["scenes"]}</td></tr>'
 table+='</table>'
 decision=read(O/'decision.json')['summary'] if (O/'decision.json').exists() else '对照已运行，真实收益待审核；不能凭合成误差宣布通过。'
 intro=f'<p><strong>{decision}</strong></p><p>同原结构、初始化、50train/25world、106目标encoder、160步、320×576和49.57M参数；唯一空间→时间self attention。保持默认CFG，未叠加r13。旧四臂76窗核对每帧RGB/H/Y后复用，19窗新r14。没有新增训练数据。</p>{table}<p>旧合成、新过程分别统计，MAE不等于实体身份或视频通过率。真实8例是已曝光开发例、隐藏GT未知；final未用。GT有地图/LiDAR辅助，一秒窗、扫过val只有一个world、dense train只有一个world等覆盖限制仍在。局部写回洞外像素保持来自合成规则，不能当神经保护能力。全部原生视频另链。人工0/1/2为空。</p><p><a href="../v77-target-protected-r13/index.html">同权重CFG1负对照</a> · <a href="../v77-target-protected-r10/index.html">r10与输入质量完整报告</a></p>'
 html=template.replace('__TITLE__','v77 r14 · 等参数量时序模块控制').replace('__INTRO__',intro).replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
 (dest/'index.html').write_text(html);dump(dest/'effect_manifest.json',data);print('HTML',len(cases))

if __name__=='__main__':main('--contacts-only' in sys.argv,'--html-only' in sys.argv)
