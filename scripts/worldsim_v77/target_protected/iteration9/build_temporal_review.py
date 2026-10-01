from pathlib import Path
import sys,json,copy
import numpy as np,cv2
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).parent));from temporal_factory import T,read,dump
from render_pairs import encode,font
O=T/'r10';R9=T/'r9';R8=T/'r8';P=Path(__file__).parent.parent/'iteration8'
TEMPLATE=(P/'review_template.html').read_text().replace('v77-r8-','v77-r10-').replace("run:'r8'","run:'r10'").replace('v77_r8_','v77_r10_').replace('原／r7／新数据','原／r7／r8／r10')
def page(path,title,intro,data):path.write_text(TEMPLATE.replace('__TITLE__',title).replace('__INTRO__',intro).replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/')))
def data_page():
 dest=O/'review';dest.mkdir(parents=True,exist_ok=True)
 for name in ['assets','contacts']:
  link=dest/('data_'+name)
  if not link.exists():link.symlink_to(R9/'data_review'/name,target_is_directory=True)
 qa={r['case_id']:r for r in read(R9/'independent_data_reviews.json')['cases']};catalog=read(O/'dataset_catalog.json');admitted={c['case_id']:c for c in catalog['cases'] if c['dataset_id'].startswith(('r10/','r10_val/'))};cases=[]
 for c in read(R9/'data_review/synthetic_manifest.json')['clips']:
  cid=c['case_id'];q=qa[cid];a=admitted.get(cid);group=a['split'] if a else '拒绝/待定/未使用';proc=c['temporal_process']
  cases.append({'id':cid,'scene':c['scene'],'type':c['process_family']+' / '+c['type'],'group':group,'note':f'{c["camera"]}；红A待擦除，绿B/蓝C只标实际活跃保护实例。A世界速度 {c["speed_signed_mps"]}m/s，实际窗口 {proc["duration_s"]:.2f}s。尺寸/位姿/十帧掩罩机器检查通过不代替视觉。','qa':f'独立AI {q["assistant_score"]}：{q["reason"]}；查看0/5/9，human未填。','review_frame':5,'videos':{r:'data_'+v for r,v in c['videos'].items()},'frame_pattern':f'data_assets/{cid}/{{i}}_{{role}}.jpg','metrics':{'temporal_process':proc,'pixel_checks':c['pixel_metrics'],'active_protected_tokens':c['protected_instances'],'human_verdict':None}})
 data={'mode':'data','roles':[{'key':k,'label':v} for k,v in [('gt','真实Y：恢复GT'),('input','灰A：完整合成X示意'),('labels','红A / 活跃真实B/C'),('condition','模型实际遮洞条件')]],'score_roles':[{'key':'data','label':'输入质量'}],'cases':cases}
 page(dest/'data_review.html','v77 遮挡过程 · 新22例逐帧输入审核','<p>新22候选全保留；AI通过才进入有限过程pilot。待定/拒绝不训练，旧标注和全部原数据保存。静止A+ego运动不等于相对扫过B；扫过代理还需实际三帧视觉支持。GT cuboid格只提供几何对应代理，不认证真实纹理已见；background证据未验证时保持未知。灰车RGB完整擦除，Y永远真实。</p>',data);dump(dest/'data_manifest.json',data)
def main(html_only=False):
 cv2.setNumThreads(1);dest=O/'review';dest.mkdir(exist_ok=True);(dest/'effects').mkdir(exist_ok=True);(dest/'effect_contacts').mkdir(exist_ok=True);plan=read(O/'evaluation_plan.json');s=read(O/'results_summary.json');qa={r['eval_id']:r for r in read(O/'assistant_effect_reviews.json').get('cases',[])} if (O/'assistant_effect_reviews.json').exists() else {};cases=[]
 for c in plan['cases']:
  cid=c['eval_id'];src=O/'evaluation'/cid;out=dest/'effects'/cid;out.mkdir(exist_ok=True);sy=c['kind']=='synthetic';panels=[]
  for i in ([] if html_only else range(10)):
   load=lambda r:np.asarray(Image.open(src/r/f'{i:05}.png').convert('RGB'));im=load('input');h=np.asarray(Image.open(src/'mask'/f'{i:05}.png'))>0;target=load('GT') if sy else im.copy();shown=im.copy()
   if not sy:
    paths=sorted((Path(c['folder'])/'core').glob('*.png'));core=np.asarray(Image.open(paths[c['frames'][i]]))>0;yy,xx=np.where(core)
    if len(xx):cv2.rectangle(target,(int(xx.min()),int(yy.min())),(int(xx.max()),int(yy.max())),(255,220,30),2)
    cv2.putText(target,f'{c["clip_id"]} target actor {c["actor_ordinal_in_scene"]}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,230,40),2);tint=np.zeros_like(im);tint[:,:,2]=255;shown[h]=np.rint(.75*shown[h]+.25*tint[h]).astype('uint8')
   panel={'target':target,'input':shown,**{a:load(a) for a in plan['arms']}};panels.append(panel)
   for k,a in panel.items():Image.fromarray(a).save(out/f'{i:03}_{k}.jpg',quality=96)
   for a in plan['arms']:Image.fromarray(load(a+'_native')).save(out/f'{i:03}_{a}_native.jpg',quality=96)
  if not html_only:
   for role in ['target','input']+plan['arms']+[a+'_native' for a in plan['arms']]:encode(out,role)
   i=5;hh=np.asarray(Image.open(src/'mask'/f'{i:05}.png'))>0;yy,xx=np.where(hh);box=np.array([xx.min()-20,yy.min()-20,xx.max()+21,yy.max()+21]);box[[0,2]]=box[[0,2]].clip(0,1024);box[[1,3]]=box[[1,3]].clip(0,576)
   sheet=Image.new('RGB',(2400,900),(15,22,33));d=ImageDraw.Draw(sheet);d.text((8,8),f'{cid} | {c.get("receiver_scene",c.get("scene"))} | f5 | {c["suite"]}',font=font(22),fill='white')
   for j,(role,arr) in enumerate(panels[i].items()):
    d.text((j*400+8,42),role,font=font(),fill='white');im=Image.fromarray(arr);sheet.paste(im.resize((400,225)),(j*400,72));crop=im.crop(tuple(box));crop.thumbnail((396,550));sheet.paste(crop,(j*400+(400-crop.width)//2,330))
   sheet.save(dest/'effect_contacts'/f'{cid}_f05.jpg',quality=96)
  else:
   for role in ['target','input']+plan['arms']+[a+'_native' for a in plan['arms']]:assert (out/(role+'.mp4')).exists()
  note=(f'真实Y是恢复GT；过程 {c.get("process_family","r8稳定遮挡验证")}，{c["type"]}。灰车完整X仅示意。' if sy else f'删除 {c["clip_id"]} actor {c["actor_ordinal_in_scene"]}；{c["camera"]}；源帧 {c["frames"][0]}–{c["frames"][-1]}。黄框只标目标身份；蓝区生成范围，真实隐藏GT未知。')
  q=qa.get(cid);qtext=q['observations'] if q else '助手固定帧观察尚未填写，人工始终空。'
  metrics=next((r for r in s['synthetic_cases'] if r['eval_id']==cid),{'actor_free_GT':None,'input_difficulty_proxy':c.get('input_difficulty_proxy'),'human_verdict':None})
  cases.append({'id':cid,'scene':c.get('receiver_scene',c.get('scene')),'type':c.get('process_family',c.get('type',c['kind'])),'group':c['suite'],'note':note,'qa':qtext,'review_frame':5,'videos':{r:f'effects/{cid}/{r}.mp4' for r in ['target','input']+plan['arms']},'frame_pattern':f'effects/{cid}/{{i}}_{{role}}.jpg','native_links':{a:f'effects/{cid}/{a}_native.mp4' for a in plan['arms']},'metrics':metrics})
 data={'mode':'effects','roles':[{'key':k,'label':v} for k,v in [('target','真实GT／原视频黄框目标'),('input','合成X／真实生成mask蓝区'),('base','原始DriveEditor'),('r7','有效r7'),('r8','r8数据覆盖版'),('r10','r10遮挡过程pilot')]],'score_roles':[{'key':a,'label':a} for a in plan['arms']],'cases':cases}
 table='<table><tr><th>分组 / 指标（scene等权MAE）</th>'+''.join('<th>'+a+'</th>' for a in plan['arms'])+'<th>case / scene</th></tr>'
 for group,g in s['groups'].items():
  for key,v in g.items():table+='<tr><td>'+group+' / '+key+'</td>'+''.join(f'<td>{v["macro_scene"][a]:.5f}</td>' if v['macro_scene'][a] is not None else '<td>N/A</td>' for a in plan['arms'])+f'<td>{v["cases"]} / {v["scenes"]}</td></tr>'
 table+='</table>';sweep=s['process_groups'].get('sweep_B',{});intro=f'<p><strong>本轮是有限遮挡过程pilot，尚不认证真实DELETE迁移。</strong>r7/r8训练均无达到阈值的扫过B过程；r8已有12例静止A＋ego运动。r9两次有界搜索未满足原覆盖目标，r9训练0步；r10只用实际技术＋独立AI2的新过程样本与r8稳定控制，同80张量160步，不改结构、loss、推理条件。</p><p>训练数据：{s["data"]["training_cases"]}例 / {s["data"]["training_scene_count"]}scene；过程 {s["data"]["training_process_counts"]}。新增扫过验证只 {s["data"]["new_sweep_validation_worlds"]} 个独立world，不能凭它声称充分泛化。其他帧证据仅GT cuboid表面格几何代理；真实隐藏纹理/身份无GT，未知保持未知。</p>{table}<p>旧合成组、新过程组分开报告，不混在总体误差里掩盖退化。真实8例为已曝光DEV；全部不是Ω重建，也没有factual重建栏。只写回原生成范围，洞外原像素保持不能当神经保护能力。每例完整10帧同步视频和逐帧空白人工评分，原生输出另链；自动单帧观察不等于视频通过率。</p><p>原模型/r7/r8/checkpoint、拒绝/待定全部保留；一秒窗口、两种共享DEV轮廓、Boston白天/GT辅助、稀疏dense覆盖均明示。没有使用final、额外seed、加权或扩大微调模块。</p>'
 if (O/'decision.json').exists():intro='<p><strong>'+read(O/'decision.json')['summary']+'</strong></p>'+intro
 page(dest/'index.html','v77 r10 · 遮挡过程同预算四权重对照',intro,data);dump(dest/'effect_manifest.json',data);dump(dest/'results_summary.json',s);data_page();print('HTML',len(cases))
if __name__=='__main__':main('--html-only' in sys.argv)
