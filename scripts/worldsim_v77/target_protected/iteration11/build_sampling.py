"""固定权重采样对照审核；原始输出和完整十帧保留。"""
from pathlib import Path
import sys,json
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(P));sys.path.insert(0,str(P/'iteration9'))
from temporal_factory import T,read,dump
from render_pairs import encode,font
from PIL import Image,ImageDraw
import numpy as np,cv2
O=T/'r13'
def main(contacts_only=False):
 cv2.setNumThreads(1);dest=O/'review';dest.mkdir(exist_ok=True);(dest/'contacts').mkdir(exist_ok=True);rows=[]
 for c in read(O/'evaluation_plan.json')['cases']:
  cid=c['eval_id'];src=O/'evaluation'/cid;prev=T/'r10/evaluation'/cid
  if not (src/'r7_cfg1_metrics.json').exists():continue
  out=dest/'effects'/cid;out.mkdir(parents=True,exist_ok=True);panels=[]
  for i in range(10):
   load=lambda folder:np.asarray(Image.open(folder/f'{i:05}.png').convert('RGB')).copy()
   x=load(prev/'input');y=load(prev/'GT') if c['kind']=='synthetic' else x.copy();h=np.asarray(Image.open(prev/'mask'/f'{i:05}.png'))>0
   blue=x.copy();blue[h]=np.rint(.7*blue[h]+.3*np.array([0,100,255])).astype('uint8')
   if c['kind']!='synthetic':
    core=np.asarray(Image.open(sorted((Path(c['folder'])/'core').glob('*.png'))[c['frames'][i]]))>0;yy,xx=np.where(core)
    if len(xx):cv2.rectangle(y,(int(xx.min()),int(yy.min())),(int(xx.max()),int(yy.max())),(255,220,30),2)
   panel={'original':y,'mask':blue,'r7_default':load(prev/'r7'),'r7_cfg1':load(src/'r7_cfg1')};panels.append(panel)
   if not contacts_only:
    for role,a in panel.items():Image.fromarray(a).save(out/f'{i:03}_{role}.jpg',quality=94)
    for role,folder in [('r7_default_native',prev/'r7_native'),('r7_cfg1_native',src/'r7_cfg1_native')]:Image.fromarray(load(folder)).save(out/f'{i:03}_{role}.jpg',quality=94)
  for name,ids in [('f5',[5]),('first5',list(range(5))),('last5',list(range(5,10)))]:
   sheet=Image.new('RGB',(1600,40+len(ids)*310),(16,23,34));d=ImageDraw.Draw(sheet);d.text((10,5),f'{cid} | same r7 weight / seed42 / 25steps | frame {ids}',font=font(20),fill='white')
   for row,i in enumerate(ids):
    h=np.asarray(Image.open(prev/'mask'/f'{i:05}.png'))>0;yy,xx=np.where(h);box=(max(0,int(xx.min())-30),max(0,int(yy.min())-30),min(1024,int(xx.max())+31),min(576,int(yy.max())+31))
    for j,(role,a) in enumerate(panels[i].items()):
     im=Image.fromarray(a).crop(box);scale=min(394/im.width,270/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS);d.text((j*400+5,40+row*310),f'f{i} {role}',font=font(18),fill='white');sheet.paste(im,(j*400+(400-im.width)//2,68+row*310))
   sheet.save(dest/'contacts'/f'{cid}_{name}.jpg',quality=96)
  if contacts_only:continue
  for role in ['original','mask','r7_default','r7_cfg1','r7_default_native','r7_cfg1_native']:encode(out,role)
  qa=read(O/'assistant_effect_reviews.json') if (O/'assistant_effect_reviews.json').exists() else {'cases':[]}
  q=next((r['observations'] for r in qa['cases'] if r['eval_id']==cid),'助手结果待检查。')
  rows.append({'id':cid,'scene':c.get('receiver_scene',c.get('scene')),'type':'已知真实Y的合成遮挡' if c['kind']=='synthetic' else '真实DELETE','group':'合成GT哨兵' if c['kind']=='synthetic' else '真实开发例','note':'同r7权重、同输入/seed/25steps，仅2D CFG线性1.2→2.0对固定1.0；真实隐藏GT未知。旧其它权重保留在r10审核。','qa':q+' 人工评分始终为空。此窗10帧约0.9秒，不能当三秒或final结论。','review_frame':5,'contact':f'contacts/{cid}_f5.jpg','videos':{r:f'effects/{cid}/{r}.mp4' for r in ['original','mask','r7_default','r7_cfg1']},'frame_pattern':f'effects/{cid}/{{i}}_{{role}}.jpg','native_links':{r:f'effects/{cid}/{r}_native.mp4' for r in ['r7_default','r7_cfg1']},'metrics':{'new':read(src/'r7_cfg1_metrics.json'),'default':read(prev/'r7_metrics.json'),'human_verdict':None}})
 if contacts_only:print('CONTACTS',sum((O/'evaluation'/c['eval_id']/'r7_cfg1_metrics.json').exists() for c in read(O/'evaluation_plan.json')['cases']));return
 assert len(rows)==10;data={'mode':'effects','cases':rows,'roles':[{'key':k,'label':v} for k,v in [('original','真实GT／原视频黄框目标'),('mask','冻结生成范围蓝区'),('r7_default','同r7：原线性CFG'),('r7_cfg1','同r7：固定CFG1控制')]],'score_roles':[{'key':r,'label':r} for r in ['r7_default','r7_cfg1']]}
 template=(P/'iteration9/review_template.html').read_text().replace('v77-r8-','v77-r13-').replace("run:'r8'","run:'r13'").replace('v77_r8_','v77_r13_')
 architecture='''<div class="box"><svg viewBox="0 0 1060 130" role="img" aria-label="采样控制组件图"><defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#91a7bd"/></marker></defs><g fill="#183049" stroke="#7698b5"><rect x="10" y="35" width="170" height="60" rx="8"/><rect x="230" y="35" width="200" height="60" rx="8"/><rect x="480" y="35" width="170" height="60" rx="8"/><rect x="700" y="35" width="150" height="60" rx="8"/><rect x="900" y="35" width="150" height="60" rx="8"/></g><g fill="#edf5ff" text-anchor="middle" font-size="16"><text x="95" y="60">冻结 RGB＋H</text><text x="95" y="82">先擦洞再条件编码</text><text x="330" y="60">同一 r7 权重</text><text x="330" y="82">seed42／25steps</text><text x="565" y="60">线性CFG／CFG1</text><text x="565" y="82">只改2D guider</text><text x="775" y="60">同范围写回</text><text x="775" y="82">另存原生输出</text><text x="975" y="60">十帧同步审核</text><text x="975" y="82">人工分数空</text></g><g stroke="#91a7bd" stroke-width="2" marker-end="url(#arr)"><path d="M180,65H225"/><path d="M430,65H475"/><path d="M650,65H695"/><path d="M850,65H895"/></g></svg></div>'''
 start=template.index('<div class="box"><svg');end=template.index('</svg></div>',start)+len('</svg></div>');template=template[:start]+architecture+template[end:];template=template.replace('<a href="data_review.html">造数据逐帧审核</a>','<a href="../v77-target-protected-r10/index.html">原/r7/r8/r10对照</a>')
 intro='<p>单一预登记采样控制，未重新训练、不改变architecture/输入/seed/step/3D guider。8真实曝光DEV与2已知GT哨兵。原生输出另链，洞外局部写回保留原像素不能当神经保护能力；未取得真实跨例收益之前不关机。人工评分空。</p>'
 html=template.replace('__TITLE__','v77 r13 · 同权重采样对照').replace('__INTRO__',intro).replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('</','<\\/'))
 (dest/'index.html').write_text(html);dump(dest/'effect_manifest.json',data);print('HTML',len(rows))
if __name__=='__main__':main('--contacts-only' in sys.argv)
