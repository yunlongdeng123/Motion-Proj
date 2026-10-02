"""r43全量空间候选三列视频；技术拒绝也展示，人工分与AI分分开。"""
from pathlib import Path
import sys,json,html,shutil
import numpy as np
import cv2
from PIL import Image,ImageDraw
sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).parents[1])]
from geometry_factory import read,dump
from build_role_condition_review import encode
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r43';OUT=O/'review'
REASONS={'protected_depth_order':'A没有位于实际受影响对象前方','insufficient_other_frame_evidence':'其他输入帧中没有足够显露依据','no_reveal_process':'未形成规定的遮挡变化',
 'insufficient_actual_occlusion':'实际遮挡不足','actual_size':'实际轮廓尺寸未达要求','nonvehicle_or_incomplete_protection':'涉及非车辆或缺少完整位姿的对象',
 'instance_identity_or_visibility_uncertain':'实例身份或可见性标签有疑点','static_foreground':'涉及靠前的静态物体','ground_fit_unavailable':'没有可靠地面支持',
 'size_border_or_ego':'尺寸、画面边界或ego保护未通过','ground_support_gap':'局部地面支持不足','collision_clearance':'世界坐标间距不足'}
def esc(v):return html.escape(str(v))


def panels(c,i):
    root=Path(c['observed_folder']);y=np.asarray(Image.open(Path(c['source_Y_quality_only'])/f'{i:05}.png')).copy()
    a=np.asarray(Image.open(root/'influence'/f'{i:05}.png'))>0;h=np.asarray(Image.open(root/'proposal_H'/f'{i:05}.png'))>0
    original_x=np.asarray(Image.open(root/'rgb'/f'{i:05}.png'));assert h[a].all() and not h[512:].any() and np.array_equal(original_x[~h],y[~h])
    x=y.copy();x[h]=127;original=y.copy();plan=y.copy();plan[a]=np.rint(plan[a]*.3+np.array([255,195,30])*.7).astype('uint8')
    cv2.drawContours(plan,cv2.findContours(h.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(240,80,220),2)
    primary=c['frames'][i]['actors'][0];bb=primary['projection']['box_xyxy'];token=primary['instance_token']
    targets=[(token,bb,True)]+[(ob['instance_token'],ob['boxes'][i],False) for ob in c['retained_instances'] if ob['instance_token']!=token and ob['boxes'][i] is not None]
    for tok,bb,main in targets:
        b=np.rint(bb).astype(int);color=(35,245,120) if main else (75,165,250)
        for im in [original,plan]:
            cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),color,2);cv2.putText(im,('source B ' if main else 'check ')+tok[:8],(max(1,b[0]),max(15,b[1]-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1,cv2.LINE_AA)
    return original,plan,x


def main():
    cv2.setNumThreads(1);OUT.mkdir(exist_ok=True)
    data=read(O/'prepared.json')['cases'];result=read(O/'instance_quality.json');rows={r['case_id']:r for r in result['cases']}
    qa={r['case_id']:r for r in read(O/'independent_quality.json')['cases']} if (O/'independent_quality.json').exists() else {}
    cards=[];manifest=[]
    for c in data:
        cid=c['case_id'];r=rows[cid];dest=OUT/cid;dest.mkdir(exist_ok=True);frames=[panels(c,i) for i in range(30)];videos=[]
        for k,role in enumerate(['Y','plan','input']):
            encode(dest/f'{role}.mp4',(p[k] for p in frames));videos.append(f'{cid}/{role}.mp4')
        # 所有30时刻均可逐帧审，不用三张抽帧推断时序稳定。
        sheets=[]
        for start in [0,10,20]:
            sheet=Image.new('RGB',(1536,1620),'#15202e');draw=ImageDraw.Draw(sheet)
            for j,i in enumerate(range(start,start+10)):
                col,row=j%2,j//2
                for k,im in enumerate(frames[i]):sheet.paste(Image.fromarray(im).resize((256,144)),(col*768+k*256,row*324+35))
                draw.text((col*768+6,row*324+7),f'{cid} f{i}: real Y | A/H | actual masked X',fill='white')
            name=f'frames_{start:02}_{start+9:02}.jpg';sheet.save(dest/name,quality=95);sheets.append(f'{cid}/{name}')
        grade=qa.get(cid,{}).get('assistant_grade','待独立QA');note=qa.get(cid,{}).get('reason','技术检查不能替代独立图像复核。')
        technical='技术候选，仍需独立输入与条件QA2' if r['technical_candidate'] else REASONS.get(r['reason'],r['reason'])
        media='<div class="grid">'+''.join(f'<div><b>{title}</b><video src="{cid}/{role}.mp4" controls muted playsinline preload="none"></video></div>' for role,title in [('Y','真实Y：训练监督／离线质检'),('plan','黄色合成A／紫色最终删除洞H'),('input','实际输入X：灰洞尚未补景')])+'</div>'
        links=''.join(f'<details><summary>{esc(Path(path).name)}</summary><a href="{path}"><img src="{path}" loading="lazy"></a></details>' for path in sheets)
        cards.append(f'<article data-case="{cid}" id="{cid}"><h2>{cid} · {esc(c["scene"])} · {esc(c["source_id"])} · {esc(c["split"])}</h2><p><strong>技术检查：{esc(technical)}；独立AI：{esc(grade)}</strong></p><p>{esc(note)}</p><p>{esc(c["camera"])}，30个真实曝光；A在世界中静止。GT间距最小{c["trajectory"]["min_GT_clearance_m"]:.2f}m仅代表包络间距，不等于完整道路合法或遮挡正确。</p>{media}<button class="play">同步播放／暂停</button><input class="seek" type="range" min="0" max="29" value="0"><span class="frame">f0</span><label>人工评分 <select class="score"><option value="">未评</option><option>0</option><option>1</option><option>2</option></select></label><input class="note" placeholder="人工备注">{links}</article>')
        manifest.append({'case_id':cid,'source_id':c['source_id'],'scene':c['scene'],'split':c['split'],'videos':videos,'images':sheets,
            'technical_candidate':r['technical_candidate'],'technical_reason':r['reason'],'assistant_grade':grade,'human_verdict':None,'model_inference':False,'training_admission':False})
    dump(OUT/'manifest.json',{'cases':manifest,'scope':'all spatial candidates including technical rejects','human_verdict':None,'training_steps':0})
    for name in ['run.json','source_summary.json','space_summary.json','instance_quality.json','gpu_plan.json','independent_quality.json']:
        if (O/name).exists():shutil.copy2(O/name,OUT/name)
    source_table=[]
    for path in sorted((O/'source_proposals').glob('*.json')):
        r=read(path);source_table.append(f'<tr><td>{esc(r["source_id"])}</td><td>{esc(r["scene"])}</td><td>{r["attempts"]}</td><td>{len(r["spatial_trajectories"])}</td><td>{esc("；".join(REASONS.get(k,k)+" "+str(v) for k,v in r["rejects"].items()))}</td></tr>')
    script="""const key='v77-r43-human-v1';let saved={};try{saved=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}document.querySelectorAll('article').forEach(a=>{const id=a.dataset.case,v=[...a.querySelectorAll('video')],s=a.querySelector('.score'),n=a.querySelector('.note');s.value=saved[id]?.score??'';n.value=saved[id]?.note??'';const save=()=>{saved[id]={score:s.value,note:n.value};localStorage.setItem(key,JSON.stringify(saved))};s.onchange=save;n.oninput=save;a.querySelector('.play').onclick=()=>{const go=v[0].paused,t=v[0].currentTime;v.forEach(x=>{x.currentTime=t;go?x.play().catch(()=>{}):x.pause()})};a.querySelector('.seek').oninput=e=>{const f=+e.target.value;a.querySelector('.frame').textContent='f'+f;v.forEach(x=>{x.pause();x.currentTime=f/10})}});document.querySelector('#export').onclick=()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(saved,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='v77_r43_human_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};"""
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v77 r43 · 空间先行数据工厂</title><style>body{{background:#121923;color:#edf2f6;font:16px/1.6 system-ui;max-width:1560px;margin:30px auto;padding:0 20px}}article{{border:1px solid #41536b;padding:20px;border-radius:10px;margin:24px 0}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}video,img{{width:100%;background:#080c12}}.flow{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}.flow span{{background:#263b55;padding:12px;border:1px solid #587999;border-radius:8px}}p{{color:#bccedf}}a{{color:#86c7ff}}button,input,select{{background:#283a50;color:white;border:1px solid #60738c;border-radius:5px;padding:7px;margin:8px}}.seek{{width:25%}}table{{border-collapse:collapse}}td,th{{padding:6px 15px;border-bottom:1px solid #40526a}}.notice{{padding:12px;background:#30301c;border-left:4px solid #e5be68}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}</style><h1>r43：先确认空间可用，再做实例与时序质检</h1><p class="notice">这里展示数据工厂候选与被拒原因。第三列是模型输入的灰洞，不是DriveEditor补景输出。本轮尚无新增训练或真实DELETE收益；人工评分保持空白。</p><div class="flow"><span>真实30帧Y<br>仅监督／离线质检</span>→<span>固定6来源<br>清楚静止B＋运动ego</span>→<span>道路／LiDAR／碰撞<br>先筛实际A空间</span>→<span>最终洞H＋SAM实例检查<br>独立QA</span>⇢<span>合法条件与训练<br>未执行</span></div><p>r42取消旧每scene前三窗口截断后，候选440→612，仍未获得既定过程来源。r43整体移除“B全程至少16米”的距离代理，交给实际空间检查；清晰度与时序要求保持。下面保留6个来源的完整分母，以及全部{len(data)}个空间候选，技术失败也展示。</p><table><tr><th>来源</th><th>scene</th><th>空间尝试</th><th>空间候选</th><th>拒绝原因</th></tr>{''.join(source_table)}</table><p>绿色框是来源筛选时选中的真实B；蓝框是最终H附近需核对的其他实例包络；黄色才是我们合成并准备删除的A。A的假RGB不作为评分依据；需判断轮廓、放置、深度顺序、时序显露，以及最终H是否完整覆盖A。</p><p>完整实例技术结果：{esc(json.dumps(result['counts'],ensure_ascii=False))}。空间无碰撞不能代替真实遮挡正确；完整Y不会进入后续actor-state条件。</p><p><a href="instance_quality.json">全部实例检查</a> · <a href="run.json">冻结规则</a> · <a href="manifest.json">媒体与空白人工评分索引</a> · <a href="../v77-target-protected-r38/index.html">旧r38与后续失败对照</a></p><button id="export">导出人工评分</button>{''.join(cards)}<script>{script}</script></html>'''
    (OUT/'index.html').write_text(page);print('SPACE_FIRST_REVIEW',len(data),len(data)*3,flush=True)


if __name__=='__main__':main()
