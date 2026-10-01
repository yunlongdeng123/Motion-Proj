"""模型对照：真实输入独立于补景输出，保留native与所有十帧。"""
from pathlib import Path
import sys,json,subprocess,os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,cv2
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,read,dump
from build_data_review import page
from render_pairs import encode,font

def main():
    cv2.setNumThreads(1);dest=O/'review';dest.mkdir(exist_ok=True);assets=dest/'effects';assets.mkdir(exist_ok=True);(dest/'effect_contacts').mkdir(exist_ok=True)
    plan=read(O/'evaluation_plan.json');s=read(O/'results_summary.json');qa=read(O/'assistant_effect_reviews.json') if (O/'assistant_effect_reviews.json').exists() else {'cases':[]};byid={c['eval_id']:c for c in qa['cases']};cases=[]
    for c in plan['cases']:
        cid=c['eval_id'];src=O/'evaluation'/cid;folder=assets/cid;folder.mkdir(exist_ok=True)
        synthetic=c['kind']=='synthetic';frames=[]
        for i in range(10):
            load=lambda r:np.asarray(Image.open(src/r/f'{i:05}.png').convert('RGB'))
            im=load('input');h=np.asarray(Image.open(src/'mask'/f'{i:05}.png'))>0
            input_panel=im.copy()
            if synthetic:
                target=load('GT');shown=im
            else:
                # 用实际目标core定位，黄色矩形仅做身份标记，不是生成删除mask。
                corepaths=sorted((Path(c['folder'])/'core').glob('*.png'));core=np.asarray(Image.open(corepaths[c['frames'][i]]))>0
                yy,xx=np.where(core)
                if len(xx):cv2.rectangle(input_panel,(int(xx.min()),int(yy.min())),(int(xx.max()),int(yy.max())),(255,220,30),2)
                cv2.putText(input_panel,f'{c["clip_id"]} target actor {c["actor_ordinal_in_scene"]}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,230,40),2)
                shown=im.copy();blend=np.zeros_like(shown);blend[:,:,2]=255;shown[h]=np.rint(.75*shown[h]+.25*blend[h]).astype('uint8');target=input_panel
            panels={'target':target,'input':shown,'base':load('base'),'r7':load('r7'),'new_data':load('new_data')};frames.append(panels)
            for k,arr in panels.items():Image.fromarray(arr).save(folder/f'{i:03}_{k}.jpg',quality=96)
            for arm in plan['arms']:Image.fromarray(load(arm+'_native')).save(folder/f'{i:03}_{arm}_native.jpg',quality=96)
        for role in ['target','input','base','r7','new_data','base_native','r7_native','new_data_native']:encode(folder,role)
        index=5;hh=np.asarray(Image.open(src/'mask'/f'{index:05}.png'))>0;yy,xx=np.where(hh);b=np.array([xx.min()-20,yy.min()-20,xx.max()+21,yy.max()+21]);b[[0,2]]=b[[0,2]].clip(0,1024);b[[1,3]]=b[[1,3]].clip(0,576)
        sheet=Image.new('RGB',(2000,900),(15,22,33));draw=ImageDraw.Draw(sheet);draw.text((8,8),f'{cid} | {c.get("receiver_scene",c.get("scene"))} | f{index} | {c["kind"]}',fill='white',font=font(22))
        for j,(role,arr) in enumerate(frames[index].items()):
            draw.text((j*400+8,42),role,fill='white',font=font());im=Image.fromarray(arr);sheet.paste(im.resize((400,225)),(j*400,72));crop=im.crop(tuple(b));crop.thumbnail((396,550));sheet.paste(crop,(j*400+(400-crop.width)//2,330))
        sheet.save(dest/'effect_contacts'/f'{cid}_f05.jpg',quality=97)
        er=byid.get(cid);note=(f'真实GT可比较洞内恢复；{c["type"]}。GT Y从未改写；灰色X仅是合成输入示意。' if synthetic else f'删除 {c["clip_id"]} / actor {c["actor_ordinal_in_scene"]} / {c["camera"]}；源帧 {c["frames"][0]}–{c["frames"][-1]}。黄框是目标core外包标识，蓝区是模型生成mask。无真实去车GT，第一栏原图不是恢复目标。')
        qtext=json.dumps(er,ensure_ascii=False) if er else '尚未填写助手效果观察；人工评分保持空。'
        metrics=s['synthetic_cases'][next(i for i,v in enumerate(s['synthetic_cases']) if v['eval_id']==cid)] if synthetic else {'GT_available':False,'input_core_nonempty':c['core_nonempty_in_selected_window'],'input_difficulty_proxy':c['input_difficulty_proxy'],'source_rule':c['input_window_rule'],'mask_issue_flags':'mask stats retained at original audit; frame contact inspection required'}
        cases.append({'id':cid,'scene':c.get('receiver_scene',c.get('scene')),'type':c.get('type',c['kind']),'kind':c['kind'],'group':'合成GT评测' if synthetic else '真实DELETE开发评测','note':note,'qa':qtext,'review_frame':5,'videos':{r:f'effects/{cid}/{r}.mp4' for r in ['target','input','base','r7','new_data']},'frame_pattern':f'effects/{cid}/{{i}}_{{role}}.jpg','native_links':{arm:f'effects/{cid}/{arm}_native.mp4' for arm in plan['arms']},'metrics':metrics})
        print('EFFECT_HTML',cid,flush=True)
    data={'mode':'effects','roles':[{'key':'target','label':'合成GT／真实原视频（黄框目标）'},{'key':'input','label':'合成X／真实输入生成范围（蓝）'},{'key':'base','label':'原始 DriveEditor'},{'key':'r7','label':'当前 r7（修复encoder）'},{'key':'new_data','label':'r8：只改数据微调'}],'score_roles':[{'key':a,'label':label} for a,label in [('base','原模型'),('r7','r7'),('new_data','r8')]],'cases':cases}
    ms=s['synthetic_metrics'];table='<table><tr><th>合成恢复MAE（越低越好）</th><th>原</th><th>r7</th><th>r8</th><th>case／scene</th></tr>'
    for key,label in [('hole','洞内'),('protected_inside_hole','洞内保护车')]:
        v=ms[key];table+=f'<tr><td>{label} · scene等权</td>'+''.join(f'<td>{v["macro_scene"][a]:.5f}</td>' for a in plan['arms'])+f'<td>{v["cases"]} / {v["scenes"]}</td></tr>'
    table+='</table>'
    intro=f'<p>训练 {s["data"]["training_cases"]} 例 / {s["data"]["training_scene_count"]} 个场景，单场景最多 {s["data"]["max_training_cases_per_scene"]} 例；三类 {s["data"]["training_type_counts"]}。同80张量、320×576、160步、原loss、seed6201；三臂推理同576×1024、10帧、seed42、25步。</p>{table}<p>合成恢复与真实DELETE分别评估。真实8例全是已经曝光的开发例，无去车GT；MAE下降不能证明真实后车身份恢复、邻车保真或时序合格。全部视频是原生推理后的局部写回，不是Ω世界重建。每栏同步播放和逐帧评分；native在折叠链接中单独保存。</p><p>数据改变包含场景扩展和已有3D网格轮廓投影，不能把变化仅归于scene数。两种已曝光网格跨train/val共享，Boston白天数据；final未使用。human verdict为空。</p>'
    page(dest/'index.html','v77 r8 · 同预算三权重／两套评测',intro,data);dump(dest/'effect_manifest.json',data);dump(dest/'results_summary.json',s)
if __name__=='__main__':main()
