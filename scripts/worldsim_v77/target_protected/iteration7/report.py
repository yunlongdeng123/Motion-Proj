"""r7根因证据和同配方修复对照；技术报告含简单组件图。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,sys,html,shutil
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r7')
REPO=Path('/root/autodl-tmp/motion_proj_v77')
sys.path.insert(0,str(REPO/'scripts/worldsim_v77/target_protected/iteration6'))
from report_pilot import encode
from audit import dump
LABELS={'base':'原始权重','r6':'r6：错误监督encoder','encoder_fixed_lowres':'仅修encoder：同80张量/160步','encoder_fixed_native':'修encoder＋原尺寸训练'}
CONTACT_LABELS={'base':'Original','r6':'r6 / wrong target encoder','encoder_fixed_lowres':'Encoder fixed / same80 /160steps','encoder_fixed_native':'Encoder fixed / native size'}
FRAME_NOTES={
    'holdout_R014':'第5帧：修复后大片模糊消失，后车主体和道路标线恢复；车底轮廓仍与GT有差异。',
    'holdout_R015':'第5帧：旧r6后车被模糊块覆盖，修复后保留车体；洞边路面色差仍可见，不能当高保真通过。',
    'holdout_R018':'第5帧：原模型有明显车形接缝，旧r6有涂抹；修复后车体与路面更连贯，细部仍需审核。',
    'holdout_R019':'第5帧：修复后保留后车主体，旧r6模糊块减轻；车底/道路接合处未获完整身份或时序认证。'}
def read(p):return json.loads(p.read_text())
def avg(m,key):
    vals=[r[key] for r in m['scores'] if r[key] is not None]
    return None if not vals else float(np.mean([v['MAE'] for v in vals]))
def main():
    dest=ROOT/'delivery';dest.mkdir(exist_ok=True)
    plan=read(ROOT/'evaluation_plan.json');state=read(ROOT/'evaluation/state.json')
    arms=[a for a in LABELS if all((ROOT/'evaluation'/c['eval_id']/f'{a}_metrics.json').exists() for c in plan['cases'])]
    assert 'encoder_fixed_lowres' in arms
    rows=[];jobs=[]
    for c in plan['cases']:
        source=ROOT/'evaluation'/c['eval_id'];out=dest/'assets'/c['eval_id'];out.mkdir(parents=True,exist_ok=True)
        row={'eval_id':c['eval_id'],'role':c['role'],'type':c['type'],'receiver_scene':c['receiver_scene'],'arms':{}}
        for arm in arms:
            m=read(source/f'{arm}_metrics.json');row['arms'][arm]={k:avg(m,k) for k in ('hole','protected_inside_hole','native_outside_hole')}
            for role in (arm,arm+'_native'):jobs.append((sorted((source/role).glob('*.png')),out/(role+'.mp4')))
        for role in ('GT','condition','input'):jobs.append((sorted((source/role).glob('*.png')),out/(role+'.mp4')))
        rows.append(row)
        # 指定第5帧；整图+相同洞邻域crop，不能从该帧判时序。
        mask=np.asarray(Image.open(Path(c['folder'])/'model_hole/005.png'))>0;yy,xx=np.where(mask)
        box=(max(0,int(xx.min())-60),max(0,int(yy.min())-60),min(1024,int(xx.max())+61),min(576,int(yy.max())+61))
        roles=['GT']+arms
        canvas=Image.new('RGB',(480*len(roles),620),'#101820');draw=ImageDraw.Draw(canvas)
        for k,role in enumerate(roles):
            im=Image.open(source/role/'00005.png').convert('RGB')
            canvas.paste(im.resize((480,270)),(480*k,25))
            crop=im.crop(box);crop.thumbnail((480,280));canvas.paste(crop,(480*k,315))
            draw.text((480*k+8,5),'GT' if role=='GT' else CONTACT_LABELS[role],fill='white')
        canvas.save(out/'frame5_comparison.jpg',quality=94)
    probe=read(ROOT/'encoder_probe/results.json')
    (dest/'probe').mkdir(exist_ok=True)
    for p in (ROOT/'encoder_probe').iterdir():
        if p.is_dir():jobs.append((sorted(p.glob('*.png')),dest/'probe'/f'{p.name}.mp4'))
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda j:encode(*j),jobs))
    held=[r for r in rows if r['role']=='heldout']
    summary={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r7','arms':arms,'cases':rows,
             'heldout_count':4,'heldout_receiver_scene_count':1,'training_diagnostic_cases':2,
             'data_audit':read(ROOT/'data_and_module_audit.json')['summary'],
             'training':{a:read(ROOT/a/'training/state.json') for a in arms if a.startswith('encoder_fixed')},
             'encoder_probe':probe,'r6_correction':read(ROOT/'r6_correction.json'),
             'heldout_means':{a:{k:float(np.mean([r['arms'][a][k] for r in held])) for k in ('hole','protected_inside_hole','native_outside_hole')} for a in arms},
             'model_promoted':False,'human_verdict':None,'assistant_frame5_observations':FRAME_NOTES,'observation_scope':'single-frame structure notes; not human score or temporal verdict','reused_r6_windows':sum(r['reused_from'] is not None for r in state['completed']),
             'actual_new_sampling_windows':sum(r['reused_from'] is None for r in state['completed'])}
    base=summary['heldout_means']['base']['protected_inside_hole'];bad=summary['heldout_means']['r6']['protected_inside_hole'];fixed=summary['heldout_means']['encoder_fixed_lowres']['protected_inside_hole']
    summary['fixed_vs_base_protected_MAE_change_percent']=(fixed/base-1)*100
    summary['fixed_vs_r6_protected_MAE_change_percent']=(fixed/bad-1)*100
    summary['fixed_vs_base_worse_cases']=sum(r['arms']['encoder_fixed_lowres']['protected_inside_hole']>r['arms']['base']['protected_inside_hole'] for r in held)
    dump(ROOT/'diagnosis_summary.json',summary);dump(dest/'summary.json',summary)
    for name in ('data_and_module_audit.json','r6_correction.json','protocol_amendment_encoder.json','evaluation_plan.json'):
        shutil.copy2(ROOT/name,dest/name)
    style='body{font:16px system-ui;background:#121820;color:#e8eff8;margin:24px}a{color:#a9d6ff}p{line-height:1.7}.warning{padding:18px;background:#57301e;border:2px solid #f2ae77}.diagram{display:flex;gap:12px;align-items:center;flex-wrap:wrap;background:#243244;padding:18px}.diagram span{padding:12px;border:1px solid #829bbe}.cols{display:grid;grid-template-columns:repeat('+str(len(arms)+2)+',1fr);gap:10px}video,img{width:100%;background:#000}article{padding:16px;margin:25px 0;border:1px solid #536276;border-radius:8px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #526479;padding:8px}button,select{padding:8px;margin:5px;background:#29425b;color:white}details{margin:12px 0}@media(max-width:1100px){.cols{grid-template-columns:repeat(2,1fr)}}'
    body='<h1>v77 r7 · 为什么r6没有改善</h1><div class="warning"><strong>确定的首要问题是训练工程错误：目标encoder的106个权重没有加载。</strong><p>r6训练日志106 missing / 推理日志0 missing被混淆。真实Y被随机初始化且冻结的encoder映射到错误latent；随后用原预训练decoder采样，训练与生成空间不一致。旧数据、旧数值和视频都保留，但r6不能证伪数据或微调范围。</p></div>'
    body+='<div class="diagram"><span>真实Y<br>固定恢复目标</span>→<span>官方SVD目标encoder<br>106权重严格恢复<br><small>r6此处随机冻结</small></span>→<span>正确latent＋噪声<br>原扩散loss</span>→<span>原DriveEditor<br>同80空间attention<br>160步</span>→<span>原预训练decoder<br>固定seed生成</span></div><div class="diagram"><span>synthetic X＋H</span>→<span>先清零H，再resize</span>→<span>原条件encoder</span>→<span>同DriveEditor</span></div>'
    body+=f'<p><strong>只修encoder的四例留出结果：</strong>洞内真实保护车MAE，原模型 {base:.5f}；错误r6 {bad:.5f}；修复后 {fixed:.5f}。修复相对原模型 {summary["fixed_vs_base_protected_MAE_change_percent"]:+.1f}%，相对错误r6 {summary["fixed_vs_r6_protected_MAE_change_percent"]:+.1f}%。{summary["fixed_vs_base_worse_cases"]}/4 例仍比原模型差。数字是所有10帧的区域均值，不是单帧分数。</p>'
    body+='<p>同数据、同48/4 split、同80张量、同学习率1e-5、同160步、同320×576训练／576×1024推理。第一修复控制只改106个目标encoder权重，不重新造数据、不换模块、不改loss。严格加载0 missing / 0 unexpected、实际80张量反向有效。原尺寸臂发现随机encoder后第3步停止留证；扩大模块臂取消，未冒称执行。原模型继续默认，修复权重是否可接受由视频审核决定。</p>'
    body+='<p>训练数据48例，29背景／19单保护车，33例来自scene-0228，仅16个receiver来源窗口；实际监督洞占画面平均3.24%，保护车隐藏部分0.188%。blank分支mask_fuse全0，空间loss归一化后全画面等权。像素占比不是VAE混合后的精确梯度份额。资料能证明覆盖偏窄、保护监督稀少，尚不能把它们认定为剩余失败的唯一原因。</p>'
    body+='<p>模块方面：原self-attention49.57M仅约官方主网络可训练1.81B的2.74%，范围偏窄是候选；DELETE valid_mask全0，SV3D生成支路跳过，不能靠微调该支路解决当前问题。四例留出共享一个scene，另两例是训练集容量诊断；技术数据2分不是输出2分。单帧检视不证明时序，所有human verdict为空，final未用。</p>'
    body+='<p><a href="summary.json">全部数值与边界</a> · <a href="data_and_module_audit.json">52例全帧审计</a> · <a href="r6_correction.json">r6纠正记录</a> · <a href="protocol_amendment_encoder.json">实验计划变更依据</a> · <a href="../v77-target-protected-r6/index.html">保留的r6报告</a></p>'
    body+='<h2>编码器独立闭环</h2><p>同一真实Y→encoder→同一原预训练decoder。只切换随机／官方encoder，使用posterior mode去掉采样差异。这里是clean-Y oracle诊断，绝不作为DELETE条件。随机控制是重新初始化的同结构encoder，未冒称恢复r6未保存的随机权重。</p>'
    body+='<table><tr><th>尺寸</th><th>encoder</th><th>保护车MAE</th><th>latent标准差</th></tr>'
    for row in probe['rows']:body+=f'<tr><td>{row["size"]}</td><td>{row["arm"]}</td><td>{row["mean_protected_MAE"]:.5f}</td><td>{row["latent_std_mean"]:.3f}</td></tr>'
    body+='</table><div class="cols">'
    for label,role in [('真实Y','576_GT'),('随机encoder重建','576_random_encoder_control'),('官方encoder重建','576_official_SVD_encoder')]:body+=f'<div><p>{label}</p><video controls muted playsinline preload="metadata" src="probe/{role}.mp4"></video></div>'
    body+='</div><h2>固定视频对照</h2><p>GT是未被A遮挡的真实Y；条件灰洞是删除合成A。真实B有可见部分留在洞外，洞内B需要补全。完整synthetic X只通过诊断链接查看，模型条件看不到它。各臂同mask、seed42、25采样步、10帧；H外写回原像素，不计恢复收益。</p><button id="export">导出人工评分</button>'
    body+='<table><tr><th>case／角色</th>'+''.join(f'<th>{LABELS[a]} B-MAE</th>' for a in arms)+'</tr>'
    for r in rows:body+='<tr><td>'+r['eval_id']+' / '+r['role']+'</td>'+''.join('<td>'+('—' if r['arms'][a]['protected_inside_hole'] is None else f'{r["arms"][a]["protected_inside_hole"]:.5f}')+'</td>' for a in arms)+'</tr>'
    body+='</table>'
    for c,r in zip(plan['cases'],rows):
        cid=c['eval_id'];body+=f'<article id="{cid}"><h3>{cid} · {c["type"]} · {c["receiver_scene"]} · {c["role"]}</h3><p>恢复真实Y中被合成A遮住的内容；只删除A，保留真实B。训练诊断例不计泛化，留出四例只属于同一个场景。</p><p>{FRAME_NOTES.get(cid,"训练集容量诊断，仅报告固定视频和区域指标，不作为留出结论。")}</p><button data-play>同步播放</button><button data-pause>暂停</button><div class="cols">'
        for role,label in [('GT','真实Y／恢复GT'),('condition','masked-X条件')]+[(a,LABELS[a]) for a in arms]:body+=f'<div><p>{label}</p><video controls muted playsinline preload="metadata" src="assets/{cid}/{role}.mp4"></video></div>'
        body+='</div><p>人工评分：'
        for a in arms:body+=f'{LABELS[a]} <select data-arm="{a}"><option value="">未评</option><option value="0">0失败</option><option value="1">1较差</option><option value="2">2可接受</option></select>'
        body+=f'</p><details><summary>第5帧／同一局部放大，点击图片打开大图</summary><a href="assets/{cid}/frame5_comparison.jpg"><img src="assets/{cid}/frame5_comparison.jpg"></a></details><p><a href="assets/{cid}/input.mp4">synthetic X：仅作诊断</a> · '+' · '.join(f'<a href="assets/{cid}/{a}_native.mp4">{LABELS[a]}原生整帧</a>' for a in arms)+'</p></article>'
    script='''document.querySelectorAll('[data-play]').forEach(b=>b.onclick=()=>{const vs=[...b.closest('article').querySelectorAll('video')];vs.forEach(v=>{v.currentTime=0;v.play().catch(()=>{});});});document.querySelectorAll('[data-pause]').forEach(b=>b.onclick=()=>b.closest('article').querySelectorAll('video').forEach(v=>v.pause()));document.getElementById('export').onclick=()=>{let rows=['case_id,arm,human_score'];document.querySelectorAll('article').forEach(a=>a.querySelectorAll('select').forEach(s=>rows.push([a.id,s.dataset.arm,s.value].join(','))));const u=URL.createObjectURL(new Blob(['\\ufeff'+rows.join('\\n')],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=u;a.download='v77_r7_human_scores.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};'''
    script+='''document.querySelectorAll('article select').forEach(s=>{const key='v77-r7-'+s.closest('article').id+'-'+s.dataset.arm;s.value=localStorage.getItem(key)||'';s.onchange=()=>localStorage.setItem(key,s.value);});'''
    (dest/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>v77 r7 训练根因排查</title><style>'+style+'</style>'+body+'<script>'+script+'</script></html>')
    print(json.dumps({k:v for k,v in summary.items() if k in ('heldout_means','fixed_vs_base_protected_MAE_change_percent','fixed_vs_r6_protected_MAE_change_percent','fixed_vs_base_worse_cases','actual_new_sampling_windows')},ensure_ascii=False,indent=2))
if __name__=='__main__':
    if '--partial' in sys.argv:
        result=[]
        for c in read(ROOT/'evaluation_plan.json')['cases']:
            row={'eval_id':c['eval_id'],'role':c['role'],'B_MAE':{}}
            for a in LABELS:
                p=ROOT/'evaluation'/c['eval_id']/f'{a}_metrics.json'
                if p.exists():row['B_MAE'][a]=avg(read(p),'protected_inside_hole')
            result.append(row)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:main()
