"""Install prepared wiring or a verified native delivery without changing RGB.

No synthetic white pixels: the image canvas stays hidden until a matching
native GPU receipt and image are published. The prepared plan is not queued.
"""
from pathlib import Path
import argparse,json,re,shutil

WORK=Path(__file__).resolve().parent
ROOT=WORK.parent.parent
CONTEXT='14705303724557273004_3105_000_3125_000'
READY=('v43_actor_10000','v43_motion_no_edit','v43_motion_delete_target','v43_motion_translate_lateral_1m','v43_motion_turn_left15','v43_shift_3m')
BLOCKED={
 'v43_actor_1000':'1K：高斯行身份可追踪，但当前归属证据测于10K。需补1K自身语义/轮廓贡献核验，再生成匹配涂白图。',
 'v43_actor_5000':'5K：高斯行身份可追踪，但当前归属证据测于10K。需补5K自身语义/轮廓贡献核验，再生成匹配涂白图。',
 'v43_main_noedit':'V1原生通路：使用原30K动态高斯，行空间不同于Actor10K；需独立核验原生分支归属，不能直接套用10K车身标签。'}
VERSION='unknown-white-release-20261030-1'

def block(s,name,content,anchor):
    b=f'<!-- {name}-start -->{content}<!-- {name}-end -->'
    pattern=rf'<!-- {re.escape(name)}-start -->.*?<!-- {re.escape(name)}-end -->'
    if re.search(pattern,s,re.S):return re.sub(pattern,lambda _:b,s,flags=re.S)
    if anchor not in s:raise ValueError('Missing insertion point: '+anchor)
    return s.replace(anchor,b+anchor,1)


def delivery_progress(plan,groups):
    expected={gid:set(groups[gid]['frames']) for gid in READY}
    total=sum(map(len,expected.values()))
    verified=sum(1 for gid,raws in expected.items()
                 for raw,entry in plan.get('groups',{}).get(gid,{}).get('frames',{}).items()
                 if raw in raws and entry.get('status')=='verified' and entry.get('group_id')==gid
                 and entry.get('raw')==int(raw))
    gpu=plan.get('gpu_executed') is True
    state=plan.get('status')
    consistent=plan.get('verified_group_frame_count')==verified
    rendered=gpu and state in ('complete','partial') and verified>0 and consistent
    if rendered and state=='complete' and verified==total:
        label=f'GPU已完成 {verified}/{total} 帧'
    elif rendered:
        label=f'GPU已完成 {verified}/{total} 帧（部分交付）'
    elif gpu or state in ('complete','partial'):
        label=f'GPU交付待核验（{verified}/{total} 帧）'
    else:
        label=f'CPU就绪，待GPU渲染（0/{total} 帧）'
    return dict(rendered=rendered,gpu_executed=gpu,status=state,
                verified_group_frames=verified,planned_group_frames=total,label=label)


def install(root=ROOT):
    root=Path(root);out=root/'outputs/GS_Scene_Diagnostics_20261027';scene=out/CONTEXT
    data=json.loads((scene/'v43_groups_data.json').read_text(encoding='utf8'))
    groups={g['id']:g for g in data['groups']}
    plan={'schema':'v43-native-unknown-white-v1','status':'prepared','context':CONTEXT,
          'classification':{'radius':5,'fraction':.2},'gpu_executed':False,'training_updates':0,'groups':{}}
    for gid in (*READY,*BLOCKED):
        g=groups[gid];raws=sorted(map(int,g['frames']))
        plan['groups'][gid]={'title':g['title'],'status':'prepared' if gid in READY else 'evidence_needed',
                            'raws':raws,'frame_count':len(raws),'frames':{}}
        if gid in BLOCKED:plan['groups'][gid]['reason']=BLOCKED[gid]
    # Keep a future verified delivery on rebuild. Prepared CPU status cannot
    # overwrite the native GPU output manifest.
    dest=scene/'unknown_white_manifest.json'
    if dest.exists():
        old=json.loads(dest.read_text(encoding='utf8'))
        if old.get('status') in ('complete','partial'):plan=old
    progress=delivery_progress(plan,groups)
    dest.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
    (scene/'unknown_white_data.js').write_text('window.UNKNOWN_WHITE_DELIVERY='+json.dumps(plan,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf8')
    shutil.copy2(WORK/'unknown_white_runtime.js',out/'scene_unknown_white.js')
    html=(scene/'index.html').read_text(encoding='utf8')
    backup=WORK/'before_index.html'
    if not backup.exists():backup.write_text(html,encoding='utf8')
    panel='''<section id="unknownWhitePanel" class="notice" style="margin-top:16px" hidden>
<strong>同帧对照 · unknown高斯涂白</strong><p id="unknownWhiteDeliveryStatus">__DELIVERY_STATUS__</p><p id="unknownWhiteFrame"></p>
<p id="unknownWhiteStatus"></p><canvas id="unknownWhiteCanvas" width="480" height="320" style="width:100%;height:auto" hidden aria-label="当前组同帧unknown高斯白色材质原生渲染"></canvas>
<p id="unknownWhitePairNote"></p><p class="note">只改unknown材质颜色；车辆核心保留原RGB。保留透明度、几何、遮挡和原环境。白底剥离、2D涂框与此图分别展示。</p>
<button id="unknownWhiteRetry">重新读取本帧</button> <a id="unknownWhiteReceipt" hidden target="_blank">本帧渲染凭据</a>
<details><summary>当前适配范围与数据流</summary><p>同帧归属行ID → 当前组位置/刚体变换 → unknown材质设白 → 原生高斯合成 → 原RGB / 涂白RGB / 3D对照。</p>
<table><tr><th>V43组</th><th>3D归属</th><th>涂白RGB</th></tr><tr><td>10K重建、无编辑、删车、横移1m/3m、左转15°</td><td>已接R2行身份；编辑按行携带原观测证据</td><td>__DELIVERY_STATUS__</td></tr><tr><td>1K、5K重建</td><td>需核验各自checkpoint证据</td><td>待证据与GPU</td></tr><tr><td>V1原生通路</td><td>独立动态行空间，需单独适配</td><td>待证据与GPU</td></tr></table>
<p>场景2/3尚未迁移继承R2。本面板不将其旧消融结果列为已适配。</p></details></section>'''
    panel=panel.replace('__DELIVERY_STATUS__',progress['label'])
    anchor='<div id="frameMetrics" class="note"></div>'
    # Insert immediately after metrics inside the LEFT panel requested by user.
    suffix='</section>\n<section class="panel"><div class="panel-title"><span class="number">02</span>'
    html=block(html,'unknown-white-panel',panel,suffix)
    html=block(html,'unknown-white-data',f'<script src="unknown_white_data.js?v={VERSION}"></script>','<script src="../viewer.js')
    html=block(html,'unknown-white-runtime',f'<script src="../scene_unknown_white.js?v={VERSION}"></script>','</body>')
    (scene/'index.html').write_text(html,encoding='utf8')
    if progress['rendered']:
        landing=out/'index.html'
        source=landing.read_text(encoding='utf8')
        links=[]
        for gid,label in (('v43_actor_10000','10K重建'),('v43_motion_no_edit','无编辑'),
                          ('v43_motion_delete_target','删车'),('v43_motion_translate_lateral_1m','横移1米'),
                          ('v43_motion_turn_left15','左转15°'),('v43_shift_3m','横移3米')):
            links.append(f'<a href="{CONTEXT}/index.html#workspace?group={gid}&amp;raw=113&amp;channel=rgb">{label}</a>')
        notice='<section class="notice"><strong>场景1：unknown高斯白材质同帧对照</strong><p>'+progress['label']+' · '+ ' · '.join(links)+'</p><p>原RGB下方查看涂白对照，与右侧灰色unknown使用同一行集合。默认5px / 20%；只改材质，保留原透明度与遮挡。</p></section>'
        landing.write_text(block(source,'unknown-white-landing',notice,'<main id="workspace">'),encoding='utf8')
    return {'CPU_wired':True,**progress,'ready_groups':list(READY),'awaiting_evidence':list(BLOCKED)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    print(json.dumps(install(a.root),ensure_ascii=False))
