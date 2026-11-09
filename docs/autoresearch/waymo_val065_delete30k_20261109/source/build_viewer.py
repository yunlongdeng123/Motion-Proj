"""Stage a fourth, synchronized30K edit column; preserve earlier viewers."""
from pathlib import Path
import argparse
import json

ASSET='native_delete30k_20261109'


def once(s,a,b):
    assert s.count(a)==1,a[:90]
    return s.replace(a,b,1)


def build(scene,output):
    output.mkdir(exist_ok=False)
    oldasset=scene/'native_continuation_20261109'
    d=json.loads((oldasset/'data.js').read_text(encoding='utf8').strip().split('=',1)[1][:-1])
    assert d['scene']=='065' and d['frame_count']==199
    d['parent_experiment_id']=d['experiment_id']
    d.update(experiment_id='WAYMO-065-DELETE-WHITE30K-20261109',schema='v44.delete30k.viewer.v1',
             continuation_url='native_continuation.html',default_raw=119,edit_iteration=30000,
             delete_track='NTjq-OhZ_97OMJ4xwJAxXg',target_description='前方白色轿车',
             jump_raws=[0,30,60,99,119,133,150,173,198])
    for row in d['frames']:
        raw=row['raw']
        row.update(native20=f'../native_continuation_20261109/native_20000/{raw}.png',
                   native30=f'../native_continuation_20261109/native_30000/{raw}.png',
                   delete_white=f'delete_white/{raw}.png',edited_keep=f'keep/{raw}.png')
    html=(scene/'index.html').read_text(encoding='utf8')
    html=once(html,'max-width:1800px','max-width:2200px')
    html=once(html,'grid-template-columns:repeat(3,minmax(0,1fr))','grid-template-columns:repeat(4,minmax(0,1fr))')
    html=once(html,'@media(max-width:760px)',
              '@media(min-width:761px) and (max-width:1300px){.native-comparison{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:760px)')
    html=once(html,'<option value="native">本次原生续训对照</option>',
              '<option value="native">30K 重建与白车删除</option><option value="continuation">原生续训对照</option>')
    html=once(html,'</select></label><button id="window">',
              '</select></label><label>第四列 <select id="edit-mode"><option value="delete" selected>删除白车 · 30K</option>'
              '<option value="keep">编辑场景保留 · 30K</option></select></label><button id="window">')
    html=once(html,'<p class="muted">从同一 10K 状态继续训练。</p></article></div>',
              '<p class="muted">从同一 10K 状态继续训练。</p></article><article><h2 id="editedTitle">删除白色轿车 · 30K</h2>'
              '<canvas id="edited" width="960" height="640" tabindex="0"></canvas><p class="muted">30K 场景编辑 · '
              '<a id="edit-video" href="native_delete30k_20261109/delete_white_30k.mp4" target="_blank" rel="noopener">播放完整视频</a></p></article></div>')
    html=html.replace('native_continuation_20261109/',ASSET+'/')
    js=(oldasset/'viewer.js').read_text(encoding='utf8')
    js=once(js,"step:'30000',epoch:0","step:'30000',edit:'delete',epoch:0")
    js=once(js,"const CANVASES=['observed','native','continued'];","const CANVASES=['observed','native','continued','edited'];")
    start,end=js.index('async function render(){'),js.index('function stop(){')
    js=js[:start]+"""async function render(){const epoch=++S.epoch,raw=S.raw,row=F.get(raw),step=S.step,edit=S.edit;S.pictures=null;placeholder(`正在加载第 ${raw} 帧…`);$('continuedTitle').textContent=`原生续训 · ${Number(step)/1000}K`;$('editedTitle').textContent=edit==='keep'?'编辑场景保留 · 30K':'删除白色轿车 · 30K';$('edit-video').href=R+(edit==='keep'?'keep_30k.mp4':'delete_white_30k.mp4');$('status').textContent=`正在加载第 ${raw} 帧…`;const paths=[row.observed,row.native,step==='20000'?row.native20:row.native30,edit==='keep'?row.edited_keep:row.delete_white];const results=await Promise.allSettled(paths.map(picture));if(epoch!==S.epoch)return;S.pictures=results.map(r=>r.status==='fulfilled'?r.value:null);S.errors=results.map(r=>r.status==='fulfilled'?'':'此帧图像加载失败');draw();$('status').textContent=`第 ${raw} 帧 · ${row.time_s.toFixed(2)} 秒 · 第三列 ${Number(step)/1000}K · 第四列 30K${results.some(r=>r.status==='rejected')?' · 部分图像暂不可用':''}`;}
"""+js[end:]
    js=once(js,"hash.set('step',S.step);","hash.set('step',S.step);hash.set('edit',S.edit);")
    js=once(js,"$('experiment').onchange=e=>{", "$('edit-mode').onchange=e=>{stop();S.edit=e.target.value==='keep'?'keep':'delete';setRaw(S.raw);};\n$('experiment').onchange=e=>{if(e.target.value==='continuation'){location.href=D.continuation_url+'#raw='+S.raw+'&step='+S.step;return;}")
    js=once(js,'`全部 ${D.training_frame_count} 帧训练 · 10K 续训至 30K · 无编辑基础重建`',
            '`全部 ${D.training_frame_count} 帧训练 · 30K 白色轿车删除`')
    js=once(js,'`${D.frame_count} 帧同步对照。第三列可切换 20K / 30K；尚未提供的节点会明确留空。此前原生结果与旧版编辑页面均保留。`',
            '`${D.frame_count} 帧同步对照。第四列固定使用30K场景，可切换删除和同场景保留；其道路恢复与表示交接均已固定。原生重建与历史结果保留在实验视图中。`')
    js=js.replace('v44.native_continuation.review.v1','v44.delete30k.review.v1')
    js=once(js,'continuation_step:Number(S.step),note:', 'continuation_step:Number(S.step),edit_mode:S.edit,note:')
    js=once(js,'native-continuation-${S.raw}.json','delete30k-${S.raw}.json')
    js=once(js,"if(r.schema==='v44.native_quality.review.v1')", "if(r.schema==='v44.native_continuation.review.v1')throw Error('这是此前原生续训记录，请在“原生续训对照”载入。');if(r.schema==='v44.native_quality.review.v1')")
    js=once(js,"$('note').value=String(r.note||'').slice(0,5000);", "S.edit=r.edit_mode==='keep'?'keep':'delete';$('edit-mode').value=S.edit;$('note').value=String(r.note||'').slice(0,5000);")
    js=once(js,"$('continuation-step').value=S.step;if(hash.has('raw')", "$('continuation-step').value=S.step;S.edit=hash.get('edit')==='keep'?'keep':'delete';$('edit-mode').value=S.edit;if(hash.has('raw')")
    assets=output/ASSET;assets.mkdir()
    (output/'index.html').write_text(html,encoding='utf8')
    (assets/'viewer.js').write_text(js,encoding='utf8')
    (assets/'data.js').write_text('window.NATIVE_QUALITY_DATA='+json.dumps(d,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf8')
    print(json.dumps(dict(scene='065',frames=199,default_raw=119,columns=4,output=str(output),installed=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.scene_root,a.output)
