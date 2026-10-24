"""只在49帧图像/几何真实验收后登记V4.1-R；保留全部历史实验。"""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path

CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023')
KEY='v4_1_R'

def export(output,campaign=CAMPAIGN):
    output=Path(output)
    ready=json.loads((output/f'data/{KEY}/ready.json').read_text())
    assert ready['status']=='V41_DIAGNOSTICS_READY' and ready['frames']==list(range(89,138))
    assert ready['validation']['status']=='PASSED'
    assert Path(ready['checkpoints']['static']['path']).parent==campaign/'arms/R/run30k'
    for name,record in ready['asset_files'].items():
        path=(output/name).resolve()
        assert path.is_relative_to(output.resolve()) and path.stat().st_size==record['bytes']
    info=dict(id=KEY,name='V4.1 R · 固定几何道路（30K）',label='V4.1 R · 固定几何道路（30K）',
        available=True,family='V4.1',arm='R',training_state='已完成30K',asset_state='READY',
        message='本次V4.1-R：原V1路径＋稠密联合/删车监督；固定道路几何。质量待评审。',
        bindings=ready['checkpoints'],road=ready['road'],campaign=str(campaign),
        status_snapshot_utc=datetime.now(timezone.utc).isoformat(),
        extra_channels=['native_v1','native_static_delete'],training_updates=0,
        source_optimization_log_available=False,gaussian_trace_available=False)
    record=dict(schema='GS_V41_EXPERIMENT_V1',experiments=[info],training_updates=0,
        preview_scope='49 actual native CUDA views; quantized display maps; not formal SSIM/LPIPS acceptance')
    data=output/'data'
    (data/'v41_experiments.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    script='''(function(){const record=PAYLOAD;const g=window.GSDATA=window.GSDATA||{};
g.v41Experiments=record;g.v4Experiments=g.v4Experiments||{experiments:[]};
g.v4Experiments.experiments=g.v4Experiments.experiments.filter(v=>v.id!=='v4_1_R').concat(record.experiments);
const scene=g.scene;if(!scene)return;scene.experiments=(scene.experiments||[]).filter(v=>v.id!=='v4_1_R').concat(record.experiments);
scene.bindings=scene.bindings||{};scene.config_bindings=scene.config_bindings||{};
for(const e of record.experiments){scene.bindings[e.id]=e.bindings;scene.config_bindings[e.id]=e.bindings.config;}
g.defaultExperiment='v4_1_R';g.defaultChannel='native_v1';
const select=document.getElementById('channel');if(select)for(const [value,label] of [['native_v1','原V1完整通路 · 本组checkpoint（默认）'],['native_static_delete','同checkpoint · 原静态删车']]){
if(!Array.from(select.options).some(o=>o.value===value)){const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o);}}
})();'''.replace('PAYLOAD',json.dumps(record,ensure_ascii=False,separators=(',',':')))
    (data/'v41_experiments.js').write_text(script,encoding='utf-8')
    return record

def integrate_index(html):
    if 'data/v41_experiments.js' not in html:
        import re
        html,n=re.subn(r'(<script src="data/v4_experiments\.js[^\"]*"></script>)',
            r'\1<script src="data/v41_experiments.js?v=v41r30k1"></script>',html,count=1)
        assert n==1,'Missing legacy experiment registry'
    import re
    html=re.sub(r'src="app\.js(?:\?[^\"]*)?"','src="app.js?v=v41r30k1"',html)
    html=html.replace('V3道路全量显示','道路分支全量显示').replace('inspection.js?v=v4results1','inspection.js?v=v41r30k1')
    return html

def patch_app(text):
    replacements=[
      ("arm:'full',raw:102,x:90,y:255,channel:'deleted_five'", "arm:window.GSDATA.defaultExperiment||'full',raw:102,x:90,y:255,channel:window.GSDATA.defaultChannel||'deleted_five'"),
      ("'dynamic_raw','road_raw','road_surface_color'].includes(ch)","'dynamic_raw','road_raw','road_surface_color','native_v1','native_static_delete'].includes(ch)"),
      ("function unavailable(canvas){", "const extraImage=ch=>['native_v1','native_static_delete'].includes(ch);\nconst supportsImage=(arm,ch)=>!extraImage(ch)||Boolean(EXPERIMENTS.find(e=>e.id===arm)?.extra_channels?.includes(ch));\nfunction unavailable(canvas,message='V1 无专用道路分支，此项不适用'){"),
      ("c.fillText('V1 无专用道路分支，此项不适用',25,45)","c.fillText(message,25,45)"),
      ("else if(isImage(channel)){const im=", "else if(extraImage(channel)&&!supportsImage(arm,channel)){unavailable($('renderCanvas'),'该实验尚未导出此通道');legend='当前通道仅绑定已明确导出的同checkpoint结果';}else if(isImage(channel)){const im="),
      ("if(isImage(channel)&&!(arm==='gn11_v1'&&channel.includes('road')))image=", "if(isImage(channel)&&supportsImage(arm,channel)&&!(arm==='gn11_v1'&&channel.includes('road')))image="),
      ("if(channel.startsWith('v1_')){if(arm!=='gn11_v1'", "if(extraImage(channel)&&!supportsImage(arm,channel)){unavailable(canvas,'该实验尚未导出此通道');}else if(channel.startsWith('v1_')){if(arm!=='gn11_v1'"),
      ("'V4真实30K原生CUDA结果＋固定道路几何'", "'真实30K原生CUDA结果＋固定道路几何'")]
    for old,new in replacements:
        assert text.count(old)==1,old
        text=text.replace(old,new,1)
    return text

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(export(a.output),ensure_ascii=True))
