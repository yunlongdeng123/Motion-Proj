"""Register the two V4 experiments without claiming unfinished assets are ready."""
import argparse,json,re
from datetime import datetime,timezone
from pathlib import Path

DEFAULT_CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v4_replay_r1_20261022')

def manifest(campaign,output=None):
    path=campaign/'serial_status.json'
    serial=json.loads(path.read_text()) if path.exists() else {}
    snapshot=datetime.now(timezone.utc).isoformat()
    experiments=[]
    for arm,description in [('R','仅真实来源'),('RP','真实来源＋先验')]:
        state=serial.get('status','NOT_STARTED')
        if arm in serial.get('completed_arms',[]):training='训练已完成，待诊断导出'
        elif serial.get('active_arm')==arm:
            training='训练未完成（当前停止）' if state.startswith('FAILED') else '训练进行中' if state=='RUNNING' else '训练未完成'
        else:training='尚未开始' if not serial.get('completed_arms') else '等待串行启动'
        ready_path=output/'data'/('v4_'+arm)/'ready.json' if output else None
        ready=json.loads(ready_path.read_text()) if ready_path and ready_path.exists() else None
        if ready:
            if ready.get('status')!='V4_DIAGNOSTICS_READY' or ready.get('frames')!=list(range(89,138)) or ready.get('iteration')!=30000:raise ValueError('incomplete V4 export receipt')
            for relative,record in ready['asset_files'].items():
                asset=(output/relative).resolve()
                if not asset.is_relative_to(output.resolve()) or not asset.is_file() or asset.stat().st_size!=record['bytes']:raise ValueError('incomplete V4 asset '+relative)
            if Path(ready['checkpoints']['static']['path']).parent!=campaign/'arms'/arm/'run30k':raise ValueError('V4 assets from another campaign')
        experiments.append(dict(id='v4_'+arm,name='V4 '+arm+' · '+description+('（30K）' if ready else '（占位）'),label='V4 '+arm+' · '+description+('（30K）' if ready else '（占位）'),available=bool(ready),
            family='V4',arm=arm,training_state=training,asset_state='NOT_EXPORTED',
            message=('已接入真实30K的49帧原生结果；道路固定几何，显示实际参与输出的高斯。' if ready else training+'；完整诊断结果尚未导出。当前仅占位，不显示其他实验的图像或指标。'),
            bindings=None if not ready else ready['checkpoints'],road=None if not ready else ready['road'],
            status_snapshot_utc=snapshot,campaign=str(campaign)))
        if ready:experiments[-1]['asset_state']='READY';experiments[-1]['training_state']='已完成30K'
    return dict(schema='GS_EXPERIMENT_SLOTS_V1',experiments=experiments,training_updates=0,
        readiness_policy='available remains false until real assets and checkpoint bindings have been exported and validated')

def export(output,campaign=DEFAULT_CAMPAIGN):
    data=manifest(campaign,output);folder=output/'data';folder.mkdir(parents=True,exist_ok=True)
    (folder/'v4_experiments.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':'))
    script='''/* Explicit slots; no checkpoint, image or metric is invented. */
(function(){const record=PAYLOAD;window.GSDATA=window.GSDATA||{};window.GSDATA.v4Experiments=record;
const scene=window.GSDATA.scene;if(!scene)return;
const experiments=scene.experiments||[{id:'full',name:'完整版'},{id:'real_only',name:'仅真实观测'},{id:'prior_only',name:'仅先验'},{id:'ownership_only',name:'仅归属修复'}];
if(scene.bindings?.gn11_v1&&!experiments.some(v=>v.id==='gn11_v1'))experiments.push({id:'gn11_v1',name:'V1 原生 GN11（历史恢复30K）'});
scene.experiments=experiments.filter(v=>!record.experiments.some(p=>p.id===v.id)).concat(record.experiments);
scene.bindings=scene.bindings||{};scene.config_bindings=scene.config_bindings||{};
for(const info of record.experiments)if(info.available){scene.bindings[info.id]=info.bindings;scene.config_bindings[info.id]=info.bindings.config;}
})();
'''.replace('PAYLOAD',payload)
    (folder/'v4_experiments.js').write_text(script,encoding='utf-8')
    return data

def integrate_index(content):
    content=content.replace('<script src="v4_embed.js?v=embed1"></script>','')
    content=content.replace('href="#v4-initialization"','href="v4_initialization.html"')
    content=content.replace('<a href="v4_initialization.html">V4初始化验收</a>','')
    content=content.replace('</nav>','<a href="v4_initialization.html">V4初始化验收</a></nav>',1)
    if 'data/v4_experiments.js' not in content:
        needle='<script src="data/scene.js"></script>'
        if needle not in content:raise ValueError('existing scene script missing')
        content=content.replace(needle,needle+'<script src="data/v4_experiments.js?v=slots1"></script>',1)
    for name in ('app','gpu','inspection','sampling','offline_geometry'):
        content=re.sub(r'src="'+name+r'\.js(?:\?[^"]*)?"','src="'+name+'.js?v=v4results1"',content)
    content=re.sub(r'src="data/v4_experiments\.js(?:\?[^"]*)?"','src="data/v4_experiments.js?v=results1"',content)
    # Keep the slots visible in source HTML too; app.bind builds the same list from metadata.
    if '<option value="v4_R">' not in content:
        start=content.index('<select id="arm">');end=content.index('</select>',start)
        options='<option value="v4_R">V4 R · 仅真实来源（占位）</option><option value="v4_RP">V4 RP · 真实来源＋先验（占位）</option>'
        content=content[:end]+options+content[end:]
    return content

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--campaign',type=Path,default=DEFAULT_CAMPAIGN);a=p.parse_args()
    print(json.dumps(export(a.output,a.campaign),ensure_ascii=False))
