"""Append the actual 2-versus-8 reference-pool result to the single existing report."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
DEFAULT_INDEX=Path(r'C:\Users\dengyunlong\Documents\Codex\2026-11-16\xian\outputs\road_material_three_rounds_20261121\index.html')
BEGIN='<!-- SOURCE_MULTIFRAME_PAIR_BEGIN -->';END='<!-- SOURCE_MULTIFRAME_PAIR_END -->'
PREFIX=Path('source_multiframe_r3')
POOLS={'two_frames':[88,115],'eight_frames':[88,90,92,94,96,99,104,115]}
QUERY=[100,110,117];STRESS=[86]


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8')
def need(condition,message):
    if not condition:raise ValueError(message)


def historical_state(raw):
    text=raw.decode('utf8');match=re.search(r'\bconst\s+D\s*=\s*',text)
    data,n=json.JSONDecoder().raw_decode(text[match.end():]);c=data['catalog']
    counts=[len(c['models']),sum(m.get('batch')!='gaussian_transfer' for m in c['models']),len(c['rows'])]
    need(counts==[58,55,744],'Original catalog changed')
    state={'counts':counts,'D_literal_sha256':hashlib.sha256(text[match.end():match.end()+n].encode('utf8')).hexdigest(),'blocks':{}}
    for name in ('SOURCE_QUALITY_AUDIT','SOURCE_FIDELITY_CHAIN'):
        pattern=('<!-- '+name+'_BEGIN -->.*?<!-- '+name+'_END -->').encode('ascii');block=re.search(pattern,raw,re.S)
        need(block is not None,'Historical evidence block missing')
        state['blocks'][name]=hashlib.sha256(block[0]).hexdigest()
    return state


def loss_plot(pair):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(11,3.4),facecolor='#101c29');ax.set_facecolor('#101c29')
    for arm,label,color in [('two_frames','两帧池','#78b7ff'),('eight_frames','八帧池','#65d8ab')]:
        rows=read(pair/arm/'history.json');x=[r['step'] for r in rows];y=[r['pre_update_batch_loss'] for r in rows]
        need(x==[1]+list(range(25,401,25)) and all(math.isfinite(v) and v>0 for v in y),'Actual loss records differ')
        ax.semilogy(x,y,'o-',markersize=3,linewidth=1.7,color=color,label=label)
    ax.set(xlabel='正式更新步数',ylabel='各自训练池目标值（对数）',xlim=(1,400));ax.tick_params(colors='#c5d9e9')
    for spine in ax.spines.values():spine.set_color('#49647c')
    ax.xaxis.label.set_color('#dce9f4');ax.yaxis.label.set_color('#dce9f4');ax.grid(color='#344b60',alpha=.55)
    legend=ax.legend(facecolor='#192d40',edgecolor='#49647c');plt.setp(legend.get_texts(),color='#e4eef7')
    fig.tight_layout();out=HERE/'loss_400_steps.svg';fig.savefig(out,facecolor=fig.get_facecolor());plt.close(fig);return out


def build(pair,exposure_root,pool_root):
    pair,exposure_root,pool_root=map(lambda p:Path(p).resolve(),(pair,exposure_root,pool_root))
    receipt=read(pair/'receipt.json');contract=receipt['contract'];cpu=read(exposure_root/'paired_cpu_audit.json')
    exposure=read(exposure_root/'shared_exposure.json');strict=read(exposure_root/'strict_nonoverlap_validation.json')
    support=read(exposure_root/'calibration_support_manifest.json');pool=read(pool_root/'pool_plan.json')
    need(receipt['complete'] and cpu['complete'] and cpu['all_required_state_data_checks_passed'],'Actual paired result/audit incomplete')
    need(pool['frozen_after_native_visual_review'] is True,'Native photo pool not frozen')
    need(contract['pools']==POOLS and contract['steps_per_arm']==400 and contract['batch_frames']==2 and contract['seed']==0,'Paired sampling/budget changed')
    need(contract['optimizer']=='Adam' and contract['RGB_lr']==.01 and contract['RGB_prior_MSE_weight']==1e-4,'Optimizer contract changed')
    need(receipt['initial_RGB_identical'] and receipt['source_geometry_unchanged'] and not receipt['target_migration'],'Source contract failed')
    need(receipt['formal_updates']==800 and receipt['discarded_smoke_restore_updates']==6 and receipt['nativeF']==1684 and receipt['nativeB']==1612,'Real resource ledger changed')
    need(cpu['native_metrics_max_recompute_delta']==0 and not cpu['query_core_used_in_loss'],'Independent metric/query audit failed')
    assets=[]
    def asset(path,relative,wh=None):
        path=Path(path).resolve(strict=True)
        if wh:
            with Image.open(path) as image:need(list(image.size)==wh,'Image crop mismatch: '+str(path))
        relative=(PREFIX/relative).as_posix();assets.append({'source':str(path),'relative':relative,'sha256':sha(path),'bytes':path.stat().st_size})
        return relative
    initial={r['raw']:r for r in receipt['initial']};arms={r['label']:r for r in receipt['arms']}
    need(set(initial)==set(contract['frames']) and set(arms)==set(POOLS),'Paired actual frames/arms incomplete')
    frame_rows=[];metric_lookup={}
    pool_records={r['raw']:r for r in pool['records']}
    for raw in contract['frames']:
        init=initial[raw];box=init['bbox_xyxy'];wh=[box[2]-box[0],box[3]-box[1]];folder=pair/'initial'/f'{raw:03d}'
        item={'raw':raw,'split':'QUERY' if raw in QUERY else 'stress' if raw in STRESS else 'TRAIN','bbox_xyxy':box,'crop_wh':wh,'images':{},'metrics':{},'score_masks':{}}
        with np.load(folder/'source_float.npz',allow_pickle=False) as data:
            reference={k:data[k].copy() for k in ('alpha','score_mask_raw','score_mask_pinhole','observation_raw','observation_pinhole_canonical')}
        for domain,obs,render,mask,metric_key in [('raw','observation_raw.png','render_raw.png','score_mask_raw.png','raw_camera_metrics'),('pinhole','observation_pinhole_canonical.png','render_pinhole_canonical.png','score_mask_pinhole.png','canonical_metrics')]:
            item['images'][domain]={'photo':asset(folder/obs,Path('initial')/f'{raw:03d}'/obs,wh),'initial':asset(folder/render,Path('initial')/f'{raw:03d}'/render,wh),'score_mask':asset(folder/mask,Path('initial')/f'{raw:03d}'/mask,wh)}
            need(int(reference['score_mask_raw' if domain=='raw' else 'score_mask_pinhole'].sum())==init[metric_key]['score_n'],'Metric mask denominator differs')
            item['metrics'][domain]={'initial':init[metric_key]}
            item['score_masks'][domain]={'sha256':sha(folder/mask),'score_n':init[metric_key]['score_n']}
            for label in POOLS:
                arm=arms[label];row=next(r for r in arm['final'] if r['raw']==raw);stage=pair/label/'final_reloaded'/f'{raw:03d}'
                need(row['bbox_xyxy']==box and row[metric_key]['score_n']==init[metric_key]['score_n'],'Arm crop/denominator differs')
                need(sha(stage/mask)==sha(folder/mask),'Fixed evaluation mask differs across arms')
                item['images'][domain][label]=asset(stage/render,Path(label)/f'{raw:03d}'/render,wh)
                item['metrics'][domain][label]=row[metric_key]
                with np.load(stage/'source_float.npz',allow_pickle=False) as data:
                    for key,value in reference.items():need(np.array_equal(data[key],value),'Shared alpha/masks/target differ: '+key)
        # The physical-core panel shows the actual native pinhole polygon, independent of scoring masks.
        uv=pool_records[raw]['core_native_uv'];canvas=Image.new('L',(1920,1280));ImageDraw.Draw(canvas).polygon([tuple(p) for p in uv],fill=255)
        core_path=HERE/'derived'/f'{raw:03d}_physical_core_pinhole.png';core_path.parent.mkdir(parents=True,exist_ok=True);canvas.crop(box).save(core_path)
        item['physical_core']=asset(core_path,Path('masks')/core_path.name,wh)
        frame_rows.append(item)
    arm_rows=[]
    for label in POOLS:
        arm=arms[label];sampling=read(pair/(label+'_sampling.json'));history=read(pair/label/'history.json')
        need(len(sampling['schedule'])==400 and all(len(x)==2 and len(set(x))==2 and set(x)<=set(POOLS[label]) for x in sampling['schedule']),'Invalid 400-step sampling schedule')
        counts=Counter(x for batch in sampling['schedule'] for x in batch);expected=400 if label=='two_frames' else 100
        need(counts==Counter({raw:expected for raw in POOLS[label]}) and sampling['counts']=={str(k):v for k,v in counts.items()},'Sampling counts differ')
        need(arm['steps']==400 and arm['reload_render_max_abs']==0 and arm['non_RGB_fields_exact'] and arm['halo_RGB_exact'],'Final reload/freeze failed')
        need([r['step'] for r in history]==[1]+list(range(25,401,25)),'Loss sampling records changed')
        arm_rows.append({'id':label,'label':'两帧池' if label=='two_frames' else '八帧池','pool':POOLS[label],'steps':400,'counts':dict(sorted(counts.items())),'history':history,'reload_render_max_abs':arm['reload_render_max_abs'],'source_packet':asset(pair/label/'source_patch_02.npz',Path('assets')/(label+'_source_patch_02.npz'))})
    evidence=[]
    paths=[(pair/'receipt.json','配对运行完整结果'),(pair/'contract.json','同预算配对合同'),(pair/'initialization_equivalence.json','共同初始化核验'),(exposure_root/'paired_cpu_audit.json','独立CPU状态与指标审计'),(exposure_root/'shared_exposure.json','共享曝光标定'),(exposure_root/'strict_nonoverlap_validation.json','严格不重叠曝光验证'),(exposure_root/'calibration_support_manifest.json','曝光支撑与计数'),(pool_root/'pool_plan.json','冻结实拍池'),(pair/'two_frames_sampling.json','两帧池400步抽样'),(pair/'eight_frames_sampling.json','八帧池400步抽样')]
    for i,(path,label) in enumerate(paths):evidence.append({'name':label,'path':asset(path,Path('evidence')/(f'{i:02d}_'+path.name))})
    payload={'schema':'source_multiframe_pair_report.v1','complete':True,'run_id':contract['run'],'contract':contract,'frames':frame_rows,'arms':arm_rows,'cpu_audit':cpu,'exposure':exposure,'strict_exposure':strict,'calibration_support':support,
             'resources':{k:receipt[k] for k in ('nativeF','nativeB','formal_updates','discarded_smoke_restore_updates','resume_next_update_RGB_max_abs','new_diffusion_calls','target_migration')},'evidence':evidence,
             'architecture':asset(HERE/'architecture_components.svg' if (HERE/'architecture_components.svg').exists() else HERE.parent/'architecture_components.svg',Path('architecture_components.svg')),'loss_curve':asset(loss_plot(pair),Path('loss_400_steps.svg'))}
    return payload,assets


def integrate(index,payload,assets):
    original=index.read_bytes();before=historical_state(original)
    text=original.decode('utf8');clean=re.sub(re.escape(BEGIN)+r'.*?'+re.escape(END),'',text,flags=re.S)
    literal=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    block=BEGIN+'\n<script id="source-multiframe-pair-data" type="application/json">'+literal+'</script>\n<script>\n'+(HERE/'pair_viewer.js').read_text(encoding='utf8')+'\n</script>\n'+END
    position=clean.rfind('</html>');need(position>=0,'Missing original closing HTML tag')
    result=(clean[:position]+block+'\n'+clean[position:]).encode('utf8');need(historical_state(result)==before,'Historical evidence changed')
    backup=HERE/'backups'/('index.'+hashlib.sha256(original).hexdigest()[:16]+'.before_multiframe.html');backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():shutil.copy2(index,backup)
    for plan in assets:
        src=Path(plan['source']);need(sha(src)==plan['sha256'],'Source artifact changed after validation')
        target=(index.parent/plan['relative']).resolve();need(target.is_relative_to((index.parent/PREFIX).resolve()),'New asset escaped its own directory')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);need(sha(target)==plan['sha256'],'Copied asset hash differs')
    temporary=index.with_name(index.name+'.multiframe.tmp');temporary.write_bytes(result);os.replace(temporary,index)
    return {'passed':True,'index':str(index),'backup':str(backup),'before_html_sha256':hashlib.sha256(original).hexdigest(),'after_html_sha256':sha(index),'historical_evidence_raw_bytes_unchanged':True,'before':before,'after':historical_state(index.read_bytes()),'new_assets':len(assets),'default_raw':100,'default_pixel_zoom':1}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pair',type=Path,default=HERE/'takeback/paired400_seed0');parser.add_argument('--exposure',type=Path,default=HERE.parent/'exposure_audit');parser.add_argument('--pool',type=Path,default=HERE.parent/'pool_audit');parser.add_argument('--index',type=Path,default=DEFAULT_INDEX);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    payload,assets=build(args.pair,args.exposure,args.pool);write(HERE/'pair_payload.json',payload);write(HERE/'pair_asset_plan.json',assets)
    if args.apply:
        audit=integrate(args.index.resolve(),payload,assets);write(HERE/'integration_audit.json',audit);print(json.dumps(audit))
    else:print(json.dumps({'validated':True,'applied':False,'frames':len(payload['frames']),'arms':len(payload['arms']),'assets':len(assets),'loss_records_per_arm':[len(a['history']) for a in payload['arms']]}))
