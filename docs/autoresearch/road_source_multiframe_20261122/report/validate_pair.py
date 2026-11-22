"""Read-only validation and macro summaries for the real paired report handoff."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
POOLS={'pool2':[88,115],'pool8':[88,90,92,94,96,99,104,115]}
QUERIES=[100,110,117]
STRESS=[86]
BANDS=[1.2,5/3,3.0]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def finite(value,label):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Expected finite value: '+label)


def require(value,message):
    if not value:
        raise ValueError(message)


def sha_string(value,label):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value),label+' needs an actual SHA256')


def validate_plan():
    if not (HERE/'plan_contract.json').exists():
        return {'plan_validated':True,'actual_results_loaded':False,'formal_steps_per_arm':400,'formal_draws_per_arm':800,'main_query_raws':QUERIES,'stress_query_raws':STRESS}
    p=read(HERE/'plan_contract.json')
    require({k:v['train_raws'] for k,v in p['arms'].items()}==POOLS,'Plan pools changed')
    require(p['main_query_raws']==QUERIES and p['stress_query_raws']==STRESS,'Plan split changed')
    require(p['optimizer']['name']=='Adam' and p['optimizer']['steps']==400 and p['optimizer']['lr']==.01 and p['optimizer']['prior_weight']==1e-4,'Plan optimizer changed')
    require(p['sampler']=={'name':'balanced_shuffle_bag','frames_per_step':2,'seed':0,'total_formal_frame_draws_per_arm':800},'Plan sampler changed')
    require(not set(QUERIES+STRESS)&set(POOLS['pool8']),'QUERY/stress overlaps source fitting')
    return {'plan_validated':True,'actual_results_loaded':False,'formal_steps_per_arm':400,'formal_draws_per_arm':800,'main_query_raws':QUERIES,'stress_query_raws':STRESS}


def validate_actual(path,audit_path=None):
    path=Path(path).resolve();data=read(path);contract=data['contract'];root=path.parent
    pools={'two_frames':POOLS['pool2'],'eight_frames':POOLS['pool8']}
    require(data['complete'] is True and contract['pools']==pools,'Actual pair incomplete or pools differ')
    require(contract['steps_per_arm']==400 and contract['batch_frames']==2 and contract['seed']==0,'Actual formal budget/sampler differs')
    require(contract['optimizer']=='Adam' and contract['RGB_lr']==.01 and contract['RGB_prior_MSE_weight']==1e-4,'Actual optimizer differs')
    require(data['initial_RGB_identical'] and data['source_geometry_unchanged'] and not data['target_migration'],'Actual init/freeze failed')
    require(data['formal_updates']==800 and data['discarded_smoke_restore_updates']==6,'Actual update ledger differs')
    require(data['resume_next_update_RGB_max_abs']==0,'Actual full-state next-update restore differs')
    initial={r['raw']:r for r in data['initial']};summaries={}
    for arm in data['arms']:
        label=arm['label'];require(label in pools and arm['pool']==pools[label] and arm['steps']==400,'Actual arm changed')
        sampling=read(root/(label+'_sampling.json'));schedule=sampling['schedule'];counts=Counter(v for batch in schedule for v in batch)
        require(len(schedule)==400 and all(len(b)==2 and len(set(b))==2 and set(b)<=set(pools[label]) for b in schedule),'Invalid actual complete sampling schedule')
        expected=400 if label=='two_frames' else 100;require(counts==Counter({raw:expected for raw in pools[label]}),'Actual counts not balanced')
        history=read(root/label/'history.json');require([r['step'] for r in history]==[1]+list(range(25,401,25)),'Actual 17-point loss log differs')
        for row in history:
            finite(row['pre_update_batch_loss'],'actual sampled loss');require(row['sampled_frames']==schedule[row['step']-1],'Actual history and full schedule disagree')
        require(arm['reload_render_max_abs']==0 and arm['non_RGB_fields_exact'] and arm['halo_RGB_exact'],'Actual asset reload/freeze failed')
        final={r['raw']:r for r in arm['final']};require(set(final)==set(initial),'Actual evaluated frame sets differ')
        for raw,row in final.items():
            for domain in ('raw_camera_metrics','canonical_metrics'):
                require(row[domain]['score_n']==initial[raw][domain]['score_n'],'Actual evaluation denominator differs')
                finite(row[domain]['RGB_MAE'],'actual RGB error')
        summaries[label]={'formal_steps':400,'formal_draws':sum(counts.values()),'draw_counts':dict(sorted(counts.items())),'loss_record_count':len(history),'primary_raw_RGB_MAE_macro':sum(final[r]['raw_camera_metrics']['RGB_MAE'] for r in QUERIES)/3}
    audit=Path(audit_path) if audit_path else HERE.parent/'exposure_audit/paired_cpu_audit.json'
    if audit.exists():
        proof=read(audit);require(proof['all_required_state_data_checks_passed'] and proof['native_metrics_max_recompute_delta']==0,'Actual independent CPU audit failed')
    return {'passed':True,'actual_results_loaded':True,'arms':summaries,'nativeF':data['nativeF'],'nativeB':data['nativeB'],'formal_updates':800,'extra_updates':6,'formal_schedule_steps_per_arm':400,'loss_records_per_arm':17}


def validate(path):
    path=Path(path).resolve();result=read(path)
    require(result.get('schema')=='source_multiframe_pair_report_input.v1' and result.get('complete') is True,'Complete real pair handoff required')
    require(result.get('pool_frozen') is True,'Photo pool review not frozen')
    shared=result['shared']
    for key in ('source_init_sha256','core_RGB_init_sha256','non_RGB_init_sha256','halo_RGB_init_sha256','core_row_ids_sha256'):
        sha_string(shared[key],key)
    for key in ('camera_calibration_id','exposure_calibration_id','metric_protocol_id','data_loss_definition'):
        require(bool(shared.get(key)),'Missing common '+key)
    frames={f['raw']:f for f in result['frames']}
    require(len(frames)==len(result['frames']),'Duplicate frame records')
    require(set(QUERIES+STRESS).issubset(frames),'Missing common QUERY or stress metadata')
    for raw,f in frames.items():
        role='QUERY' if raw in QUERIES else 'stress' if raw in STRESS else 'TRAIN'
        require(f['split']==role,'TRAIN/QUERY/stress role differs: '+str(raw))
        box=f['bbox_xyxy'];require(len(box)==4 and f['crop_wh']==[box[2]-box[0],box[3]-box[1]],'Invalid crop metadata')
        for domain in ('raw','pinhole'):
            d=f['domains'][domain];require(type(d['score_n']) is int and d['score_n']>0,'Empty or invalid source score domain')
            sha_string(d['score_mask_sha256'],'source score mask');sha_string(d['target_sha256'],'shared target')
        e=f['exposure'];finite(e['gain'],'exposure gain');require(e['gain']>0 and e['source_core_RGB_samples_used']==0,'Exposure uses source core or invalid gain')
        for key in ('fit_world_cells','validation_world_cells','matching_world_queries'):
            require(type(e[key]) is int and e[key]>=0,'Calibration count missing: '+key)
        strict=e['strict_validation']
        require(strict['no_bilinear_support_pixel_used_in_any_fit'] is True and strict['source_and_anchor_both_checked'] is True,'Exposure validation support overlaps fitting')
        if role!='stress':
            require(strict['minimum8cells_pass'] is True and strict['residual_thresholds_pass'] is True,'Main TRAIN/QUERY exposure validation failed')
    arms={a['id']:a for a in result['arms']}
    require(set(arms)==set(POOLS) and len(result['arms'])==2,'Expected exactly paired pool2/pool8')
    summaries={}
    for arm_id,a in arms.items():
        require(a['train_raws']==POOLS[arm_id],'Arm training pool differs')
        require(a['source_init_sha256']==shared['source_init_sha256'] and a['exposure_calibration_id']==shared['exposure_calibration_id'],'Arms do not share initialization/exposure')
        require(a['formal_optimizer_updates']==400 and a['non_RGB_unchanged'] is True and a['halo_RGB_unchanged'] is True,'Formal budget or fixed-source contract failed')
        opt=a['optimizer'];require(opt['name']=='Adam' and opt['lr']==.01 and opt['prior_weight']==1e-4 and opt['prior_kind']=='MSE_to_common_original_RGB','Optimizer contract differs')
        trace_path=Path(a['trace']);trace_path=trace_path if trace_path.is_absolute() else path.parent/trace_path
        trace=[json.loads(line) for line in trace_path.read_text(encoding='utf8').splitlines() if line.strip()]
        require(len(trace)==400 and [r['optimizer_step'] for r in trace]==list(range(1,401)),'Formal trace is not 400 ordered updates')
        counts=Counter()
        for record in trace:
            samples=record['sampled_raws'];require(len(samples)==2 and len(set(samples))==2 and set(samples).issubset(POOLS[arm_id]),'Frame draw outside assigned pool or duplicate draw')
            counts.update(samples)
            for key in ('data_loss','prior_mse','total_loss'):
                finite(record[key],key)
        expected=400 if arm_id=='pool2' else 100
        require(counts==Counter({raw:expected for raw in POOLS[arm_id]}),'Unbalanced formal shuffle-bag counts')
        reload=a['reload'];require(reload['checkpoint_step']==400,'Reload does not restore step400')
        for key in ('RGB_equal','optimizer_equal','RNG_equal','sampler_equal'):
            require(reload.get(key) is True,'Full reload proof failed: '+key)
        require(any(c['raw'] in QUERIES for c in reload['render_checks']),'Reload needs a real common-QUERY render check')
        for c in reload['render_checks']:
            finite(c['max_abs'],'reload render difference');require(c['max_abs']>=0,'Negative reload render error')
        resources=a['resources']
        for key in ('nativeF','nativeB','extra_optimizer_updates'):
            require(type(resources.get(key)) is int and resources[key]>=0,'Actual resource counter missing: '+key)
        require(resources.get('formal_optimizer_updates')==400,'Resource ledger formal step count differs')
        require(resources.get('new_diffusion_calls')==0 and resources.get('new_target_migration_updates')==0,'Unexpected generation or target migration')
        metrics={}
        for row in a['metrics']:
            key=(row['raw'],row['domain']);require(key not in metrics,'Duplicate final metric record')
            require(row['raw'] in frames and row['domain'] in {'raw','pinhole'},'Unknown metric frame/domain')
            domain=frames[row['raw']]['domains'][row['domain']]
            for field in ('score_n','score_mask_sha256','target_sha256'):
                require(row[field]==domain[field],'Metric denominator/target differs from the fixed common domain')
            finite(row['RGB_MAE'],'RGB_MAE');require(row['RGB_MAE']>=0,'Negative RGB error')
            require(len(row['spectral'])==3,'Three actual native-pixel spectral bands required')
            for sigma in BANDS:
                bands=[b for b in row['spectral'] if abs(b['sigma_native_px']-sigma)<1e-6]
                require(len(bands)==1,'Missing or duplicate sigma')
                for field in ('HF_cosine','HF_RMS_ratio'):
                    if bands[0][field] is not None:
                        finite(bands[0][field],field)
            metrics[key]=row
        require(all((raw,d) in metrics for raw in QUERIES+STRESS for d in ('raw','pinhole')),'Primary QUERY/stress final metrics missing')
        summary={'formal_optimizer_updates':400,'formal_draw_counts':dict(sorted(counts.items())),'main_QUERY':{},'stress_raws':STRESS}
        for domain in ('raw','pinhole'):
            rows=[metrics[(raw,domain)] for raw in QUERIES]
            values=[next(b for b in row['spectral'] if abs(b['sigma_native_px']-5/3)<1e-6)['HF_cosine'] for row in rows]
            summary['main_QUERY'][domain]={'aggregation':'equal_frame_macro','frame_count':3,'score_n_by_frame':{r['raw']:r['score_n'] for r in rows},'RGB_MAE_macro':sum(r['RGB_MAE'] for r in rows)/3,'HF_sigma1p667_cosine_macro':sum(values)/3 if all(v is not None for v in values) else None}
        summaries[arm_id]=summary
    return {'passed':True,'actual_results_loaded':True,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'paired_arms':summaries,'exposure_and_source_score_denominators_separate':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--cpu-audit',type=Path)
    args=parser.parse_args()
    if args.plan_only:
        result=validate_plan()
    elif args.input:
        validate_plan();result=validate_actual(args.input,args.cpu_audit) if 'contract' in read(args.input) and 'arms' in read(args.input) else validate(args.input)
    else:
        parser.error('Use --plan-only or --input REAL_HANDOFF.json')
    if args.output:
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))
