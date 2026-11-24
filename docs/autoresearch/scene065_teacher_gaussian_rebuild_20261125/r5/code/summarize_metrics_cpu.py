"""One-shot lightweight R5 metrics from fixed evaluations and execution records."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,math
import numpy as np

R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_gaussian_rebuild_20261125_r5')
GROUPS=('xyz','log_scales','raw_quaternion','opacity_logits','dc')
KINDS=('birth','clone','split','prune')

def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except FileNotFoundError:return default

def pin(path):
    path=Path(path);return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def balanced(values,groups):
    return float(np.mean([np.mean([values[raw]for raw in group])for group in groups]))

def fixed_summary(point,cfg):
    rows=point['rows'];groups=cfg['view_groups'];weights=cfg['category_weights'];cats={};rgb={}
    assert sorted(x['raw']for x in rows)==sorted(cfg['fixed_monitor_frames'])
    for row in rows:
        raw=row['raw'];rgb[raw]=0.
        for kind,item in row['categories'].items():
            q=cats.setdefault(kind,dict(pixels=0,fixed_weight_denominator=0.,weighted_error_numerator=0.,by_raw={}))
            mse=item['weighted_MSE'];den=float(item['weight_denominator']);q['pixels']+=item['pixels'];q['fixed_weight_denominator']+=den
            q['by_raw'][raw]=float(mse)if mse is not None else 0.
            if mse is not None:q['weighted_error_numerator']+=den*mse;rgb[raw]+=weights[kind]*mse
    for q in cats.values():
        q['balanced_group_MSE']=balanced(q.pop('by_raw'),groups)
        q['denominator_pooled_MSE']=q['weighted_error_numerator']/q['fixed_weight_denominator']if q['fixed_weight_denominator']else None
    parts={}
    for name in ['protection_RGB','opacity_inside','opacity_outside']:
        den=sum(row[name]['pixels']for row in rows);mse={row['raw']:row[name]['MSE']for row in rows}
        part=dict(fixed_pixels=den,balanced_group_MSE=balanced(mse,groups),
                  denominator_pooled_MSE=sum(row[name]['pixels']*row[name]['MSE']for row in rows)/den if den else None)
        if all(row[name].get('mean_alpha')is not None for row in rows):
            part['balanced_group_mean_alpha']=balanced({row['raw']:row[name]['mean_alpha']for row in rows},groups)
            part['denominator_pooled_mean_alpha']=sum(row[name]['pixels']*row[name]['mean_alpha']for row in rows)/den if den else None
        parts[name]=part
    return dict(unique_step=int(point['step']),RGB_categories=cats,balanced_RGB_MSE_only=balanced(rgb,groups),
        protection_RGB=parts['protection_RGB'],coverage_inside=parts['opacity_inside'],coverage_outside=parts['opacity_outside'],
        balanced_fixed_total_without_prior=balanced({row['raw']:row['fixed_objective_without_prior']for row in rows},groups),
        rows=rows)

def main():
    run=R/'training_v1';phases={};bindings={};all_losses=[];curves=[]
    for phase in ['zero','probe','train']:
        folder=run/phase;receipt=read(folder/'receipt.json');losses=read(folder/'losses.json',[])
        binding=read(folder/'bindings.json')or(receipt or{}).get('bindings',{})
        if binding:bindings=binding
        phases[phase]=dict(receipt=receipt,losses_source=str(folder/'losses.json'),recorded_loss_rows=len(losses))
        all_losses.extend((phase,x)for x in losses)
        for point in read(folder/'fixed_eval_curve.json',[]):curves.append((phase,point))
        native=read(folder/'native/native30k/native30k_load_receipt.json')
        composition=read(folder/'native/composition.json')
        phases[phase]['frozen_native_evidence']=dict(native_load_receipt=native,composition=composition,
            frozen_assets=binding.get('frozen_assets'))
    cfg=read(bindings['configuration']['path']);cache_spec=bindings['cache_manifest'];cache=read(cache_spec['path'])
    assert pin(cache_spec['path'])['sha256']==cache_spec['sha256']
    fixed=[dict(phase=phase,**fixed_summary(point,cfg))for phase,point in curves]
    gradient={name:dict(updates=0,finite_updates=0,positive_gradient_updates=0,positive_delta_updates=0,
        gradient_nonzero_values=0,changed_parameter_values=0,gradient_l2_min=None,gradient_l2_max=0.,delta_l2_min=None,delta_l2_max=0.)for name in GROUPS}
    events=[];counts={k:0 for k in KINDS};point_history=[dict(step=0,rows=3126,source='teacher_seed')]
    for phase,entry in all_losses:
        record=entry['record']
        for name in GROUPS:
            g=record['gradients_by_group'][name];d=record['parameter_delta_by_group'][name];q=gradient[name];q['updates']+=1
            q['finite_updates']+=int(g['finite']);q['positive_gradient_updates']+=int(g['l2']>0);q['positive_delta_updates']+=int(d['l2']>0)
            q['gradient_nonzero_values']+=g['nonzero_count'];q['changed_parameter_values']+=d['nonzero_count']
            q['gradient_l2_min']=g['l2']if q['gradient_l2_min']is None else min(q['gradient_l2_min'],g['l2'])
            q['delta_l2_min']=d['l2']if q['delta_l2_min']is None else min(q['delta_l2_min'],d['l2'])
            q['gradient_l2_max']=max(q['gradient_l2_max'],g['l2']);q['delta_l2_max']=max(q['delta_l2_max'],d['l2'])
        event=entry.get('event')or record.get('topology')
        if event:
            summary={k:event[k]for k in ['logical_step','rows_before','rows_after',*KINDS]}
            summary['phase']=phase;events.append(summary);point_history.append(dict(step=entry['step'],rows=event['rows_after'],source='unique topology event'))
            for kind in KINDS:counts[kind]+=int(event[kind])
    probe=phases['probe']['receipt'];train=phases['train']['receipt'];replay=read(run/'train/step32_replay.json')
    unique=len(all_losses);latest=max((x['step']for _,x in all_losses),default=0)
    replay_count=int(bool(replay));actual=unique+replay_count
    replay_topology={k:0 for k in KINDS}
    if replay:
        original32=next((e for e in events if e['logical_step']==32),None)
        if original32:replay_topology={k:original32[k]for k in KINDS}
    if train:
        ledger=train.get('execution_ledger',{});unique=ledger.get('unique_updates',unique);actual=ledger.get('actual_optimizer_updates',actual);replay_count=ledger.get('engineering_replay_updates',replay_count)
        counts=ledger.get('topology_unique',counts);replay_topology=ledger.get('topology_replay',replay_topology)
    wiring=read(R/'gpu_wiring_v1/receipt.json',{});active=sum(x['active_seconds']for _,x in all_losses)
    accounting=dict(latest_saved_unique_step=latest,formal_unique_updates=unique,formal_actual_optimizer_updates=actual,
        engineering_replay_updates=replay_count,engineering_wiring_optimizer_updates=wiring.get('engineering_optimizer_updates',0),
        recorded_physical_optimizer_updates=actual+wiring.get('engineering_optimizer_updates',0),
        counter_source='closed phase execution ledger'if train else 'completed optimizer rows in saved losses plus successful fresh-process replay',
        topology_unique=counts,topology_replay=replay_topology,
        unique_training_seconds_through_saved_losses=active,projected_unique_training_seconds_from_probe=probe.get('projected_unique_training_seconds')if probe else None,
        source_native_parameters_in_optimizer=False,old_r4_checkpoint_or_color_loaded=False)
    readouts=[]
    for phase in ['zero','probe','train']:
        for p in sorted((run/phase).glob('actual_step*')):
            receipt=read(p/'receipt.json')
            if receipt:readouts.append(dict(phase=phase,step=int(p.name.removeprefix('actual_step')),source=str(p),receipt=receipt))
    result=dict(schema='scene065.full_teacher_gaussian.metrics.v1',updated_utc=datetime.now(timezone.utc).isoformat(),CPU_only=True,GPU_started=False,
        complete=bool(train and train['complete']and train['logical_step']==2000),bindings=bindings,
        definitions=dict(RGB='fixed eight-frame canonical-linear category MSE, reported separately from coverage/protection and stochastic loss',
            balance='mean of four group means [[106..110],[120],[137],[165]]; absent RGB category contributes0 to its sampled view',
            RGB_weights=cfg['category_weights'],protection_RGB='LDR MSE against frozen retained context, preservation reference only',
            coverage='inside(red minus fixed green): alphaT->1; full-image outside: alphaT->0; separate immutable denominators',
            split='new child rows; prune includes removed split parents',engineering_state_used_as_formal_initialization=False,
            source='R5 direct original FULL teacher RGB + explicitly nonindependent reprojections; no R4 LF/fitted appearance comparison'),
        fixed_monitors=fixed,cache_denominators=dict(source=cache_spec,inventory=cache['inventory'],frames=cache['frame_rows']),
        five_parameter_group_updates=gradient,point_count_history=point_history,topology_events_unique=events,accounting=accounting,
        fresh_process_replay=replay,phase_evidence=phases,available_actual_readouts=readouts)
    path=R/'results/metrics.json';path.parent.mkdir(exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)
    wanted=[0,32,100,200,300,400]
    print(json.dumps(dict(output=str(path),selected_fixed=[{k:r[k]for k in ['unique_step','balanced_RGB_MSE_only','RGB_categories','protection_RGB','coverage_inside','coverage_outside']}for r in fixed if r['unique_step']in wanted],
        accounting=accounting,point_history=point_history,all_five_finite_and_changed=all(q['finite_updates']==q['updates']and q['positive_delta_updates']==q['updates']for q in gradient.values())),ensure_ascii=False))

if __name__=='__main__':main()
