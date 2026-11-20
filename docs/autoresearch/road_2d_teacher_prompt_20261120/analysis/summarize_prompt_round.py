"""Read-only CPU review of the complete60 prompt/seed scores.

No truth images, generated images, models or GPU are opened. Smoke calls are not
accepted as formal scores. This program summarizes every frozen seed; it never
selects a best seed, chosen image, automatic prompt winner or second round.

python summarize_prompt_round.py --score scores.json --plan plan.json \
  --predictions predictions.json --config ../next_round_config.json --out-dir review
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path

IDS=['visible_scene081_raw000_v0','visible_scene081_raw000_v1','visible_scene081_raw050_v0','visible_scene081_raw050_v1','scene065_raw086','scene065_raw110']
SEEDS=[101,202,303,404,505]
PROMPTS=['legacy','material_match_v1']
ARM='full1152_T0'
CHECKPOINT_SHA='a75423e126b93eca22e0ce587782b3dc4edd76654d6a929302118aa45bda14ae'
INFERENCE={'model_wh':[1152,768],'view':'full1152','adapter':'T0','scheduler':'DDIM','inference_steps':20,'guidance':3.,'eta':0.,'strength':1.,'dtype':'bfloat16','weight_variant':'fp16','negative_prompt':''}
BANDS=[[.025,.0625],[.0625,.125],[.125,.2],[.2,.3],[.3,.5]]
ROLES={'synthetic_certified_hidden_target':'synthetic_certified_hidden_target','known_visible_road_preservation_only':'real_known_visible_control'}
FRAMES={sid:('scene081_raw000' if 'raw000' in sid else 'scene081_raw050') for sid in IDS[:4]}
FRAMES.update({sid:sid for sid in IDS[4:]})

class ReviewError(RuntimeError):pass
def need(x,msg):
    if not x:raise ReviewError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def numeric(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def exact_mean(v):return statistics.mean(v) if v and all(numeric(x) for x in v) else None
def logdistance(v):return abs(math.log2(max(v,1e-12))) if numeric(v) and v>=0 else None
def lower_is_closer(field):return field in ['deltaE76_mean','RGB_MAE_0_255'] or '_abs_log2_' in field

def checked_pin(record):
    need(isinstance(record,dict) and isinstance(record.get('path'),str) and isinstance(record.get('sha256'),str),'runtime pin malformed')
    path=Path(record['path']);need(path.is_file(),'pinned runtime evidence unavailable: '+str(path))
    need(sha(path)==record['sha256'],'runtime evidence pin changed: '+str(path))
    return path

def checked_flight(flight,config):
    need(flight.get('schema')=='road_2d_teacher_next_round_preflight.v1','runtime preflight source schema mismatch')
    need(flight.get('runtime_preflight_passed') is True and flight.get('runtime_assets_verified') is True,'runtime preflight/assets failed')
    need(flight.get('planned_diffusion_calls')==60 and flight.get('historical_replay_expected')==12,'runtime preflight60/12 contract changed')
    need(flight.get('GT_opened') is False and flight.get('training_updates')==0,'runtime preflight scope changed')
    token=flight['token_check']
    need(token.get('checked') is True and token.get('status')=='passed_no_truncation' and token.get('truncation_allowed') is False,'runtime tokenizer gate failed')
    need(len(token['prompts'])==2 and {r['prompt_id'] for r in token['prompts']}==set(PROMPTS),'runtime prompt token identities changed')
    counts={r['prompt_id']:r['tokens_including_BOS_EOS'] for r in token['prompts']}
    need(all(numeric(v) and 0<v<=77 for v in counts.values()) and all(r.get('exceeds_limit') is False for r in token['prompts']),'prompt token truncation')
    if config['current_prompt_comparison_round']==1:
        need(counts=={'legacy':23,'material_match_v1':40},'frozen round1 prompt token23/40 mismatch')
    return token

def runtime_evidence(predictions,plan,config):
    """Read existing execution evidence; never insert fields into predictions."""
    p=predictions
    if 'worker_receipts' not in p:
        need('runtime_preflight' in p,'runtime evidence missing: single preflight or four pinned workers required')
        token=checked_flight(p['runtime_preflight'],config)
        return {'runtime_execution_mode':'single_process','runtime_worker_count':1,'token_check':token,'runtime_preflight':p['runtime_preflight']}
    need(plan is not None,'worker evidence requires the bound frozen plan')
    records=p['worker_receipts'];need(isinstance(records,list) and len(records)==4,'exactly four worker receipt pins required')
    need(len({r.get('path') for r in records})==4,'duplicate worker receipt path')
    evidence=[];indices=set();job_ids=[];source_hashes=None;common_tokens=None
    for record in records:
        path=checked_pin(record);w=read(path)
        need(w.get('schema')=='road_2d_teacher_prompt_round_worker.v1' and w.get('mode')=='worker','formal worker receipt source schema/mode mismatch')
        need(w.get('complete') is True and w.get('saved_receipt_and_outputs_restored') is True,'worker receipt incomplete/not restored')
        need(w.get('manifest')==p['manifest'],'worker manifest/prediction binding mismatch')
        need(w.get('GT_opened') is False and w.get('training_updates')==0,'worker scope changed')
        need(w.get('adapter_unchanged') is True and w.get('checkpoint_unchanged') is True,'worker parameters changed')
        index=w.get('worker_index');need(isinstance(index,int) and index in range(4) and index not in indices and w.get('worker_count')==4,'worker index/count mismatch')
        indices.add(index);selected=[j['id'] for n,j in enumerate(plan['jobs']) if n%4==index]
        need(w.get('selected_job_ids')==selected and len(selected)==15,'worker selected jobs differ from fixed partition')
        need(w.get('attempted_diffusion_calls')==w.get('completed_diffusion_calls')==len(w['jobs'])==15,'worker15 completed calls required')
        need([j['id'] for j in w['jobs']]==selected,'worker returned job order/identities changed')
        lookup={j['id']:j for j in p['jobs']}
        need(all(j==lookup.get(j['id']) for j in w['jobs']),'merged job differs from pinned worker receipt')
        sources=w.get('execution_sources',{})
        need(set(sources)=={'infer_next_round.py','teacher_engine.py','smoke_and_workers.py'},'worker execution source package mismatch')
        for source in sources.values():checked_pin(source)
        hashes={name:source['sha256'] for name,source in sources.items()}
        need(hashes['infer_next_round.py']==plan['source_pins']['runner']['sha256'],'worker inference entry differs from frozen plan')
        if source_hashes is None:source_hashes=hashes
        need(hashes==source_hashes,'workers executed different source packages')
        token=checked_flight(w['runtime_preflight'],config)
        if common_tokens is None:common_tokens=token
        need(token==common_tokens,'workers used different prompt/tokenizer checks')
        job_ids.extend(selected)
        evidence.append({'worker_index':index,'receipt':record,'schema':w['schema'],'manifest':w['manifest'],
          'completed_calls':15,'runtime_preflight':w['runtime_preflight'],'execution_sources':sources})
    need(indices==set(range(4)) and len(job_ids)==len(set(job_ids))==60,'worker union is not the fixed60 jobs')
    return {'runtime_execution_mode':'four_pinned_worker_receipts','runtime_worker_count':4,'token_check':common_tokens,
      'runtime_workers':sorted(evidence,key=lambda r:r['worker_index']),'runtime_execution_source_sha256':source_hashes}

def distribution(values,field):
    n=sum(numeric(v) for v in values)
    out={'seed_count_expected':5,'valid_seed_count':n,'status':'complete' if len(values)==n==5 else 'incomplete_no_partial_denominator'}
    if len(values)!=5 or n!=5:
        return out|{k:None for k in ['mean','minimum','maximum','range','population_variance','population_std','worst_reference_error']}
    out.update(mean=statistics.mean(values),minimum=min(values),maximum=max(values),range=max(values)-min(values),
               population_variance=statistics.pvariance(values),population_std=statistics.pstdev(values),
               worst_reference_error=max(values) if lower_is_closer(field) else None)
    return out

def validate(score,config,plan=None,predictions=None,allow_mock=False):
    need(score.get('schema')=='road_2d_teacher_prompt_round_postfreeze_CPU_score.v1' and score.get('complete') is True,'new60 score schema/incomplete')
    mock=score.get('mode')=='CPU_MOCK_VALIDATION_ONLY'
    need(not mock or allow_mock,'CPU fixture is not a GPU result; explicit --allow-cpu-mock required')
    need(mock or score.get('mode')=='FROZEN_PREDICTION_CPU_SCORING','unknown score mode')
    need(score.get('scored_jobs')==60 and score.get('GT_opened_after_gate') is True,'formal60 scores must be gated before GT')
    gate=score['gate']
    need(gate.get('passed') is True and gate.get('jobs')==gate.get('completed_jobs')==60 and gate.get('historical_replays')==12,'60/12 freeze gate incomplete; smoke is excluded')
    need(gate.get('actual_W_outside_exact') is True and gate.get('actual_K_exact') is True,'actual scope gates missing')
    need(gate.get('GT_opened') is False,'generation freeze gate used GT')
    need(config.get('schema')=='road_2d_teacher_next_round.v1' and config.get('selected_arm')==ARM,'wrong frozen configuration')
    need(config['expected_case_ids']==IDS and config['seeds']==SEEDS and [p['id'] for p in config['prompts']]==PROMPTS,'fixed cases/seeds/prompts changed')
    need(config['inference']==INFERENCE and config['planned_diffusion_calls']==config['maximum_diffusion_calls']==60,'method/budget changed')
    for name in ['automatic_candidate_selection','reuse_historical_outputs','second_prompt_round_automatically_enabled','three_dimensional_rendering_changes','GT_used_for_generation_or_candidate_selection']:
        need(config.get(name) is False,'forbidden policy: '+name)
    need(config['training_updates']==config['additional_VAE_control_calls']==0,'unexpected training/control budget')
    need(config['current_prompt_comparison_round'] in [1,2] and config['maximum_prompt_comparison_rounds']==2,'prompt round budget changed')
    expected={(sid,prompt,seed) for sid in IDS for prompt in PROMPTS for seed in SEEDS}
    keys={(j['case_id'],j['prompt_id'],j['seed']) for j in score['jobs']}
    need(len(score['jobs'])==len(keys)==60 and keys==expected,'missing, duplicate or extra formal score combinations')
    cases={c['id']:c for c in score['cases']};need(len(score['cases'])==6 and set(cases)==set(IDS),'six fixed score cases required')
    support=defaultdict(set)
    for j in score['jobs']:
        sid=j['case_id'];role=j['metric_role']
        need(j['arm']==ARM and j['view']=='full1152' and role in ROLES,'arm/view/domain changed')
        need(cases[sid]['metric_role']==role and (sid in IDS[:4])==(role=='synthetic_certified_hidden_target'),'real hidden GT must not be invented')
        for grid in ['native','same_view']:
            m=j[grid];spacing=1. if grid=='native' else 5/3
            need(abs(m['spacing_native_px']-spacing)<1e-12,'grid physical spacing changed')
            need(m['grid_wh']==([1920,1280] if grid=='native' else [1152,768]),'grid dimensions changed')
            need(m['PSD']['band_edges_cycles_per_native_px']==BANDS,'physical frequency bands changed')
            need(abs(m['PSD']['axis_nyquist_cycles_per_native_px']-.5/spacing)<1e-12,'frequency unit changed')
            support[(sid,grid,'color')].add(m['valid_pixels']);support[(sid,grid,'tiles')].add(m['PSD']['fixed_native_tile_count'])
            for key,h in m['HP'].items():
                support[(sid,grid,'HP'+key)].add(h['valid_pixels'])
                if h.get('status')=='insufficient':continue
                need(abs(h['sigma_native_px']-float(key))<1e-12 and abs(h['sigma_this_grid_px']*spacing-float(key))<1e-10,'HP physical sigma changed')
    need(all(len(v)==1 for v in support.values()),'quality denominator changed with prompt/seed')
    fairness={'frozen60_complete':True,'fixed_support_all_prompts_seeds':True,'runtime_plan_checked':False}
    if not mock:need(plan is not None and predictions is not None,'production review requires frozen plan and predictions for pairing audit')
    if plan is not None:
        need(plan['schema']=='road_2d_teacher_prompt_round_plan.v1' and plan['complete'] and plan['configuration']==config,'plan/config mismatch')
        need(plan['checkpoint']['sha256']==CHECKPOINT_SHA,'step600 checkpoint changed')
        need(plan['GT_opened'] is False and plan['training_updates']==0 and plan['automatic_candidate_selection'] is False,'plan scope changed')
        need(plan.get('best_of_N_score_implemented') is False,'best-of-N must remain disabled')
        need({(j['case_id'],j['prompt_id'],j['seed']) for j in plan['jobs']}==expected and len(plan['jobs'])==60,'formal plan combinations changed')
        for case in plan['cases']:
            need(case['view']['model_wh']==[1152,768] and case['view']['plan']=={'output_wh':[1152,768],'window_xyxy':[0,0,1920,1280]},'paired FOV/resolution changed')
            # One immutable legal view record per case is shared by both prompts.
            need(all(k in case['view'] for k in ['I_geo','H_input','C_mat','roles_model','construction']),'shared legal input binding missing')
        fairness['runtime_plan_checked']=True
    if predictions is not None:
        p=predictions
        need(p['schema']=='road_2d_teacher_prompt_round_predictions.v1' and p['complete'],'prediction schema/incomplete')
        need(p['attempted_diffusion_calls']==p['completed_diffusion_calls']==60 and p['historical_replays_passed']==12,'formal60/replay12 mismatch')
        need(p['first_historical_replay_passed'] and p['adapter_unchanged'] and p['checkpoint_unchanged'],'frozen parameters/replays failed')
        need(p['GT_opened'] is False and p['training_updates']==0 and p['automatic_candidate_selection'] is False,'prediction scope changed')
        actual={(j['case_id'],j['prompt_id'],j['seed']):j for j in p['jobs']}
        need(len(actual)==len(p['jobs'])==60 and set(actual)==expected,'prediction combinations changed')
        if plan is not None:
            for template in plan['jobs']:
                j=actual[(template['case_id'],template['prompt_id'],template['seed'])]
                need(all(j.get(k)==template.get(k) for k in ['id','case_id','prompt_id','seed','arm','view','adapter']),'pair metadata changed')
        if not mock:
            fairness.update(runtime_evidence(p,plan,config))
    return mock,fairness

def flatten(j,grid):
    m=j[grid];c=m['color'];p=m['PSD'];out={'case_id':j['case_id'],'base_frame':FRAMES[j['case_id']],'domain':ROLES[j['metric_role']],
      'prompt_id':j['prompt_id'],'seed':j['seed'],'grid':grid,'deltaE76_mean':c['deltaE76_mean'],'RGB_MAE_0_255':c['RGB_MAE_0_255']}
    for k,v in zip(['L','a','b'],c['mean_Lab_bias']):out['signed_Lab_bias_'+k]=v
    for sigma in [2.5,5.,10.]:
        h=m['HP'].get(str(sigma),{});tag=str(sigma).replace('.','p')
        out['HP_RMS_ratio_sigma_'+tag+'_native']=h.get('RMS_ratio')
        out['HP_abs_log2_RMS_ratio_sigma_'+tag+'_native']=logdistance(h.get('RMS_ratio'))
        out['HP_winsor95_abs_log2_RMS_ratio_sigma_'+tag+'_native']=logdistance(h.get('winsor95_RMS_ratio'))
    ratios=p['band_power_ratio'];errors=[logdistance(v) for v in ratios]
    out['PSD_common3_abs_log2_energy_ratio']=exact_mean(errors[:3])
    out['PSD_shared4_abs_log2_energy_ratio']=exact_mean(errors[:4])
    for i,value in enumerate(ratios):out['PSD_band'+str(i)+'_energy_ratio']=value
    return out

def tables(score):
    rows=[flatten(j,g) for j in score['jobs'] for g in ['native','same_view']]
    fields=[k for k in rows[0] if k not in ['case_id','base_frame','domain','prompt_id','seed','grid']]
    groups=defaultdict(list)
    for row in rows:groups[(row['case_id'],row['prompt_id'],row['grid'])].append(row)
    case_stats=[]
    for (sid,prompt,grid),rs in groups.items():
        need(len(rs)==5 and sorted(r['seed'] for r in rs)==SEEDS,'case prompt missing seed')
        case_stats.append({'case_id':sid,'base_frame':FRAMES[sid],'domain':rs[0]['domain'],'prompt_id':prompt,'grid':grid,
          'metrics':{f:distribution([r[f] for r in rs],f) for f in fields}})
    frame_groups=defaultdict(list)
    for r in rows:frame_groups[(r['base_frame'],r['domain'],r['prompt_id'],r['seed'],r['grid'])].append(r)
    frame_rows=[]
    for (frame,domain,prompt,seed,grid),rs in frame_groups.items():
        frame_rows.append({'base_frame':frame,'domain':domain,'prompt_id':prompt,'seed':seed,'grid':grid,'task_variants':len(rs),**{f:exact_mean([r[f] for r in rs]) for f in fields}})
    macro_groups=defaultdict(list)
    for r in frame_rows:macro_groups[(r['domain'],r['prompt_id'],r['seed'],r['grid'])].append(r)
    macros=[]
    for (domain,prompt,seed,grid),rs in macro_groups.items():
        need(len(rs)==2,'expected two base frames within each domain')
        macros.append({'domain':domain,'prompt_id':prompt,'seed':seed,'grid':grid,'base_frame_count':2,**{f:exact_mean([r[f] for r in rs]) for f in fields}})
    dist_groups=defaultdict(list)
    for r in macros:dist_groups[(r['domain'],r['prompt_id'],r['grid'])].append(r)
    prompt_stats=[{'domain':domain,'prompt_id':prompt,'grid':grid,'base_frame_count':2,'seed_count':5,
      'metrics':{f:distribution([r[f] for r in rs],f) for f in fields}} for (domain,prompt,grid),rs in dist_groups.items()]
    lookup={(r['case_id'],r['prompt_id'],r['seed'],r['grid']):r for r in rows};deltas=[]
    for r in rows:
        if r['prompt_id']!='material_match_v1':continue
        old=lookup[(r['case_id'],'legacy',r['seed'],r['grid'])]
        deltas.append({k:r[k] for k in ['case_id','base_frame','domain','seed','grid']}|{'difference':'material_match_v1 minus legacy',
          **{f+'_delta':r[f]-old[f] if numeric(r[f]) and numeric(old[f]) else None for f in fields if lower_is_closer(f)}})
    return rows,case_stats,frame_rows,macros,prompt_stats,deltas

def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def flat_distribution(rows):
    return [{k:v for k,v in r.items() if k!='metrics'}|{field+'_'+stat:value for field,stats in r['metrics'].items() for stat,value in stats.items()} for r in rows]

def observations(score,filled=None):
    fields=['case_id','prompt_id','seed','domain','review_status','suspected_marking','color_brightness_note','material_texture_note','marking_note']
    template=[dict(zip(fields,[j['case_id'],j['prompt_id'],j['seed'],ROLES[j['metric_role']],'not_reviewed','','','',''])) for j in score['jobs']]
    if filled is None:return template,{'status':'not_reviewed','expected_rows':60,'reviewed_rows':0,'note':'Unreviewed is not artifact-free; no automated visual detector or seed ranking is implemented.'}
    with Path(filled).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    keys={(r['case_id'],r['prompt_id'],int(r['seed'])) for r in rows}
    need(len(rows)==len(keys)==60 and keys=={(r['case_id'],r['prompt_id'],r['seed']) for r in template},'visual review must retain all60 rows')
    tally=[]
    for sid in IDS:
        for prompt in PROMPTS:
            rs=[r for r in rows if r['case_id']==sid and r['prompt_id']==prompt]
            need(all(r['review_status'] in ['not_reviewed','reviewed'] for r in rs),'unknown visual review state')
            reviewed=[r for r in rs if r['review_status']=='reviewed']
            need(all(r['suspected_marking'] in ['none_observed','possible','clear_incompatibility','not_assessable'] for r in reviewed),'unknown suspected-marking observation')
            tally.append({'case_id':sid,'prompt_id':prompt,'reviewed_seed_count':len(reviewed),'unreviewed_seed_count':5-len(reviewed),
              'marking_observation_counts':{label:sum(r['suspected_marking']==label for r in reviewed) for label in ['none_observed','possible','clear_incompatibility','not_assessable']}})
    return template,{'status':'manual_fixed_all60_row_review','reviewed_rows':sum(r['review_status']=='reviewed' for r in rows),'tally':tally,
      'note':'Real hidden road has no GT; observations concern compatibility with visible evidence, not verified hidden marking truth. No seed/image is selected for reuse.'}

def summarize(score_path,config_path,out_dir,plan_path=None,predictions_path=None,review_csv=None,allow_mock=False):
    score=read(score_path);config=read(config_path);plan=read(plan_path) if plan_path else None;p=read(predictions_path) if predictions_path else None
    mock,fairness=validate(score,config,plan,p,allow_mock)
    if plan_path:
        need(score['gate']['manifest']['sha256']==sha(plan_path),'score/plan pin mismatch')
        need(plan['source_pins']['config']['sha256']==sha(config_path),'plan/config pin mismatch')
    if predictions_path:
        need(score['gate']['predictions']['sha256']==sha(predictions_path),'score/prediction pin mismatch')
        if plan_path:need(p['manifest']['sha256']==sha(plan_path),'prediction/plan pin mismatch')
    out=Path(out_dir);need(not out.exists() or not any(out.iterdir()),'preserve existing review; choose empty output directory')
    rows,cases,frames,macros,prompt_stats,deltas=tables(score);template,visual=observations(score,review_csv)
    result={'schema':'road_2d_teacher_prompt_seed_review.v1','complete':True,'mode':score['mode'],'CPU_only':True,'truth_or_generated_images_opened':False,
      'source_score':pin(score_path),'source_config':pin(config_path),'fairness':fairness,'automatic_winner_selected':False,'automatic_seed_selection':False,
      'counts':{'formal_calls':60,'smoke_calls_included':0,'prompts':2,'seeds_per_case_prompt':5,'synthetic_tasks':4,'synthetic_clean_base_frames':2,'synthetic_source_scenes':1,'real_known_visible_frames':2,'real_source_scenes':1},
      'case_seed_distributions':cases,'prompt_seed_macros':macros,'prompt_distributions':prompt_stats,'paired_prompt_deltas':deltas,'visual_observations':visual,
      'notes':['Five seeds characterize generation variability on the existing fixed inputs; they do not add scenes, source frames or independent synthetic ground truths.',
        'For each seed, task variants are averaged within the same clean base frame, then the two base frames are weighted equally; only then are the five-seed mean, worst error, population variance and range summarized. No significance test or confidence interval is claimed.',
        'Worst means the maximum descriptive reference-error value, not a selected worst image/seed. No best/worst seed ID, best-of-N score, selected output or automatic prompt winner is emitted.',
        'DeltaE/RGB MAE and absolute log2 spectral mismatch are closer to their references when lower. HP RMS/PSD energy ratios target1, not the maximum. No scalar combines color, texture and artifact observations.',
        'Native and same_view remain separate; HP sigma is in native pixels. Primary PSD uses common0.025–0.2 cycles/native bands; the additional0.2–0.3 band shared by both1152 prompts is retained separately and in shared4 spectral mismatch. Native0.3–0.5 is interpolation-tail diagnostic, not independently generated bandwidth.',
        'All prompts and seeds use identical frozen masks/physical patch support. The inherited scoring support may use the old common-view window; it is not expanded to claim full-FOV qualification.',
        'Synthetic GT is used only in independent postfreeze evaluation of the whole prompt condition. It is never used to choose an image or seed for deployment. Real hidden road has no GT; known-visible preservation scores and visual compatibility observations remain separate.',
        'Same-case, same-size numerical seed pairs use the same intended noise draw; only frozen prompt text changes. Legal inputs/masks, LoRA/checkpoint, inference settings and runtime tokenization must pass the fairness audit.',
        'Four smoke calls are engineering checks and are excluded from the formal60 statistics. Prompt comparison is limited to the configured one or two rounds; no second round is triggered here.',
        'A fixed all60-row manual observation sheet records color/texture and suspected markings. Unreviewed entries are not counted as clean outputs; visible compatibility is not hidden-road truth.']}
    out.mkdir(parents=True,exist_ok=True);write(out/'comparison.json',result)
    for filename,data in [('per_seed.csv',rows),('case_seed_distributions.csv',flat_distribution(cases)),('base_frame_seed_macros.csv',frames),('prompt_seed_macros.csv',macros),('prompt_distributions.csv',flat_distribution(prompt_stats)),('paired_prompt_deltas.csv',deltas),('observations_template.csv',template)]:csv_write(out/filename,data)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['score','config','out-dir']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['plan','predictions','review-csv']:p.add_argument('--'+name,type=Path)
    p.add_argument('--allow-cpu-mock',action='store_true');a=p.parse_args()
    r=summarize(a.score,a.config,a.out_dir,a.plan,a.predictions,a.review_csv,a.allow_cpu_mock)
    print(json.dumps({'complete':r['complete'],'mode':r['mode'],'formal_calls':60,'automatic_seed_selection':False,'out_dir':str(a.out_dir.resolve())}))
if __name__=='__main__':main()
