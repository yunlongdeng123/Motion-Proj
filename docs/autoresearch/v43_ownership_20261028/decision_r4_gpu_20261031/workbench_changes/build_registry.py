from pathlib import Path
import json,gzip,math,statistics,csv,sys
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path('C:/Users/dengyunlong/Documents/Codex/2026-10-31/g/outputs/V43_R4_GPU_20261031');old=p/'actor_screen_existing/existing_normalized_v1';hf=p/'actor_screen_horizonforge';folders=sorted([d for d in old.iterdir() if d.is_dir() and (d/'actors_native_merged.json').is_file()]+[d for d in hf.glob('scene_*') if (d/'actors.json').is_file()]);assert len(folders)==15
reg=[];old_queue=[];assets=[];ids=set()
for folder in folders:
 isold=(folder/'actors_native_merged.json').is_file(); data=json.loads((folder/('actors_native_merged.json' if isold else 'actors.json')).read_text(encoding='utf8'));summary=json.loads((folder/'summary.json').read_text(encoding='utf8')); rows=data['rows'];assert len(rows)==summary['all_vehicle_tracks'];context=data['context_id'];obs={}
 if isold:
  with gzip.open(folder/'all_vehicle_observations.jsonl.gz','rt',encoding='utf8') as f:
   for line in f:
    v=json.loads(line);obs.setdefault(v['track_id'],[]).append(v)
 for a in rows:
  key=(context,a['track_id']);assert key not in ids;ids.add(key);assert a.get('formal_eligibility') is None
  mask=a.get('body_mask_support');mapping=a.get('native_GS_actor_mapping');window=a.get('native_model_window');status=a.get('editable_status') if isold else 'needs_mask_GS_and_independent_candidates'
  r={'dataset':'Waymo','source_label':folder.name,'context_id':context,'track_id':a['track_id'],'raw_frames_denominator':a['total_raw_frames'],'scene_vehicle_tracks_denominator':len(rows),'track_labelled_frames':a['labelled_frames'],'front_projected_frames':a['front_projected_frames'],'front_projected_box_observation_count':a['front_projected_frames'],'median_front_bbox_truncation_fraction':a['median_bbox_truncation_fraction'],'median_front_box_overlap_proxy':a['median_bbox_overlap_proxy'],'median_front_bbox_area_ratio_proxy':a['median_front_bbox_area_ratio'],'median_front_box_lidar_annotation_count_proxy':a['median_front_lidar_annotation_count'],'front_view_sectors_proxy':a['front_view_sectors'],'native_GS_49_frame_window':window,'native_asset_mapping':mapping,'editable_status':status,'mask_evidence':mask,'background_exposure_support':None,'formal_P':None}
  reg.append(r)
  if isold and mapping and mapping.get('status')=='asset_and_raw_track_bound':assets.append({'context_id':context,'track_id':a['track_id'],'asset_index':mapping['dynamic_actor_index'],'native_GS_49_frame_window':[window['start_raw'],window['end_raw']],'editable_status':status,'immediately_native_editable':a['immediately_native_editable'],'mask_evidence':mask,'formal_P':None,'note':'Bound input asset; operation success not assumed. Legacy restore requires verification.'})
  if isold and window['development_rule_pass']:
   start,end=window['start_raw'],window['end_raw'];vs=[v for v in obs[a['track_id']] if start<=v['frame_index']<=end and v['front_bbox_xyxy'] is not None];score=sum(math.sqrt(v['front_bbox_visible_area_ratio'])*min(v['annotation_lidar_points_in_box']/50,1)*(1-v['front_bbox_overlap_proxy'])*(1-v['front_bbox_truncation_fraction']) for v in vs)/49
   old_queue.append({'context_id':context,'track_id':a['track_id'],'source_label':folder.name,'recommended_window_first':start,'recommended_window_last':end,'recommended_window_front_frames':len(vs),'recommended_window_support_score':score,'development_screen_scope':'native_49_frame_window','native_preparation_state':status,'formal_P':None,'used_edit_result':False})
queue=[];hfdata=json.loads((hf/'preedit_preparation_queue.json').read_text(encoding='utf8'));assert len(hfdata['scenes'])==11
for scene in hfdata['scenes']:
 selected=scene['selected_preparation_candidates'];assert len(selected)==3
 for a in selected:
  assert (a['context_id'],a['track_id']) in ids and a['formal_eligibility'] is None
  queue.append({'context_id':a['context_id'],'track_id':a['track_id'],'source_label':'HorizonForge '+a['scene_number'],'recommended_window_first':a['recommended_window_first'],'recommended_window_last':a['recommended_window_last'],'recommended_window_front_frames':a['recommended_window_front_frames'],'recommended_window_support_score':a['recommended_window_support_score'],'development_screen_scope':'full_sequence_candidate_best_49_frame_window','native_preparation_state':a['native_preparation_state'],'formal_P':None,'used_edit_result':False,'preparation_priority_within_scene':a['preparation_priority_within_scene']})
for context in sorted({r['context_id'] for r in old_queue}):
 candidates=sorted([r for r in old_queue if r['context_id']==context],key=lambda r:(-r['recommended_window_support_score'],r['track_id']));assert len(candidates)>=3
 for rank,r in enumerate(candidates[:3],1):r['preparation_priority_within_scene']=rank;queue.append(r)
assert len(reg)==1228 and len(queue)==45 and len(assets)==8 and len({(r['context_id'],r['track_id']) for r in queue})==45
recipe={'rule':'Within each existing native 49-frame window, require development_rule_pass; rank by mean 49-frame support = sum(sqrt(clipped bbox/image area) * min(box lidar/50,1) * (1-overlap proxy) * (1-truncation))/49; missing frames contribute zero. HF rows copied from the existing top-3 queue and its chosen 49-frame windows.','interpretation':'Preparation ordering only. Not formal eligibility, successful editing, background visibility or an optimal actor claim. Top-6 preview images use a different full-sequence ordering.','formal_P':None}
def write(name,rows,meta):
 (p/(name+'.json')).write_text(json.dumps({'schema':name+'_v1','metadata':meta,'rows':rows},ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
 fields=list(rows[0]);
 with (p/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else ('null' if v is None else v) for k,v in row.items()})
write('actor_registry',reg,{'scenes':15,'vehicles':len(reg),'denominator':'all raw annotated vehicle tracks, including non-front; all other actor types preserved in existing full-type audit','background_exposure':'unknown','formal_P':None})
write('actor_preparation_queue',queue,recipe)
(p/'existing_bound_assets.json').write_text(json.dumps({'schema':'existing_bound_assets_v1','count':len(assets),'note':'All 8 existing bound assets; not a ranking or optimal actor claim. Legacy 5 prepared assets need runtime restore verification.','rows':assets},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('registry',len(reg),'queue',len(queue),'hf',sum(r['source_label'].startswith('HorizonForge') for r in queue),'existing_assets',len(assets))
