from pathlib import Path
import argparse,json,shutil,re
p=argparse.ArgumentParser();p.add_argument('source_root',type=Path);p.add_argument('page_dir',type=Path);p.add_argument('--additional-root',type=Path,action='append',default=[]);args=p.parse_args();args.page_dir.mkdir(parents=True,exist_ok=True)
scenes=[];seen=set()
for folder in sorted(folder for root in [args.source_root]+args.additional_root for folder in root.iterdir()):
 if not folder.is_dir():continue
 f=folder/'actors_native_merged.json'
 if not f.exists():f=folder/'actors.json'
 if not f.is_file():continue
 data=json.loads(f.read_text(encoding='utf8'));context=data['context_id']
 if not re.fullmatch(r'\d+_\d+_\d+_\d+_\d+',context):raise ValueError('invalid context: '+context)
 if context in seen:raise ValueError('duplicate context: '+context)
 seen.add(context);actors=[]
 for row in data['rows']:
  preview=row.get('preview_image');image=None
  if preview:
   rel=Path(preview)
   if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe preview')
   source=folder/rel
   if not source.is_file():raise FileNotFoundError(source)
   target=args.page_dir/'scenes'/context/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);image=target.relative_to(args.page_dir).as_posix()
  body=row.get('body_mask_support');window=row.get('native_model_window');actors.append({'track_id':row['track_id'],'preview_image':image,'preview_frame':row.get('best_preview_frame') if image else None,'labelled_frames':row.get('labelled_frames'),'front_projected_frames':row.get('front_projected_frames'),'front_view_sector_count':len(row['front_view_sectors']) if isinstance(row.get('front_view_sectors'),list) else None,'median_front_bbox_area_ratio':row.get('median_front_bbox_area_ratio'),'median_front_lidar_annotation_count':row.get('median_front_lidar_annotation_count'),'body_mask_frames':body.get('frames') if isinstance(body,dict) else None,'editable_status':row.get('editable_status') or ('hf_prepare_inputs' if re.fullmatch(r'scene_\d{3}',folder.name) else None),'window_label':f"原始帧 {window['start_raw']}–{window['end_raw']}" if isinstance(window,dict) else None})
 if not actors:continue
 scenes.append({'context_id':context,'display_label':('HorizonForge '+folder.name[-3:]+' · Waymo '+context) if re.fullmatch(r'scene_\d{3}',folder.name) else ('Waymo · '+context),'actors':actors})
if not scenes:raise ValueError('no scene datasets')
target=args.page_dir/'data.js';target.write_text('window.ACTOR_CANDIDATES='+json.dumps(scenes,ensure_ascii=True,separators=(',',':'))+';\n',encoding='utf8')
print('scene_count',len(scenes),'actor_count',sum(len(x['actors']) for x in scenes),'image_count',sum(bool(a['preview_image']) for x in scenes for a in x['actors']))
