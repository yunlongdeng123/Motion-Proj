from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import hashlib,json,re,subprocess
zpath=Path('C:/Users/dengyunlong/Downloads/Wang_HorizonForge_Driving_Scene_CVPR_2026_supplemental.zip');w=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-31/g/work/horizon_trace');o=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-31/g/outputs/HorizonForge_Waymo_Source_Trace_20261031');o.mkdir(parents=True,exist_ok=True)
js=(w/'index.js').read_text(encoding='utf8'); ids=re.findall(r"\bid:\s*'(scene_\d+_[^']+)'",js); assert len(ids)==32 and len(set(ids))==32
by=defaultdict(list)
with ZipFile(zpath) as z:
 names=z.namelist(); originals=[n for n in names if n.endswith('/original.mp4')];assert len(originals)==32
 for n in originals:
  sample=n.split('/')[-2];match=re.match(r'scene_(\d+)_',sample);assert match and sample in ids
  b=z.read(n);by[match.group(1)].append({'sample_id':sample,'zip_path':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 print('scene IDs:',sorted(by),'count',len(by))
 scenes=[]
 for scene,items in sorted(by.items()):
  hashes=sorted(set(i['sha256'] for i in items)); n=items[0]['zip_path'];video=w/f'scene_{scene}_original.mp4';video.write_bytes(z.read(n))
  proc=subprocess.run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(video)],capture_output=True,text=True,check=True);metadata=json.loads(proc.stdout)
  frame=w/f'scene_{scene}_first.png'; subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(video),'-frames:v','1',str(frame)],capture_output=True,text=True,check=True)
  strings=[x.decode('ascii','ignore') for x in re.findall(rb'[\x20-\x7e]{8,}',video.read_bytes())]
  matches=[s[:300] for s in strings if re.search(r'(segment[-_]|context|waymo|\d{15,})',s,re.I)]
  scenes.append({'scene_number':scene,'sample_count':len(items),'originals':items,'same_scene_originals_byte_identical':len(hashes)==1,'distinct_original_sha256_count':len(hashes),'analysis_video':str(video),'first_frame':str(frame),'ffprobe':metadata,'embedded_string_hits':matches[:30]})
 texts=[x for x in names if x.lower().endswith(('.js','.json','.html','.txt','.yaml','.yml','.csv')) and not re.search(r'(bulma|fontawesome)',x,re.I)]
 text_hits=[]
 for n in texts:
  text=z.read(n).decode('utf8','replace')
  for lineno,line in enumerate(text.splitlines(),1):
   if re.search(r'waymo|segment[-_]|context|\d{15,}',line,re.I):text_hits.append({'path':n,'line':lineno,'text':line[:500]})
 result={'zip_path':str(zpath),'zip_sha256':hashlib.sha256(zpath.read_bytes()).hexdigest(),'entry_count':len(names),'index_js_source':'43826_supp/static/js/index.js','original_description':'Raw Waymo clip and recorded trajectory overlay.','original_video_file':'original.mp4','original_trajectory_file':'trajectory_original.mp4','edited_description':'Edited scene paired with our generated rollout.','sample_count':len(ids),'category_count':8,'unique_scene_count':len(by),'scene_numbers':sorted(by),'scenes':scenes,'text_files_scanned':texts,'text_identifier_hits':text_hits,'limitations':['网页将 original.mp4 称作原始 Waymo 视频；这不证明其未经裁剪、缩放或重编码。','scene_XXX 是补充材料展示编号；未找到可以据此直接映射原始 Waymo segment/context 的表。','展示描述与视频元数据不构成编辑后结果的真实标签或对原始 segment 的身份确认。']}
 (o/'supplemental_inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print('hash consistency',[(s['scene_number'],s['sample_count'],s['same_scene_originals_byte_identical']) for s in scenes]);print('text hits',text_hits);print('embedded string counts',[(s['scene_number'],len(s['embedded_string_hits'])) for s in scenes]);print('saved',o/'supplemental_inventory.json')
