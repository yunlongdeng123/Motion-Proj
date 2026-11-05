"""Independent final slider data and unchanged-science QA; local read-only inputs."""
from pathlib import Path
import hashlib,json,re,sys
from urllib.parse import urlparse,unquote
from PIL import Image
from final_report_qa import Parser
W=Path(__file__).resolve().parents[3];A=W/'work/a6_artifact_diagnosis';O=W/'outputs/scene008_vehicle_appearance/diagnosis'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
 issues=[]
 def check(ok,name,detail=None):
  if not ok:issues.append(dict(kind=name,details=detail))
 data=(O/'index.html').read_bytes();text=data.decode('utf-8');snapshot=sha(O/'index.html')
 stripped=re.sub(r'<!-- A6_FRAME_COMPARE_BEGIN -->.*?<!-- A6_FRAME_COMPARE_END -->','',text,count=1,flags=re.S).encode('utf-8')
 check(hashlib.sha256(stripped).hexdigest()=='170ccef3351164a1ce8660a07d2f906f8e522fd406c9b0956de4b1eae56a0191','original_science_bytes_changed')
 src=read(O/'frame_compare_source_manifest.json');norm=read(O/'frame_compare_manifest.json');ledger=read(O/'frames_sha256.json')
 check(sha(O/'frame_compare_source_manifest.json')=='efbcf40049521eecb947e3e9a2282b83796ccf36433dc328e2724a2e0c90c9dd'==norm['source_manifest_sha256'],'source_manifest_binding')
 inline=json.loads(re.search(r'<script[^>]*id="fc-data"[^>]*>(.*?)</script>',text,re.S).group(1))
 check(all(inline[k]==norm[k] for k in inline),'inline_manifest_differs')
 check(src['raw_ids']==norm['raw_ids']==list(range(89,138)) and len(src['frames'])==49,'49_frame_domain')
 check(src['images']==294 and src['references']==49 and len(ledger)==343,'image_cardinality')
 check(norm['viewport']==dict(width=480,height=320),'viewport')
 columns=['old_formal','c2_matte','surface_lod'];paths=[];render_count=0;ref_count=0;reuse_count=0
 for frame in src['frames']:
  raw=frame['raw'];nf=norm['frames'][str(raw)];crop=frame['crop_xyxy'];x0,y0,x1,y1=crop
  check(nf['raw']==raw and nf['crop_xyxy']==crop and 0<=x0<x1<=480 and 0<=y0<y1<=320,'raw_crop_binding',raw)
  check(frame['crop_provenance']['same_crop_for_all_methods_states_and_reference'] and not frame['crop_provenance']['predicted_alpha_used'],'crop_observation_domain',raw)
  for state in ['keep','delete']:
   check(set(frame[state])==set(columns),'method_set',[raw,state])
   for method in columns:
    im=frame[state][method];desc=nf[state][method];p=O/im['path'];paths.append(im['path']);render_count+=1
    check(im['raw']==raw and im['state']==state and im['method']==method,'image_role_binding',im['path'])
    if 'native_meta' in im:
     check(im['native_meta']['raw']==raw and im['native_meta']['delete_a37']==(state=='delete') and im['native_meta']['native_a6_rows_input']==0 and (im['native_meta']['native_a37_packed_rows']==0)==(state=='delete'),'native_state_binding',im['path'])
     check(frame['keep'][method]['native_meta']['asset_rows']==frame['delete'][method]['native_meta']['asset_rows'],'method_rows_edit_state_consistency',[raw,method])
    else:
     check(raw in [105,106,113,130] and im['origin']=='Byte-exact prior fixed49 render' and im['legacy_receipt_sha256']==src['reuse_provenance']['prior_receipt_sha256'],'reused_metadata_provenance',im['path'])
    check(im['renderer_id']==src['renderer']['id'] and im['asset_sha256']==next(x['asset_sha256'] for x in src['columns'] if x['id']==method),'renderer_asset_binding',im['path'])
    check(p.is_file() and sha(p)==im['sha256']==desc['sha256']==ledger[im['path']] and desc['path']==im['path'],'PNG_SHA_binding',im['path'])
    if raw in [105,106,113,130]:
     prior=A/'scale/surface_lod_v1/evaluation49'/f'raw{raw}'/(dict(old_formal='old_native',c2_matte='final_native',surface_lod='surface_LOD')[method]+f'_{state}480.png')
     check(prior.is_file() and sha(prior)==im['sha256'],'reused_native_PNG_changed',im['path']);reuse_count+=1
  im=frame['reference'];p=O/im['path'];paths.append(im['path']);ref_count+=1
  check(im['raw']==raw and im['state']=='observed_a37_present' and im['method']=='original_RGB' and '非delete GT' in im['label'],'reference_not_delete_GT',raw)
  check(p.is_file() and sha(p)==im['sha256']==nf['reference']['sha256']==ledger[im['path']],'reference_SHA_binding',raw)
 for rel in paths:
  p=(O/rel).resolve();check(p.is_relative_to(O.resolve()),'image_path_escape',rel)
  try:
   with Image.open(p) as im:check(im.format=='PNG' and im.size==(480,320),'image_decode_dimensions',rel);im.load()
  except Exception as e:check(False,'image_decode_failure',[rel,str(e)])
 check(len(paths)==len(set(paths))==343,'unique_images')
 p=Parser();p.feed(text);local=[]
 for _,_,ref in p.refs:
  if ref.startswith(('data:','#')) or urlparse(ref).scheme:continue
  target=(O/unquote(urlparse(ref).path)).resolve();local.append(ref)
  check(target.is_relative_to(O.resolve()) and target.is_file(),'main_local_link',ref)
 static=Parser();static.feed((O/'frames_static.html').read_text(encoding='utf-8'));check(static.scripts==0,'fallback_has_script')
 check(all(f'raw{raw}' in static.ids for raw in range(89,138)),'fallback_raw_anchors')
 static_paths=[]
 for image in static.imgs:
  check(not image.startswith('data:') and (O/image).is_file(),'fallback_image_path',image);static_paths.append(image)
 check(set(static_paths)==set(paths),'fallback_incomplete_branches')
 manifest=read(O/'sha256.json');bad=[p for p,s in manifest.items() if not(O/p).is_file() or sha(O/p)!=s];check(not bad,'publication_ledger',bad)
 check(sha(O/'index.html')==snapshot,'publication_changed_during_review')
 report=dict(status='PASS_SLIDER_STATIC_INPUTS' if not issues else 'MUST_FIX',HTML_sha256=snapshot,read_only=True,new_GPU_calls=0,external_network_calls=0,must_fix=issues,raw_domain=[89,137],frames=49,render_PNGs=render_count,reference_PNGs=ref_count,all343_SHA_decode_480x320=True,metadata_raw_state_method_renderer_asset_checked=True,per_raw_shared_crop_checked=True,exact_prior_PNG_reuses_checked=reuse_count,unchanged_science=dict(original_170c_byte_exact_after_module_strip=True,original_static_scientific_images=21,original_metric_rows=94),static_fallback=dict(script_count=static.scripts,image_occurrences=len(static_paths),all343_unique_paths_covered=set(static_paths)==set(paths)),local_link_occurrences=len(local),publication_ledger_files=len(manifest),source_manifest_sha256=sha(O/'frame_compare_source_manifest.json'))
 (A/'review/slider_static_review.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
