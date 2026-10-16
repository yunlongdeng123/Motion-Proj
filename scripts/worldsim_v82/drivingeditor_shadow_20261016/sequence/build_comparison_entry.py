"""Build a saved-result viewer and append it without changing any legacy markup."""
import json,os,re,datetime,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];WORK=BASE/'work/shadow_sequence_v2';OUT=BASE/'outputs';E=OUT/'shadow_sequence_v2'
UNIFIED=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-15/file-c-users-dengyunlong-documents-codex/outputs/DrivingEditor_unified_report.html')
PAGE=OUT/'DrivingEditor_shadow_sequence_comparison.html'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rel(p):return os.path.relpath(p,OUT).replace('\\','/')
rows=read(E/'final49_unselected_v1/per_frame.json');summary=read(E/'final49_unselected_v1/summary.json');m=summary['metrics_mean_frames'];weighted=summary['metrics_pixel_weighted']
frames=[];old=UNIFIED.parent/'drivingeditor_nearby_actors_30k/scene0008_089_137/edits/actor0_delete'
for i,row in enumerate(rows):
 fid=row['frame'];paths={'comparison':E/'reviews/final49_unselected_v1/comparison_frames'/(fid+'.png'),'gt':old/(f'{i:03d}_{fid}_gt.png'),'old_delete':old/(f'{i:03d}_{fid}_after.png')}
 for p in paths.values():assert p.is_file(),p
 frames.append({**row,**{k:rel(p) for k,p in paths.items()}})
assert len(frames)==49
road=100*(1-m['final']['surface_clean_MAE']/m['original']['surface_clean_MAE']);delete=100*(1-m['final']['deleted_target_shadow_area_clean_MAE']/m['original']['deleted_target_shadow_area_clean_MAE'])
table=''
for label,key in [('支持道路，干净参考','surface_clean_MAE'),('删除后原目标车影区域','deleted_target_shadow_area_clean_MAE'),('原车位阴影伪标注','surface_shadow_MAE')]:
 count=sum(r['final'].get(key) is not None for r in rows)
 table+='<tr><td>'+label+'</td><td>'+str(count)+'/49</td>'+''.join('<td>%.6f</td>'%m[v][key] for v in ['original','deferred','final'])+'</tr>'
html=(WORK/'comparison_template.html').read_text(encoding='utf-8')
replace={'UNIFIED':UNIFIED.as_uri(),'ROAD':f'{road:.1f}','DELETE':f'{delete:.1f}','SUMMARY_ROWS':table,'DATA':json.dumps({'frames':frames},ensure_ascii=False,separators=(',',':')).replace('</','<\\/'),'TIME_ROAD':f"{summary['mean_fixed_supported_road_time_MAE']:.2g}",'TIME_FULL':f"{summary['mean_fixed_full_image_time_MAE']:.6f}"}
for name,key in [('SHADOW_ORIG','original'),('SHADOW_OLD','deferred'),('SHADOW_NEW','final')]:replace[name]=f"{weighted[key]['surface_shadow_MAE']:.6f}"
for k,v in replace.items():html=html.replace('@@'+k+'@@',v)
assert '@@' not in html;PAGE.write_text(html,encoding='utf-8')
begin='<!-- SHADOW_SEQUENCE_V2_ENTRY_START -->';end='<!-- SHADOW_SEQUENCE_V2_ENTRY_END -->';url=PAGE.as_uri()
entry=f'''{begin}
<aside id="shadow-sequence-v2-entry" style="max-width:1292px;margin:20px auto;padding:20px 24px;border:1px solid #548f81;border-left:4px solid #78dcc3;border-radius:12px;background:#102a2b;color:#deebf4;font:15px/1.65 system-ui,Microsoft YaHei,sans-serif">
<div style="font-size:12px;color:#78dcc3">新增 · 2026-10-16 · 跨时刻道路去影 + 车辆关联阴影</div>
<h2 style="font-size:22px;margin:5px 0 9px">完整49帧诊断与18k分阶段联合微调</h2>
<p style="color:#b7ced0">同帧比较真实GT、原30k、旧deferred_R4与最终版；支持原模型 / 最终版删车前后对照、49帧进度条、固定相机、阴影开关和连续编辑视频。</p>
<p style="color:#b7ced0">支持道路参考误差下降 <b>{road:.1f}%（49帧）</b>；删除后原车影误差下降 <b>{delete:.1f}%（21个有效帧）</b>。<b style="color:#e1bd7a">范围外黑片、±30°资产破面与车辆资产去影认证仍未解决。</b> GPU已按指示关机，本页使用保存结果。</p>
<a href="{url}" target="_blank" style="display:inline-block;margin:6px 0;color:#78dcc3;font-weight:650">单独打开：49帧道路去影与车辆关联阴影对比 ↗</a>
<details open style="margin-top:12px;border-color:#345a55"><summary style="cursor:pointer;color:#bdd7ce">在统一入口展开 · 新实验交互对比</summary><iframe loading="lazy" src="{url}" title="49帧跨时刻道路去影与车辆关联阴影对比" style="display:block;width:100%;height:1660px;margin-top:12px;border:1px solid #345a55;border-radius:9px;background:#09131f"></iframe></details>
</aside>
{end}'''
prior=UNIFIED.read_bytes();raw=prior.decode('utf-8-sig');backup=WORK/('DrivingEditor_unified_report.before_shadow_sequence.'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.html');backup.write_bytes(prior)
pattern=re.escape(begin)+r'.*?'+re.escape(end)
if begin in raw:
 match=re.search(pattern,raw,flags=re.S);assert match
 updated=raw[:match.start()]+entry+raw[match.end():]
 preserved=updated.replace(entry,match.group(0),1)==raw
else:
 anchor=re.search(r'<aside\s+id="unified-report-intro".*?</aside>',raw,flags=re.S);assert anchor
 updated=raw[:anchor.end()]+entry+raw[anchor.end():]
 preserved=updated.replace(entry,'',1)==raw
assert preserved and updated.count(begin)==1
UNIFIED.write_bytes((b'\xef\xbb\xbf' if prior.startswith(b'\xef\xbb\xbf') else b'')+updated.encode('utf-8'))
receipt={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'page':str(PAGE),'unified':str(UNIFIED),'backup':str(backup),'legacy_content_preserved_exactly':preserved,'existing_base_href_unchanged':True,'absolute_entry_URL_avoids_legacy_base_href':url,'frames':49,'GT_and_legacy_delete_pairs_verified':49,'new_page_sha256':hashlib.sha256(PAGE.read_bytes()).hexdigest(),'unified_before_sha256':hashlib.sha256(prior).hexdigest(),'unified_after_sha256':hashlib.sha256(UNIFIED.read_bytes()).hexdigest(),'no_GPU_operation':True}
(E/'evidence/unified_comparison_entry.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if 'sha256' not in k},ensure_ascii=False))
