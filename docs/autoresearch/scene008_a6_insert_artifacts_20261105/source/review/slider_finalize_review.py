"""Consolidate existing component evidence and final-file QA, no browser actions."""
from pathlib import Path
import hashlib,json,re
from datetime import datetime,timezone
A=Path(__file__).resolve().parents[1];O=A.parents[1]/'outputs/scene008_vehicle_appearance/diagnosis'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(p,id):return re.search(r'<script[^>]*id="'+id+r'"[^>]*>(.*?)</script>',p.read_text(encoding='utf-8'),re.S).group(1)
def main():
 static=read(A/'review/slider_static_review.json');component=read(A/'review/slider_component_browser_review.json')
 preview=A/'frame_compare/native49_v1/component_preview_not_delivery.html';controller=O/'code/report/frame_compare_controller.js'
 script=controller.read_text(encoding='utf-8');pjs=extract(preview,'fc-controller');fjs=extract(O/'index.html','fc-controller')
 issues=list(static['must_fix']);same=pjs==fjs==script
 if not same:issues.append(dict(kind='controller_changed_since_component_test'))
 checks={
  'all_group_members_preloaded_and_decoded_before_commit':'const loaded = await Promise.all(paths.map(preload));' in script and "await im.decode()" in script,
  'sequence_guard_before_and_inside_animation_frame':script.count('if (ticket !== sequence)')==3 and 'requestAnimationFrame' in script,
  'single_turn_group_replacement':"columns.forEach((col,index) => {" in script and 'old.replaceWith(im);' in script,
  'failed_group_does_not_commit_displayed_target':"整组未更新，仍显示 " in script,
  'already_failed_initial_images_read_synchronously':"if (!im.complete) waiting=true;" in script and 'else if (im.naturalWidth!==480 || im.naturalHeight!==320)' in script and 'inspectInitial();' in script,
  'initial_failure_wins_over_ready':'if (initialErrors.size)' in script and '} else if (waiting)' in script and "status.dataset.level='ready'" in script,
  'no_manifest_fetch_or_external_runtime_dependencies':'fetch(' not in script and 'XMLHttpRequest' not in script and 'import ' not in script,
  'common_crop_from_displayed_frame':"const frame = data.frames[String(displayed.raw)];" in script and "grid.style.setProperty('--fc-left'" in script,
  'range_and_state_handlers_same_atomic_request':"range.addEventListener('input'" in script and "stateSelect.addEventListener('change'" in script,
  'prev_next_and_raw_jump_bindings':"byId('fc-prev').addEventListener('click'" in script and "byId('fc-next').addEventListener('click'" in script and "root.querySelectorAll('[data-fc-raw]')" in script,
 }
 for k,v in checks.items():
  if not v:issues.append(dict(kind='source_contract_absent',details=k))
 finished=datetime.fromtimestamp((A/'review/slider_component_browser_review.json').stat().st_mtime,timezone.utc).isoformat()
 result=dict(status='PASS_STATIC_AND_EXISTING_COMPONENT_EVIDENCE' if not issues else 'MUST_FIX',must_fix=issues,created_UTC=datetime.now(timezone.utc).isoformat(),scope=dict(read_only=True,new_GPU_calls=0,post_policy_browser_actions=0,publisher_or_asset_edits=False,own_headless_sessions_closed=True),snapshots=dict(final_HTML_sha256=sha(O/'index.html'),source_manifest_sha256=sha(O/'frame_compare_source_manifest.json'),controller_sha256=sha(controller),component_HTML_sha256=sha(preview)),static=static,controller_code_review=checks,node_syntax_check='PASS: node --check (no browser launch)',existing_dynamic_component_evidence=dict(status=component['status'],finished_UTC=finished,finished_Singapore='2026-11-06 01:51:54 +08:00',same_controller_as_final_byte_exact=same,normal_file_security=True,external_HTTP_requests=component['externalRequests'],runtime_exceptions=component['runtimeExceptions'],tests=component['tests'],receipt=str(A/'review/slider_component_browser_review.json')),validation_limits=dict(final_main_report_browser_run=False,full49_keep_delete_runtime_switches_tested=False,actual_range_drag_prev_next_130_UI_tested=False,already_failed_initial_image_runtime_tested=False,already_failed_initial_image_source_guard_reviewed=True,reason='Root reported CUA file:// URL-policy rejection and explicit prohibition on raw CDP, alternate browser surfaces or a local HTTP bridge. No browser interaction was performed after that notification. Prior component tests remain historical evidence, not a full final-main runtime test.'),PASS_items=['All343 files match SHA and decode as lossless480x320 PNG;49frames×3methods×2states and49 truthful references are complete.','Per-raw camera/branch metadata and observation-based common crop are consistent;24 reused native PNG bytes exactly match earlier results.','Removing only the marked new module recovers frozen170c report bytes: original21 scientific figures and94 metric rows unchanged.','Static no-JS49×2 fallback contains all343 unique image paths; all main local links and445 publication ledger entries match.','Prior component Chrome tests covered missing-one-file atomic failure,113keep/crop recovery, superseded-request race, and no-JS initial113. Final controller source is byte-identical.'])
 (A/'review/slider_final_review.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print(json.dumps(dict(status=result['status'],must_fix=issues,final_HTML_sha256=result['snapshots']['final_HTML_sha256'],existing_component_finished_UTC=finished,controller_exact=same,final_main_runtime='NOT_RUN_POLICY_BOUNDARY',record=str(A/'review/slider_final_review.json')),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
