from pathlib import Path
import json,shutil,tarfile
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');O=C/'waymo_val065_delete30k_gpu_diagnosis_20261111_r1';R=C/'waymo_val065_delete30k_20261109_r1';D=O/'delivery';D.mkdir(exist_ok=False)
load=lambda p:json.loads(p.read_text())
receipts={}
for key in ['joint_v1','interaction_v1','gradient_v1','selection_trace_v1']:
    receipts[key]=load(O/key/'receipt.json');assert receipts[key]['status'].startswith('COMPLETE'),key
    shutil.copy2(O/key/'receipt.json',D/(key+'_receipt.json'))
for f in (O/'review').glob('*'):
    if f.suffix in ['.png','.json']:shutil.copy2(f,D/f.name)
selection=load(O/'selection_trace_v1/selection_summary.json');shutil.copy2(O/'selection_trace_v1/selection_summary.json',D/'selection_summary.json')
traces=[]
for raw in [119,198]:
    value=load(O/'selection_trace_v1'/f'{raw}_summary.json')
    for region in value['regions'].values():
        for k in ['top_static_rows','top_selected_rows','top_unselected_rows']:region.pop(k,None)
    traces.append(value)
    (D/f'{raw}_trace_summary.json').write_text(json.dumps(value,indent=2)+'\n')
original=load(R/'registration.json');model=Path(original['source_model']);now={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in model.glob('*30000.pth')}
assert now==original['native_checkpoint_files'],'Native checkpoint metadata changed'
summary=dict(task='WAYMO-065-GPU-DIAGNOSIS-20261111',date_singapore='2026-11-11',frames=[60,119,150,198],
 same_sort_RGB_sum_maxabs=max(x['RGB_additivity_maxabs'] for key in ['joint_v1','interaction_v1'] for x in receipts[key]['rows']),
 published_RGB_parity_maxabs=max(x.get('saved_RGB_maxabs',0) for x in receipts['joint_v1']['rows']),
 semantic_RGB_parity_maxabs=max(x['semantic_baseline_RGB_maxabs'] for x in receipts['interaction_v1']['rows']),
 native_checkpoint_metadata_unchanged=True,selection_reproduced=selection['eligible_ids_equal_saved'],
 raster_forward_calls=len(receipts['joint_v1']['rows'])+len(receipts['interaction_v1']['rows'])+len(receipts['gradient_v1']['records'])+len(selection['participating_frames'])+len(traces),
 successful_module_seconds={k:v['seconds'] for k,v in receipts.items()},
 training_updates=0,optimizer_created=False,static_or_dynamic_gaussian_updates=0,automatic_shutdown=False,
 failure_ledger_refs=['V82-F06'],failure_ledger_delta='updated:V82-F06',human_verdict=None,
 diagnosis_only=True,production_code_or_viewer_replaced=False,
 road_only_label='road plus environment: native objects removed for diagnostics, not a valid recovery method')
(D/'diagnosis_summary.json').write_text(json.dumps(summary,indent=2)+'\n');shutil.copy2(O/'registration.json',D/'registration.json');shutil.copytree(O/'source',D/'source')
(D/'retrieval_files.json').write_text(json.dumps([dict(path=str(p.relative_to(D)),bytes=p.stat().st_size) for p in sorted(D.rglob('*')) if p.is_file()],indent=2)+'\n')
with tarfile.open(O/'delivery.tar.gz','w:gz') as t:
    for p in sorted(D.rglob('*')):
        if p.is_file():t.add(p,arcname=str(p.relative_to(D)),recursive=False)
print(json.dumps(dict(archive=str(O/'delivery.tar.gz'),bytes=(O/'delivery.tar.gz').stat().st_size,summary=summary),indent=2))
