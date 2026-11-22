"""Root-gated isolated-index publication. No commit/push command is exposed."""
from pathlib import Path,PurePosixPath
import json,os,sys
import isolated_publication_base as p
from status_tail_replacement import replace_status_tail
from origin_failure_index_cpu import rebuild as rebuild_failure_index
import time

ROOT=Path(os.environ.get('PUBLICATION_RUN_ROOT','/root/autodl-tmp/drivingeditor_repro/campaign/old_raw_teacher_joint_20261123_r9'))
R5=Path(os.environ.get('PUBLICATION_NATIVE_HISTORY_ROOT',str(ROOT.parent/'road_native_source_transfer_20261122_r5')))
PREFIX='docs/autoresearch/road_vehicle_joint_20261123'
LABEL='ROAD-VEHICLE-JOINT-20261123'
LEGACY=['road_gaussian_transfer_20261122','road_gaussian_source_quality_20261122','road_source_fidelity_20261122','road_source_multiframe_20261122','road_clean_core_transfer_20261122']
CORRECTION_MARKER='ROAD-NATIVE-SOURCE-CORRECTION-20261122'
p.RUN=ROOT;p.OUT=ROOT/'publication';p.PREFIX=PREFIX;p.LABEL=LABEL
p.R3_PARENT_COMMIT='fb9a9e83ce93d7f77c2fac6619b52ddfd6ea5eb7'
p.R3_PARENT_README='docs/autoresearch/road_clean_core_transfer_20261122/README.md'
p.DOC_PATHS=['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V82-F06.md']+['docs/autoresearch/'+name+'/README.md' for name in LEGACY]
p.ALLOWED_TYPES={'.py','.json','.md','.svg','.js'}

def final_narrative(n):
    if n.get('finalized_by_root') is not True or n.get('task')!=LABEL:
        raise RuntimeError('Root final narrative is required before staging')
    if n.get('failure_ledger_refs')!=['V82-F06'] or n.get('failure_ledger_delta')!='updated:V82-F06' or n.get('new_failure_family',False):
        raise ValueError('Only existing V82-F06 may be updated')
    if 'human_verdict' not in n or 'problem_solved' not in n or n['human_verdict'] is not None or n['problem_solved'] is not None:
        raise ValueError('human_verdict/problem_solved must remain null')
    for key in ['formal400_and_fresh_reviewed','official_DGGT_control_reviewed','archive_verified','report_QA_passed','takeback_verified','source_00_02_correction_preserved','R7_NOT_ACCEPTED_preserved']:
        if n.get(key) is not True:raise RuntimeError('Final publication gate is not ready: '+key)
    for key in ['status_replacement','experiment_entry','failure_append']:
        if not isinstance(n.get(key),str) or LABEL not in n[key]:raise ValueError('Final narrative is missing task-bound text: '+key)
    return n

def current_stage_review(root,n,selected):
    """Snapshot review is explicit; a partial negative result may be delivered."""
    if not selected.get('requires_current_stage_review'):return
    ref=selected['current_stage_snapshot'];path=root/ref['source']
    if p.digest(path.read_bytes())!=ref['sha256']:raise RuntimeError('Current-stage snapshot changed')
    if n.get('current_stage_snapshot_reviewed') is not True or n.get('current_stage_snapshot_sha256')!=ref['sha256']:
        raise RuntimeError('Root must review the exact current R8/R9/R10 snapshot')
    if n.get('R10_archive_takeback_reviewed') is not True:
        raise RuntimeError('Current R10 archive and local takeback review is pending')
    status=json.loads(path.read_text(encoding='utf-8-sig'))
    if status.get('R10_full_archive_takeback_verified') is not True:
        raise RuntimeError('Actual R10 full local archive verification receipt is missing from the reviewed snapshot')
    delivery_path=root/'publication/final_delivery.json'
    if not delivery_path.is_file():raise RuntimeError('Explicit root delivery pointer is missing')
    d=json.loads(delivery_path.read_text(encoding='utf-8-sig'))
    if not(d.get('ready') is True and d.get('finalized_by_root') is True and d.get('current_stage_snapshot_sha256')==ref['sha256']):
        raise RuntimeError('Explicit root delivery is not ready on the reviewed snapshot')
    if not isinstance(d.get('delivery_scope'),str) or not d['delivery_scope'].strip() or type(d.get('whole_scene_restoration_accepted')) is not bool:
        raise RuntimeError('Root delivery must name its scope and explicit whole-scene verdict')


def selection():
    value=json.loads((p.OUT/'selected_files.json').read_text(encoding='utf-8'))
    if value.get('run_id')!=ROOT.name or value.get('docprefix')!=PREFIX:raise ValueError('Wrong publication task')
    names=[]
    for row in value['files']:
        source=PurePosixPath(row['source']);destination=PurePosixPath(row['destination'])
        if source.is_absolute() or destination.is_absolute() or '..' in source.parts or '..' in destination.parts:raise ValueError('Unsafe publication path')
        if source.suffix not in p.ALLOWED_TYPES or destination.suffix not in p.ALLOWED_TYPES:raise ValueError('Binary payload belongs in archive')
        path=(ROOT/Path(*source.parts)).resolve()
        if not path.is_relative_to(ROOT.resolve()):raise ValueError('Source escapes current run')
        if path.is_file():
            if path.stat().st_size>=300000:raise ValueError('Each selected input must be smaller than 300000 bytes')
            if row.get('sha256') and p.digest(path.read_bytes())!=row['sha256']:raise ValueError('Root-reviewed source changed: '+str(source))
        names.append(row['destination'])
    if len(names)!=len(set(names)):raise ValueError('Duplicate publication destination')
    return value

def correction(old,kind):
    if CORRECTION_MARKER in old:
        if kind!='road_source_fidelity_20261122' or 'native_static' in old:return old
        return old+'\n两组独立native_static保持formal30k原生静态对照身份。\n'
    first,separator,rest=old.partition('\n')
    text='**来源纠正（'+CORRECTION_MARKER+'）**：本组主00／02路块来自加工道路`road_surface_residual_20261121_r2/appearance_fit/asset`。原始历史数据与数值保留为加工道路对照；不作为未编辑原生重建源迁移证据。'
    if kind=='road_source_fidelity_20261122':text+='两组独立native_static直接来自formal30k静态checkpoint，保持原生静态对照身份。'
    text+='[原生来源与可见集合历史诊断](../road_native_source_transfer_20261122/README.md)。'
    return first+'\n\n'+text+'\n'+rest

def updates(parent,n):
    final_narrative(n);files={}
    status_path=p.DOC_PATHS[0];status=p.origin(parent,status_path).decode('utf-8')
    if LABEL in status:raise RuntimeError('Task already exists in origin; inspect before republishing')
    replacement,status_record=replace_status_tail(status,n['status_replacement'])
    files[status_path]=replacement.encode('utf-8')
    p.write(p.OUT/'status_replacement_receipt.json',dict(status_record,parent=parent,origin_sha256=p.digest(status.encode('utf-8')),replacement_sha256=p.digest(files[status_path]),shared_checkout_written=False))
    for path,key in zip(p.DOC_PATHS[1:3],['experiment_entry','failure_append']):
        old=p.origin(parent,path).decode('utf-8')
        if LABEL in old:raise RuntimeError('Task already exists in origin; inspect before republishing')
        files[path]=(old.rstrip()+'\n\n'+n[key].strip()+'\n').encode('utf-8')
    for kind,path in zip(LEGACY,p.DOC_PATHS[3:]):
        old=p.origin(parent,path).decode('utf-8');new=correction(old,kind)
        if new!=old:files[path]=new.encode('utf-8')
    for filename in ['README.md','selected_run.json','architecture_components.svg','run_summary_compact.json']:
        path=R5/filename
        if not path.is_file() or path.stat().st_size>=300000:raise RuntimeError('Required lightweight native history missing/oversized: '+str(path))
        files['docs/autoresearch/road_native_source_transfer_20261122/'+filename]=path.read_bytes()
    if any(name.startswith('docs/research_failures/entries/') and name!='docs/research_failures/entries/V82-F06.md' for name in files):raise ValueError('New failure family is forbidden')
    return files

p.source_manifest=selection;p.document_updates=updates
def actual_failure_index(parent,files):
    return rebuild_failure_index(parent,files,p.OUT/('origin_failure_rebuild_'+str(time.time_ns())),p.git,p.origin)
p.failure_index=actual_failure_index

def main():
    if '--stage' in sys.argv:
        narrative_path=p.OUT/'narrative.json';final_narrative(json.loads(narrative_path.read_text(encoding='utf-8')))
        chosen=selection()
        if chosen.get('finalized_by_root') is not True:raise RuntimeError('Root must finalize selected_files.json')
        if chosen.get('root_narrative_sha256')!=p.digest(narrative_path.read_bytes()):raise RuntimeError('Root narrative changed after final input review')
        current_stage_review(ROOT,json.loads(narrative_path.read_text(encoding='utf-8')),chosen)
        for row in chosen['files']:
            path=ROOT/row['source']
            if not path.is_file():raise RuntimeError('Missing final input: '+row['source'])
            if row.get('require_ready') and json.loads(path.read_text(encoding='utf-8-sig')).get('ready') is not True:raise RuntimeError('Final input is not ready: '+row['source'])
    p.main()

if __name__=='__main__':main()
