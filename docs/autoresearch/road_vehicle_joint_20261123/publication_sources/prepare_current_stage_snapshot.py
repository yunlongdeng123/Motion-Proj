"""Snapshot lightweight current evidence; never creates ready or runs Git."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,shutil

LIMIT=300000
ALLOWED={'.py','.js','.json','.md','.svg'}
R8_NAME='road_vehicle_joint_dggt_20261123_r8'
R10_NAME='road_underbody_local_20261123_r10'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)


def prepare(root,stamp=None,archive_receipt=None,r10_archive_receipt=None):
    root=Path(root).resolve();r8=root.parent/R8_NAME;r10=root.parent/R10_NAME
    stamp=stamp or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=root/'publication/current_stages'/stamp;out.mkdir(parents=True,exist_ok=False)
    entries=[];omitted=[];sources=[];seen={}
    def add(source,destination):
        source=Path(source)
        if not source.is_file():return
        if source.suffix not in ALLOWED:return
        if source.stat().st_size>=LIMIT:
            omitted.append(dict(path=str(source),reason='at_least_300000_bytes_archive_only'));return
        digest=sha(source)
        if destination in seen:
            if seen[destination]!=digest:raise RuntimeError('Duplicate destination changed during snapshot: '+destination)
            return
        seen[destination]=digest
        # Logical publication paths remain hierarchical; physical snapshots are
        # flat so Windows MAX_PATH cannot block long evidence filenames.
        target=out/(f'{len(entries):04d}_'+digest[:12]+source.suffix)
        shutil.copy2(source,target)
        if sha(target)!=digest or sha(source)!=digest:raise RuntimeError('Input changed during snapshot: '+str(source))
        entries.append(dict(source=target.relative_to(root).as_posix(),destination=destination,require_ready=False,exists_at_preparation=True,sha256=digest,bytes=target.stat().st_size))
        sources.append(dict(original=str(source),snapshot=target.relative_to(root).as_posix(),sha256=digest))
    def metadata(run,label,relative):
        folder=run/relative
        if folder.is_file():add(folder,f'{label}/{relative}');return
        if not folder.is_dir():return
        # Only named evidence directories; binaries/full runtime and datasets stay archived.
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or path.suffix not in {'.json','.md','.svg'}:continue
            if any(k in path.parts for k in ['__pycache__','native','input_refs','retained_runtime_snapshot','takeback']):continue
            # The served-report full hash inventory belongs to the archive,
            # while its actual QA/root review remain lightweight evidence.
            if path.name.endswith('.manifest.json') or path.name in ['manifest.json','files_sha256.json','final_report_freeze.actual.json']:continue
            add(path,f'{label}/'+path.relative_to(run).as_posix())
    for run,label in [(r8,'r8'),(root,'r9'),(r10,'r10')]:
        for folder in ['code','report']:
            if (run/folder).is_dir():
                for path in sorted((run/folder).iterdir()):
                    if path.is_file() and path.suffix in {'.py','.js','.svg'}:add(path,f'{label}/{folder}/{path.name}')
    for relative in [
        'formal400_summary_v1','joint_rgb_smoke6_v1','joint_rgb_smoke6_v2','joint_rgb_smoke6_v2_fresh',
        'joint_rgb_formal400_v1','joint_rgb_formal400_v1_fresh','joint_colour_guard_v1','joint_persistent_asset_audit_v1',
        'forward_smoke_five_views_v1','forward_smoke_five_views_recovered_v2','dggt_no_diff_control_v1',
        'audit/dggt_control_CPU_v1','residual_rgb_groups_v1/CPU_attribution_v1','residual_rgb_groups_v1/CPU_attribution_v2_focus']:
        metadata(r8,'r8',relative)
    for folder in sorted(r8.glob('joint_rgb_*fresh*')):
        if folder.is_dir():metadata(r8,'r8',folder.relative_to(r8).as_posix())
    for relative in ['source_audit','world_observations_v2','joint_shared_continuous_v2','cpu_review/joint_shared_continuous_v2']:
        metadata(root,'r9',relative)
    for relative in ['user_execution_contract.json','baseline_vehicle0_road400.json','local_measure_contract_v1.json','local_measure_contract_v2.json',
        'measure_v1','retire0_strict_v1','body_overlap_policy_v1','body_overlap_diagnostic_v1','zero_step_review_v1','native_identity_gate_v1',
        'ground_unowned_policy_v1','local_texture_seed_v1','local_world_observations_v1','local_shared_continuous_v1','local_seed_surface_review_v1',
        'local_shared_surface_review_v1','root_geometry_zero_approval_v1','new_local_asset_zero_v1','new_local_zero_v1',
        'audit/local_surface_candidate_v2','audit/local_surface_geometry_v1','audit/official_ground_geometry_v1',
        'audit/ground_contact_prior_raw_GT_v2','audit/new_local_wiring_cpu_v1','audit/blue_row_eligibility_cpu_v1',
        'audit/blue_world_obb_cpu_v1','audit/blue_world_obb_cpu_v2','audit/depth_order_v1']:
        metadata(r10,'r10',relative)
    r10_prefixes=('local_','new_local_','edit_footprint_','footprint_limited_','ground_unowned_','root_geometry_','native_identity_','measure_','retire0_','body_overlap_','zero_step_review_','replacement_contained_','expanded_','timewindow17_','opacity_','intrinsic_','asset_intrinsic_')
    if r10.is_dir():
        # Future actual stages are copied only once files exist; names confer no completion.
        metadata(r10,'r10','audit')
        metadata(r10,'r10','docs')
        metadata(r10,'r10','provenance/previous_stage_archive_reference.json')
        metadata(r10,'r10','provenance/original_A20_r3')
        metadata(r10,'r10','root_current_stage.json')
        for rel in ['native_early_termination_source_evidence_correction_v1.json','source_label_interpretation_correction.json','root_final_stage_review_v1.json']:
            metadata(r10,'r10',rel)
        for folder in sorted(r10.iterdir()):
            if folder.is_dir() and folder.name.startswith(r10_prefixes):
                metadata(r10,'r10',folder.name)
        for path in sorted(r10.glob('*contract*.json')):add(path,'r10/'+path.name)
        for path in sorted(r10.glob('root_*.json')):add(path,'r10/'+path.name)
    if archive_receipt:
        add(Path(archive_receipt),'archive/local_takeback_receipt.json')
    else:
        candidate=root.parent.parent/'outputs'/R8_NAME/'method_archive/local_takeback_receipt.json'
        if candidate.is_file():archive_receipt=candidate;add(candidate,'archive/local_takeback_receipt.json')
    r10_archive=None
    if r10_archive_receipt:
        r10_archive_receipt=Path(r10_archive_receipt);r10_archive=read(r10_archive_receipt)
        if not (r10_archive.get('schema')=='R10.safe_local_archive_verification.v1' and
                r10_archive.get('complete') is True and r10_archive.get('decision')=='PASS' and
                r10_archive.get('all_member_SHA_verified') is True):
            raise ValueError('R10 needs the actual local full-archive/member verification receipt')
        add(r10_archive_receipt,'archive/R10_local_takeback_receipt.json')
    def state(run,relative):
        path=run/relative
        if not path.is_file():return dict(available=False,status='not_present_in_this_snapshot')
        value=read(path)
        keep=['schema','complete','decision','mode','logical_steps','formal_steps','actual_optimizer_updates','nativeF','nativeB','actual_nativeF','updates','quality_accepted','whole_scene_restoration_accepted','new_diffusion_calls','new_semantic_calls','formal_actual_complete','all_fresh_and_independent_evidence_ready','actual_resources','ready','method_stage_frozen','stage_status','blocked_technical','authorization_pending','bindings','source_qualification','required_next_evidence','resources_at_preparation','R10_actual_nativeF','R10_actual_nativeB','new_local_optimizer_updates','existing_representation_formal_updates','existing_representation_limit','R10','historical_separate','DGGT','current_existing_representation_budget','live_ledger_not_final_freeze']
        return dict(available=True,reference=record(path),observed={k:value[k] for k in keep if k in value})
    stages=dict(
        formal400=state(r8,'formal400_summary_v1/compact_result.json'),formal400_run=state(r8,'joint_rgb_formal400_v1/receipt.json'),
        DGGT_original_scene=state(r8,'audit/dggt_control_CPU_v1/control_summary.json'),
        R10_old_representation_diagnostics=state(r10,'zero_step_review_v1/review_summary.json'),
        R10_shared_extension=state(r10,'local_shared_continuous_v1/receipt.json'),
        R10_actual_new_zero=state(r10,'new_local_zero_v1/receipt.json'),
        R10_noncolour_counterfactual=state(r10,'ground_unowned_policy_v1/scope.json'),
        R10_root_active_stage=state(r10,'root_current_stage.json'),
        R10_actual_resource_ledger=state(r10,'docs/current_resource_ledger.json'),
        R10_publication_preparation=state(r10,'docs/current_stage_publication_inputs.pending.json'))
    if r10.is_dir():
        stages['R10_actual_stage_receipts']={folder.name:state(r10,folder.name+('/resources.json' if (folder/'resources.json').is_file() else '/receipt.json')) for folder in sorted(r10.iterdir())
            if folder.is_dir() and folder.name.startswith(r10_prefixes) and ((folder/'receipt.json').is_file() or (folder/'resources.json').is_file())}
    archive=read(archive_receipt) if archive_receipt and Path(archive_receipt).is_file() else None
    status=dict(schema='R8_R9_R10.publication_current_stage_snapshot.v1',complete=True,prepared_UTC=datetime.now(timezone.utc).isoformat(),
        snapshot_only=True,ready=False,finalized_by_root=False,narrative_generated=False,Git_operations_executed=False,stages=stages,
        full_archive_takeback_verified=bool(archive and archive.get('complete') and archive.get('local_archive_and_all_1700_member_SHA_verified')),
        R10_full_archive_takeback_verified=bool(r10_archive),
        R10_archive_takeback_reference=record(r10_archive_receipt) if r10_archive else None,
        whole_scene_restoration_claimed=False,official_DGGT_scope='five original input views, A9 retained; literal double-alpha and single-alpha diagnostic remain distinct; no accepted metric native import',
        current_vehicle_parent='original accepted A20 r3 ab4fa110; historical 400-step actor DC update is retained as a control, not automatically selected',
        source_target_parent='R2 road provides geometry/initialization. Current candidate geometry/source pins and qualified-real changes are recorded separately in R10_publication_preparation; old parent fields remain immutable and inferred material is not relabelled as real.',
        failure_family='V82-F06',human_verdict=None,problem_solved=None,
        final_requirements=['root explicit delivery pointer and narrative','current R10 actual results/root review','report QA','R10 archive and local takeback verification'],
        omitted_large_metadata=omitted,archive_only_types=['PNG','NPZ','PTH','full retained runtime tree','full member hash inventories'],
        sources=sources)
    status_path=out/'actual_stage_status.json';status_path.write_text(json.dumps(status,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
    entries.append(dict(source=status_path.relative_to(root).as_posix(),destination='actual_stage_status.json',require_ready=False,exists_at_preparation=True,sha256=sha(status_path),bytes=status_path.stat().st_size))
    result=dict(schema='R8_R9_R10.publication_current_stage_selection.v1',complete=True,finalized_by_root=False,ready=False,files=entries,
        actual_stage_status=record(status_path),omitted_large_metadata=omitted,Git_operations_executed=False)
    path=out/'selection.pending.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return dict(path=str(path),sha256=sha(path),files=entries,status=record(status_path),finalized_by_root=False,ready=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--archive-receipt',type=Path);p.add_argument('--r10-archive-receipt',type=Path)
    a=p.parse_args();value=prepare(a.root,archive_receipt=a.archive_receipt,r10_archive_receipt=a.r10_archive_receipt);value.pop('files');print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
