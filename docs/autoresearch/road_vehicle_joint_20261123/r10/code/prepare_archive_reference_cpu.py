"""Bind existing verified R8/R9 archive and external immutable checkpoints; no archive creation."""
from pathlib import Path
import hashlib,json,argparse


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();W=a.workspace
    R=W/'work/road_underbody_local_20261123_r10';A=W/'outputs/road_vehicle_joint_dggt_20261123_r8/method_archive'
    M=A/'completed_R8_R9_snapshot_20261123.manifest.json';T=A/'local_takeback_receipt.json';V=A/'completed_R8_R9_snapshot_20261123.tar.gz.verify.json'
    m,t=read(M),read(T);assert t['complete'] and t['local_archive_and_all_1700_member_SHA_verified'] and len(m['members'])==1700
    assert t['archive_sha256']==m['archive']['sha256']=='41f79eda42cd03385fa135fbc783e279aca2a85d63a5a0b4997b36006f5d544c'
    npath=W/'work/native_source_correction_20261122_r5/provenance/native_source_manifest.json';n=read(npath)
    dpath=W/'work/road_vehicle_joint_dggt_20261123_r8/audit/dggt_control_CPU_v1/forward_prediction_audit.json';d=read(dpath)
    result=dict(schema='R10.previous_stage_external_dependencies.v1',complete=True,
        previous_verified_method_archive=dict(**m['archive'],member_count=1700,local_directory=str(A),
            verified_receipt=pin(T),member_manifest=pin(M),verification_receipt=pin(V),
            archived_payload='Completed R8/R9 methods, their new persisted assets and retained C++/Python runtime sources; see exact member manifest',
            repeated_2GB_archive_read_not_performed_here=True,prior_allmember_verification_used=True),
        native_checkpoint_retention=dict(mode='external_immutable_checkpoint',source_evidence=pin(npath),files=n['native_checkpoint_files']),
        DGGT_main_checkpoint_retention=dict(mode='external_immutable_checkpoint',file=d['checkpoint'],source_evidence=pin(dpath),
            contained_in_previous_method_tar=False,not_needed_to_replay_R10_DrivingEditor_renderer=True),
        source_dataset_camera_photo_dependencies='Remain external where existing R8/R9/R7/R10 contracts explicitly pin them; R10 new NPZ evidence and inputs are included by the R10 archive.',
        restoration_boundary='R10 archive plus verified prior archives restore new local fields, evidence and code. Complete native scene rendering still requires the external original static/dynamic/environment checkpoints and original data/config dependencies. DGGT checkpoint is needed only for DGGT control regeneration.',
        large_weights_copied_into_R10=False,remote_GPU_stays_on=True,HTML_snapshot_separate=True)
    write(R/'provenance/previous_stage_archive_reference.json',result)
    # Deliberately not a ready signal: Root fills the final footprint receipt only after it completes.
    template=dict(schema='R10.method_snapshot_ready.v1',run='road_underbody_local_20261123_r10',complete=False,
        method_stage_frozen=False,overall_run_complete=False,whole_scene_restoration_accepted=False,GPU_stays_on=True,
        report_HTML_included=False,pending_next_footprint_experiment=True,
        final_footprint_receipt=dict(path='REPLACE_WITH_FINAL_COMPLETED_FOOTPRINT_RECEIPT_RELATIVE_TO_R10',sha256='REPLACE_WITH_SHA256'),
        required_complete_receipts=[],critical_members=[],
        snapshot_scope='Stable R10 method/assets/evidence, including failed versions; ongoing overall task and final shared HTML snapshot are separate.',
        resources_summary='Fill actual completed receipts; do not infer quality from method_stage_frozen')
    write(R/'docs/final_method_snapshot_receipt.template.json',template)
    print(json.dumps(dict(reference=pin(R/'provenance/previous_stage_archive_reference.json'),template=pin(R/'docs/final_method_snapshot_receipt.template.json'))))


if __name__=='__main__':main()
