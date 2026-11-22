"""Make a minimal independent17-frame copy; never alter the prior8-frame files."""
import hashlib,json
from pathlib import Path

code=Path(__file__).resolve().parent
source=code/'select_expanded_curvature_loo_cpu.py'
target=code/'select_timewindow17_curvature_loo_cpu.py'
text=source.read_text(encoding='utf-8')
changes=[
('TRAIN = [60, 70, 80, 95, 110, 125, 145, 165]','TRAIN = list(range(90, 171, 5))'),
("evidence_path=R/'audit/ground_contact_prior_raw_GT_v2/GT_bottom_and_road_evidence.npz'","evidence_path=R/'audit/timewindow17_geometry_v1/training_points.npz'"),
("X=e['official_training_road_worldXYZ']; raw=e['official_training_road_source_raw']; rowid=e['official_training_road_lidar_row_id']","X=e['world_xyz']; raw=e['source_raw']; rowid=e['source_lidar_row_id']"),
('Leave one of8 training frames out','Leave one of17 time-window training frames out'),
('R10.expanded_training_LOO_curvature_contract.v1','R10.timewindow17_training_LOO_curvature_contract.v1'),
('R10.expanded_official_ground_training_data.v1','R10.timewindow17_official_ground_training_data.v1'),
('R10.expanded_training_only_curvature_selection.v1','R10.timewindow17_training_only_curvature_selection.v1'),
]
for old,new in changes:
    assert text.count(old)==1,old
    text=text.replace(old,new)
if target.exists():assert target.read_text(encoding='utf-8')==text
else:target.write_text(text,encoding='utf-8')
audit_source=code/'audit_expanded_training_folds_cpu.py';audit_target=code/'audit_timewindow17_training_folds_cpu.py'
audit=audit_source.read_text(encoding='utf-8');old="D=R/'audit/expanded_curvature_loo_v1'";assert audit.count(old)==1
audit=audit.replace(old,"D=R/'audit/timewindow17_geometry_v1/loo_selection'")
audit=audit.replace('R10.expanded_original_point_LOO_height_audit.v1','R10.timewindow17_original_point_LOO_height_audit.v1')
if audit_target.exists():assert audit_target.read_text(encoding='utf-8')==audit
else:audit_target.write_text(audit,encoding='utf-8')
receipt=dict(schema='R10.timewindow17_minimal_selector_copy.v1',complete=True,
    old_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),new_source_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    replacements=[dict(old=o,new=n) for o,n in changes],loss_family_CV_math_oneSE_bootstrap_unchanged=True,
    only_training_input_and_frame_set_and_schema_labels_changed=True,old_source_not_modified=True,GPU_calls=0)
receipt_path=code.parent/'audit/timewindow17_geometry_v1/selector_copy_receipt.json';receipt_path.parent.mkdir(parents=True,exist_ok=True)
assert not receipt_path.exists()
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
