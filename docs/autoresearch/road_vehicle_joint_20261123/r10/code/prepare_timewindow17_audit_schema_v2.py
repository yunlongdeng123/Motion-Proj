"""Correct legacy8-frame field labels in a fresh17-frame audit; numerical code unchanged."""
import hashlib,json
from pathlib import Path
code=Path(__file__).resolve().parent
source=code/'audit_expanded_training_folds_cpu.py';target=code/'audit_timewindow17_training_folds_cpu_v2.py';assert not target.exists()
text=source.read_text(encoding='utf-8')
changes=[("D=R/'audit/expanded_curvature_loo_v1'","D=R/'audit/timewindow17_geometry_v1/loo_selection'"),
    ('R10.expanded_original_point_LOO_height_audit.v1','R10.timewindow17_original_point_LOO_height_audit.v2'),
    ('training_fold_original_height_audit.json','training_fold_original_height_audit_v2.json'),
    ('training_original_point_predictions.npz','training_original_point_predictions_v2.npz'),
    ('all8_','all17_'),('raw60=','first_frame='),('raw70=','second_frame=')]
for old,new in changes:
    assert old in text,old;text=text.replace(old,new)
target.write_text(text,encoding='utf-8')
receipt=dict(complete=True,schema='R10.timewindow17_audit_schema_correction.v2',
    numerical_audit_unchanged=True,original_legacy_named_audit_preserved=True,
    actual_frame_count=17,correction='Original minimal copy calculated all17 but retained historical all8 field names; v2 labels them all17. No data/model/number selection changes.',
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),replacements=changes,GPU_calls=0)
p=code.parent/'audit/timewindow17_geometry_v1/audit_schema_correction_v2.json';p.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
