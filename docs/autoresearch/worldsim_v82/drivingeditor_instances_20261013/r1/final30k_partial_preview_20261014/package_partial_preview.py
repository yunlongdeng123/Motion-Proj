"""Create a new offline preview package, preserving the old full-report package."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json

root=Path(__file__).resolve().parents[2]
outputs=root/'outputs'
destination=outputs/'DrivingEditor_final_reports_instances_preview.zip'
if destination.exists():
    raise FileExistsError('Preserve previous package')
extra=[outputs/'DrivingEditor_final_report.html',outputs/'DrivingEditor_instance_editor_report.md',
       outputs/'DrivingEditor_instance_editor_status.json',outputs/'DrivingEditor_instance_editor_components.png',
       outputs/'DrivingEditor_instance_editor_components.svg']
extra.extend(path for path in (outputs/'drivingeditor_GMC_instances_30k').rglob('*') if path.is_file())
names={path.relative_to(outputs).as_posix() for path in extra}
with ZipFile(outputs/'DrivingEditor_final_reports.zip','r') as previous, ZipFile(destination,'x',ZIP_DEFLATED) as package:
    for item in previous.infolist():
        if item.filename not in names:
            package.writestr(item,previous.read(item.filename))
    for path in sorted(extra):
        package.write(path,path.relative_to(outputs).as_posix())
with ZipFile(destination,'r') as package:
    if package.testzip() is not None:
        raise ValueError('Offline package integrity failure')
    assert len(package.namelist())==len(set(package.namelist()))
    assert 'DrivingEditor_final_report.html' in package.namelist()
    assert sum(name.startswith('drivingeditor_GMC_instances_30k/') and name.endswith('.png')
               for name in package.namelist())==420
receipt={'status':'ACTUAL_UNIFIED_PARTIAL_PREVIEW_OFFLINE_PACKAGE_CREATED',
         'package':str(destination),'bytes':destination.stat().st_size,'actual_observed_frames':28,
         'new_instance_PNG_count':420,'fixed_frames':0,'edit_operations_30k':0,
         'old_package_preserved':True,'single_top_level_unified_entry':'DrivingEditor_final_report.html',
         'human_quality_verdict':None}
with (root/'work/drivingeditor_instances/partial_preview_offline_package_receipt.json').open('x',encoding='utf-8') as stream:
    json.dump(receipt,stream,ensure_ascii=False,indent=2)
print(json.dumps(receipt))
