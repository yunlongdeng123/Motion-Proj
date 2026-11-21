"""Compare the essential independently audited artifacts across remote and local copies."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--make',action='store_true')
a=p.parse_args();root=a.root.resolve()
manifest=root/'audit_delivery_manifest.json'
if a.make:
    files=['gate.md','independent_grey_audit.md','grey_gate_result.json','preregistered_gate.json',
           'bbox_axis_audit.md','original_bbox_visual_findings.json','bbox_frame_fixed_visual_findings.json','architecture_components.png']
    files += ['source/'+n for n in ['audit_hq_cpu.py','shape_audit_cpu.py','preregistered_gate.json',
              'render_exact_orthographic_cpu.py','spatial_residuals_cpu.py','build_independent_report.py','build_audit_diagram.py']]
    files += [n+'/audit.json' for n in ['uniform_audit','registered_audit','bbox_frame_fixed_uniform_audit','bbox_frame_fixed_registered_audit']]
    for prefix in ['','bbox_frame_fixed_']:
        files += [prefix+'uniform_initialization/'+n for n in ['semantic_uniform_transform.json','candidate_actor_uniform_init.ply']]
        files += [prefix+'bounded_registration/'+n for n in ['bounded_registration.json','candidate_actor_bounded_registered.ply']]
    for stage in ['bbox_frame_fixed_uniform_audit','bbox_frame_fixed_registered_audit']:
        files += [stage+'/'+n for n in ['point_surface_residuals.npz','grey_six_orthographic.png','measured_points_six_orthographic.png',
                  'second_component_locator.png','exact_orthographic_receipt.json','projection_086.png','projection_110.png']]
    files += ['bbox_frame_fixed_registered_audit/'+n for n in ['spatial_residual_breakdown.json','largest_100_residuals.csv']]
    records=[{'relative_path':n,'bytes':(root/n).stat().st_size,'sha256':sha(root/n)} for n in files]
    result={'schema':'a20.hq.audit_delivery_manifest.v1','files':records,'file_count':len(records),
            'total_bytes':sum(r['bytes'] for r in records),'reference':'Generated from the completed local audit delivery.'}
    manifest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
else:
    result=json.loads(manifest.read_text(encoding='utf-8'))
    for r in result['files']:
        q=(root/r['relative_path']).resolve()
        assert q.is_relative_to(root)
        assert q.stat().st_size==r['bytes'],r['relative_path']
        assert sha(q)==r['sha256'],r['relative_path']
    receipt={'schema':'a20.hq.audit_delivery_receipt.v1','verified':True,'file_count':result['file_count'],
             'total_bytes':result['total_bytes'],'manifest_sha256':sha(manifest),
             'method':'All essential artifact byte sizes and SHA256 values matched the local delivery manifest.',
             'gpu_used':False,'geometry_modified':False}
    (root/'audit_delivery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))
