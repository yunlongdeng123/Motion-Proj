from pathlib import Path
import json,hashlib,time

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
manifest=P/'trellis2/input_r1/manifest.json';d=json.loads(manifest.read_text())
assert d['status']=='PASS_CPU_INPUT' and d['source_frames']==[86,110]
assert d['patch']['vertices']==16384 and d['patch']['geometry_supported_vertices']==16384
assert d['reference_valid_rows']+d['reference_unknown_rows']==16384
assert d['patch']['actual_GS_world_vertex_error_m']<1e-7
assert not d['historical_generated_output_reused'] and d['GPU_operations']==0
for name,key in [('mesh','mesh_sha256'),('reference','reference_sha256')]:
    assert hashlib.sha256(Path(d[name]).read_bytes()).hexdigest()==d[key]
receipt=dict(schema='root_TRELLIS_input_and_exclusive_lease.v1',
 source_manifest=str(manifest),source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
 input_source_CPU_qualified=True,reference_unknown_explicit=True,
 inference_calls_authorized=1,training_updates=0,contract=str(P/'trellis2/execution_contract.json'),
 authorized_at_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),GPU_keep_on=True)
(P/'trellis2/root_input_acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n')
lease=json.loads((P/'gpu_lease.json').read_text());assert lease['active'] is False
(P/'gpu_lease.json').write_text(json.dumps(dict(gpu_authorized=True,active=True,owner='static_handoff',keep_GPU_on=True,
 scope='one TRELLIS2 fixed geometry inference; no native raster or training yet'),indent=2)+'\n')
print('TRELLIS exclusive lease granted: one inference, zero training')
