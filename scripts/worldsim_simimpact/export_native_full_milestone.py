"""保存已完成运行的分析状态和小型复现证据，不重跑实验。"""
import json,shutil,tarfile,time
from pathlib import Path
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
D=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-DETECTOR-01/20260915-r1')
F=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FF-PERCEPTION-01/20260915-r1')
C=N/'closed_loop/scene0004-step030000-full1';V=N/'validation/scene0004-step030000-full1'
out=D/'milestone6_evidence'
if out.exists():raise RuntimeError(f'Preserve {out}')
out.mkdir();shutil.copy2(__file__,out/'export_source_snapshot.py')
specs={'real':D/'real_scene0004_r1','native':D/'native_scene0004_step030000_r1','constant13':D/'constant13_scene0004_r1',
       'logged_pattern':D/'logged_pattern_detection_scene0004_r1','local':D/'local_excavator_detection_r1','ff':F/'detection_r1',
       'fixed_scans':D/'native_scans_scene0004_step030000_r1','logged_scans':D/'logged_pattern_scans_scene0004_r1',
       'loop':C,'validation':V}
counts={}
for name,root in specs.items():
    p=root/'registration.json';r=json.loads(p.read_text());assert r['completed'],root
    shutil.copy2(p,root/'registration.before_F19_analysis.json')
    r.update(failure_ledger_delta='V74-H2-F19',human_verdict=None,
        analysis_status='Full fit and downstream measurements complete. Excavator local sensor recovery has a strong intensity alternative on median readout; no geometry-asset causal repair or independent safety confirmation. See milestone6 report.')
    p.write_text(json.dumps(r,indent=2));dest=out/name;dest.mkdir()
    for fn in ['registration.json','summary.json','source_snapshot.py','policy_execution_summary.json','downstream_audit.json','sensor_interface_qa.json']:
        if (root/fn).exists():shutil.copy2(root/fn,dest/fn)
    if (root/'source_snapshot').exists():shutil.copytree(root/'source_snapshot',dest/'source_snapshot')
    counts[name]={k:r[k] for k in ['detector_forwards','native_lidar_renders','policy_predictions','pdm_forecasts','closed_loop_runs','checkpoint_step','completed'] if k in r}
for fn in ['fit_manifest.json','config.yml']:
    shutil.copy2(N/'fits/scene-0004/splatad/native-r2'/fn,out/fn)
for root,name in [(D/'object_audit_r2','object_audit'),(D/'local_excavator_inputs_r1','local_inputs'),(F,'ff_inputs')]:
    dest=out/name;dest.mkdir()
    for fn in ['registration.json','input_manifest.json','source_snapshot.py','prepare_source_snapshot.py','objects.json','instance_summary.json']:
        if (root/fn).exists():shutil.copy2(root/fn,dest/fn)
shutil.copy2(D/'input_alignment_audit.json',out/'input_alignment_audit.json')
manifest={'milestone':'6','completed_unix':time.time(),'counts':counts,
    'total_new_detector_forwards':sum(c.get('detector_forwards',0) for c in counts.values()),
    'historical_totals':{'geometry_forwards':72,'policy_PDM_forecasts':802,'HUGSIM_LTF_loops':18,'native_SplatAD_feedback_loops':4},
    'failure_ledger_delta':'V74-H2-F19','human_verdict':None,'goal_status':'active'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(D/'milestone6_evidence.tar.gz','w:gz') as t:
    for p in out.rglob('*'):
        if p.is_file():t.add(p,arcname=str(p.relative_to(out)))
print(json.dumps(manifest,indent=2))
