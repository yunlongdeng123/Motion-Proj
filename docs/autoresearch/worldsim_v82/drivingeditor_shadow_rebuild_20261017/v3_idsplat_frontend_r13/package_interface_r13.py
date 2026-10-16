from pathlib import Path
import json,zipfile
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'idsplat_frontend_r13'
paths=list((D/'pointsets').glob('*.npz'))+list((D/'dinov3_validation/matches').glob('*.npz'))+[D/'data_IDSplat'/x for x in ['trajectory.npz','manifest.json','seed.npz']]+[D/x for x in ['protocol.json','frontend_complete.json','module_coverage_diagnostic.json','pointset_QUERY_gate.json','actual_data_audit.json']]+[D/'dinov3_validation/result.json']
with zipfile.ZipFile(D/'evidence_and_interface.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in paths:z.write(p,p.relative_to(D).as_posix())
print(json.dumps({'archive':str(D/'evidence_and_interface.zip'),'files':len(paths),'bytes':(D/'evidence_and_interface.zip').stat().st_size,'contains_uncertain_raw_source_pool':True,'contains_measured_actor_interface':True,'new_scene_training_updates':0}))
