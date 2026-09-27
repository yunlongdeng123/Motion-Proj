"""检查实际RGB写回/证据来源与DELETE身份；不把像素合同当视觉通过。"""
from hybrid_common import *
rows=[];admission=[]
guards=read(ROOT/'guard_summary.json')['scenes']
for s in read(ROOT/'registration.json')['scenes']:
 out=ROOT/s['name'];inst=read(pathlib.Path(s['data'])/'instances/instances_info.json');actor=inst[s['actor']];assert actor['id']==s['track_id'];assert all(f in actor['frame_annotations']['frame_idx'] for f in s['source_frames'])
 index=read(out/'geometry_source_index.json');donors=np.load(OLD/s['name']/'donors.npz')
 temporal=0;geometry=0
 for i,f in enumerate(s['source_frames']):
  m=load_masks(out,i);original=rgb(out/'rgb'/f'{i:05}.png');final=rgb(out/'final'/f'{i:05}.png');e=rgb(out/'evidence'/f'{i:05}.png');native=rgb(out/'native_png'/f'{i:05}.png')
  assert not (m['delete']&m['protect']).any();assert not (m['generate']&m['protect']).any();assert np.array_equal(m['generate'],m['observed']|m['residual_generate']);assert not (m['observed']&m['residual_generate']).any()
  for area,a,b in [(m['protect'],final,original),(~m['generate'],final,original),(m['observed'],final,e),(m['residual_generate'],final,native),(~m['observed'],e,original)]:assert np.array_equal(a[area],b[area]),(s['name'],f)
  with np.load(out/'provenance'/f'{i:05}.npz') as z:
   for sf in np.unique(z['temporal_source']):
    if sf<0:continue
    use=z['temporal_source']==sf;uv=z['temporal_uv'][use];src=rgb(pathlib.Path(s['data'])/'images'/f'{int(sf):03}_{s["camera"]}.jpg');assert np.array_equal(e[use],src[uv[:,1],uv[:,0]]);assert (np.abs(z['temporal_second'][use]-sf)>=5).all();temporal+=int(use.sum())
   flat=z['geometry_target_index'];used=z['geometry_used'].reshape(-1)[flat];ids=z['geometry_source_id'][used];pids=z['geometry_source_point'][used];flat=flat[used]
   for sid in np.unique(ids):
    use=ids==sid;col=donors[f'{sid}_rgb'][pids[use]];assert np.array_equal(e.reshape(-1,3)[flat[use]],col);assert (donors[f'{sid}_lidar_distance'][pids[use]]<=.200001).all();geometry+=int(use.sum())
  rows.append(dict(scene=s['name'],frame=f,protect_unchanged=True,outside_generate_unchanged=True,observed_locked=True,native_written_exactly_to_residual=True))
 donors.close()
 guard=next(g for g in guards if g['scene']==s['name']);blocked=guard['arms']['final']['blocked_frames'];assert blocked
 admission.append(dict(scene=s['name'],actor=s['actor'],operation='DELETE',track_id=s['track_id'],identity_frames=30,new_occupied_volume=0,legality_scope='target identity and removal semantics; no MOVE and no claim of full traffic-rule legality',admitted=False,status='FAIL_ACTOR_REGENERATION',blocked_frames=blocked,background_input_dir=None,new_omega_forwards=0,new_glb_assets=0,temporal_evidence_pixels_checked=temporal,geometry_evidence_pixels_checked=geometry,human_verdict=None))
dump(ROOT/'validation.json',dict(frames=rows,total_frames=60,identity_checks=2,contract_passed=True,scope='pixel and provenance contract only, visual quality failed',human_verdict=None))
dump(ROOT/'background_admission.json',dict(scenes=admission,stop_rule_triggered=True,default_pipeline_changed=False,human_verdict=None))
print('VALIDATED 60 frames; both candidates blocked from Omega')
