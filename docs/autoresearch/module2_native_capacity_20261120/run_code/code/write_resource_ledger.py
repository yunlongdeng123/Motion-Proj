from pathlib import Path
import json
O=Path(__file__).resolve().parents[2]/'outputs/road_native_capacity_20261120_r1'
rows=[
 dict(phase='native_preflight',F=8,B=2,engineering_accepted=0,seconds=17.82),
 dict(phase='full_lifecycle_profile_and_completion',F=58,B=16,engineering_accepted=7,seconds=70.25),
 dict(phase='actual_smoke_six_nonoverlap_phases',F=11931,B=6414,engineering_accepted=3205,seconds=520.9650422800332),
 dict(phase='registered_QUERY_four_arms',F=34834,B=14804,formal_Adam=2400,formal_box_accepted=5000,seconds=804.46),
 dict(phase='QUERY_public_fiveview',F=82,B=0,seconds=134.027),
 dict(phase='QUERY_numeric_lifecycle_smoke',F=14,B=16,engineering_accepted=3,seconds=29.76),
 dict(phase='QUERY_numeric_completion',F=4002,B=4000,formal_FG=2000,formal_box_accepted=1925,seconds=338.951339725405),
 dict(phase='QUERY_numeric_public_fiveview',F=20,B=0,seconds=34.449),
 dict(phase='BUILD_initialization_lifecycle_and_two_arms',F=6430,B=6426,formal_Adam=1200,formal_FG=2000,formal_box_accepted=1933,engineering_accepted=9,seconds=475.7209029365331),
 dict(phase='BUILD_public_fiveview',F=40,B=0,seconds=60.216)]
result=dict(schema='native_capacity_actual_phase_resource_ledger.v1',phases=rows,
 native_forwards=sum(r['F']for r in rows),native_per_view_backwards=sum(r['B']for r in rows),
 formal_Adam_updates=sum(r.get('formal_Adam',0)for r in rows),formal_box_accepted_steps=sum(r.get('formal_box_accepted',0)for r in rows),
 engineering_accepted_steps=sum(r.get('engineering_accepted',0)for r in rows),
 recorded_native_phase_wall_seconds=sum(r['seconds']for r in rows),
 wall_time_note='Some rounded phase seconds; includes loading/save/CPU work inside native phase, not kernel time or total instance uptime. Detailed raw receipts remain authoritative.',
 units_note='F=actual native renderer forward, B=per-view native backward. Adam steps, accepted bounded steps and FG queries are different units; line-search trials and replay are included in actual calls.',
 CPU_evaluation_native_calls=0,CPU_audit_native_calls=0,extra_generation_model_calls=0,
 new_geometry_or_density=0,validation_selected_hyperparameters=False,default_scene_modified=False,GPU_keep_on=True)
(O/'resource_ledger.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps({k:result[k]for k in ['native_forwards','native_per_view_backwards','formal_Adam_updates','formal_box_accepted_steps','engineering_accepted_steps']}))
