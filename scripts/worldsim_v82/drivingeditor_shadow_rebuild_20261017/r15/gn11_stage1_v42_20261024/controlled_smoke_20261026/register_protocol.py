from pathlib import Path
import json,datetime
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_v42_controlled_20261026');C.mkdir(exist_ok=True)
p=dict(task_id='WS-V82-GN11-V42-CONTROLLED-SMOKE-20261026',authorization='User GPU ready; both routes with controls and short smoke approved 2026-10-26',created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 master='V41-B30K four frozen native checkpoints; original native render is default complete reconstruction',
 arms=['matched49 observed road','full199 observed road','full199 observed road plus FreeInpaint only in residual unknown'],
 factors=['static without road','road without static suppression','static suppression only','static suppression plus road'],
 common_edit_gate='Fixed3D diagnostic selection from raw102 a37/a6 dilated2px ray volume,0.25..3m above observed road and nearest support<=0.4m; not ground truth contamination classifier',
 input_roles=dict(BUILD='raw0..198 RGB, LiDAR, known poses, all labels for source visibility',QUERY='raw89..137 fixed view diagnostic incl raw102 centers; shares BUILD, no unseen-generalization claim',
 generative_probe='masked a37/a6 from source raw102/109/116, same source images/masks across cases; recovered full199 ground is an observed reference, not truly unseen ground'),
 inference=dict(freeinpaint='frozen source9ac03f14 + official weights; seed0; three views initial; one forward engineering then3pass STR; no optimizer',
 metric_alignment='Known camera pose metric scale; unmasked scene LiDAR anchors for fit/check split. Do not force a road plane into generated center; reject invalid alignment before 3D edit insertion'),
 short_training=dict(stage='observed road colour/opacity only; fixed geometry; frozen B',proposed_steps=1000,smoke_wall_minutes_target=10,max_smoke_steps=1000,
 checkpoint_steps=[100,500,1000],required=['finite forward/backward','save all trainable+optimizer+RNG+sampler','cross-process restore','next-step comparison','evaluation','next-stage input'],
 note='Step target is short engineering budget, not ten-thousand formal updates; external frozen inference has no optimizer'),
 original_native_updates=0,smoke_and_retry_separate=True,quality_accepted=False,human_verdict=None,failure_ledger_refs=['V82-F01','V82-F06'])
f=C/'protocol.json'
if f.exists():raise FileExistsError(f)
f.write_text(json.dumps(p,indent=2,ensure_ascii=False))
print(f)
