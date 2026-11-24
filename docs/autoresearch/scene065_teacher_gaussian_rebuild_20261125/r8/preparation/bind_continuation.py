from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;R7=R.parent/'road_teacher_region_rebuild_20261125_r7'
def read(p):return json.loads(Path(p).read_text())
def pin(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
cfg=read(R7/'config/training.json');rc=read(R7/'config/runtime.json');ex=read(R7/'config/execution.json')
cfg.update(target_unique_updates=2600,background_added_at_unique_step=2000,background_unique_updates=600,
    continuation_probe_unique_updates=16,readout_at=[2100,2400,2600],reconstruction_revision='r8_visible_BG_continuation')
cfg['category_weights']['real_visible_background']=1.
cfg['source_fixed_cache_manifest']=str(R7/'fixed_training_cache_v1/manifest.json')
cfg['source_fixed_cache_sha256']=pin(cfg['source_fixed_cache_manifest'])['sha256']
rc['candidate_pool']=pin(R/'initialization/combined_candidate_pool.npz')
rc['real_background_manifest']=pin(R/'manifest.json')
rc['retained_fragments']=pin(R/'initialization/retained_fragments_background_v1.npz')
rc['densification'].update(start_step=2001,special_steps=[2016],until_step=2400,every=50,birth_per_event=4096,max_densify_parents=512)
rc['reconstruction_revision']='r8_visible_BG_continuation'
ex.update(unique_optimizer_budget=2600,teacher_targets=pin(R/'config/targets8.remote.json'),candidate_pool=rc['candidate_pool'],
    parent_checkpoint=pin(R7/'training_v1/train/checkpoint_final.pt'),background_seed=pin(R/'initialization/bg_seed.npz'),
    background_manifest=rc['real_background_manifest'],previous_road_unique_updates=2000,new_background_unique_updates=600,
    continuation_preserves_road_optimizer_sampler_RNG=True,old_R1_student_or_optimizer_loaded=False,
    revision_reason='Rebuild only lost orange static support with legal same-position real photos; continue the already fresh 3D road field and residual densification')
write(R/'config/training.json',cfg);write(R/'config/runtime.json',rc);write(R/'config/execution.json',ex)
print(json.dumps(dict(parent_checkpoint=ex['parent_checkpoint'],runtime=pin(R/'config/runtime.json'),config=pin(R/'config/training.json'),execution=pin(R/'config/execution.json'))))
