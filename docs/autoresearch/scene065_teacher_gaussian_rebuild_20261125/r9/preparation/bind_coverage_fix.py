from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parent;R8=R.parent/'road_teacher_background_continuation_20261125_r8'
def read(p):return json.loads(Path(p).read_text())
def pin(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
data=read(R8/'manifest.json');write(R/'manifest.json',data)
binding=read(R8/'config/targets8.remote.json');rows={int(x['raw']):x for x in data['frames']}
for row in binding['frames']:
    if int(row['raw'])in rows:row['background_geometry']=rows[int(row['raw'])]['fixed_masks']
write(R/'config/targets8.remote.json',binding)
with np.load(R8/'initialization/bg_seed.npz',allow_pickle=False)as z:
    seed={k:z[k][:0] if z[k].ndim else z[k] for k in z.files}
(R/'initialization').mkdir(exist_ok=True);np.savez_compressed(R/'initialization/empty_append.npz',**seed)
cfg=read(R8/'config/training.json');rc=read(R8/'config/runtime.json');ex=read(R8/'config/execution.json')
cfg.update(target_unique_updates=2800,background_unique_updates=800,continuation_start_step=2600,readout_at=[2700,2800],
    reconstruction_revision='r9_unknown_BG_geometry_coverage_fix',unknown_BG_RGB_loss_weight=0,
    protection_RGB_normalization='Full original protected denominator; unknown_BG direct RGB weight0, no frozen-render truth there')
cfg['source_fixed_cache_manifest']=str(R8/'fixed_training_cache_v1/manifest.json');cfg['source_fixed_cache_sha256']=pin(cfg['source_fixed_cache_manifest'])['sha256']
rc['densification'].update(start_step=2601,special_steps=[2616],until_step=2700,every=50,birth_per_event=1024,max_densify_parents=256)
rc['reconstruction_revision']='r9_unknown_BG_geometry_coverage_fix'
rc['retained_fragments']=pin(R/'initialization/retained_fragments_background_v1.npz')
ex.update(unique_optimizer_budget=2800,teacher_targets=pin(R/'config/targets8.remote.json'),parent_checkpoint=pin(R8/'training_v1/train/checkpoint_final.pt'),
    background_seed=pin(R/'initialization/empty_append.npz'),previous_road_unique_updates=2600,new_background_unique_updates=800,
    revision_reason='Separate geometric occupancy from RGB permission; preserve old proxy RGB veto and fixed evaluation range; no new RGB source')
for name,value in [('training.json',cfg),('runtime.json',rc),('execution.json',ex)]:write(R/'config'/name,value)
print(json.dumps(dict(configuration=pin(R/'config/training.json'),runtime=pin(R/'config/runtime.json'),execution=pin(R/'config/execution.json'))))
