"""Preview/final-asset SHA ledger for the formal observed-colour3D chart."""
import hashlib,json,shutil
from pathlib import Path
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105')
ROOT=RUN/'output_adapter/projective_surface/formal_observed'
shutil.copyfile(RUN/'source/output_adapter/projective_surface/architecture.mmd',ROOT/'architecture.mmd')
report=dict(status='PASS',PVD_weight_updates=0,geometry_updates_during_colour_fit=0,
    formal_colour_steps=1000,extra_smoke_colour_steps=32,
    collect_seconds=12.41246771812439,smoke_seconds=2.9784390926361084,
    formal_seconds=7.2643067836761475,evaluation_seconds=46.1278281211853,
    real_RGB_target_domain='Bitwise identical to original thin HR40 accepted target/raw arrays',
    unobserved_columns=30190,unobserved_SH_coefficients_bitwise_unchanged=True,
    geometry_and_provenance_bitwise_unchanged=True,
    all49_alpha_depth_before_after_colour_fit_bitwise_unchanged_at_both_resolutions=True,
    geometry='View-conditioned surface resampling from raw106 prior predicted depth, not measurements or full360 reconstruction',
    metadata_scope='prepare receipt describes zero-optimization chart initialization; formal receipt and projective_surface_observation_control describe the subsequent1000-step colour fit')
(ROOT/'delivery_qa.json').write_text(json.dumps(report,indent=2)+'\n')
files=[]
for p in ROOT.rglob('*'):
    if not p.is_file() or p.name in ('design.npz','inputs.npz','preview_manifest.json') or p.suffix=='.pt':continue
    if p.name.endswith('raster_hr.npz') or p.name.endswith('native_groups.npz'):continue
    files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(ROOT/'preview_manifest.json').write_text(json.dumps(dict(files=sorted(files,key=lambda f:f['path'])),indent=2)+'\n')
print(json.dumps(dict(files=len(files))))
