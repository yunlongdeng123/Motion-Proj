"""Compact final restore/evidence core; excludes intermediate float caches/checkpoints."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tarfile
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']=''


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    c=read(root/'contract.json');base=Path(c['baseline']['root']);run=root/'runs/formal80_r1';r=read(run/'receipt.json')
    source=np.load(base/'assets_object_clean_color_r1/teacher_actor.npz');actor=np.load(run/'actor.npz');obs=np.asarray(actor['observed_appearance_prior_geometry'],bool)
    proof={}
    for name in ('xyz','scales','rotation','opacity'):
        if not np.array_equal(source[name],actor[name]):raise AssertionError('Final geometry changed: '+name)
        proof[name]=dict(bitwise_equal=True,sha256=ah(actor[name]))
    if not np.array_equal(source['coefficients'][~obs],actor['coefficients'][~obs]) or not np.array_equal(source['coefficients'][:,1:],actor['coefficients'][:,1:]):raise AssertionError('Frozen colors changed')
    proof['unknown_DC']=dict(bitwise_equal=True,sha256=ah(actor['coefficients'][~obs,0]))
    proof['higher_SH']=dict(bitwise_equal=True,sha256=ah(actor['coefficients'][:,1:]))
    np.savez_compressed(root/'appearance_support.npz',observed_appearance_support=obs,unknown_appearance_support=~obs,
        observed_row_ids=np.flatnonzero(obs).astype(np.int32),geometry_is_generated_prior=np.asarray(True),measurement_gaussians_added=np.asarray(0))
    import torch
    torch.set_num_threads(2);checkpoint=torch.load(run/'checkpoint_final.pt',map_location='cpu',weights_only=False)
    if not np.array_equal(checkpoint['observed_dc'].numpy(),actor['coefficients'][obs,0]):raise AssertionError('Full mutable state differs from NPZ')
    if checkpoint['completed']!=80 or checkpoint['role']!='formal80' or checkpoint['immutable_full_geometry_and_unknown_DC_sha256']!=sha(root/'assets_native_410k_r1/control_init.npz'):raise AssertionError('Full restore state binding mismatch')
    adam_steps=[int(s['step']) for s in checkpoint['optimizer']['state'].values()]
    if adam_steps!=[80]:raise AssertionError('Formal control Adam budget is not exactly 80')
    if torch.cuda.is_initialized():raise AssertionError('CPU core builder initialized CUDA')
    verification=dict(schema='a20.native_support_control.final_CPU_proof.v1',passed=True,actor_sha256=sha(run/'actor.npz'),
        checkpoint_sha256=sha(run/'checkpoint_final.pt'),array_proof=proof,observed_rows=int(obs.sum()),unknown_rows=int((~obs).sum()),
        observed_checkpoint_equals_standard_NPZ=True,Adam_step=80,formal_started_from_clean_control_init=True,
        complete_restore_definition='Mutable observed-DC/Adam/four RNG checkpoint plus the SHA-bound complete frozen control_init.npz in this same core.',
        no_full_compression_distillation_claim=True,baseline40_and_its_closed_material80_unchanged=True,cuda_initialized=False)
    write(root/'final_CPU_freeze_proof.json',verification)
    selected=[root/'contract.json',root/'root_approval_formal80_r1.json',root/'appearance_support.npz',root/'final_CPU_freeze_proof.json',
        root/'assets_native_410k_r1/control_init.npz',root/'assets_native_410k_r1/assets_receipt.json',root/'assets_native_410k_r1/real_RGB_source_trace.npz',
        root/'assets_native_410k_r1/run_control.prepared.py',run/'actor.npz',run/'checkpoint_final.pt',run/'receipt.json',run/'entry_coverage.json',run/'run_control.executed.py',
        root/'runs/smoke_r1/receipt.json',root/'runs/smoke_r1/checkpoint_replay_source_step002.pt',root/'disk_reload_r1/receipt.json']
    selected+=sorted((root/'code').glob('*.py'))+sorted((run/'eval').rglob('*.png'))+sorted((run/'coverage/final').glob('*.npz'))
    selected+=sorted((root/'disk_reload_r1/NPZ_reload').rglob('*.png'))
    for path in sorted(root.glob('*.md')):selected.append(path)
    selected=list(dict.fromkeys(selected));entries=[]
    for path in selected:
        if not path.is_file() or root not in path.resolve().parents:raise ValueError('Invalid core member '+str(path))
        entries.append(dict(path=str(path.relative_to(root)).replace('\\','/'),sha256=sha(path),bytes=path.stat().st_size))
    manifest=dict(schema='a20.native_support_control.core.v1',files=entries,
        excluded=['intermediate float teacher caches','intermediate formal checkpoints','full scene/model/environment'],
        external_renderer_source=dict(baseline_root=str(base),runtime_sha256=c['baseline']['runtime_sha256'],contract_sha256=c['baseline']['contract_sha256']),
        final_restore_files=['assets_native_410k_r1/control_init.npz','runs/formal80_r1/checkpoint_final.pt','runs/formal80_r1/actor.npz'])
    write(root/'core_manifest.json',manifest);archive=root/'native_support_control_final_core_r1.tar.gz'
    if archive.exists():raise ValueError('Do not overwrite a sealed final core')
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for path in selected:tar.add(path,arcname=str(Path('core')/path.relative_to(root)),recursive=False)
        tar.add(root/'core_manifest.json',arcname='core/core_manifest.json',recursive=False)
    write(root/'core_archive_receipt.json',dict(schema='a20.native_support_control.core_archive.v1',archive_sha256=sha(archive),bytes=archive.stat().st_size,
        manifest_sha256=sha(root/'core_manifest.json'),files=len(entries),ready_for_takeback=True))
    print(json.dumps(read(root/'core_archive_receipt.json'),indent=2))
