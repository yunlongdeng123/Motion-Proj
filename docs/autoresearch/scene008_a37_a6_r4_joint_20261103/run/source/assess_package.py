"""All49 acceptance, measured-domain summary and portable R4+addon package."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,math,tarfile,subprocess,hashlib,io,shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import imageio_ffmpeg
RUN=Path(__file__).resolve().parent.parent;OLD=RUN.with_name('shadow_a5_a37_199_20261103_r4')
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
    assert (RUN/'queue_complete.json').exists();frames=RUN/'evaluation/all49';dest=RUN/'quality';dest.mkdir(exist_ok=False)
    q=json.loads((frames/'metrics.json').read_text());logs=[json.loads(x) for x in (RUN/'formal/training_from0.jsonl').read_text().splitlines()]
    assert [x['step'] for x in logs]==list(range(1,1001)) and all(math.isfinite(x['loss']) for x in logs)
    ck=torch.load(RUN/'formal/checkpoint001000.pt',map_location='cpu',weights_only=False);assert {int(v['step']) for v in ck['optimizer']['state'].values()}=={1000}
    restored={str(r):(frames/('%03d_joint.png'%r)).read_bytes()==(RUN/'formal/step001000'/('%03d_joint.png'%r)).read_bytes() for r in [101,105,113]};assert all(restored.values())
    baseline={str(r):(frames/('%03d_R4.png'%r)).read_bytes()==(OLD/'evaluation/all49'/('%03d_repaired_delete5.png'%r)).read_bytes() for r in [101,105,113]};assert all(baseline.values())
    aggregates={}
    for actor in ['6','37']:
        rows=[x['actors'][actor] for x in q['rows'] if x['actors'][actor]['samples']>0];den=sum(x['samples'] for x in rows)
        before=sum(x['R4_MAE']*x['samples'] for x in rows)/den;after=sum(x['joint_MAE']*x['samples'] for x in rows)/den
        aggregates[actor]=dict(R4_MAE=before,joint_MAE=after,reduction=1-after/before,frames=len(rows),samples=den,improved_frames=sum(x['joint_MAE']<x['R4_MAE'] for x in rows))
    shadow={}
    for actor in ['5','37']:
        rows=[x['original_shadow'][actor] for x in q['rows'] if x['original_shadow'][actor]['samples']>0];den=sum(x['samples'] for x in rows)
        shadow[actor]=dict(samples=den,frames=len(rows),R4_MAE=sum(x['R4_MAE']*x['samples'] for x in rows)/den,joint_MAE=sum(x['joint_MAE']*x['samples'] for x in rows)/den)
    for raw in range(89,138):
        imgs=[Image.open(frames/('%03d_R4.png'%raw)),Image.open(frames/('%03d_joint.png'%raw))];im=Image.new('RGB',(960,320));im.paste(imgs[0],(0,0));im.paste(imgs[1],(480,0));im.save(dest/('%03d_comparison.png'%raw))
    video=RUN/'R4_joint_a37_a6_all49.mp4';subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-n','-framerate','10','-start_number','89','-i',str(dest/'%03d_comparison.png'),'-frames:v','49','-c:v','libx264','-crf','17','-pix_fmt','yuv420p',str(video)],check=True)
    it=imageio_ffmpeg.read_frames(str(video));next(it);decoded=sum(1 for _ in it);assert decoded==49
    summary=dict(joint_actors=[37,6],deletion_set=[5,26,18,6,37],aggregates=aggregates,original_shadow=shadow,rows=q['rows'],
        outside_R4_MAE_mean=float(np.mean([x['outside_MAE'] for x in q['rows']])),outside_R4_MAE_max=max(x['outside_MAE'] for x in q['rows']),
        restored_PNG_exact=restored,R4_baseline_PNG_exact=baseline,frames_decoded=decoded,formal_updates=1000,smoke_updates=32,replay_updates=2,total_updates=1034,
        prior_updates=10204,cumulative_updates=11238,training_seconds=json.loads((RUN/'formal/receipt.json').read_text())['seconds'],
        CUDA_peak_allocated=max(x['cuda_peak_allocated'] for x in logs),CUDA_peak_reserved=max(x['cuda_peak_reserved'] for x in logs),
        invariants=json.loads((RUN/'evaluation/invariants.json').read_text()),input=json.loads((RUN/'inputs/receipt.json').read_text()),
        selection=json.loads((RUN/'selection_receipt.json').read_text()),all199_BUILD=True,reference_is_proxy=True,independent_test=False,unknown_reference_not_filled=True,
        parent_unchanged=q['parent_unchanged'],quality_accepted=False,human_verdict=None,GPU_shutdown=False,failure_ledger_refs=['V82-F01','V82-F06'],failure_ledger_delta='none')
    write(RUN/'result_summary.json',summary);print(json.dumps(dict(actors=aggregates,original_shadow=shadow,outside=summary['outside_R4_MAE_mean'])),flush=True)
    # Minimal complete R4 rendering dependency: preserve exact tensors needed
    # by the original loader while omitting unused source RGB sample arrays.
    dependency=RUN/'dependency'/OLD.name;(dependency/'inputs/views').mkdir(parents=True,exist_ok=False);(dependency/'formal').mkdir()
    r=np.load(OLD/'inputs/road_reference.npz');fields=['keys','active','model_xyz','gaussian_sigma_m','normal_world','reference_srgb']
    np.savez_compressed(dependency/'inputs/road_reference.npz',**{k:r[k] for k in fields});shutil.copy2(OLD/'inputs/contributors.npz',dependency/'inputs/contributors.npz');shutil.copy2(OLD/'formal/checkpoint001000.pt',dependency/'formal/checkpoint001000.pt')
    for raw in range(89,138):
        d=np.load(OLD/'inputs/views'/('%03d.npz'%raw));fields=['domain_a5','domain_a37','shadow_proxy','reference_linear']
        np.savez_compressed(dependency/'inputs/views'/('%03d.npz'%raw),**{k:d[k] for k in fields})
    with tarfile.open(RUN/'review_light.tar.gz','w:gz',compresslevel=3) as t:
        for folder in ['source','quality','diagnostic_joint']:
            t.add(RUN/folder,arcname='run/'+folder,filter=lambda x:None if '__pycache__' in x.name else x)
        for rel in ['formal/checkpoint001000.pt','formal/contract.json','formal/receipt.json','formal/training_from0.jsonl','evaluation/restore.json','evaluation/invariants.json','inputs/receipt.json','selection_receipt.json','smoke_receipt.json','result_summary.json']:
            t.add(RUN/rel,arcname='run/'+rel)
        t.add(video,arcname=video.name)
    archive=RUN/'joint_a37_a6_replay.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=3) as t:
        for folder in ['source','inputs','quality','evaluation','formal','smoke','smoke_restore','smoke_repeat','diagnostic_joint']:
            t.add(RUN/folder,arcname=RUN.name+'/'+folder,filter=lambda x:None if '__pycache__' in x.name else x)
        for rel in ['joint_contributors.npz','selection_receipt.json','smoke_receipt.json','queue_complete.json','result_summary.json',video.name,'build_evidence.log','train.log','eval.log']:
            t.add(RUN/rel,arcname=RUN.name+'/'+rel)
        t.add(dependency,arcname=OLD.name)
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    write(RUN/'package_receipt.json',dict(bytes=archive.stat().st_size,sha256=h.hexdigest(),file=archive.name,R4_render_dependency_included=True,
        original_B_weights_and_V3_bundle_external=True,GPU_shutdown=False));print(json.dumps(dict(archive_bytes=archive.stat().st_size,sha256=h.hexdigest())),flush=True)
if __name__=='__main__':main()
