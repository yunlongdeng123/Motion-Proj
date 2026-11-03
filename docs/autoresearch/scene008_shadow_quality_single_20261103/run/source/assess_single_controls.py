"""Fixed yellow preservation metrics for actual fresh actor-only 1k runs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,math,subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import imageio_ffmpeg
RUN=Path(__file__).resolve().parent.parent;OLD=RUN.with_name('shadow_a5_a37_199_20261103_r4')
def rgb(p):return np.asarray(Image.open(p),np.float32)/255.
def main():
    assert (RUN/'single_controls_complete.json').exists();dest=RUN/'single_controls_review';dest.mkdir(exist_ok=False);rows=[];checks={}
    for actor in [5,37]:
        p=RUN/('single_a%d'%actor);log=[json.loads(x) for x in (p/'formal/training_from0.jsonl').read_text().splitlines()]
        assert [x['step'] for x in log]==list(range(1,1001)) and all(math.isfinite(x['loss']) for x in log)
        a=torch.load(p/'smoke_restore/checkpoint000033.pt',map_location='cpu',weights_only=False);b=torch.load(p/'smoke_repeat/checkpoint000033.pt',map_location='cpu',weights_only=False)
        dif=max(float((v-b['repair'][k]).abs().max()) for k,v in a['repair'].items() if v.is_floating_point());assert dif<1e-4
        pngexact={str(r):(p/'evaluation/all49'/('%03d_repaired_delete5.png'%r)).read_bytes()==(p/'formal/step001000'/('%03d_repaired_delete5.png'%r)).read_bytes() for r in [101,105,113]};assert all(pngexact.values())
        ck=torch.load(p/'formal/checkpoint001000.pt',map_location='cpu',weights_only=False);assert set(int(x['step']) for x in ck['optimizer']['state'].values())=={1000}
        checks[str(actor)]=dict(finite_steps=1000,resumed_next_step_maxabs=dif,formal_restore_PNG_exact=pngexact,invariants=json.loads((p/'evaluation/invariants.json').read_text()),input=json.loads((p/'inputs/receipt.json').read_text()))
    for raw in range(89,138):
        base=rgb(OLD/'evaluation/all49'/('%03d_baseline_delete5.png'%raw));d=np.load(OLD/'inputs/views'/('%03d.npz'%raw));yellow=(base[:,:,0]>1.22*base[:,:,2])&(base[:,:,1]>1.08*base[:,:,2])&(base[:,:,0]>.32)&d['repair_mask']&(d['actor_labels']==0)
        images=[base,rgb(RUN/'single_a5/evaluation/all49'/('%03d_repaired_delete5.png'%raw)),rgb(RUN/'single_a37/evaluation/all49'/('%03d_repaired_delete5.png'%raw)),rgb(OLD/'evaluation/all49'/('%03d_repaired_delete5.png'%raw))]
        record=dict(raw=raw,samples=int(yellow.sum()),**{name:float(np.abs(im-base)[yellow].mean()) for name,im in zip(['B','a5_only','a37_only','joint_R4'],images)})
        rows.append(record);canvas=Image.new('RGB',(1920,320))
        for k,im in enumerate(images):canvas.paste(Image.fromarray(np.uint8(np.round(im*255))),(k*480,0))
        canvas.save(dest/('%03d_comparison.png'%raw))
    n=sum(x['samples'] for x in rows);agg={name:sum(x[name]*x['samples'] for x in rows)/n for name in ['a5_only','a37_only','joint_R4']}
    actor_metrics={}
    for actor in [5,37]:
        data=json.loads((RUN/('single_a%d'%actor)/'evaluation/all49/metrics.json').read_text())['rows'];a=[x['actors'][str(actor)] for x in data];a=[x for x in a if x['shadow_MAE'] is not None];nshadow=sum(x['shadow_samples'] for x in a)
        actor_metrics[str(actor)]=dict(shadow_MAE=sum(x['shadow_MAE']*x['shadow_samples'] for x in a)/nshadow,shadow_samples=nshadow,frames=len(a))
    video=RUN/'single_B_a5_a37_joint_all49.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-n','-framerate','10','-start_number','89','-i',str(dest/'%03d_comparison.png'),'-frames:v','49','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(video)],check=True)
    reader=imageio_ffmpeg.read_frames(str(video));next(reader);decoded=sum(1 for _ in reader);assert decoded==49
    result=dict(rows=rows,aggregates=dict(samples=n,**agg),target_shadow_metrics=actor_metrics,checks=checks,formal_updates=2000,smoke_updates=64,replay_updates=4,frames_decoded=49,
        interpretation='Single actor changes road domain, its contributors and protection boundary. Source B, road representation, colors, optimizer, seed, camera sampling, loss coefficients and 1000-step budget are unchanged.',
        deletion_set=[5,26,18,6,37],independent_test=False,human_verdict=None,GPU_shutdown=False)
    (dest/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(aggregates=result['aggregates'],shadow=actor_metrics)),flush=True)
if __name__=='__main__':main()
