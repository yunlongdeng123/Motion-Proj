"""Fixed-domain quality audit: shadow, markings, clean asphalt and seams."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,math,subprocess,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation,binary_erosion
import imageio_ffmpeg
RUN=Path(__file__).resolve().parent.parent
OLD=RUN.with_name('shadow_a5_a37_199_20261103_r4')
VERSION=RUN.name.rsplit('_',1)[-1].upper()

def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def readimage(p):return np.asarray(Image.open(p),np.float32)/255.
def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def main():
    assert (RUN/'queue_complete.json').exists()
    dest=RUN/'quality';dest.mkdir(exist_ok=False);frames=RUN/'evaluation/all49';rows=[]
    for raw in range(89,138):
        d=np.load(RUN/'inputs/views'/('%03d.npz'%raw));base=readimage(frames/('%03d_baseline_delete5.png'%raw))
        old=readimage(OLD/'evaluation/all49'/('%03d_repaired_delete5.png'%raw));new=readimage(frames/('%03d_repaired_delete5.png'%raw))
        # Masks are fixed by B, old evidence and old repair domain, never by the
        # after image or a new metric qualification mask.
        ref=d['reference_linear'];b=linear(base);y=b@np.array([.2126,.7152,.0722]);ry=ref@np.array([.2126,.7152,.0722])
        domain=d['repair_mask'];yellow=(base[:,:,0]>1.22*base[:,:,2])&(base[:,:,1]>1.08*base[:,:,2])&(base[:,:,0]>.32)&domain&(d['actor_labels']==0)
        dark=domain&(y<.78*np.maximum(ry,.01))&((ry-y)>.025)
        clean=domain&~binary_dilation(dark,iterations=5)&(d['actor_labels']==0)&~binary_dilation(yellow,iterations=2)&(np.abs(b-ref).max(2)<.035)
        left=np.indices(y.shape)[1]<220;leftclean=clean&left
        ring=binary_dilation(domain,iterations=3)&~binary_erosion(domain,iterations=3)&~binary_dilation(dark,iterations=5)&~d['other_vehicle_guard']
        newdomain=d['edit_weight']>.5
        leftseam=binary_dilation(newdomain,iterations=3)&~binary_erosion(newdomain,iterations=3)&left&~binary_dilation(dark,iterations=2)&~d['other_vehicle_guard']
        def measure(im):
            diff=np.abs(im-base);res=im-base;g=np.zeros(y.shape)
            g[:,1:]+=np.abs(np.diff(res,axis=1)).mean(2);g[1:]+=np.abs(np.diff(res,axis=0)).mean(2)
            return dict(yellow_MAE=float(diff[yellow].mean()) if yellow.any() else None,
                clean_road_MAE=float(diff[clean].mean()) if clean.any() else None,
                left_clean_MAE=float(diff[leftclean].mean()) if leftclean.any() else None,
                left_seam_residual_gradient=float(g[leftseam].mean()) if leftseam.any() else None,
                left_original_boundary_residual_gradient=float(g[ring&left].mean()) if (ring&left).any() else None,
                repair_boundary_residual_gradient=float(g[ring].mean()) if ring.any() else None)
        row=dict(raw=raw,yellow_samples=int(yellow.sum()),clean_samples=int(clean.sum()),left_samples=int(leftclean.sum()),left_seam_samples=int(leftseam.sum()),left_original_boundary_samples=int((ring&left).sum()),boundary_samples=int(ring.sum()),R4=measure(old));row[VERSION]=measure(new);rows.append(row)
    aggregates={}
    for key,count in [('yellow_MAE','yellow_samples'),('clean_road_MAE','clean_samples'),('left_clean_MAE','left_samples'),('left_seam_residual_gradient','left_seam_samples'),('left_original_boundary_residual_gradient','left_original_boundary_samples'),('repair_boundary_residual_gradient','boundary_samples')]:
        den=sum(x[count] for x in rows);aggregates[key]=dict(samples=den,**{version:sum(x[version][key]*x[count] for x in rows if x[version][key] is not None)/den for version in ['R4',VERSION]})
    metrics=json.loads((frames/'metrics.json').read_text());actors={}
    for actor in ['5','37']:
        actors[actor]={}
        for key,count in [('shadow_MAE','shadow_samples'),('road_MAE','road_samples')]:
            rec=[x['actors'][actor] for x in metrics['rows'] if x['actors'][actor][key] is not None];den=sum(x[count] for x in rec)
            actors[actor][key]=dict(samples=den,frames=len(rec),B=sum(x['baseline_'+key]*x[count] for x in rec)/den);actors[actor][key][VERSION]=sum(x[key]*x[count] for x in rec)/den
    log=[json.loads(x) for x in (RUN/'formal/training_from0.jsonl').read_text().splitlines()]
    assert [x['step'] for x in log]==list(range(1,1001)) and all(math.isfinite(x['loss']) for x in log)
    ck=torch.load(RUN/'formal/checkpoint001000.pt',map_location='cpu',weights_only=False)
    assert set(int(x['step']) for x in ck['optimizer']['state'].values())=={1000}
    exact={str(r):(frames/('%03d_repaired_delete5.png'%r)).read_bytes()==(RUN/'formal/step001000'/('%03d_repaired_delete5.png'%r)).read_bytes() for r in [101,105,113]}
    assert all(exact.values())
    for raw in range(89,138):
        ims=[Image.open(frames/('%03d_baseline_delete5.png'%raw)),Image.open(OLD/'evaluation/all49'/('%03d_repaired_delete5.png'%raw)),Image.open(frames/('%03d_repaired_delete5.png'%raw))]
        canvas=Image.new('RGB',(1440,320))
        for k,im in enumerate(ims):canvas.paste(im,(480*k,0))
        canvas.save(dest/('%03d_comparison.png'%raw))
    video=RUN/('B_R4_%s_all49.mp4'%VERSION)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-n','-framerate','10','-start_number','89','-i',str(dest/'%03d_comparison.png'),'-frames:v','49','-c:v','libx264','-crf','17','-pix_fmt','yuv420p',str(video)],check=True)
    reader=imageio_ffmpeg.read_frames(str(video));next(reader);decoded=sum(1 for _ in reader);assert decoded==49
    write(dest/'metrics.json',dict(rows=rows,aggregates=aggregates,actors=actors,frames_decoded=decoded,restored_PNG_exact=exact,
        formal_updates=1000,smoke_updates=32,replay_updates=2,parent_unchanged=metrics['parent_unchanged'],
        invariants=json.loads((RUN/'evaluation/invariants.json').read_text()),
        train_seconds=json.loads((RUN/'formal/receipt.json').read_text())['seconds'],CUDA_peak_allocated=max(x['cuda_peak_allocated'] for x in log),
        CUDA_peak_reserved=max(x['cuda_peak_reserved'] for x in log),quality_accepted=False,human_verdict=None,
        failure_ledger_refs=['V82-F01','V82-F06'],failure_ledger_delta='none',all_199_BUILD=True,independent_test=False,
        quality_reference='native B preservation masks, fixed for both methods; shadow uses unchanged original strong proxy'))
    print(json.dumps(aggregates),flush=True);print(json.dumps(actors),flush=True)

if __name__=='__main__':main()
