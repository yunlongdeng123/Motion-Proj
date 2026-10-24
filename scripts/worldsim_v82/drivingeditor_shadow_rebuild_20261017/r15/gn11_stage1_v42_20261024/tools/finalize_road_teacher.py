import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from road_teacher_cpu import Teacher,measured_maps
from road_teacher_bridge import cpu_splat

def save(im,p):Image.fromarray(np.uint8(np.clip(im,0,1)*255+.5)).save(p)

def run(a):
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'frames').mkdir();(a.output/'generative_inputs').mkdir()
    summary=json.loads((a.results/'summary.json').read_text());v=json.loads((a.results/'validation.json').read_text())
    atlas=dict(np.load(a.results/'observed_atlas.npz'));cloud=dict(np.load(a.results/'road_birth_proposals.npz'))
    ids=json.loads((a.inputs/'manifest.json').read_text())['frame_order'];checks=[]
    for fid in ids:
        f=dict(np.load(a.inputs/f'{fid}.npz'));p=dict(np.load(a.results/'frames'/f'{fid}.npz'))
        # Deliberately label this composited RGB preview as a pseudo-target, not a render.
        target=f['rgb'].copy();m=p['source_kind']>0;target[m]=p['pseudo_rgb'][m]
        weight=f['rgb']*.3;weight[m]=plt.get_cmap('viridis')(p['loss_weight'][m])[...,:3]
        save(f['rgb'],a.output/'frames'/f'{fid}_observed.webp')
        save(target,a.output/'frames'/f'{fid}_pseudo.webp');save(weight,a.output/'frames'/f'{fid}_weight.webp')
        if fid in ['0008102','0008129','0008137']:
            rend=cpu_splat(cloud,f);np.savez_compressed(a.results/f'{fid}_cpu_splat.npz',**rend)
            save(rend['rgb'],a.output/f'{fid}_road_splat.png')
            for x,y in ([(255,240),(256,219)] if fid=='0008102' else []):
                checks.append(dict(frame=fid,pixel=[x,y],alpha=float(rend['alpha'][y,x]),
                    rgb=rend['rgb'][y,x].tolist(),depth=float(rend['depth'][y,x]),native_cuda=False))
            # Portable model inputs, not a claim that an external model has been executed.
            folder=a.output/'generative_inputs'/fid;folder.mkdir();mask=f['actor_labels']==5
            save(f['rgb'],folder/'observed_rgb.png');save(mask.astype(float),folder/'inpaint_mask.png')
            masked=f['rgb'].copy();masked[mask]=.5;save(masked,folder/'masked_rgb.png')
            fmap=measured_maps(f,atlas['ground_plane']);known=fmap['depth'].copy();known[mask]=0
            np.save(folder/'known_ground_depth_m.npy',known)
            np.save(folder/'road_prior_depth_m.npy',p['pseudo_depth'])
            np.savez_compressed(folder/'camera.npz',K=f['K'],local_to_camera=f['w2l'],camera_to_local=np.linalg.inv(f['w2l']))
            (folder/'contract.json').write_text(json.dumps(dict(frame=fid,pixel_center='+0.5',
                known_depth='same-frame ground LiDAR triangles; zeros unknown, NOT static polluted render',
                road_prior='inferred, not a measurement',mask='a37 contour only; white=inpaint',
                next_step='RGB inpainting first; then metric-anchored depth completion; fuse in one shared road coordinate system before Gaussian insertion',
                external_model_executed=False,do_not_use='original vehicle RGB inside mask as road labels'),indent=2))
    summary['cpu_splat_checks']=checks
    # Show the evidence atlas and predictions in road coordinates, not only image boundaries.
    xx,yy=np.meshgrid(np.arange(5,35,.1),np.arange(-4,8,.1));q=np.c_[xx.ravel(),yy.ravel()]
    pred=Teacher(atlas).predict(q)
    fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    ok=(atlas['xy'][:,0]>=5)&(atlas['xy'][:,0]<35)
    axes[0].scatter(atlas['xy'][ok,0],atlas['xy'][ok,1],c=atlas['rgb'][ok],s=4,marker='s')
    axes[1].imshow(pred['rgb'].reshape(*xx.shape,3),origin='lower',extent=[5,35,-4,8],aspect='auto')
    for ax,title in zip(axes,['Measured visible-road donor cells','Gaussian road prior (inferred appearance)']):
        ax.set_title(title);ax.set_xlabel('Road longitudinal x (m)');ax.set_ylabel('Lateral y (m)');ax.set_xlim(5,35);ax.set_ylim(-4,8)
        for c in summary['targets']['critical']:
            ax.scatter(c['xyz'][0],c['xyz'][1],marker='+',color='red',s=80)
    fig.savefig(a.output/'road_atlas.png',dpi=150);plt.close(fig)
    # One held-out interior example. All observations inside this rectangle were removed.
    cx,cy=20,2;xy=atlas['xy'];hole=(abs(xy[:,0]-cx)<4)&(abs(xy[:,1]-cy)<1);keep=~hole
    fig,axes=plt.subplots(1,4,figsize=(15,4),constrained_layout=True)
    ids2=np.flatnonzero(hole);methods=['reference','nearest','isotropic','road_gaussian'];teacher=Teacher(atlas,keep)
    for ax,method in zip(axes,methods):
        col=atlas['rgb'][ids2] if method=='reference' else teacher.predict(xy[ids2],method)['rgb']
        ax.scatter(xy[ids2,0],xy[ids2,1],c=col,s=12,marker='s');ax.set_title(method)
        ax.set_xlim(16,24);ax.set_ylim(1,3);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
        ax.add_patch(plt.Rectangle((17,1.4),6,1.2,fill=False,color='red'))
    fig.savefig(a.output/'withdrawn_hole.png',dpi=150);plt.close(fig)
    # Aggregate geometry error weighted by number of actually scored core cells.
    summary['geometry_validation']={}
    for split in ['development','validation']:
        rows=[r for r in v['holes'] if r['split']==split]
        summary['geometry_validation'][split]={method:sum(r['methods'][method]['core_depth_mae']*r['core_count'] for r in rows)/sum(r['core_count'] for r in rows) for method in methods[1:]}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));(a.output/'validation.json').write_text(json.dumps(v,indent=2))
    print(json.dumps(dict(critical=summary['targets']['critical'],splat=checks,geometry=summary['geometry_validation']),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);run(ap.parse_args())
