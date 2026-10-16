"""Full 49-frame actual GPU baseline and edit diagnostic. Parent read-only."""
import os,sys,json,time,copy,math,traceback,importlib.util
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
BASE=ROOT/'shadow_BC_20261016_v1'
NEW=ROOT/'shadow_sequence_20261016_v2'
sys.path.insert(0,str(BASE/'implementation_deferred_v1'))
import numpy as np
import torch
from PIL import Image,ImageDraw
from types import SimpleNamespace
from run_deferred_refinement import DeferredExperiment
from run_experiments import write,save_image,now
from bc_model import srgb_to_linear,sample_render
import renderer

def encode(directory,fps=10):
    import cv2
    files=sorted(directory.glob('*.png'))
    if not files:return None
    shape=np.asarray(Image.open(files[0])).shape
    dest=directory.with_suffix('.mp4')
    writer=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'mp4v'),fps,(shape[1],shape[0]))
    if not writer.isOpened():return None
    for f in files:writer.write(cv2.cvtColor(np.asarray(Image.open(f)),cv2.COLOR_RGB2BGR))
    writer.release()
    cap=cv2.VideoCapture(str(dest));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ok,frame=cap.read();cap.release()
    assert n==len(files) and ok,(dest,n,len(files))
    return {'file':str(dest),'frames':n,'fps':fps,'width':shape[1],'height':shape[0]}

def run():
    out=NEW/'diagnostic49';args=SimpleNamespace(output=str(out),steps=0,warmup=1)
    e=DeferredExperiment(args)
    try:
        warm=torch.load(BASE/'runs/deferred_1000_v1/R4/addon_001000.pt',map_location='cpu',weights_only=False)
        e.warm=warm;e.activate('R4',e.shadow());e.active.load_addon_state(warm['road_addon'])
        n=e.data['plane_model'][:3].float();n=n/n.norm();seed=torch.eye(3,device='cuda')[int(n.abs().argmin())]
        tangent=torch.linalg.cross(n,seed);tangent=tangent/tangent.norm()
        names=['original_observed','original_static','retained_observed','retained_static','delete_target','delete_all','translate_smooth','yaw_smooth','fixed_original_static','fixed_retained_static']
        for name in names:(out/name).mkdir()
        rows=[];anchor=None;start=time.time()
        with torch.no_grad():
            for i in range(49):
                e.priority();fid=e.frame_ids[i];t=i/48
                edits_move={0:{'translation':(tangent*(math.sin(2*math.pi*t))*e.data['world_scale']).tolist()}}
                theta=math.radians(15)*math.sin(2*math.pi*t)
                edits_yaw={0:{'quaternion':[math.cos(theta/2),*(n*math.sin(theta/2)).tolist()]}}
                renderer.prepare_static=e.original_prepare;saved=e.active;e.active=None
                original=e.render(i);static0=e.render(i,instances=False)
                camera=copy.copy(e.cameras[12]);camera.timestamp=e.cameras[i].timestamp
                fixed0=e.render(i,instances=False,camera=camera)
                e.active=saved;renderer.prepare_static=e.active.patch
                seen=e.render(i,shaded=True);clean=e.render(i,instances=False)
                deleted=e.render(i,shaded=True,edits={0:{'delete':True}})
                allgone=e.render(i,shaded=True,edits={k:{'delete':True} for k in range(e.instances.num_instances)})
                move=e.render(i,shaded=True,edits=edits_move);yaw=e.render(i,shaded=True,edits=edits_yaw)
                fixed=e.render(i,instances=False,camera=camera)
                packs=[original,static0,seen,clean,deleted,allgone,move,yaw,fixed0,fixed]
                for name,pkg in zip(names,packs):save_image(out/name/(fid+'.png'),pkg['render'])
                if i==0:anchor=fixed['render'].clone()
                pred=srgb_to_linear(sample_render(clean['render'],e.data['uv'][i]));obs=srgb_to_linear(sample_render(seen['render'],e.data['uv'][i]))
                accepted=e.data['accepted_reference'];sm=e.data['shadow'][i];cm=e.data['clean'][i]
                control=torch.ones((320,480),device='cuda',dtype=torch.bool);control[205:305,20:205]=False
                diff=abs(seen['render']-original['render']).mean(0)
                row={'frame':fid,'clean_road_MAE':float(abs(pred[accepted]-e.data['reference_linear'][accepted]).mean()),
                    'shadow_MAE':float(abs(obs[sm]-e.data['observed_linear'][i,sm]).mean()) if sm.any() else None,
                    'clean_observed_MAE':float(abs(obs[cm]-e.data['observed_linear'][i,cm]).mean()) if cm.any() else None,
                    'shadow_pseudo_pixels':int(sm.sum()),'control_MAE':float(diff[control].mean()),
                    'control_pixels_change_gt_05':int((diff[control]>.05).sum()),
                    'delete_all_identity':float(abs(allgone['render']-clean['render']).max()),
                    'fixed_camera_full_MAE_vs_first':float(abs(fixed['render']-anchor).mean())}
                rows.append(row);write(out/'per_frame.json',rows)
                e.update(state='running',phase='49_frame_diagnostic',frame=i+1,frames=49,elapsed_seconds=time.time()-start)
                print(json.dumps(row),flush=True)
                if i in [0,6,12,18,24,36,48]:
                    sheet=Image.new('RGB',(480*5,344*2),'white');d=ImageDraw.Draw(sheet)
                    for k,(name,pkg) in enumerate(zip(names,packs)):
                        im=Image.fromarray((pkg['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8'))
                        x=k%5*480;y=k//5*344;sheet.paste(im,(x,y+24));d.text((x+5,y+4),fid+' '+name,fill='black')
                    sheet.save(out/('review_'+fid+'.jpg'))
        video=[encode(out/name) for name in names]
        e.verify_sources();write(out/'videos.json',video)
        summary={'frames':49,'renders':490,'elapsed_seconds':time.time()-start,'max_delete_identity':max(r['delete_all_identity'] for r in rows),
            'mean_clean_road_MAE':float(np.mean([r['clean_road_MAE'] for r in rows])),
            'mean_shadow_MAE':float(np.mean([r['shadow_MAE'] for r in rows if r['shadow_MAE'] is not None])),
            'max_control_MAE':max(r['control_MAE'] for r in rows),
            'scope':'full49 images, same evidenced local world road patch; no full-road claim','video':video}
        write(out/'summary.json',summary);e.update(state='complete',phase='49_frame_diagnostic_complete',summary=summary)
    finally:renderer.prepare_static=e.original_prepare;e.slot.close()

if __name__=='__main__':run()
