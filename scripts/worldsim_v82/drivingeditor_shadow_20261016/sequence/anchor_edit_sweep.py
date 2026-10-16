"""49 edit states at fixed raw101, distinct from49 actual-time observations."""
import argparse,math,json,time
import torch
import numpy as np
from train_surface import SurfaceExperiment,NEW
from bc_model import edited_casters
from rigid_model import quaternion_to_matrix
from run_experiments import write,save_image
from sequence_diagnostic import encode
import renderer
def run():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');a=parser.parse_args()
    if a.final:
        started=time.time()
        while time.time()-started<3600:
            path=NEW/'final49_unselected_v1_follow.json'
            if path.exists():
                state=json.loads(path.read_text())['state']
                if state=='complete':break
                if state in ['evaluation_failed','training_needs_review']:raise SystemExit('Preserve failed run for review')
            time.sleep(10)
        else:raise SystemExit('Final sweep waiting deadline')
    args=argparse.Namespace(output=str(NEW/('anchor_edit_sweep_final' if a.final else 'anchor_edit_sweep')),steps=0,warmup=1,restore=str(NEW/('unselected_clean_v1' if a.final else 'contribution_frozen_v1')/'final.pt'),probe=True)
    e=SurfaceExperiment(args)
    try:
        for name in ['translation','yaw30']:(e.out/name).mkdir()
        tangent=e.surface_data['basis'][:,0];normal=e.surface_data['plane'][:3];normal=normal/normal.norm()
        refs={};last={};centroids=[];pose_error=0.;q=e.surface_data['xyz']
        with torch.no_grad():
            for step in range(49):
                if step%7==0:e.priority()
                signal=0. if step in [0,48] else math.sin(2*math.pi*step/48)
                move={0:{'translation':(tangent*signal*e.data['world_scale']).tolist()}}
                angle=math.radians(30)*signal;yaw={0:{'quaternion':[math.cos(angle/2),*(normal*math.sin(angle/2)).tolist()]}}
                for name,edits in [('translation',move),('yaw30',yaw)]:
                    pkg=e.render(12,shaded=True,edits=edits);save_image(e.out/name/('%03d.png'%step),pkg['render'])
                    if step==0:refs[name]=pkg['render'].clone()
                    last[name]=pkg['render'].clone()
                    cast=edited_casters(e.instances,e.cameras[12].timestamp,edits,e.data['sizes_model'][12]);world=e.instances.evaluate(e.cameras[12].timestamp,edits)
                    j=int(torch.nonzero(world.point_ids==0)[0]);pose_error=max(pose_error,float(abs(cast[0][0]-world.object_translation[j]).max()),float(abs(cast[1][0]-quaternion_to_matrix(world.object_rotation[j])).max()))
                    if name=='translation':
                        valid=cast[3].clone();valid[1:]=False
                        weight=e.surface.shadow.mask(q,cast[0],cast[1],cast[2],valid)
                        center=(q*weight[:,None]).sum(0)/weight.sum().clamp_min(1e-8)/e.data['world_scale']
                        centroids.append({'step':step,'translation_m':signal,'shadow_centroid_m':center.tolist()})
                e.update(state='running',phase='fixed_raw101_edit_sweep',step=step+1,steps=49)
        closure={name:bool(torch.equal(refs[name],last[name])) for name in refs};assert all(closure.values())
        reference=np.array(centroids[0]['shadow_centroid_m']);errors=[]
        direction=tangent.cpu().numpy()
        for row in centroids:errors.append(float(np.linalg.norm(np.array(row['shadow_centroid_m'])-reference-row['translation_m']*direction)))
        assert pose_error<2e-6 and max(errors)<.08,(pose_error,max(errors))
        videos=[encode(e.out/name) for name in ['translation','yaw30']];e.verify_sources()
        result={'states':49,'fixed_real_frame':'0008101','not49_different_observation_times':True,'translation_amplitude_m':1,'yaw_amplitude_deg':30,
            'both_positive_and_negative_edits_with_target_visible':True,'body_caster_pose_max_abs':pose_error,'shadow_centroid_translation_error_max_m':max(errors),
            'closed_loop_RGB_bitwise':closure,'centroids':centroids,'videos':videos,'paired_edit_GT':False,'original_asset_yaw_defects_not_repaired':True}
        write(e.out/'summary.json',result);e.update(state='complete',phase='anchor_edit_sweep_complete',summary=result)
    finally:renderer.prepare_static=e.original_prepare;renderer.prepare_instances=e.original_instances_prepare;e.slot.close()
if __name__=='__main__':run()
