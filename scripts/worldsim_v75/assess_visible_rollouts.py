"""在冻结点身份上比较生成轨迹，并应用已过真实基线的三角测量规则。"""
import json
import time
import numpy as np
import torch
from PIL import Image,ImageDraw
from prepare_visible_sources import OUT
from audit_natural_observation import advance,RAFTFlow
from triangulate_visible_state import geometry,fit_points,FRAMES
from prepare_argoverse import project_track

ARMS=['gt_clean','dvgt_metric','ordinary_bbox','reference_lidar']

def main():
    queue=json.loads((OUT/'rollout_queue_result.json').read_text());assert queue['status']=='complete'
    path=OUT/'assessment_result.json';assert not path.exists()
    selected=json.loads((OUT/'observation_selection.json').read_text());raft=RAFTFlow();torch.set_num_threads(4);start=time.monotonic();cases=[]
    for log in queue['completed']:
        case=OUT/'cases'/log;roll=case/'rollouts';folder=case/'assessment';folder.mkdir()
        selection=next(r for r in selected['logs'] if r['log_id']==log);target=selection['selected_target']
        observation=next(r for r in selection['candidate_measurements'] if r['target']==target)
        points=np.array(observation['initial_points'],np.float32);real=np.load(OUT/'real'/log/target/'tracks.npz')
        initial=np.array(Image.open(OUT/'real'/log/'reference-000.png'));K,cameras,motion=geometry(case/'base',target)
        metric_real=json.loads((OUT/'triangulation'/log/'real_result.json').read_text())
        fit_real=np.load(OUT/'triangulation'/log/'real_fit.npz')
        tracked={};fitted={};fit_results={};scenes={};bounds={}
        trajectory=np.load(case/'base/trajectory.npz')
        for arm in ARMS:
            dest=folder/arm;dest.mkdir();array=np.load(roll/arm/'clean.npy',mmap_mode='r')
            q,valid,error=advance(raft,initial,array[0],points,np.ones(len(points),bool),dest/'initial-alignment.npz')
            positions=[q.copy()];masks=[valid.copy()]
            for a,b in zip(FRAMES[:-1],FRAMES[1:]):
                q,valid,error=advance(raft,array[a],array[b],q,valid,dest/f'step-{b:03d}.npz')
                positions.append(q.copy());masks.append(valid.copy())
            pos=np.array(positions);mask=np.array(masks);np.savez(dest/'tracks.npz',points=pos,valid=mask)
            tracked[arm]={'points':pos,'valid':mask}
            fit,xyz,accepted=fit_points(pos,mask,K,cameras);fit_results[arm]=fit
            fitted[arm]={'points':xyz,'accepted':accepted}
            np.savez(dest/'fit.npz',points_actor=xyz,accepted=accepted)
            (dest/'fit_result.json').write_text(json.dumps(fit,indent=2)+'\n')
            t=next(t for t in json.loads((roll/arm/'scene.json').read_text())['tracks'] if t['id']==target);scenes[arm]=t
            bounds[arm]=[project_track(t,f,trajectory['camera_world'],trajectory['K'])['bounds'] for f in FRAMES]
            print(json.dumps({'log_id':log,'arm':arm,'tracked':[int(x.sum()) for x in masks],'rigid_fit_accepted':int(accepted.sum())}),flush=True)
        common=real['valid'].copy()
        for tr in tracked.values():common &= tr['valid']
        rows=[]
        for i,f in enumerate(FRAMES):
            keep=common[i];count=int(keep.sum());ok=count>=8 and count/len(points)>=.25;values={}
            for arm in ARMS:
                vector=np.median(tracked[arm]['points'][i,keep]-real['points'][i,keep],axis=0) if ok else None
                paired=np.median(tracked[arm]['points'][i,keep]-tracked['gt_clean']['points'][i,keep],axis=0) if ok else None
                b=np.array(bounds[arm][i]);c=np.array(bounds['gt_clean'][i]);shift=(b[:2]+b[2:]-c[:2]-c[2:])/2
                values[arm]={'to_real_vector_px':None if vector is None else vector.tolist(),'to_real_px':None if vector is None else float(np.linalg.norm(vector)),
                             'paired_clean_vector_px':None if paired is None else paired.tolist(),'projection_shift_px':shift.tolist()}
            rows.append({'frame':f,'common_points':count,'admitted':ok,'arms':values})
        rigid=fit_real['accepted'].copy()
        for fit in fitted.values():rigid &= fit['accepted']
        nr=int(rigid.sum());admitted=metric_real['real_metric_admitted'] and nr>=8 and nr/len(points)>=.25
        metrics={}
        for arm,fit in fitted.items():
            delta=(fit['points'][rigid]-fit_real['points_actor'][rigid])@cameras[0,:3,:3]
            v=np.median(delta,axis=0) if admitted else None
            metrics[arm]={'median_camera_xyz_delta_m':None if v is None else v.tolist(),
                          'median_point_distance_to_real_m':float(np.median(np.linalg.norm(delta,axis=1))) if admitted else None,
                          'p90_point_distance_to_real_m':float(np.percentile(np.linalg.norm(delta,axis=1),90)) if admitted else None,
                          'rigid_fit_accepted_points':fit_results[arm]['accepted_count'],
                          'initial_points':len(points)}
        np.savez(folder/'common_geometry.npz',common=rigid,real_actor=fit_real['points_actor'],cameras=cameras,
                 **{arm:fit['points'] for arm,fit in fitted.items()})
        review=json.loads((roll/'review_result.json').read_text())
        summary={'log_id':log,'target':target,'initial_points':len(points),'frames':rows,
                 'flow_full_window_admitted':all(r['admitted'] for r in rows),
                 'flow_means_px':{a:float(np.mean([r['arms'][a]['to_real_px'] for r in rows])) if all(r['admitted'] for r in rows) else None for a in ARMS},
                 'detector_means_px':{r['variant']:r['mean_distance_to_real_px'] if r['valid_matches']==r['scheduled'] else None for r in review['variants']},
                 'rigid_geometry_common_points':nr,'metric_comparison_admitted':bool(admitted),'metric_point_results':metrics,
                 'real_metric_reference':{k:v for k,v in metric_real.items() if k not in ['points','association_pixel_distances']},
                 'boundary':'conditional rigid surface-point diagnostic using known camera and shared GT actor motion; not a measured generated vehicle box or policy effect',
                 'human_verdict':None}
        (folder/'result.json').write_text(json.dumps(summary,indent=2)+'\n');cases.append(summary)
        # 同一真实ROI显示所有帧，不让裁剪跟随预测平移。
        labels=['Real RGB',*ARMS];sheet=Image.new('RGB',(1500,5*214+28),'#101b2b');d=ImageDraw.Draw(sheet)
        for j,label in enumerate(labels):d.text((j*300+5,7),label,fill='white')
        arrays={a:np.load(roll/a/'clean.npy',mmap_mode='r') for a in ARMS}
        for i,f in enumerate(FRAMES):
            cx,cy=observation['detections'][i]['center'];crop=[round(cx-180),round(cy-108),round(cx+180),round(cy+108)];y=28+i*214
            d.text((5,y+3),f'{f/30:.1f}s | common {int(common[i].sum())}/{len(points)}',fill='white')
            images=[Image.open(OUT/'real'/log/f'reference-{f:03d}.png')]+[Image.fromarray(arrays[a][f]) for a in ARMS]
            for j,im in enumerate(images):
                im=im.copy();draw=ImageDraw.Draw(im);tr=real if j==0 else tracked[ARMS[j-1]]
                for k,(x,yj) in enumerate(tr['points'][i]):
                    if tr['valid'][i,k]:draw.ellipse((x-2,yj-2,x+2,yj+2),fill='#00e5b5' if common[i,k] else '#ffe081')
                sheet.paste(im.crop(crop).resize((300,180)),(j*300,y+23))
        sheet.save(folder/'tracking-comparison.jpg',quality=95)
        print(json.dumps({k:v for k,v in summary.items() if k in ['log_id','flow_full_window_admitted','flow_means_px','detector_means_px','rigid_geometry_common_points','metric_comparison_admitted','metric_point_results']}),flush=True)
    result={'status':'complete','cases':cases,'seed':42,'world_model_calls_added_by_assessment':0,'human_verdict':None,'failure_ledger_delta':'none',
            'wall_s':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
    path.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
