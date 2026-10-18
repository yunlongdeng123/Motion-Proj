"""Re-export isolated branch with the exact original neg_fov camera contract."""
from run_fragment_audit_20261018 import *

def main():
    slot,admission=require_gpu_slot('/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock')
    torch.set_num_threads(2)
    audit=json.loads((OUT/'audit.json').read_text());spec=audit['old_spec']
    saved=torch.load(CK,map_location='cuda',weights_only=False)
    cache=OUT/'camera_contract_verification';cache.mkdir()
    args,scene,static,instances,opt,env,exclusion=load_models(spec,cache,restore=saved)
    ids=list(instances.instance_ids);current_ids=[x['track_id'] for x in audit['mapping']]
    old_records=[];pairs=[]
    with torch.no_grad():
        for i,camera in enumerate(scene.getTrainCameras()):
            fid=Path(camera.image_name).stem;raw=int(fid[-3:]);folder=OUT/'old'/fid
            masks,known=masks_for_frame(spec,instances,camera)
            for k,track in enumerate(ids):
                actor=audit['mapping'][k]['r15_actor'];rgb,alpha=old_isolated(camera,instances,args,k)
                png(folder/f'a{actor:02d}_rgb.png',rgb);png(folder/f'a{actor:02d}_alpha.png',alpha)
                body=masks[k]>0;outside=~body
                old_records.append({'raw':raw,'actor':actor,'body_pixels':int(body.sum()),
                    'alpha_mass_exterior':float(alpha[outside].sum()),
                    'exterior_alpha_pixels_gt_005':int(((alpha>.05)&outside).sum()),
                    'inside_alpha_mean':float(alpha[body].mean()) if body.any() else None})
                # Exact RGB/alpha parity against normal production renderer with all
                # other actor/static masks removed. No new asset ownership inferred.
                if i==16:
                    class Proxy:
                        instance_ids=instances.instance_ids
                        def evaluate(self,t,edits=None):
                            w=instances.evaluate(t,edits=edits);m=w.point_ids==k
                            return type(w)(**{key:(value[m] if torch.is_tensor(value) else value)
                                for key,value in vars(w).items()})
                    p=render(camera,static,Proxy(),args,torch.zeros(3,device='cuda'),
                        mask=torch.zeros(len(static.get_xyz),device='cuda',dtype=torch.bool),
                        env_map=None,instance_ids=[],render_separate=False)
                    delta=float((p['render']-rgb).abs().max());ad=float((p['alpha'][0]-alpha).abs().max())
                    assert delta==0 and ad==0,(delta,ad)
                    pairs.append({'raw':raw,'actor':actor,'max_rgb_error':delta,'max_alpha_error':ad})
            old_original=ROOT/'shadow_sequence_20261016_v2/diagnostic49/original_observed'/(fid+'.png')
            p=np.asarray(Image.open(old_original).convert('RGB'))
            q=np.asarray(Image.open(folder/'full.png').convert('RGB'))
            if i==0:full_deltas=[]
            full_deltas.append(int(np.abs(p.astype(int)-q.astype(int)).max()))
            if i%14==0:print(json.dumps({'correct_camera_isolated_frames':i+1}),flush=True)
    audit['old_records']=old_records
    audit['camera_contract']={'neg_fov':bool(args.neg_fov),'exact_production_renderer_parity':pairs,
        'original_baseline_full_png_max_byte_error_all49':max(full_deltas),
        'original_baseline_full_png_bitwise_equal_frames':sum(x==0 for x in full_deltas),
        'provisional_positive_tangent_isolated_exports_replaced_before_review':True}
    assert max(full_deltas)==0
    assert audit['old_checkpoint']==snap(CK)
    write(OUT/'audit.json',audit)
    print(json.dumps(audit['camera_contract']),flush=True);slot.close()

if __name__=='__main__':main()
