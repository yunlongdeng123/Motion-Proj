"""可配置视图的背景联合修复基线；继承保背景原型目标，隔离主副本。"""
import argparse,json,random,signal,sys,time
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp')
from editor_common import *
from scipy.ndimage import distance_transform_edt


def main():
    import resource_control
    resource_control.require_start(repair_training=True)
    requested_stop=[]
    for sig in [signal.SIGTERM,signal.SIGINT]:
        signal.signal(sig,lambda signum,frame:requested_stop.append(signum))
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--min-seconds',type=float,default=0)
    p.add_argument('--resume');a=p.parse_args();config=read(a.config);dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    n=Native();geometry_weight=config.get('surface_moment_weight',0.)
    if geometry_weight:
        import surface_moments
        check=surface_moments.install(n)
        (dest/'surface_moment_cuda_check.json').write_text(json.dumps(check,indent=2))
    road=Road(Path(config['initialization']));views=config['train_views']
    data=[dict(np.load(Path(config['reference_root'])/f'{raw}.npz')) for raw in views];cams=[n.camera(raw) for raw in views]
    selected=np.zeros(len(n.static.get_xyz),bool);scale=config['meter_to_model'];sigma=cpu(n.static.get_scaling).max(1)
    with torch.no_grad():
        for camera,d in zip(cams,data):
            xyz=prepare_rasterization(camera,n.static,n.args)['means3D'];world=torch.cat((xyz,torch.ones_like(xyz[:,:1])),1)
            z=cpu(world@camera.world_view_transform)[:,2];clip=cpu(world@camera.full_proj_transform)
            uv=(clip[:,:2]/np.maximum(clip[:,3:],1e-9)+1)*np.array([camera.image_width,camera.image_height])/2-.5
            ij=np.clip(np.round(uv).astype(int),[0,0],[camera.image_width-1,camera.image_height-1])
            distance=distance_transform_edt(~d['target'])[ij[:,1],ij[:,0]]
            radius=3*sigma*max(camera.fx,camera.fy)/np.maximum(z,.001)
            depths=d['depth'][d['target']&np.isfinite(d['depth'])]
            if not len(depths):continue
            far=float(np.percentile(depths,95))*scale
            selected|=(distance<=radius)&(z>.1*scale)&(z<far)&(uv[:,0]+radius>=0)&(uv[:,0]-radius<camera.image_width)&(uv[:,1]+radius>=0)&(uv[:,1]-radius<camera.image_height)
    ids=torch.tensor(np.flatnonzero(selected),device='cuda');initial=config['initial_static_multiplier']
    gate=torch.nn.Parameter(torch.full((len(ids),),float(np.log(initial/(1-initial))),device='cuda'))
    optimizer=torch.optim.Adam([{'params':[road.dc],'lr':.0025},{'params':[road.opacity],'lr':.05},{'params':[gate],'lr':.05}],eps=1e-15)
    sampler=torch.Generator();sampler.manual_seed(config['sampler_seed']);start_step=0
    def state(step):
        return dict(step=step,config=config,dc=road.dc.detach().clone(),opacity=road.opacity.detach().clone(),gate=gate.detach().clone(),ids=ids,
            optimizer=optimizer.state_dict(),sampler=sampler.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),master=n.before)
    if a.resume:
        ck=torch.load(a.resume,weights_only=False);assert ck['config']==config and ck['master']==n.before and torch.equal(ids,ck['ids'])
        with torch.no_grad():road.dc.copy_(ck['dc']);road.opacity.copy_(ck['opacity']);gate.copy_(ck['gate'])
        optimizer.load_state_dict(ck['optimizer']);sampler.set_state(ck['sampler']);torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng']);start_step=ck['step']
        checks=dict(parameters=all(torch.equal(x,y) for x,y in [(road.dc,ck['dc']),(road.opacity,ck['opacity']),(gate,ck['gate'])]),
            sampler=torch.equal(sampler.get_state(),ck['sampler']),torch_cpu_rng=torch.equal(torch.get_rng_state(),ck['torch_rng']),
            cuda_rng=all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),ck['cuda_rng'])),
            numpy_rng=np.array_equal(np.random.get_state()[1],ck['numpy_rng'][1]),python_rng=random.getstate()==ck['python_rng'],
            optimizer=all(torch.equal(value,ck['optimizer']['state'][key][field]) if torch.is_tensor(value) else value==ck['optimizer']['state'][key][field]
                for key,values in optimizer.state_dict()['state'].items() for field,value in values.items()))
        assert all(checks.values());(dest/f'restore_{start_step}.json').write_text(json.dumps(dict(checks=checks,loaded_state_exact=True,continued_cuda_trajectory_bitwise_claim=False),indent=2))
    with torch.no_grad():
        reference=[n.render(camera)['rgb'].detach() for camera in cams]
        master=n.render(cams[0],dynamic=True)['rgb'].clone()
    dense=[]
    for raw in views:
        src=min([102,130,137],key=lambda x:abs(x-raw));dense.append((src,dict(np.load(OUT/'dense_training'/f'{src:03d}.npz'))))
    (dest/'contract.json').write_text(json.dumps(dict(config=config,requested_final_step=a.steps,min_active_training_seconds=a.min_seconds,
        selected_static_rows=len(ids),baseline='保背景联合修复，原损失与学习率继承；改变训练视图用于同片段转移基线',
        evaluation_role='同片段开发一致性，不作为跨场景泛化',geometry_fixed=True,master_frozen=True),indent=2))
    def evaluate(step):
        rows=[]
        with torch.no_grad():
            mult=torch.ones(len(selected),device='cuda').index_copy(0,ids,gate.sigmoid())
            output=dest/f'evaluation_{step}';output.mkdir(exist_ok=True)
            for raw,camera,d,ref in zip(views,cams,data,reference):
                result=n.render(camera,road,static_multiplier=mult);mask=torch.tensor(d['valid'],device='cuda')
                target=torch.tensor(d['rgb'].transpose(2,0,1),device='cuda');protect=~torch.tensor(d['target'],device='cuda')
                save(output/f'{raw:03d}.png',result['rgb'])
                rows.append(dict(raw=raw,labeled_pixels=int(mask.sum()),full_target=int(d['target'].sum()),
                    MAE=float((result['rgb']-target).abs()[:,mask].mean()),outside_change=float((result['rgb']-ref).abs()[:,protect].mean())))
            assert float((n.render(cams[0],dynamic=True)['rgb']-master).abs().max())==0
        (output/'summary.json').write_text(json.dumps(rows,indent=2))
    evaluate(start_step);start=time.time();step=start_step;torch.cuda.reset_peak_memory_stats()
    with (dest/f'loss_from{start_step}.jsonl').open('x') as log:
        while step<a.steps or time.time()-start<a.min_seconds:
            if requested_stop or resource_control.stop_requested():
                break
            step+=1;i=int(torch.randint(len(views),(1,),generator=sampler));camera=cams[i];d=data[i]
            mask=torch.tensor(d['valid'],device='cuda');protect=~torch.tensor(d['target'],device='cuda');target=torch.tensor(d['rgb'].transpose(2,0,1),device='cuda')
            optimizer.zero_grad(set_to_none=True);mult=torch.ones(len(selected),device='cuda').index_copy(0,ids,gate.sigmoid())
            result=n.render(camera,road,static_multiplier=mult)
            deleted=(result['rgb']-target).abs()[:,mask].mean();preserve=(result['rgb']-reference[i]).abs()[:,protect].mean()
            visible=(.25-result['contribution'][0,mask]).clamp_min(0).mean()
            src,obs=dense[i];visible_mask=torch.tensor(obs['visible'],device='cuda');gt=torch.tensor(obs['rgb'].transpose(2,0,1),device='cuda')
            direct=render_road(n.camera(src),road,n.args,for_training=True);road_loss=(direct['rgb']-gt).abs()[:,visible_mask].mean()
            loss=deleted+preserve+.1*visible+.1*road_loss
            geometry=loss.new_zeros(())
            if geometry_weight:
                geometry=surface_moments.loss(result,torch.tensor(d['depth'],device='cuda'),mask,scale,config['surface_tolerance_m'])
                loss=loss+geometry_weight*geometry
            assert torch.isfinite(loss);loss.backward()
            assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in [gate,road.dc,road.opacity])
            optimizer.step()
            record=dict(step=step,raw=views[i],loss=float(loss),deleted=float(deleted),protect=float(preserve),road=float(road_loss),geometry=float(geometry),
                seconds=time.time()-start,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
            log.write(json.dumps(record)+'\n');log.flush()
            if step%100==0:print(json.dumps(record),flush=True)
            if step%1000==0 or step in [100,101]:
                torch.save(state(step),str(dest/f'checkpoint_{step}.pth'));evaluate(step)
            del loss,result,direct,mult
    torch.save(state(step),str(dest/'checkpoint_final.pth'));evaluate(step);assert n.unchanged()
    receipt=dict(completed=not(requested_stop or resource_control.stop_requested()),resource_stop=bool(requested_stop or resource_control.stop_requested()),start_step=start_step,final_step=step,updates=step-start_step,active_training_seconds=time.time()-start,
        min_seconds_requested=a.min_seconds,master_unchanged=True,quality_accepted=False,human_verdict=None)
    (dest/f'receipt_from{start_step}.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    from gpu_slot import exclusive
    with exclusive():main()
