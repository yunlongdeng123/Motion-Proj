"""约十分钟编辑接线：冻结B，只在独立副本学习静态透射门和道路颜色/alpha。
真实跨帧像素标签优先；未标注车底不借本帧车身RGB伪造监督。
"""
import argparse,time,random,json
from scipy.ndimage import distance_transform_edt
from editor_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1000);p.add_argument('--arm',default='full199',choices=['full199','window49']);p.add_argument('--resume');p.add_argument('--name',default='edit_smoke');p.add_argument('--initial-gate',type=float,default=.001);a=p.parse_args()
    dest=OUT/a.name/a.arm;dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0);np.random.seed(0);random.seed(0);n=Native();road=Road(FULL/f'{a.arm}_initialization.npz')
    sources=[102,129,137];data=[dict(np.load(OUT/'crossview'/f'{raw}.npz')) for raw in sources];cams=[n.camera(raw) for raw in sources]
    # Fixed global row set based on projected finite extent, all three development views.
    # No claim this set is exclusively polluted: the outside-domain loss must reveal damage.
    selected=np.zeros(len(n.static.get_xyz),bool);m=read('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json');scale=m['meter_to_model'];sigma=cpu(n.static.get_scaling).max(1)
    with torch.no_grad():
        for c,d in zip(cams,data):
            xyz=prepare_rasterization(c,n.static,n.args)['means3D'];world=torch.cat((xyz,torch.ones_like(xyz[:,:1])),1);cc=cpu(world@c.world_view_transform)[:,:3];z=cc[:,2];fx=c.image_width/(2*np.tan(c.FoVx/2));fy=c.image_height/(2*np.tan(c.FoVy/2))
            # Actual Camera projection matrix preserves principal point and native viewport.
            clip=cpu(world@c.full_proj_transform);uv=(clip[:,:2]/np.maximum(clip[:,3:],1e-9)+1)*np.array([480,320])/2-.5
            ij=np.clip(np.round(uv).astype(int),[0,0],[479,319]);distance=distance_transform_edt(~d['target'])[ij[:,1],ij[:,0]];radius=3*sigma*max(fx,fy)/np.maximum(z,.001)
            target_depth=d['depth'][d['target']&np.isfinite(d['depth'])];far=float(np.percentile(target_depth,95))*scale
            selected|=(distance<=radius)&(z>.1*scale)&(z<far)&(uv[:,0]+radius>=0)&(uv[:,0]-radius<480)&(uv[:,1]+radius>=0)&(uv[:,1]-radius<320)
    ids=torch.tensor(np.flatnonzero(selected),device='cuda');gate=torch.nn.Parameter(torch.full((len(ids),),float(np.log(a.initial_gate/(1-a.initial_gate))),device='cuda'))
    opt=torch.optim.Adam([{'params':[road.dc],'lr':.0025},{'params':[road.opacity],'lr':.05},{'params':[gate],'lr':.05}],eps=1e-15)
    sampler=torch.Generator();sampler.manual_seed(20261026);start_step=0
    def state(step):return dict(step=step,arm=a.arm,dc=road.dc.detach().clone(),opacity=road.opacity.detach().clone(),gate=gate.detach().clone(),ids=ids,optimizer=opt.state_dict(),sampler=sampler.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),master=n.before)
    if a.resume:
        ck=torch.load(a.resume,weights_only=False);assert ck['master']==n.before and ck['arm']==a.arm and torch.equal(ids,ck['ids'])
        with torch.no_grad():road.dc.copy_(ck['dc']);road.opacity.copy_(ck['opacity']);gate.copy_(ck['gate'])
        opt.load_state_dict(ck['optimizer']);sampler.set_state(ck['sampler']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng']);start_step=ck['step']
        assert torch.equal(road.dc,ck['dc']) and torch.equal(road.opacity,ck['opacity']) and torch.equal(gate,ck['gate'])
        (dest/'restore_loaded_exact.json').write_text(json.dumps(dict(loaded_parameters_exact=True,sampler_exact=torch.equal(sampler.get_state(),ck['sampler']),step=start_step)))
    with torch.no_grad():reference=[n.render(c)['rgb'].detach() for c in cams];master=n.render(cams[0],dynamic=True)['rgb'].clone()
    (dest/'contract.json').write_text(json.dumps(dict(seed=0,selected_static_rows=len(ids),training_frames=sources,source_frames=[138,169],static_initial_multiplier=a.initial_gate,updates_budget=a.steps,loss='crossview L1 + static outside-target protect L1 + 0.1 actual-road contribution shortfall0.25 +0.1 dense visible isolated-road L1',geometry_fixed=True,master_frozen=True,experimental_selection_not_safe_semantic_filter=True),indent=2))
    start=time.time()
    def evaluate(step):
        with torch.no_grad():
            multiplier=torch.ones(len(selected),device='cuda').index_copy(0,ids,gate.sigmoid());results=[]
            for i,(raw,c,d) in enumerate(zip(sources,cams,data)):
                out=n.render(c,road,static_multiplier=multiplier);save(dest/f'{step}_{raw}_edit.png',out['rgb']);target=torch.tensor(d['rgb'].transpose(2,0,1),device='cuda');mask=torch.tensor(d['valid'],device='cuda');protect=~torch.tensor(d['target'],device='cuda')
                results.append(dict(raw=raw,valid=int(mask.sum()),target=int(d['target'].sum()),MAE=float((out['rgb']-target).abs()[:,mask].mean()),static_MAE=float((reference[i]-target).abs()[:,mask].mean()),outside_change=float((out['rgb']-reference[i]).abs()[:,protect].mean()),critical={f'{x},{y}':dict(contribution=float(out['contribution'][0,y,x]),rgb=cpu(out['rgb'][:,y,x]).tolist()) for x,y in [(255,240),(256,219)]}))
            repeat=n.render(cams[0],dynamic=True)['rgb'];assert float((repeat-master).abs().max())==0
            (dest/f'evaluation{step}.json').write_text(json.dumps(results,indent=2))
    evaluate(start_step)
    with (dest/f'loss_{start_step}.jsonl').open('x') as log:
        for step in range(start_step+1,a.steps+1):
            i=int(torch.randint(3,(1,),generator=sampler));c=cams[i];d=data[i];mask=torch.tensor(d['valid'],device='cuda');protect=~torch.tensor(d['target'],device='cuda');target=torch.tensor(d['rgb'].transpose(2,0,1),device='cuda')
            opt.zero_grad(set_to_none=True);multiplier=torch.ones(len(selected),device='cuda').index_copy(0,ids,gate.sigmoid());out=n.render(c,road,static_multiplier=multiplier)
            deleted=(out['rgb']-target).abs()[:,mask].mean();preserve=(out['rgb']-reference[i]).abs()[:,protect].mean();visible=(.25-out['contribution'][0,mask]).clamp_min(0).mean()
            # Same-frame road masks exist for102; for129/137 choose nearest prepared frame but render its camera.
            src=min([102,130,137],key=lambda v:abs(v-sources[i]));obs=dict(np.load(OUT/'dense_training'/f'{src:03d}.npz'));om=torch.tensor(obs['visible'],device='cuda');gt=torch.tensor(obs['rgb'].transpose(2,0,1),device='cuda');direct=render_road(n.camera(src),road,n.args,for_training=True)
            road_loss=(direct['rgb']-gt).abs()[:,om].mean();loss=deleted+preserve+.1*visible+.1*road_loss
            assert torch.isfinite(loss);loss.backward();assert torch.isfinite(gate.grad).all() and torch.isfinite(road.dc.grad).all();opt.step()
            record=dict(step=step,raw=sources[i],loss=float(loss),deleted=float(deleted),preserve=float(preserve),contribution_shortfall=float(visible),road=float(road_loss),seconds=time.time()-start,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved());log.write(json.dumps(record)+'\n');log.flush()
            if step%100==0 or step==start_step+1:print(json.dumps(record),flush=True)
            if step in [100,101,500,1000] or step==a.steps:torch.save(state(step),dest/f'checkpoint{step}_from{start_step}.pth');evaluate(step)
            del out,direct,loss,multiplier
    assert n.unchanged();(dest/f'receipt_{start_step}_{a.steps}.json').write_text(json.dumps(dict(completed=True,updates=a.steps-start_step,seconds=time.time()-start,master_unchanged=True,quality_accepted=False,human_verdict=None)))
if __name__=='__main__':main()
