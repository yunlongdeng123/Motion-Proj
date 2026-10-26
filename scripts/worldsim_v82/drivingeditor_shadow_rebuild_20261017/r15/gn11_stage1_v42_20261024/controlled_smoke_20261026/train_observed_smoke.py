"""独立编辑道路短程：真实稠密可见区域训练；完整状态/跨进程恢复验证。"""
import argparse,random,time,json,subprocess
from editor_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['window49','full199'],required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--resume');p.add_argument('--name',default='observed_smoke');a=p.parse_args()
    dest=OUT/a.name/a.arm;dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    n=Native();road=Road(FULL/f'{a.arm}_initialization.npz');gate=torch.as_tensor(np.load(OUT/'fixed_gate_footprint.npz')['suppress'],device='cuda')
    # Geometry is fixed; independent buffers, optimizer does not contain B tensors.
    opt=torch.optim.Adam([{'params':[road.dc],'lr':.0025},{'params':[road.opacity],'lr':.05}],eps=1e-15)
    paths=sorted((OUT/'dense_training').glob('*.npz'));assert len(paths)==18
    data=[dict(np.load(f)) for f in paths];frames=[int(f.stem) for f in paths];cams=[n.camera(raw) for raw in frames]
    sampler=torch.Generator(device='cpu');sampler.manual_seed(20261026)
    begin=0
    def checkpoint(step):
        return dict(step=step,dc=road.dc.detach().clone(),opacity=road.opacity.detach().clone(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),sampler=sampler.get_state(),frames=frames,arm=a.arm,config=dict(lr_dc=.0025,lr_opacity=.05,loss='dense visible road premultipliedRGB L1 +0.1 relu(0.8-alpha); no fake hidden-body target',geometry='fixed'),master_files=n.before)
    if a.resume:
        ck=torch.load(a.resume,weights_only=False);assert ck['arm']==a.arm and ck['frames']==frames and ck['master_files']==n.before
        with torch.no_grad():road.dc.copy_(ck['dc']);road.opacity.copy_(ck['opacity'])
        opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng']);sampler.set_state(ck['sampler']);begin=ck['step']
    started=time.time();nvml=0
    with (dest/f'steps_{begin}.jsonl').open('x') as log:
        for step in range(begin+1,a.steps+1):
            i=int(torch.randint(len(data),(1,),generator=sampler));d=data[i];mask=torch.as_tensor(d['visible'],device='cuda');gt=torch.as_tensor(d['rgb'].transpose(2,0,1),device='cuda')
            opt.zero_grad(set_to_none=True);r=render_road(cams[i],road,n.args,for_training=True)
            l1=(r['rgb']-gt).abs()[:,mask].mean();coverage=(.8-r['alpha'][0,mask]).clamp_min(0).mean();loss=l1+.1*coverage
            if not torch.isfinite(loss):raise FloatingPointError(step)
            loss.backward();assert road.dc.grad is not None and torch.isfinite(road.dc.grad).all();assert torch.isfinite(road.opacity.grad).all();opt.step()
            record=dict(step=step,raw=frames[i],loss=float(loss),l1=float(l1),coverage=float(coverage),visible_pixels=int(mask.sum()),cuda_allocated_peak=torch.cuda.max_memory_allocated(),cuda_reserved_peak=torch.cuda.max_memory_reserved())
            log.write(json.dumps(record)+'\n');log.flush()
            if step%100==0 or step==begin+1:print(json.dumps(record),flush=True)
            if step in [100,101,500,1000] or step==a.steps:
                torch.save(checkpoint(step),dest/f'checkpoint{step}_from{begin}.pth')
                with torch.no_grad():
                    for raw in [102,129,137]:
                        c=n.camera(raw);rr=render_road(c,road,n.args);save(dest/f'{step}_raw{raw}_road.png',rr['rgb']);com=n.render(c,road,gate);save(dest/f'{step}_raw{raw}_edited.png',com['rgb'])
                        (dest/f'{step}_raw{raw}_contribution.json').write_text(json.dumps({f'{x},{y}':dict(alpha=float(rr['alpha'][0,y,x]),contribution=float(com['contribution'][0,y,x])) for x,y in [(255,240),(256,219)]}))
            del r,loss,gt,mask
    assert n.unchanged()
    (dest/f'receipt_{begin}_{a.steps}.json').write_text(json.dumps(dict(status='completed',start_step=begin,end_step=a.steps,updates_executed=a.steps-begin,seconds=time.time()-started,checkpoints_saved=True,evaluation_entry=True,next_stage='checkpoint exported to native independent editing',native_updates=0,quality_accepted=False,human_verdict=None),indent=2))

if __name__=='__main__':main()
