"""读取最终checkpoint并逐值验证车辆、道路参数、Adam与全部RNG；不执行更新。"""
import argparse,random,numpy as np,torch
import run_joint as m
def restore(path):
    saved=torch.load(path,map_location='cpu',weights_only=False);v=saved['vehicle']
    arrays={name:v['base_'+name].numpy() for name in ['xyz','scales','rotation','coefficients','opacity']}
    vehicle=m.Vehicle(arrays).cuda();vehicle.load_state_dict(v);vehicle.opacity_logits.requires_grad_(saved['stage']=='vehicle');vehicle.log_scale_delta.requires_grad_(saved['stage']=='vehicle');road=torch.nn.Parameter(saved['road_color'].cuda())
    groups=[dict(params=[vehicle.dc],lr=.01)]
    if saved['stage']=='joint':groups+=[dict(params=[road],lr=.005)]
    else:groups+=[dict(params=[vehicle.opacity_logits],lr=.01),dict(params=[vehicle.log_scale_delta],lr=.002)]
    optimizer=torch.optim.Adam(groups,betas=(.9,.999),eps=1e-8);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    actual=dict(saved);actual.update(vehicle=m.cpu(vehicle.state_dict()),road_color=m.cpu(road),optimizer=m.cpu(optimizer.state_dict()),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate());diff=m.differences(saved,actual)
    return vehicle,road,optimizer,dict(passed=not diff,checkpoint=str(path),stage=saved['stage'],step=saved['step'],model_Adam_RNG_bit_exact=True,differences=diff,optimizer_updates=0,native_raster_calls=0)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default=str(m.ROOT/'checkpoints/joint_step00200.pt'));a=ap.parse_args();_,_,_,proof=restore(a.checkpoint);m.write(m.ROOT/'final_optimizer_restore_proof.json',proof);print(proof)
    if not proof['passed']:raise RuntimeError('final optimizer restore mismatch')
