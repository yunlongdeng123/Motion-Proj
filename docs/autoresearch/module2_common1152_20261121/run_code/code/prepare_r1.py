"""Compile the unchanged round-one GH15 physical reads, CPU only."""
import sys,time
from runtime1152 import *
paths()
P=C/'road_multiscale_material_20261118_r1'
sys.path.insert(0,str(P/'code'))
import torch
torch.set_num_threads(2)
from fixed_material_read_operator import ReadCompiler
from world_material_field import WorldMaterialField

def main():
    out=ROOT/'r1_inputs';out.mkdir(exist_ok=True)
    if (out/'receipt.json').exists():return
    t=time.monotonic()
    source=P/'formal_r1/ABC800/asset/world_field.pt'
    model,metadata=WorldMaterialField.load(source,device='cpu')
    with torch.no_grad(): model.low.zero_();model.high.zero_()
    s=load(P/'method_scope_r1/scope.npz');eligible=s['eligible'];ids=np.flatnonzero(eligible)
    compiler=ReadCompiler(model.A0.numpy(),model.valid[...,0].numpy(),{'observed':model.mH[...,0].numpy()},low_stride=6)
    for start in range(0,len(ids),512):
        ix=ids[start:start+512]
        compiler.add(model.sample_uv[ix].numpy(),model.weights.numpy(),model.geometric_sample_valid[ix].numpy())
    op=compiler.finish(s['active_world_rows'][eligible],dict(kind='original_GH15_fixed_world_field_read',source=str(source),source_SHA256=sha(source),geometry_unchanged=True))
    op.save(out/'operator')
    init=dict(common_base_srgb=model.A0.numpy(),common_material_valid=model.valid[...,0].numpy(),
        observed_H_prior=model.mH[...,0].numpy(),common_latent_mask=model.high_parameter_mask[...,0].numpy(),
        common_reg_support=model.high_output_mask[...,0].numpy(),low_mask=model.low_parameter_mask[...,0].numpy(),
        active_world_rows=s['active_world_rows'],eligible=eligible)
    np.savez_compressed(out/'initial.npz',**init)
    rng=np.random.default_rng(101)
    with torch.no_grad():
        model.low.copy_(torch.from_numpy(rng.normal(0,.01,model.low.shape).astype(np.float32)))
        model.high.copy_(torch.from_numpy(rng.normal(0,.02,model.high.shape).astype(np.float32)))
        chosen=np.linspace(0,len(ids)-1,96,dtype=int)
        L,H=model.fields();reference=model.query(indices=torch.as_tensor(ids[chosen]))[0].numpy()
        pred=op.numpy_query(L.numpy(),H.numpy(),'observed')[chosen]
    error=float(np.max(np.abs(reference-pred)))
    if error>1e-6:raise RuntimeError('R1 exact lowering differs: '+str(error))
    write(out/'receipt.json',dict(complete=True,CPU_only=True,source=str(source),source_SHA256=sha(source),
        rows=len(ids),high_bound=.04,original_GH15=True,read_RGB_max=error,seconds=time.monotonic()-t))
    print('R1_INPUTS_READY',len(ids),error,time.monotonic()-t,flush=True)

if __name__=='__main__':main()
