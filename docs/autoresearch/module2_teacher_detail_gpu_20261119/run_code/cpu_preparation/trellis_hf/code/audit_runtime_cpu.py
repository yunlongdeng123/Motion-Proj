"""Actual shared shader algebra on CPU: H seed fields, bilinear reader and compact roundtrip."""
from pathlib import Path
import argparse, os, json, hashlib, importlib.util
import numpy as np
from trellis_hf_seed import sha, blur
from audit_seed_readers import sample0


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU only')
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=Path, required=True)
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--buffers', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    spec = importlib.util.spec_from_file_location('shared_world_runtime_CPU_only', a.runtime)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    import torch
    torch.set_num_threads(1)
    with np.load(a.seed)as z:d={k:z[k]for k in z.files}
    with np.load(a.buffers)as z:b={k:z[k]for k in ['common_base_srgb','common_material_valid','mH_union','common_latent_mask']}
    before = hashlib.sha256(np.ascontiguousarray(b['common_base_srgb']).tobytes()).hexdigest()
    seed = torch.from_numpy(d['H_seed_latent'].copy())
    mask = torch.as_tensor(b['common_latent_mask'][...,None],dtype=torch.float32)
    raw = torch.tanh(seed)*mask
    high = .17*(raw-module.blur(raw,3.))
    assert high.device.type=='cpu'
    numpy_raw = np.tanh(d['H_seed_latent'].astype(np.float64))*b['common_latent_mask'][...,None]
    reference_high = .17*(numpy_raw-blur(numpy_raw))
    field_error = float(np.max(np.abs(high.numpy()-reference_high)))
    y0,x0=d['crop_origin_yx'];H,W=d['mH_combined'].shape
    assert np.array_equal(b['mH_union'][y0:y0+H,x0:x0+W],d['mH_combined'])
    assert np.all(b['common_latent_mask'][np.any(d['H_seed_latent']!=0,axis=-1)])
    rng=np.random.default_rng(101)
    uv=np.stack([rng.uniform(x0,x0+W-1,2048),rng.uniform(y0,y0+H-1,2048)],-1)
    uv=np.concatenate([uv,np.array([[-.25,.25],[0,0],[b['mH_union'].shape[1]-1+.25,b['mH_union'].shape[0]-1-.25]])])
    sampled_H=module.sample_bilinear(high,torch.as_tensor(uv[None],dtype=torch.float64))[0].numpy()
    sampled_m=module.sample_bilinear(torch.as_tensor(b['mH_union'][...,None]),torch.as_tensor(uv[None],dtype=torch.float64))[0].numpy()
    numpy_sample=sample0(high.numpy(),uv)*sample0(b['mH_union'][...,None],uv)
    sampler_error=float(np.abs(sampled_H*sampled_m-numpy_sample).max())
    h_abs=np.abs(high.numpy().astype(np.float64))
    max_H=np.maximum.reduce([h_abs[:-1,:-1],h_abs[:-1,1:],h_abs[1:,:-1],h_abs[1:,1:]])
    m=b['mH_union'];max_m=np.maximum.reduce([m[:-1,:-1],m[:-1,1:],m[1:,:-1],m[1:,1:]])
    continuous_bound=float(np.max(max_m[...,None]*max_H))
    valid_error=float(np.abs(module.sample_bilinear(torch.as_tensor(b['common_material_valid'][...,None],dtype=torch.float64),
        torch.as_tensor(uv[None],dtype=torch.float64))[0].numpy()-sample0(b['common_material_valid'][...,None].astype(float),uv)).max())
    a.output.mkdir(parents=True)
    saved=dict(H_seed_crop=seed[y0:y0+H,x0:x0+W].clone(),origin=torch.as_tensor([y0,x0]),global_shape=seed.shape)
    torch.save(saved,a.output/'CPU_seed_roundtrip.pt')
    restored=torch.load(a.output/'CPU_seed_roundtrip.pt',map_location='cpu',weights_only=False)
    rebuilt=torch.zeros(tuple(restored['global_shape']),dtype=torch.float32)
    rebuilt[y0:y0+H,x0:x0+W]=restored['H_seed_crop']
    assert torch.equal(rebuilt,seed)
    after=hashlib.sha256(np.ascontiguousarray(b['common_base_srgb']).tobytes()).hexdigest()
    assert before==after
    checks=dict(Numpy_vs_runtime_H_field=field_error<1e-6,Numpy_vs_runtime_continuous_query=sampler_error<1e-12,
        zero_padded_source_valid_sampler=valid_error<1e-12,serialized_seed_exact=True,
        base_buffer_exact=True,only_common_latent_nodes_seeded=True)
    checks['continuous_effective_HF_init_cap']=continuous_bound<=.02
    if not all(checks.values()):raise AssertionError(checks)
    report=dict(status='PASS_CPU_SHADER_ALGEBRA_ONLY',checks=checks,H_field_max=field_error,
        continuous_query_max=sampler_error,source_valid_query_max=valid_error,
        continuous_effective_HF_abs_upper_bound=continuous_bound,
        continuous_bound_method='cell max(mH corners)*max(abs(unmodulated H corners)); bilinear convex hull, not only sampled pixels',
        common_base_buffer_sha256=before,runtime_source_sha256=sha(a.runtime),input_buffers_sha256=sha(a.buffers),
        seed_sha256=sha(a.seed),GPU_operations=0,pretrained_model_inference_calls=0,native_renderer_calls=0,
        training_updates=0,actual_A200_or_Adam_restore_not_executed=True,GPU_entry_permitted=False)
    (a.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
