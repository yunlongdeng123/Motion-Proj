"""Matched direct-atlas/fusion-atlas supervision, one fixed parent road chart.

400 paired asset steps per variant, identical sampler/masks/Adam, projected RGB.
No geometry/opacity/view policy/original layer can receive an optimizer update.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse
import importlib.util
import json
import random
import shutil
import sys
import time
import numpy as np
from shared_fit_common import OUT,RUN,BUILD,HELD,SEED,Sampler,sha,load,read,write,tree_hash,cpu_tree,rng_state,restore_rng
from fusion_colour_runtime import FusionColourRuntime,PARENT

SMOKE=(90,128)
SOURCES=('fit_shared_material.py','fusion_colour_runtime.py','shared_fit_common.py','qualify_shared_road.py')


def target_path(root,raw):return root/f'raw{raw:03d}.npz'


def freeze(args):
    shared=read(args.output/'shared_contract.json');q=load(args.output/'qualified_parents.npz');masks=load(args.output/'supervision_masks.npz')
    assert sha(args.output/'qualified_parents.npz')==shared['qualified_parents_sha256']
    ready=read(args.target_manifest)
    if ready.get('fusion_ready') is not True and not args.placeholder:
        raise ValueError('Final fusion_ready=true target manifest is required; temporary raw direct target forbidden')
    files={};stats=[]
    for raw in BUILD:
        path=target_path(args.targets,raw);data=load(path);files[str(path.resolve())]=sha(path)
        for key in ('direct_rgb','fusion_rgb','observation_rgb'):
            image=data[key];assert image.shape==(320,480,3) and image.dtype==np.float32 and np.isfinite(image).all(),(raw,key)
            assert image.min()>=0 and image.max()<=1,(raw,key)
        mask=data['fit_mask'];replace=data['fusion_replace_mask'].astype(bool)
        assert mask.dtype==bool and np.array_equal(mask,masks[f'raw{raw:03d}']),raw
        assert np.array_equal(data['direct_rgb'][~replace],data['fusion_rgb'][~replace]),raw
        assert np.array_equal(data['direct_rgb'][~replace],data['observation_rgb'][~replace]),raw
        stats.append(dict(raw=raw,fit_pixels=int(mask.sum()),atlas_replace_fit_pixels=int((mask&replace).sum()),
            direct_fusion_fit_RGB_abs_mean=float(np.abs(data['direct_rgb'][mask]-data['fusion_rgb'][mask]).mean())))
    source=Path(__file__).parent;hashes={name:sha(source/name) for name in SOURCES}
    for name,digest in shared['source_sha256'].items():assert hashes[name]==digest,name
    destination=args.output/'fit_contract.json'
    if destination.exists():raise FileExistsError('Keep prior frozen target contract')
    contract=dict(schema='MATCHED_190SOURCE_DIRECT_ATLAS_VS_POISSON_ATLAS_PARENT_RGB_V1',
        shared_contract_sha256=sha(args.output/'shared_contract.json'),qualification_sha256=sha(args.output/'qualified_parents.npz'),
        supervision_masks_sha256=sha(args.output/'supervision_masks.npz'),target_manifest=str(args.target_manifest.resolve()),
        target_manifest_sha256=sha(args.target_manifest),targets_directory=str(args.targets.resolve()),target_files_sha256=files,
        baseline_cache_directory=str(args.baseline_cache.resolve()),placeholder_engineering_only=args.placeholder,
        target_roles=(dict(direct='temporary original raw RGB engineering fixture; not a scientific comparison',
            fusion='unexecuted identity placeholder, original raw RGB',observation='original raw RGB') if args.placeholder else
            dict(direct='same190 nonHELD real source pool, unfused atlas projection',
            fusion='same190 nonHELD real source pool, shared Poisson-fused atlas projection',observation='original raw RGB for evaluation')),
        BUILD=list(BUILD),HELD=list(HELD),states=['keep','delete'],variant_order=['direct','fusion'],target_statistics=stats,
        optimizer=dict(name='Adam',lr=.01,betas=[.9,.999],eps=1e-8,formal_steps=400,epochs=10),
        loss='.5*(mean_RGB_MSE_keep+mean_RGB_MSE_delete) on identical fixed fit_mask; no regularization or extra protection loss',
        protected_domain='independent evaluation only, no hidden or delete-render target',
        trainable_parent_rows=int(q['eligible'].sum()),trainable_scalars=3*int(q['eligible'].sum()),
        zero_mass_rows='absent from trainable tensor and all Adam moments; export initial bytes exactly',
        colour_range=[0,1],projection='clamp eligible RGB after each Adam step',
        smoke=dict(two_training_frames=list(SMOKE),paired_steps=2,initial_parity_forwards=4,evaluation_forwards=4,
            disposable_original_next_steps=2,fresh_restore_next_steps=2,restore_required=['parameter','Adam','Python/NumPy/TorchCPU/CUDA RNG','sampler']),
        source_sha256=hashes,original_parameter_updates=0,geometry_updates=0,opacity_updates=0,
        whole_scene_adoption_claim=False,hidden_GT_used=False,
        holdout_note='new target source pool excludesHELD9; prior frozen parent material/source proxy may previously have used199 sources')
    snap=args.output/'fit_source_snapshot';snap.mkdir()
    for name in hashes:shutil.copy2(source/name,snap/name)
    write(destination,contract);print(json.dumps(dict(contract=str(destination),sha256=sha(destination),eligible=contract['trainable_parent_rows'])))


def locked(args):
    c=read(args.output/'fit_contract.json');shared=read(args.output/'shared_contract.json')
    assert sha(args.output/'shared_contract.json')==c['shared_contract_sha256']
    assert sha(args.output/'qualified_parents.npz')==c['qualification_sha256']
    assert sha(args.output/'supervision_masks.npz')==c['supervision_masks_sha256']
    assert sha(c['target_manifest'])==c['target_manifest_sha256']
    assert c['placeholder_engineering_only'] or read(c['target_manifest']).get('fusion_ready') is True
    for name,digest in c['source_sha256'].items():assert sha(Path(__file__).with_name(name))==digest,name
    for path,digest in c['target_files_sha256'].items():assert sha(path)==digest,path
    return c,shared,load(args.output/'qualified_parents.npz'),load(args.output/'supervision_masks.npz')


class Material:
    def __init__(self,torch,q):
        self.torch=torch;self.initial=torch.as_tensor(q['initial_colours_srgb'],device='cuda')
        self.ids=torch.as_tensor(q['eligible_parent_ids'],device='cuda',dtype=torch.long)
        self.value=torch.nn.Parameter(self.initial.index_select(0,self.ids).clone())
        self.optimizer=torch.optim.Adam([self.value],lr=.01,betas=(.9,.999),eps=1e-8)
    def colours(self):return self.initial.index_copy(0,self.ids,self.value)
    def project(self):
        with self.torch.no_grad():self.value.clamp_(0,1)


def state_dict(material,sampler,step,c,torch):
    return dict(step=step,value=material.value.detach().cpu().clone(),optimizer=cpu_tree(material.optimizer.state_dict()),
        sampler=sampler.state(),rng=cpu_tree(rng_state(torch)),next_two_frames=sampler.peek(2),
        contract_sha256=sha(c),parameter_parent_ids=material.ids.cpu().numpy().copy())


def checkpoint(folder,material,sampler,step,contract_path,torch,label=''):
    state=state_dict(material,sampler,step,contract_path,torch);path=folder/f'checkpoint{step:06d}{label}.pt'
    if path.exists():raise FileExistsError(path)
    torch.save(state,path);restored=torch.load(path,map_location='cpu',weights_only=False)
    assert tree_hash(state)==tree_hash(restored)
    return path,state


def step(model,material,sampler,variant,c,masks,torch):
    raw=sampler.next();target=load(target_path(Path(c['targets_directory']),raw))
    goal=torch.as_tensor(target[variant+'_rgb'].transpose(2,0,1).copy(),device='cuda')
    domain=torch.as_tensor(masks[f'raw{raw:03d}'],device='cuda',dtype=torch.bool)
    material.optimizer.zero_grad(set_to_none=True);total=0.;started=time.time()
    for state in ('keep','delete'):
        out=model.render(raw,state,material.colours());loss=.5*(out['rgb'][:,domain]-goal[:,domain]).square().mean()
        assert torch.isfinite(loss);loss.backward();total+=float(loss.detach().cpu());del out,loss
    assert material.value.grad is not None and torch.isfinite(material.value.grad).all()
    gradient=float(material.value.grad.norm().detach().cpu());material.optimizer.step();material.project()
    colour=material.colours().detach();eligible=torch.zeros(10098,device='cuda',dtype=torch.bool);eligible[material.ids]=True
    assert torch.isfinite(colour).all() and colour.min()>=0 and colour.max()<=1
    assert torch.equal(colour[~eligible],material.initial[~eligible])
    model.rt.assert_unchanged();assert model.model.extension.unchanged()
    model.rt._data.clear()
    return dict(raw=raw,MSE=total,gradient_norm=gradient,seconds=time.time()-started,
        clipped_channels=int(((material.value==0)|(material.value==1)).sum().detach().cpu()))


def image(path,colour):
    from PIL import Image
    Image.fromarray(np.uint8(np.clip(colour.transpose(1,2,0),0,1)*255+.5)).save(path)


def evaluate(model,material,folder,c,torch,baseline=False):
    folder.mkdir(exist_ok=False);records=[]
    with torch.no_grad():
        for raw in SMOKE:
            target=load(target_path(Path(c['targets_directory']),raw));mask=target['fit_mask'];protect=target['protect_mask'].astype(bool)
            for state in ('keep','delete'):
                out=model.render(raw,state,material.colours());rgb=out['rgb'].cpu().numpy()
                geometry={key:out[key].cpu().numpy() for key in ('alpha','depth_m','group_alpha_T','newroad_alpha_T')}
                name=f'raw{raw:03d}_{state}';path=folder/(name+'.npz')
                np.savez_compressed(path,rgb=rgb,**geometry);image(folder/(name+'.png'),rgb)
                errors={role:float(np.abs(rgb.transpose(1,2,0)[mask]-target[role+'_rgb'][mask]).mean()) for role in ('direct','fusion','observation')}
                equality=None
                if baseline:
                    cache=load(Path(c['baseline_cache_directory'])/f'{name}.npz')
                    assert np.array_equal(rgb,cache['rgb']) and np.array_equal(geometry['group_alpha_T'],cache['group_alpha_T'])
                else:
                    reference=load(folder.parent/'baseline'/f'{name}.npz')
                    equality={key:np.array_equal(value,reference[key]) for key,value in geometry.items()}
                    assert all(equality.values()),equality
                records.append(dict(raw=raw,state=state,fit_pixels=int(mask.sum()),target_MAE=errors,
                    protected_observation_MAE=float(np.abs(rgb.transpose(1,2,0)[protect]-target['observation_rgb'][protect]).mean()) if protect.any() else None,
                    geometry_bitwise_checks=equality,group_conservation=out['group_conservation_maxabs'],RGB_conservation=out['RGB_conservation_maxabs']))
            model.rt._data.clear()
    write(folder/'evaluation.json',dict(records=records,forwards=4,baseline_cache_parity=baseline))


def run(args):
    c,shared,q,masks=locked(args);branch=args.output/args.variant;branch.mkdir(exist_ok=True)
    if args.phase=='formal':
        if c['placeholder_engineering_only']:raise ValueError('Engineering placeholder cannot enter formal fitting')
        smoke=read(branch/'smoke/receipt.json');restore=read(branch/'restore/receipt.json')
        assert restore['restore_state_all_exact'] and restore['steps_executed']==2
        if smoke['formal_ETA_seconds']>=3600:raise RuntimeError('Need600s real smoke before a >=1h formal run')
    folder=branch/args.phase;folder.mkdir(exist_ok=False)
    import torch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);random.seed(SEED);np.random.seed(SEED)
    spec=importlib.util.spec_from_file_location('_fit_preflight',PARENT/'canonical_road_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);preflight=module.gpu_preflight()
    started=time.time();model=FusionColourRuntime();material=Material(torch,q);sampler=Sampler(smoke=args.phase!='formal')
    assert torch.equal(material.initial,model.initial)
    rows=[];state_step=0;restore_checks=None;nextstep_comparison=None
    if args.phase=='smoke':evaluate(model,material,folder/'baseline',c,torch,baseline=True)
    if args.phase=='restore':
        saved=torch.load(branch/'smoke/checkpoint000002.pt',map_location='cpu',weights_only=False)
        assert saved['contract_sha256']==sha(args.output/'fit_contract.json')
        with torch.no_grad():material.value.copy_(saved['value'].cuda())
        material.optimizer.load_state_dict(saved['optimizer']);sampler.restore(saved['sampler']);restore_rng(torch,saved['rng']);state_step=2
        restore_checks=dict(parameter=tree_hash(material.value)==tree_hash(saved['value']),Adam=tree_hash(material.optimizer.state_dict())==tree_hash(saved['optimizer']),
            sampler=tree_hash(sampler.state())==tree_hash(saved['sampler']),RNG=tree_hash(rng_state(torch))==tree_hash(saved['rng']),
            parent_ids=np.array_equal(material.ids.cpu().numpy(),saved['parameter_parent_ids']),next_two_frames=sampler.peek(2)==saved['next_two_frames'])
        assert all(restore_checks.values());write(folder/'restore_checks.json',restore_checks)
    budget=400 if args.phase=='formal' else 2
    for index in range(budget):
        record=step(model,material,sampler,args.variant,c,masks,torch);state_step+=1;record['step']=state_step;rows.append(record)
        if state_step%100==0 or index==budget-1:checkpoint(folder,material,sampler,state_step,args.output/'fit_contract.json',torch)
        if index%20==0 or index==budget-1:print(json.dumps(record),flush=True)
    if args.phase=='smoke':
        evaluate(model,material,folder/'evaluation',c,torch)
        disposable=[]
        for _ in range(2):disposable.append(step(model,material,sampler,args.variant,c,masks,torch));state_step+=1
        checkpoint(folder,material,sampler,state_step,args.output/'fit_contract.json',torch,label='_disposable_expected')
        write(folder/'disposable_steps.json',dict(steps=disposable,forwards=4,backwards=4,formal_steps=0))
    elif args.phase=='restore':
        expected=torch.load(branch/'smoke/checkpoint000004_disposable_expected.pt',map_location='cpu',weights_only=False)
        actual=state_dict(material,sampler,4,args.output/'fit_contract.json',torch)
        nextstep_comparison={key:tree_hash(actual[key])==tree_hash(expected[key]) for key in actual}
        nextstep_comparison['parameter_maxabs']=float((actual['value']-expected['value']).abs().max())
        write(folder/'nextstep_comparison.json',nextstep_comparison)
    receipt=dict(status='ENGINEERING_RUN_COMPLETE',phase=args.phase,variant=args.variant,steps_executed=budget,
        formal_steps=budget if args.phase=='formal' else 0,original_disposable_steps=2 if args.phase=='smoke' else 0,
        fresh_restore_disposable_steps=2 if args.phase=='restore' else 0,forwards=model.forwards,
        backwards=2*budget+(4 if args.phase=='smoke' else 0),records=rows,
        restore_checks=restore_checks,restore_state_all_exact=all(restore_checks.values()) if restore_checks else None,
        nextstep_comparison=nextstep_comparison,seconds=time.time()-started,
        formal_ETA_seconds=float(np.median([r['seconds'] for r in rows])*400),preflight=preflight,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        eligible_parents=len(material.ids),excluded_zero_parents=10098-len(material.ids),Adam_parameter_shape=list(material.value.shape),
        parent_geometry_updates=0,original_parameter_updates=0,opacity_updates=0,gpu_fraction=args.gpu_fraction,
        placeholder_engineering_only=c['placeholder_engineering_only'],
        runtime=model.receipt(),actual_command=sys.argv,fit_contract_sha256=sha(args.output/'fit_contract.json'))
    write(folder/'receipt.json',receipt)
    if args.phase=='formal':
        colour=material.colours().detach().cpu().numpy();assert np.array_equal(colour[~q['eligible']],q['initial_colours_srgb'][~q['eligible']])
        np.savez_compressed(branch/'material_override.npz',colours_srgb=colour,initial_colours_srgb=q['initial_colours_srgb'],
            eligible=q['eligible'],eligible_parent_ids=q['eligible_parent_ids'],parent_ids=np.arange(10098),
            asset_sha256=np.array(shared['canonical_asset_sha256']),parent_asset_sha256=np.array(shared['canonical_asset_sha256']),
            fit_contract_sha256=np.array(sha(args.output/'fit_contract.json')),target_manifest_sha256=np.array(c['target_manifest_sha256']),
            variant=np.array(args.variant),source_roles=np.array(json.dumps(c['target_roles'],sort_keys=True)),
            formal_steps=np.array(400),original_parameter_updates=np.array(0),geometry_updates=np.array(0),opacity_updates=np.array(0))
        write(branch/'export_receipt.json',dict(material=str(branch/'material_override.npz'),sha256=sha(branch/'material_override.npz'),
            rows=10098,variant=args.variant,formal_steps=400,initial_unqualified_rows_byte_exact=True,unknown_rows_absent_from_Adam=True))
    locked(args);print(json.dumps({key:receipt[key] for key in ('phase','variant','forwards','backwards','seconds','formal_ETA_seconds')}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--phase',choices=['freeze','smoke','restore','formal'],required=True)
    p.add_argument('--variant',choices=['direct','fusion'],default='direct');p.add_argument('--targets',type=Path,default=RUN/'targets/projected_BUILD40')
    p.add_argument('--target-manifest',type=Path,default=RUN/'targets/target_manifest.json');p.add_argument('--gpu-fraction',type=float,default=.4)
    p.add_argument('--baseline-cache',type=Path,default=OUT/'baseline_cache');p.add_argument('--placeholder',action='store_true')
    args=p.parse_args();freeze(args) if args.phase=='freeze' else run(args)


if __name__=='__main__':main()
