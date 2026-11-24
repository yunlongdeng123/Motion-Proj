"""Shared formal/microtest teacher-RGB and differentiable visibility objective."""
from pathlib import Path
import hashlib, json
import numpy as np

CATEGORIES = dict(real_visible=1., teacher_pseudo=.25, similar_reference=.15, real_visible_background=1.)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def pin(path):
    return dict(path=str(path), sha256=sha(path))

def load(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def weights(cfg):
    return dict(categories=cfg.get('category_weights', CATEGORIES), protection=float(cfg.get('protection_RGB_weight', 1.)),
                inside=float(cfg.get('opacity_inside_weight', .05)), outside=float(cfg.get('opacity_outside_weight', .05)),
                A20_visibility=float(cfg.get('A20_visibility_weight',1.)),
                regularization=cfg.get('regularization_weights', {k:1e-4 for k in ['geometry_prior','scale_prior','rotation_prior','color_prior']}))

def build_cache(rt, binding_path, mask_root, output, cfg):
    output = Path(output); assert not output.exists(), 'Reuse an existing immutable cache through load_cache'
    output.mkdir(parents=True); binding = read(binding_path); byraw = {int(r['raw']):r for r in binding['frames']}
    fixed_source=Path(cfg['source_fixed_cache_manifest']);fixed_manifest=read(fixed_source)
    assert sha(fixed_source) == cfg['source_fixed_cache_sha256']
    fixed_byraw={int(r['raw']):r for r in fixed_manifest['frame_rows']}
    inventory = []; frame_rows = []; t = rt.torch
    for raw in cfg['train_frames']:
        row = byraw[raw]; mask_path = Path(mask_root)/f'{raw:03d}'/'native.npz'; masks = load(mask_path)
        with t.no_grad(): context = {k:v.detach().cpu().numpy() for k,v in rt.render(raw, component='context').items()}
        fixed_spec = fixed_byraw[raw]['cache']
        assert sha(fixed_spec['path']) == fixed_spec['sha256']
        fixed_arrays=load(fixed_spec['path'])
        green=masks['keep_green'].astype(bool)|(np.squeeze(context['actor_alpha'])>float(cfg.get('green_alpha_threshold',.95)))
        orange=masks['keep_orange'].astype(bool);red=masks['red_rebuild'].astype(bool)
        background=load(row['background_completion']['path'])
        assert sha(row['background_completion']['path'])==row['background_completion']['sha256']
        background_support=background['support'].astype(bool)&orange&~green
        inside=(red|background_support)&~green;outside=~inside;protection=green|orange
        transform = row['colour_transform']; gain = np.asarray(transform['gain_linear_RGB'], np.float32); bias = np.asarray(transform['bias_linear_RGB'], np.float32)
        foreground=np.clip(context['retained_foreground_alpha'],0,1)
        arrays = dict(reference_RGB_LDR=np.clip(context['rgb'],0,1), green=green, orange=orange, red=red,
                      protection=protection, alpha_inside=inside, alpha_outside=outside, gain=gain, bias=bias)
        arrays.update(alpha_target=inside.astype(np.float32)*(1-foreground),retained_foreground_alpha=foreground,
                      A20_alpha_reference=np.squeeze(context['actor_alpha']))
        arrays['background_rebuild']=background_support
        target_linear=(background['rgb_common_linear']-bias)/gain
        target_srgb=np.where(target_linear<=.0031308,12.92*target_linear,1.055*np.maximum(target_linear,0)**(1/2.4)-.055)
        arrays['reference_RGB_LDR'][:,background_support]=np.clip(target_srgb.transpose(2,0,1)[:,background_support],0,1)
        categories = {}
        for name, spec in row['categories'].items():
            assert name in CATEGORIES and sha(spec['path']) == spec['sha256']
            target = load(spec['path']); original = target['support'].astype(bool); support = original&~green
            weight=target['loss_weight'].copy();weight[~support]=0
            arrays[f'{name}__rgb'] = target['rgb_common_linear'].transpose(2,0,1)
            arrays[f'{name}__support'] = support; arrays[f'{name}__weight'] = weight
            item = dict(raw=raw,category=name,original_pixels=int(original.sum()),excluded_fixed_green_pixels=int((original&green).sum()),
                        final_pixels=int(support.sum()),original_weight_denominator=float(target['loss_weight'][original].sum()),
                        final_weight_denominator=float(weight[support].sum()),source=spec)
            inventory.append(item); categories[name] = item
        frame_dir = output/f'{raw:03d}'; frame_dir.mkdir(); path = frame_dir/'cache.npz'; np.savez_compressed(path,**arrays)
        if raw == 110:
            from PIL import Image
            image = np.rint(np.clip(context['rgb'].transpose(1,2,0),0,1)*255).astype(np.uint8)
            Image.fromarray(image).save(frame_dir/'actual_retained_context.png')
        frame_rows.append(dict(raw=raw,cache=pin(path),source_mask=pin(mask_path),categories=categories,
            green_pixels=int(green.sum()),orange_pixels=int(orange.sum()),protection_pixels=int(protection.sum()),
            alpha_inside_pixels=int(inside.sum()),alpha_outside_pixels=int(outside.sum()),
            fixed_alpha_target_inside_mean=float(arrays['alpha_target'][inside].mean())if inside.any()else None,
            A20_visibility_weight_denominator=float(arrays['A20_alpha_reference'].sum()),
            semantic21_alpha_source='frozen retained native context render; threshold fixed before new-field training'))
    receipt = dict(schema='scene065.teacher_gaussian.fixed_objective_cache.v1',complete=True,cache_root=str(output),manifest=str(output/'manifest.json'),
        frames=list(cfg['train_frames']),frame_rows=frame_rows,inventory=inventory,target_bindings=pin(binding_path),
        weights=weights(cfg),views_per_optimizer_step=int(cfg['views_per_optimizer_step']),frozen_assets=dict(rt.frozen_assets),
        protection_reference='frozen retained context-only LDR shader RGB; preservation reference, not recovery ground truth',
        RGB_domain='canonical linear from LDR-clamped shader sRGB',selection='fixed masks; excludes green once; no learned-alpha selection',
        source_fixed_cache_manifest=pin(fixed_source),RGB_fixed_support_and_denominators_inherited_exactly=False,
        mask_correction='Original target RGB/source masks unchanged; fixed A20 occlusion now uses actual checkpoint source21, not transferred semantic labels. New denominators locked before training; compare R5 only on separately reported common domain.',
        opacity_target='inside:1-frozen retained foreground alpha; outside:0; effective native-raster alphaT',
        A20_visibility='one-sided decrease penalty against frozen context semantic21 alpha, weighted/normalized by reference alpha',
        engineering_optimizer_updates=0)
    write(output/'manifest.json',receipt); rt.assert_fixed(); return receipt

def load_cache(torch, cache_root):
    root = Path(cache_root); manifest = read(root/'manifest.json'); assert manifest['complete']
    cache = dict(manifest=manifest,frames={},weights=manifest['weights'])
    for frame in manifest['frame_rows']:
        path = Path(frame['cache']['path']); assert sha(path)==frame['cache']['sha256']; arrays = load(path)
        item = {k:torch.as_tensor(v,device='cuda') for k,v in arrays.items() if '__' not in k}
        item['categories'] = {}
        for name, record in frame['categories'].items():
            item['categories'][name] = dict(rgb=torch.as_tensor(arrays[f'{name}__rgb'],device='cuda'),
                support=torch.as_tensor(arrays[f'{name}__support'],device='cuda'),
                weight=torch.as_tensor(arrays[f'{name}__weight'],device='cuda'),denominator=record['final_weight_denominator'],pixels=record['final_pixels'])
        cache['frames'][frame['raw']] = item
    return cache

def canonical(torch, rgb, item):
    x = rgb.clamp(0,1); linear = torch.where(x<=.04045,x/12.92,((x+.055)/1.055).pow(2.4))
    return linear*item['gain'][:,None,None]+item['bias'][:,None,None]

def masked_mean(value, mask):
    return value[mask].mean() if bool(mask.any()) else value.sum()*0

def view_loss(rt, out, item, w):
    t=rt.torch; pred=canonical(t,out['rgb'],item); categories={}; rgb_error=t.zeros_like(out['local_effective_alpha']); rgb_weight=t.zeros_like(rgb_error)
    total=pred.sum()*0
    for name,target in item['categories'].items():
        support=target['support']; denominator=target['denominator']
        error=(pred-target['rgb']).square().mean(0)
        loss=(error[support]*target['weight'][support]).sum()/denominator if denominator else pred.sum()*0
        categories[name]=loss; total=total+w['categories'][name]*loss
        rgb_error=rgb_error+w['categories'][name]*error*target['weight']; rgb_weight=rgb_weight+w['categories'][name]*target['weight']
    protection=masked_mean((out['rgb'].clamp(0,1)-item['reference_RGB_LDR']).square().mean(0),item['protection'])
    alpha=out['local_effective_alpha']; inside=masked_mean((alpha-item['alpha_target']).square(),item['alpha_inside']); outside=masked_mean(alpha.square(),item['alpha_outside'])
    actor_ref=item['A20_alpha_reference'];actor_den=actor_ref.sum()
    visibility=(actor_ref*(actor_ref-out['actor_alpha']).clamp_min(0).square()).sum()/actor_den if float(actor_den)>0 else alpha.sum()*0
    total=total+w['protection']*protection+w['inside']*inside+w['outside']*outside+w['A20_visibility']*visibility
    # Birth targets missing teacher-supported road coverage; outside visibility
    # remains a full-image differentiable loss and drives geometry/opacity/prune.
    residual=(rgb_error/rgb_weight.clamp_min(1e-12)+(alpha-item['alpha_target']).square())*item['alpha_inside']
    return total,categories,dict(protection_RGB=protection,opacity_inside=inside,opacity_outside=outside,A20_visibility=visibility),residual

def step(rt,opt,sampler,cache,cfg,logical_step):
    t=rt.torch; w=weights(cfg); opt.zero_grad(set_to_none=True); draws=[]; terms=[]; views=[]; residuals={}; logical=0.; screen_stats=[]
    count=int(cfg['views_per_optimizer_step'])
    for _ in range(count):
        raw=int(sampler.next());draws.append(raw);item=cache['frames'][raw];out=rt.render(raw)
        loss,categories,parts,residual=view_loss(rt,out,item,w)
        assert bool(t.isfinite(loss)); (loss/count).backward();rt.backward_calls+=1
        screen=rt.last_local_means2D; grad=screen.grad
        screen_stats.append(dict(raw=raw,gradient_finite=bool(grad is not None and t.isfinite(grad).all()),
            gradient_norm=float(grad.norm()) if grad is not None else None,
            nonzero_gradient_rows=int((grad[:,:2].norm(dim=-1)>0).sum()) if grad is not None else 0,
            visible_new_rows=int((rt._render_records[-1][1]>0).sum()) if rt._render_records else 0))
        rt.accumulate_render_stats();residuals[raw]=residual.detach();logical+=float(loss.detach())/count
        terms.extend((k,float(v.detach()))for k,v in categories.items())
        views.append(dict(raw=raw,category_MSE={k:float(v.detach())for k,v in categories.items()},**{k:float(v.detach())for k,v in parts.items()}))
    priors=rt.regularization_losses();prior_total=sum(w['regularization'].get(k,1e-4)*v for k,v in priors.items())
    assert bool(t.isfinite(prior_total));prior_total.backward();logical+=float(prior_total.detach())
    gradients={};before={}
    for name,p in rt.named_new_parameters().items():
        assert p.grad is not None and bool(t.isfinite(p.grad).all()),name
        gradients[name]=dict(l2=float(p.grad.norm()),finite=True,nonzero_count=int((p.grad!=0).sum()));before[name]=p.detach().clone()
    opt.step();delta={name:dict(l2=float((p.detach()-before[name]).norm()),max_abs=float((p.detach()-before[name]).abs().max()),
        nonzero_count=int((p.detach()!=before[name]).sum()))for name,p in rt.named_new_parameters().items()}
    rt.post_optimizer_step(); event=rt.densify_and_prune(opt,int(logical_step),residual=residuals);opt.zero_grad(set_to_none=True);rt.assert_fixed()
    record=dict(raws=draws,logical_loss=logical,category_losses=terms,view_terms=views,prior_terms={k:float(v.detach())for k,v in priors.items()},
        gradient_norm=float(sum(x['l2']**2 for x in gradients.values())**.5),gradients_by_group=gradients,
        parameter_delta_by_group=delta,screen_gradient_stats=screen_stats,new_rows=len(rt.field.xyz),topology=event)
    return record,event

def evaluate(rt,cache,frames):
    rows=[];t=rt.torch
    with t.no_grad():
        for raw in frames:
            item=cache['frames'][int(raw)];out=rt.render(int(raw));loss,categories,parts,_=view_loss(rt,out,item,cache['weights'])
            rows.append(dict(raw=int(raw),categories={name:dict(pixels=item['categories'][name]['pixels'],
                weight_denominator=item['categories'][name]['denominator'],weighted_MSE=float(value) if item['categories'][name]['denominator'] else None)
                for name,value in categories.items()},fixed_objective_without_prior=float(loss),
                protection_RGB=dict(pixels=int(item['protection'].sum()),MSE=float(parts['protection_RGB'])),
                A20_visibility=dict(weight_denominator=float(item['A20_alpha_reference'].sum()),MSE=float(parts['A20_visibility'])),
                opacity_inside=dict(pixels=int(item['alpha_inside'].sum()),MSE=float(parts['opacity_inside']),mean_alpha=float(out['local_effective_alpha'][item['alpha_inside']].mean())if bool(item['alpha_inside'].any())else None),
                opacity_outside=dict(pixels=int(item['alpha_outside'].sum()),MSE=float(parts['opacity_outside']),mean_alpha=float(out['local_effective_alpha'][item['alpha_outside']].mean())if bool(item['alpha_outside'].any())else None)))
    rt.assert_fixed();return rows
