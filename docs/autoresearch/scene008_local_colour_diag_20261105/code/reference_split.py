"""Attribute native RGB error to fixed opacity shares, on observed pixels only.

This is a diagnostic reference substitution, not a proposed rendering method.
"""
from diagnose import *


@torch.no_grad()
def main():
    output = ROOT / 'reference_split'
    output.mkdir(exist_ok=False)
    rt = Runtime(fine_road=True)
    y, x = np.indices((320, 480))
    records = []
    for raw in PILOTS:
        data = rt.raw(raw)
        static = prepare_rasterization(data['cam'], rt.n.static, rt.pipe)
        static['means2D'] = torch.zeros_like(static['means3D'])
        dynamic = rt.n.actor_kwargs(data['cam'], delete_a37=False)
        rows = merge_rows(tagged(static, 'legacy_static'), tagged(dynamic, owners=rt.n.keep_a6))
        scene = dict(rows=rows, cam=data['cam'], env=data['env'], bg=rt.n.bg)
        pkg = draw(rt, raw, scene)
        reference = cpu(data['rgb'])
        actors = cpu(data['actor_labels']) > 0
        errors = {name: pkg['parts'][name]-pkg['weights'][i:i+1]*reference for i, name in enumerate(GROUPS)}
        errors['environment'] = pkg['parts']['environment']-(1-pkg['alpha'][None])*reference
        largest = float(np.abs(sum(errors.values())-(pkg['rgb']-reference)).max())
        assert largest < 3e-6
        masks = dict(observed_road=cpu(data['road']).astype(bool) & ~actors)
        for name, (x0, y0, x1, y1) in BOXES.items():
            masks[name] = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & ~actors
        records.append(dict(raw=raw, reconstruction_maxabs=largest,
                            regions={name: component_stats(errors, m) for name, m in masks.items()}))
        for name in ('static', 'environment'):
            save(output/f'{raw:03d}_{name}_blue_error.png', heat(blue(errors[name])))
        np.savez_compressed(output/f'{raw:03d}_exact.npz', reference=reference, native=pkg['rgb'],
                            alpha=pkg['alpha'], weights=pkg['weights'],
                            **{f'{k}_error': v for k, v in errors.items()})
        rt._data.clear()
    assert rt.modelunchanged()
    report = dict(status='REFERENCE_SHARE_DIAGNOSTIC_COMPLETE', training_updates=0, rows=records,
                  definition='e_g=C_g-w_g*originalRGB; e_env=T*(env-originalRGB). Their sum exactly equals native-originalRGB.',
                  limitation='Reference-share substitution identifies the source term of the observed RGB residual. It is not an independent ground truth for each layer or a physical illumination estimate.',
                  frozen_scene_unchanged=True)
    (output/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
