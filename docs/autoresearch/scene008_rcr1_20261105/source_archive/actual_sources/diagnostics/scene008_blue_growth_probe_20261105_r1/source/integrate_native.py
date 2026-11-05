"""Checked integration points for the immutable owner5 training engine."""
from pathlib import Path


def integrate(train, out, objective, growth_probe=None):
    path = Path(train.__file__)
    source = path.read_text()

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f'Native integration point changed: {old[:100]}')
        source = source.replace(old, new)

    start = source.index('        render_pkg = render(viewpoint_cam')
    stop = source.index('        gt_image = ', start)
    replace(source[start:stop], """        evidence, markers = _ownership.evidence(viewpoint_cam)
        render_pkg = _ownership.draw(viewpoint_cam, args, gaussians, dynamic_gaussians,
                                     background, env_map, other, dynamic_other,
                                     shift=time_shift, training=True, markers=markers)

""")
    replace('        viewspace_point_tensor = render_pkg["viewspace_points"]\n'
            '        visibility_filter = render_pkg["visibility_filter"]\n',
            '        visibility_filter = render_pkg["radii"] > 0\n')
    start = source.index('            from utils.hybrid30k_losses import compose_hybrid_loss')
    stop = source.index('        else:\n', source.index('            dynamic_alpha = dynamic_alpha * dynamic_mask', start))
    replace(source[start:stop], """            from utils.hybrid30k_runtime import record_step
            terms = _ownership.terms(render_pkg, evidence, gt_image, viewpoint_cam.pts_depth,
                                     sky_mask, dynamic_mask, args, iteration)
            if _growth_probe is not None:
                _growth_probe.prepare(terms, render_pkg, evidence, gt_image, dynamic_mask,
                                      args, iteration, gaussians, dynamic_gaussians, env_map)
            if iteration in (1, 1000):
                _ownership.audit_gradients(terms, gaussians, dynamic_gaussians, render_pkg,
                                           args, iteration)
            loss = sum(terms.values())
            log_dict.update({name: value.item() for name, value in terms.items()})
            log_dict['loss_instance_total'] = loss.item()
            record_step(args, iteration, viewpoint_cam, time_shift, gaussians, dynamic_gaussians, log_dict)
""")
    replace('        loss.backward()\n', """        loss.backward(retain_graph=bool(_growth_probe is not None and _growth_probe.needs_main_graph))
        if _growth_probe is not None:
            _growth_probe.after_backward(terms, render_pkg, evidence, gt_image, dynamic_mask,
                                         args, iteration)
""")
    start = source.index('            cur_viewspace_point_tensor = slice_with_grad(viewspace_point_tensor, 0, end_idx)')
    stop = source.index('            gaussians.optimizer.step()', start)
    replace(source[start:stop], """            static_xy = render_pkg['static_growth']
            if _growth_probe is not None:
                static_xy = _growth_probe.statistics_tensor(render_pkg, iteration)
            static_radii = render_pkg['static_radii']
            static_t_grad = gaussians._t.grad
            gaussians._t.grad = render_pkg['static_growth_t'].grad
            static_time_parameter = gaussians._t
            try:
                densification_and_optimization(gaussians, args, iteration, static_xy,
                                               static_radii > 0, scene, static_radii, dynamic=False)
            finally:
                # Native topology changes replace Parameter objects and, as in
                # the control, those new objects receive no old step gradient.
                if gaussians._t is static_time_parameter:
                    gaussians._t.grad = static_t_grad
            dynamic_radii = render_pkg['dynamic_radii']
            densification_and_optimization(dynamic_gaussians, args, iteration,
                                           render_pkg['dynamic_growth'], dynamic_radii > 0,
                                           scene, dynamic_radii, dynamic=True)

""")
    # Only replace the training function so evaluator, optimizer and controller
    # hooks retain their native definitions.
    a = source.index('def training(args):')
    b = source.index('\ndef slice_with_grad', a)
    executed = out / 'executed_training_function.py'
    executed.write_text(source[a:b])
    train.__dict__['_ownership'] = objective
    train.__dict__['_growth_probe'] = growth_probe
    exec(compile(source[a:b], str(executed), 'exec'), train.__dict__)
