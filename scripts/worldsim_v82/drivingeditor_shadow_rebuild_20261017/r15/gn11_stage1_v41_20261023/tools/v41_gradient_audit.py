"""Read-only gradient diagnostics, separate from optimizer updates and stochastic training."""
import torch

AUDIT_FRAMES = ('0008102', '0008129', '0008137')
ACTORS = (5, 26, 18, 6, 37)


def gradient_l1(loss, parameters):
    if not loss.requires_grad:
        return dict(dc=0., opacity=0.)
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return {key:float(g.abs().sum()) if g is not None else 0.
            for key,g in zip(('dc','opacity'),grads)}


def domain_gradient_audit(package, reference, labelled_domains, full_domains, parameters, target=.25):
    result = {}
    contribution = package['road_visible_alpha'][0]
    for i,actor in enumerate(ACTORS):
        labelled, full = labelled_domains[i], full_domains[i]
        n, total = int(labelled.sum()), int(full.sum())
        row = dict(labelled_pixels=n, target_pixels=total,
            color_status='NO_REFERENCE', rgb_loss=None, rgb_gradient_l1=None,
            labelled_contribution=None, contribution_gradient_l1=None,
            hinge_loss=None, hinge_gradient_l1=None,
            full_contribution=None, full_contribution_gradient_l1=None)
        if total:
            mean = contribution[full].mean()
            row.update(full_contribution=float(mean), full_contribution_gradient_l1=gradient_l1(mean,parameters))
        if n:
            rgb = (package['rgb'][:,labelled]-reference[:,labelled]).abs().mean()
            mean = contribution[labelled].mean()
            hinge = torch.relu(target-contribution[labelled]).square().mean()
            row.update(color_status='MEASURED_PATH',rgb_loss=float(rgb),
                rgb_gradient_l1=gradient_l1(rgb,parameters), labelled_contribution=float(mean),
                contribution_gradient_l1=gradient_l1(mean,parameters),
                hinge_loss=float(hinge),hinge_gradient_l1=gradient_l1(hinge,parameters))
        result[str(actor)] = row
    return result


def a37_path_failures(audits):
    failures = []
    for frame in AUDIT_FRAMES:
        if frame not in audits:
            failures.append(frame+':missing audit'); continue
        row = audits[frame]['37']
        if not row['labelled_pixels']:
            if frame != '0008137': failures.append(frame+':a37 expected labelled hard region missing')
            continue  # raw137 has no label: explicitly unknown, never a color-path pass.
        if row['rgb_loss'] > 1e-7 and row['rgb_gradient_l1']['dc'] <= 0:
            failures.append(frame+':a37 color residual but no DC gradient')
        if row['hinge_loss'] > 1e-10 and row['hinge_gradient_l1']['opacity'] <= 0:
            failures.append(frame+':a37 contribution deficit but no opacity gradient')
    return failures
