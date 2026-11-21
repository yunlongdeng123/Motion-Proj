"""Surface-bound a20 student; CPU import is default, GPU phases require bound gates.

Reuses the proven R2 renderer/trajectory. Geometry, normals and scales stay fixed.
Teacher-gradient isolation is an interface repair, independent of prior quality.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback

RUN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_completion_prior_20261122_r1')
LEGACY = Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1/code/frozen_formal_r2/run_joint.py')
SIDECAR = Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1/assets/handoff_fixed_guarded/a9_ownership_sidecar.npz')
ROAD = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2/appearance_fit/asset')
FRAMES = (86, 110, 134, 179, 198)
TRAIN = (86, 110)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def reference(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size)


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    q.replace(p)


def load_legacy(path, student_root):
    """Import definitions only, then redirect ROOT before constructing anything."""
    spec = importlib.util.spec_from_file_location('_surface_student_frozen_r2', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    old_root = module.ROOT.resolve()
    root = Path(student_root).resolve()
    if root == old_root or old_root in root.parents:
        raise ValueError('student ROOT must not be within the old run')
    module.ROOT = root
    assert tuple(module.TRAIN) == TRAIN and tuple(module.FRAMES) == FRAMES
    return module


def read_array_bank(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def validate_surface(path, expected_rows=25000):
    import numpy as np
    a = read_array_bank(path); n = len(a['xyz'])
    shapes = dict(xyz=(n, 3), scales=(n, 3), rotation=(n, 4), coefficients=(n, 16, 3),
                  opacity=(n,), surface_face_id=(n,), surface_barycentric=(n, 3),
                  surface_normal_actor=(n, 3), surface_uv=(n, 2),
                  surface_rgb_generated_srgb=(n, 3), is_observed=(n,), is_generated_prior=(n,))
    if n != expected_rows:
        raise ValueError(f'Expected {expected_rows} surface Gaussians, got {n}')
    for key, shape in shapes.items():
        if key not in a or a[key].shape != shape:
            raise ValueError(f'Surface field {key} must have shape {shape}')
        if a[key].dtype.kind in 'fci' and not np.isfinite(a[key]).all():
            raise ValueError('Nonfinite surface field: ' + key)
    if (a['scales'] <= 0).any() or ((a['opacity'] <= 0) | (a['opacity'] >= 1)).any():
        raise ValueError('Invalid initial Gaussian sigma/opacity')
    if np.asarray(a['is_observed']).any() or not np.asarray(a['is_generated_prior']).all():
        raise ValueError('Generated geometry provenance must remain explicit')
    if not np.allclose(a['surface_barycentric'].sum(1), 1, atol=2e-5):
        raise ValueError('Invalid surface barycentric coordinates')
    return a


def check_readiness(path, surface, phase=None):
    gate = json.loads(Path(path).read_text())
    if gate.get('shape_gate_passed') is not True or gate.get('material_gate_passed') is not True:
        raise ValueError('Both shape and material gates must pass before GPU construction')
    if gate.get('surface_npz_sha256') != digest(surface):
        raise ValueError('Readiness receipt is not bound to this surface NPZ')
    requested=gate.get('requested_phases')
    needed=['preview','smoke','formal'] if phase=='all' else [phase]
    if requested is not None and any(item not in requested for item in needed):
        raise ValueError('This entry receipt does not authorize the requested phase: '+str(phase))
    return reference(path)


def create_classes(m, args):
    np, torch, F = m.np, m.torch, m.F

    class SurfaceVehicle(m.Vehicle):
        def __init__(self, a):
            super().__init__(a)
            self.metadata = {k: np.array(v, copy=True) for k, v in a.items()}
            n = len(a['xyz'])
            support = np.asarray(a.get('train_visible_support', np.zeros(n, bool)), bool)
            self.register_buffer('visible_support', m.ten(support, torch.bool))
            self.register_buffer('base_opacity_logits', torch.logit(self.base_opacity.clamp(1e-5, 1-1e-5)))
            self.log_scale_delta.requires_grad_(False)

        def coeff(self):
            dc = self.base_coefficients[:, 0] + self.visible_support[:, None] * (self.dc-self.base_coefficients[:, 0])
            return torch.cat([dc[:, None], self.base_coefficients[:, 1:]], 1)

        def scales(self):
            return self.base_scales

        def opacity(self):
            delta = args.opacity_delta_max * (self.opacity_logits-self.base_opacity_logits).tanh()
            return (self.base_opacity + self.visible_support * delta).clamp(1e-5, 1-1e-5)

        def rows(self, pose, cam):
            rows = super().rows(pose, cam)
            rows['opacities'] = self.opacity()[:, None]
            return rows

        def export(self):
            a = {k: np.array(v, copy=True) for k, v in self.metadata.items()}
            a.update(xyz=m.cpu(self.base_xyz).numpy(), scales=m.cpu(self.base_scales).numpy(),
                     rotation=m.cpu(self.base_rotation).numpy(), coefficients=m.cpu(self.coeff()).numpy(),
                     opacity=m.cpu(self.opacity()).numpy(), train_visible_support=m.cpu(self.visible_support).numpy(),
                     visibility_evaluated=np.asarray(True), opacity_delta_max=np.asarray(args.opacity_delta_max),
                     appearance_adapted_from_real_build=np.asarray(True), geometry_optimized=np.asarray(False))
            return a

    m.Vehicle = SurfaceVehicle

    class SurfaceExperiment(m.Experiment):
        def __init__(self):
            self.source_surface = Path(args.surface_npz).resolve()
            self.surface = validate_surface(self.source_surface)
            self.surface_ref = reference(self.source_surface)
            self.sidecar_ref = reference(args.sidecar)
            self.frozen_sidecar = read_array_bank(args.sidecar)
            self.sidecar_applied = False
            self.gradient_isolation = []
            self.input_domains = {}
            self.base_binding = dict(road=reference(Path(args.road_source)/'road.npz'),
                                     manifest=reference(Path(args.road_source)/'persistent/scene.json'),
                                     proof=reference(Path(args.road_source)/'persistent/proof.json'))
            init = m.ROOT/'initial_vehicle_r2.npz'
            if init.exists() and digest(init) != self.surface_ref['sha256']:
                raise ValueError('New ROOT already has a different initial surface; use a new student directory')
            super().__init__(Path(args.road_source))
            if len(self.road_train_ids) != 1191:
                raise ValueError('Expected exactly the inherited 1191 approved road color rows')
            self.pose_initial = {raw: m.cpu(p) for raw, p in self.poses.items()}
            self.derive_visible_support()
            self.original_state = m.cpu(self.vehicle.state_dict())
            m.write(m.ROOT/'surface_source.json', dict(surface=self.surface_ref, legacy=reference(args.legacy),
                    base_binding=self.base_binding, sidecar=self.sidecar_ref, rows=len(self.surface['xyz']),
                    no_native_initialization=True, support='actual unified KEEP feature VJP over BUILD real visible core',
                    teacher_gradient_isolation='interface repair; not independent prior-model gain'))

        def build_vehicle(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.source_surface, path)
            if digest(path) != self.surface_ref['sha256']:
                raise RuntimeError('Surface bootstrap copy mismatch')

        def apply_fixed_sidecar(self):
            if self.sidecar_applied:
                return
            ns, nd = len(self.scene.static.get_xyz), len(self.scene.dynamic.get_xyz)
            side = self.frozen_sidecar
            ids = np.asarray(side['global_row_ids'], np.int64)
            if len(ids) != 75 or len(np.unique(ids)) != 75 or (ids < 0).any() or (ids >= ns+nd).any():
                raise ValueError('Expected the exact fixed 75-row native a9 sidecar')
            if int(side['native_static_rows']) != ns or int(side['native_dynamic_rows']) != nd:
                raise ValueError('Frozen sidecar/native topology mismatch')
            selected = m.ten(ids, torch.long); receipts = []
            for raw in FRAMES:
                with torch.no_grad():
                    values, original_owners, _, _ = self.scene._rows(raw, (m.TARGET,), (), False)
                    global_ids = torch.nonzero(values['mask']).reshape(-1)
                for container, packet_ids in [(self.native_packets, global_ids),
                                              (self.packets, global_ids[original_owners[global_ids] != m.TARGET])]:
                    rows, owners, ri = container[raw]
                    if not torch.equal(values['means3D'][packet_ids], rows['means3D']):
                        raise RuntimeError('Exact native global row mapping mismatch')
                    match = torch.isin(packet_ids, selected)
                    if ((owners[match] > 0) & (owners[match] != m.DELETE)).any():
                        raise RuntimeError('Frozen sidecar would alter another retained actor')
                    owner_copy = owners.clone(); owner_copy[match] = m.DELETE
                    container[raw] = (rows, owner_copy, ri)
                receipts.append(dict(raw=raw, matched_rows=int(torch.isin(global_ids, selected).sum())))
            self.sidecar_applied = True
            m.write(m.ROOT/'a9_fixed_sidecar_receipt.json', dict(source=self.sidecar_ref, rows=75,
                    reevaluated_or_selected=False, applied_to_native_and_surface_packets=True, per_frame=receipts))

        def build_domains(self):
            self.apply_fixed_sidecar()
            super().build_domains()
            # Keep legacy comparison masks, but train only conservative real V and known B.
            for raw in TRAIN:
                folder = Path(args.conditioning)/'train'/f'{raw:03d}'
                meta = m.read(folder/'mapping.json'); w, h = meta['image_native_size']
                x0, y0, x1, y1 = meta['crop_native_xyxy_half_open']
                masks = {}
                for name in ['visible_mask', 'unknown_potential_mask', 'known_background_mask']:
                    default = np.ones((h, w), np.uint8)*255 if name == 'known_background_mask' else np.zeros((h, w), np.uint8)
                    crop = np.asarray(m.Image.open(folder/(name+'.png')).convert('L'))
                    default[y0:y1, x0:x1] = crop
                    masks[name] = m.ten(np.asarray(m.Image.fromarray(default).resize((1152,768), m.Image.Resampling.NEAREST)) > 0, torch.bool)
                visible, unknown, background = [masks[k] for k in ['visible_mask','unknown_potential_mask','known_background_mask']]
                if (visible & unknown).any() or (visible & background).any() or (unknown & background).any():
                    raise ValueError('Real tri-state domains overlap')
                self.targets[raw]['loss_mask'] = visible
                self.targets[raw]['negative_alpha_valid'] = background
                self.domains[raw]['legal'] &= ~(visible | unknown)
                self.domains[raw]['teacher'] = self.domains[raw]['teacher'].detach()
                self.input_domains[raw] = {name:reference(folder/(name+'.png')) for name in
                                           ['visible_mask','unknown_potential_mask','known_background_mask']}
                p = m.ROOT/'delivery/references'/f'{raw:03d}'
                m.png(p/'student_real_visible_core.png',visible); m.png(p/'student_unknown_ignore.png',unknown)
                m.png(p/'vehicle_negative_valid.png',background); m.png(p/'road_legal.png',self.domains[raw]['legal'])
            m.write(m.ROOT/'surface_supervision_domains.json',dict(train=list(TRAIN), diagnostic_only=[134,179,198],
                    source=self.input_domains, RGB='original observed RGB only, never amodal/teacher',
                    positive='conservative observed V',negative='known B only; all U ignored',
                    road_teacher='inherited A/C/S legal domain additionally excluding V and U'))

        def render(self, raw, mode='keep', branch='full', native=False, *, detach_road=False,
                   detach_vehicle=False, support_weights=None):
            cam = self.cameras[raw]
            packet, owners, road_index = (self.native_packets if native else self.packets)[raw]
            rows = dict(packet)
            if branch == 'road':
                choose = road_index >= 0; rows=m.subset(packet,choose); road_index=road_index[choose]; owners=owners[choose]
            elif branch == 'vehicle' and native:
                choose=owners==m.TARGET; rows=m.subset(packet,choose); road_index=road_index[choose]; owners=owners[choose]
            if branch != 'vehicle':
                ids=torch.nonzero(road_index>=0).reshape(-1)
                colors=self.colors()
                if detach_road:
                    colors=colors.detach()
                rows=dict(rows,colors_precomp=rows['colors_precomp'].index_copy(0,ids,colors[road_index[ids]]))
            if not native and branch != 'road':
                car=self.vehicle.rows(self.poses[raw],cam)
                if support_weights is not None:
                    feature=car['features'].clone(); feature[:,2]=support_weights
                    car=dict(car,features=feature)
                if detach_vehicle:
                    car={k:v.detach() if torch.is_tensor(v) else v for k,v in car.items()}
                    if any(torch.is_tensor(v) and v.requires_grad for v in car.values()):
                        raise RuntimeError('Incomplete vehicle packet detach')
                if branch == 'vehicle':
                    rows=car; owners=torch.full((len(car['means3D']),),m.TARGET,device='cuda',dtype=torch.long)
                else:
                    rows=self.scene._bindings().merge_rows(rows,car)
                    owners=torch.cat([owners,torch.full((len(car['means3D']),),m.TARGET,device='cuda',dtype=torch.long)])
            if mode == 'delete' and branch == 'full':
                rows=dict(rows,mask=rows['mask'] & (owners != m.DELETE))
            out=self.scene._bindings().raster(cam,self.scene.pipe,0,rows,self.scene._background)
            self.native_calls+=1
            result=dict(rgb=out['rgb'],alpha=out['alpha'].squeeze(),vehicle_alpha=out['extra'][0],road_alpha=out['extra'][1])
            if branch == 'full':
                result['rgb']=result['rgb']+(1-out['alpha'])*self.env[raw]
            return result

        def derive_visible_support(self):
            weights=torch.ones(len(self.vehicle.base_xyz),device='cuda',requires_grad=True)
            contributions=[]
            for raw in TRAIN:
                out=self.render(raw,detach_road=True,support_weights=weights)
                score=out['vehicle_alpha'][self.targets[raw]['loss_mask']].sum()
                g=torch.autograd.grad(score,weights)[0].detach()
                if not torch.isfinite(g).all() or g.min() < -1e-7:
                    raise RuntimeError('Invalid common-raster support contribution')
                contributions.append(g.clamp_min(0))
            total=torch.stack(contributions).sum(0); support=total>1e-8
            if not support.any():
                raise RuntimeError('No surface Gaussian reaches any real BUILD visible core')
            self.vehicle.visible_support.copy_(support)
            np.savez_compressed(m.ROOT/'visible_support.npz',train_raw=np.asarray(TRAIN),
                    contribution_by_raw=m.cpu(torch.stack(contributions)).numpy(),
                    actual_visible_contribution=m.cpu(total).numpy(),
                    train_visible_support=m.cpu(support).numpy(), source_surface_sha256=np.asarray(self.surface_ref['sha256']))
            m.write(m.ROOT/'visible_support.json',dict(method='actual common KEEP renderer feature VJP',
                    threshold=1e-8,raw=list(TRAIN),supported_rows=int(support.sum()),total_rows=len(support),
                    generated_alpha_used=False,geometric_center_mask_used=False,holdout_used=False))

        def optimizer(self,stage):
            self.vehicle.dc.requires_grad_(True)
            self.vehicle.opacity_logits.requires_grad_(stage=='vehicle')
            self.vehicle.log_scale_delta.requires_grad_(False)
            # Remains a gradient target in vehicle stage solely to prove cross-loss isolation.
            self.road_color.requires_grad_(True)
            groups=[dict(params=[self.vehicle.dc],lr=.01)]
            if stage=='vehicle': groups.append(dict(params=[self.vehicle.opacity_logits],lr=.01))
            elif stage=='joint': groups.append(dict(params=[self.road_color],lr=.005))
            else: raise ValueError(stage)
            return torch.optim.Adam(groups,betas=(.9,.999),eps=1e-8)

        @staticmethod
        def grad_max(g):
            return 0. if g is None else float(g.detach().abs().max())

        def step(self,stage,step,opt,role):
            raw=TRAIN[step%len(TRAIN)]
            self.vehicle.zero_grad(set_to_none=True); self.road_color.grad=None; opt.zero_grad(set_to_none=True)
            target=self.targets[raw]; mask=target['loss_mask']
            keep=self.render(raw,detach_road=True); alpha=keep['vehicle_alpha']
            rgb=m.l1(keep['rgb'],target['rgb'],mask)
            ssim=(1-m.masked_mean(m.ssim_map(keep['rgb'],target['rgb']),mask))/2
            ain=m.masked_mean((alpha-1).square(),mask)
            aout=m.masked_mean(alpha.square(),target['negative_alpha_valid'])
            vehicle_loss=.8*rgb+.2*ssim+.1*ain+.1*aout
            active_vehicle=[('vehicle_dc',self.vehicle.dc)]
            if self.vehicle.opacity_logits.requires_grad:
                active_vehicle.append(('vehicle_opacity',self.vehicle.opacity_logits))
            vp=[p for _,p in active_vehicle]
            if not torch.isfinite(vehicle_loss): raise FloatingPointError('nonfinite vehicle loss')
            # One VJP per actual loss both trains its branch and audits the other branch.
            vg=torch.autograd.grad(vehicle_loss,vp+[self.road_color],allow_unused=True)
            self.backward_calls+=1
            if self.grad_max(vg[-1]) != 0:
                raise RuntimeError('Lvehicle leaked into road_color')
            if self.grad_max(vg[0]) == 0:
                raise RuntimeError('Lvehicle active color branch disconnected')
            for (_,p),g in zip(active_vehicle,vg[:-1]): p.grad=None if g is None else g.detach()
            road_loss=vehicle_loss.detach()*0
            rg=None
            if stage=='joint':
                deleted=self.render(raw,'delete',detach_vehicle=True)
                legal=self.domains[raw]['legal']; teacher=self.domains[raw]['teacher'].detach()
                road_loss=m.l1(deleted['rgb'],teacher,legal)+.1*m.edge(deleted['rgb'],teacher,legal)
                if not torch.isfinite(road_loss): raise FloatingPointError('nonfinite road loss')
                rg=torch.autograd.grad(road_loss,[self.road_color]+vp,allow_unused=True)
                self.backward_calls+=1
                if any(self.grad_max(g)!=0 for g in rg[1:]):
                    raise RuntimeError('Lroad leaked into vehicle parameters')
                if self.grad_max(rg[0])==0: raise RuntimeError('Lroad active road branch disconnected')
                self.road_color.grad=rg[0].detach()
            grads=dict(vehicle_dc=self.grad_max(self.vehicle.dc.grad),
                       vehicle_opacity=self.grad_max(self.vehicle.opacity_logits.grad), vehicle_scale=0.,
                       road_color=self.grad_max(self.road_color.grad))
            for _,p in active_vehicle:
                if p.grad is not None and not torch.isfinite(p.grad).all(): raise FloatingPointError('nonfinite vehicle gradient')
            if self.road_color.grad is not None and not torch.isfinite(self.road_color.grad).all():
                raise FloatingPointError('nonfinite road gradient')
            audit=dict(role=role,stage=stage,step=step+1,raw=raw,
                       Lvehicle_to_road_max=self.grad_max(vg[-1]), Lvehicle_to_vehicle_dc_max=self.grad_max(vg[0]),
                       Lroad_to_vehicle_max=max([self.grad_max(g) for g in rg[1:]],default=0.) if rg else None,
                       Lroad_to_road_max=self.grad_max(rg[0]) if rg else None,
                       road_detached_in_vehicle_loss=True,complete_vehicle_packet_detached_in_road_loss=stage=='joint')
            self.gradient_isolation.append(audit)
            before={name:p.detach().clone() for name,p in [('vehicle_dc',self.vehicle.dc),('road_color',self.road_color)]}
            opt.step()
            with torch.no_grad(): self.road_color.clamp_(0,1)
            updates={name:float((p-before[name]).abs().max()) for name,p in [('vehicle_dc',self.vehicle.dc),('road_color',self.road_color)]}
            if updates['vehicle_dc']<=0 or (stage=='joint' and updates['road_color']<=0):
                raise RuntimeError('Active branch did not update')
            if stage=='vehicle' and updates['road_color']!=0: raise RuntimeError('Road updated during vehicle stage')
            record=dict(role=role,stage=stage,step=step+1,raw=raw,loss=float(vehicle_loss.detach()+road_loss.detach()),
                        vehicle_rgb=float(rgb.detach()),vehicle_dssim=float(ssim.detach()),alpha_inside=float(ain.detach()),
                        alpha_outside=float(aout.detach()),road=float(road_loss.detach()),gradient_max=grads,
                        update_max=updates,gradient_isolation=audit,elapsed=time.monotonic()-self.started)
            self.records.append(record)
            return record

        def snapshot(self,stage,step,opt,role):
            result=super().snapshot(stage,step,opt,role)
            result.update(schema='a20.surface_student.fullstate.v1',surface_binding=self.surface_ref,
                          road_base_binding=self.base_binding,sidecar_binding=self.sidecar_ref)
            return result

        def restore(self,state,opt):
            if state.get('surface_binding')!=self.surface_ref or state.get('sidecar_binding')!=self.sidecar_ref:
                raise RuntimeError('Checkpoint surface/sidecar binding mismatch')
            if state.get('road_base_binding')!=self.base_binding: raise RuntimeError('Checkpoint road base mismatch')
            return super().restore(state,opt)

        def asset(self,stage):
            p=m.ROOT/'assets'/stage; p.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(p/'vehicle.npz',**self.vehicle.export())
            np.savez_compressed(p/'poses.npz',raw=np.asarray(FRAMES),
                                model_from_actor=np.stack([m.cpu(self.poses[r]).numpy() for r in FRAMES]))
            np.savez_compressed(p/'road_rgb_patch.npz',world_row_ids=self.road_train_ids,
                                reference_srgb=m.cpu(self.road_color).numpy(),
                                base_road_sha256=np.asarray(self.base_binding['road']['sha256']))
            shutil.copy2(args.sidecar,p/'a9_ownership_sidecar.npz')
            m.write(p/'scene_binding.json',dict(schema='a20.surface_student.compact_asset.v1',stage=stage,
                    vehicle=str(p/'vehicle.npz'),poses=str(p/'poses.npz'),road_rgb_patch=str(p/'road_rgb_patch.npz'),
                    base_binding=self.base_binding,source_surface=self.surface_ref,sidecar=self.sidecar_ref,
                    ownership_sidecar=str(p/'a9_ownership_sidecar.npz'),target_actor=m.TARGET,target_track=m.TRACK,
                    delete_actor=m.DELETE,loader=str(Path(__file__).resolve()),
                    full_poses_base='/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1/assets/handoff_fixed_guarded/poses_full.npz',
                    row_count=len(self.vehicle.base_xyz),surface_metadata_preserved=True,road_full_copy_written=False,
                    readback='Construct from pinned base, apply exact 1191 world-row RGB patch, fixed a9 sidecar, then load vehicle and poses.'))
            return p

        def reload_compact_asset(self,path):
            path=Path(path); binding=m.read(path/'scene_binding.json')
            if binding['base_binding']!=self.base_binding: raise RuntimeError('Compact road base binding mismatch')
            if digest(path/'a9_ownership_sidecar.npz')!=self.sidecar_ref['sha256']:
                raise RuntimeError('Compact asset sidecar mismatch')
            self.vehicle=SurfaceVehicle(m.load(path/'vehicle.npz')).cuda()
            patch=m.load(path/'road_rgb_patch.npz')
            if not np.array_equal(patch['world_row_ids'],self.road_train_ids): raise RuntimeError('Road patch row mapping mismatch')
            if str(patch['base_road_sha256'])!=self.base_binding['road']['sha256']: raise RuntimeError('Road patch base mismatch')
            with torch.no_grad(): self.road_color.copy_(m.ten(patch['reference_srgb']))
            pose=m.load(path/'poses.npz')
            self.poses={int(r):m.ten(p) for r,p in zip(pose['raw'],pose['model_from_actor'])}

        def freeze_proof(self):
            result=super().freeze_proof()
            result.update(local_scales_equal=bool(torch.equal(self.vehicle.base_scales,m.ten(self.initial['scales']))),
                          local_scales_no_delta=bool((self.vehicle.log_scale_delta==0).all()),
                          all_surface_metadata_preserved=all(np.array_equal(v,self.vehicle.metadata[k]) for k,v in self.surface.items()
                                                           if k not in ['coefficients','opacity']),
                          hidden_color_equal=bool(torch.equal(self.vehicle.coeff()[~self.vehicle.visible_support],
                                                             self.vehicle.base_coefficients[~self.vehicle.visible_support])),
                          hidden_opacity_equal=bool(torch.equal(self.vehicle.opacity()[~self.vehicle.visible_support],
                                                               self.vehicle.base_opacity[~self.vehicle.visible_support])),
                          opacity_max_abs_delta=float((self.vehicle.opacity()-self.vehicle.base_opacity).abs().max()),
                          opacity_delta_bound=args.opacity_delta_max,
                          trajectory_equal=all(torch.equal(m.cpu(self.poses[r]),self.pose_initial[r]) for r in FRAMES),
                          fixed_a9_sidecar_sha256=self.sidecar_ref['sha256'],fixed_a9_rows=75,
                          teacher_gradient_isolation_all_recorded=all(x['Lvehicle_to_road_max']==0 and
                              (x['Lroad_to_vehicle_max'] in [None,0.]) for x in self.gradient_isolation),
                          separate_active_loss_gradients_nonzero=all(x['Lvehicle_to_vehicle_dc_max']>0 and
                              (x['Lroad_to_road_max'] is None or x['Lroad_to_road_max']>0) for x in self.gradient_isolation))
            if not all(result[k] for k in ['local_xyz_equal','local_rotation_equal','local_scales_equal','local_scales_no_delta',
                       'all_surface_metadata_preserved','hidden_color_equal','hidden_opacity_equal','trajectory_equal',
                       'teacher_gradient_isolation_all_recorded','separate_active_loss_gradients_nonzero']):
                raise RuntimeError('Surface/gradient freeze proof failed: '+str(result))
            return result

        def smoke(self):
            m.status('surface_smoke'); initial=m.cpu(self.vehicle.state_dict()); road_initial=m.cpu(self.road_color)
            opt=self.optimizer('vehicle')
            for i in range(4): self.step('vehicle',i,opt,'smoke')
            saved=self.snapshot('vehicle',4,opt,'smoke'); m.save(m.ROOT/'smoke/checkpoint.pt',saved)
            one=self.step('vehicle',4,opt,'smoke_replay'); state1=m.cpu(self.vehicle.state_dict()); opt1=m.cpu(opt.state_dict())
            self.restore(torch.load(m.ROOT/'smoke/checkpoint.pt',map_location='cpu',weights_only=False),opt)
            if m.differences(saved,self.snapshot('vehicle',4,opt,'smoke')):
                raise RuntimeError('Full checkpoint restore is not bit exact')
            two=self.step('vehicle',4,opt,'smoke_replay'); state2=m.cpu(self.vehicle.state_dict()); opt2=m.cpu(opt.state_dict())
            delta={k:float((state1[k]!=state2[k]).any()) if state1[k].dtype==torch.bool else
                   float((state1[k]-state2[k]).abs().max()) for k in state1}
            maximum=max(delta.values()); loss_delta=abs(one['loss']-two['loss'])
            continuation=dict(saved_state_restore_bit_exact=True,next_step_parameter_max=maximum,
                    next_step_parameter_differences=delta,next_step_loss_abs=loss_delta,
                    next_step_optimizer_bit_exact=not m.differences(opt1,opt2),
                    parameter_abs_tolerance=2e-5,loss_abs_tolerance=1e-6)
            m.write(m.ROOT/'smoke/continuation_numeric.json',continuation)
            if maximum>2e-5 or loss_delta>1e-6: raise RuntimeError('Smoke replay tolerance failed')
            self.export('smoke_vehicle'); compact=self.asset('smoke_vehicle')
            with torch.no_grad(): before=self.render(134,'delete')['rgb'].detach().clone()
            previous=self.vehicle
            self.reload_compact_asset(compact)
            with torch.no_grad(): after=self.render(134,'delete')['rgb'].detach()
            reload_difference=float((before-after).abs().max()); self.vehicle=previous
            if reload_difference>1e-5: raise RuntimeError('Compact surface asset reload render mismatch')
            optj=self.optimizer('joint')
            for i in range(3): self.step('joint',i,optj,'smoke')
            m.save(m.ROOT/'smoke/joint_checkpoint.pt',self.snapshot('joint',3,optj,'smoke'))
            proof=dict(passed=True,real_cuda=True,checkpoint_saved=True,optimizer_and_sampler_rng_restore=True,
                    evaluation=True,joint_next_stage=True,continuation=continuation,actual_smoke_optimizer_updates=9,
                    formal_updates=0,persistent_vehicle_render_reload_max=reload_difference,freeze=self.freeze_proof(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            m.write(m.ROOT/'smoke/proof.json',proof)
            self.vehicle.load_state_dict(initial)
            with torch.no_grad(): self.road_color.copy_(road_initial.cuda())
            m.write(m.ROOT/'gradient_isolation.json',dict(rows=self.gradient_isolation))
            return proof

        def formal(self,vehicle_steps,joint_steps):
            if vehicle_steps!=200 or joint_steps!=200:
                raise ValueError('Formal budget is fixed at 200 vehicle + 200 joint updates')
            if not (m.ROOT/'smoke/proof.json').exists() or not m.read(m.ROOT/'smoke/proof.json').get('passed'):
                raise RuntimeError('A completed GPU smoke is required before formal')
            super().formal(vehicle_steps,joint_steps)
            m.write(m.ROOT/'gradient_isolation.json',dict(rows=self.gradient_isolation,
                    teacher_detach_is_interface_repair_not_prior_effect=True))

    return SurfaceVehicle, SurfaceExperiment


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['cpu-import','preview','smoke','formal','all','reload'],default='cpu-import')
    ap.add_argument('--student-root',type=Path,default=RUN/'student')
    ap.add_argument('--legacy',type=Path,default=LEGACY)
    ap.add_argument('--surface-npz',type=Path)
    ap.add_argument('--gate-receipt',type=Path)
    ap.add_argument('--conditioning',type=Path,default=RUN/'conditioning')
    ap.add_argument('--road-source',type=Path,default=ROAD)
    ap.add_argument('--sidecar',type=Path,default=SIDECAR)
    ap.add_argument('--opacity-delta-max',type=float,default=.1)
    ap.add_argument('--reload-asset',type=Path)
    args=ap.parse_args()
    if not 0<args.opacity_delta_max<=.1: raise ValueError('Opacity bound must be in (0,0.1]')
    if args.phase=='cpu-import': os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    m=load_legacy(args.legacy,args.student_root)
    _,Experiment=create_classes(m,args)
    if args.phase=='cpu-import':
        if m.torch.cuda.is_initialized(): raise RuntimeError('CPU import initialized CUDA')
        report=dict(schema='a20.surface_student.cpu_import.v1',passed=True,phase=args.phase,
                    GPU_initialized=False,Experiment_instantiated=False,legacy_root_redirected=str(m.ROOT),
                    legacy_source=reference(args.legacy),script=reference(__file__),
                    classes=['SurfaceVehicle','SurfaceExperiment'],formal_budget=[200,200],
                    GPU_launch_authorized=False)
        if args.surface_npz: report['surface_rows']=len(validate_surface(args.surface_npz)['xyz'])
        write_json(m.ROOT/'cpu_import.json',report); print(json.dumps(report,indent=2)); return
    if args.surface_npz is None or args.gate_receipt is None:
        raise ValueError('GPU phases require --surface-npz and --gate-receipt')
    readiness=check_readiness(args.gate_receipt,args.surface_npz,args.phase)
    validate_surface(args.surface_npz)
    contract=dict(task='a20_surface_student_20261122_r1',phase=args.phase,readiness=readiness,student_script=reference(__file__),
        surface=reference(args.surface_npz),legacy_source=reference(args.legacy),root=str(m.ROOT),
        train=list(TRAIN),diagnostic_only=[134,179,198],vehicle_steps=200,joint_steps=200,seed=101,
        trainable=dict(vehicle=['supported DC','supported bounded opacity'],joint=['supported DC','1191 road RGB']),
        frozen=['geometry','normals','scales','rotation','higher SH','poses','all other actors','native scene'],
        vehicle_loss=dict(L1=.8,DSSIM=.2,alpha_inside=.1,alpha_outside=.1),road_loss=dict(L1=1.,edge=.1),
        unknown_is_negative=False,generated_RGB_loss=False,opacity_delta_bound=args.opacity_delta_max,
        teacher_gradient_isolation='new interface repair; not isolated prior-model gain',sidecar=reference(args.sidecar),
        roadmap='new assets contain full surface metadata and compact 1191-row RGB patch with pinned base',
        GPU_keep_on=True,power_operations=False)
    m.write(m.ROOT/'contract.json',contract)
    m.torch.cuda.reset_peak_memory_stats()
    e=Experiment()
    if args.phase in ['preview','all']:
        e.export('native_before',native=True); e.export('canonical_prior_init')
        if args.phase=='preview':
            m.write(m.ROOT/'status.json',dict(state='preview_complete',complete=False)); return
    if args.phase in ['smoke','all']: e.smoke()
    if args.phase in ['formal','all']: e.formal(200,200)
    if args.phase=='reload':
        if not args.reload_asset: raise ValueError('--reload-asset is required')
        e.reload_compact_asset(args.reload_asset); e.export('joint_fit_reloaded')
        rows=[]
        for raw in FRAMES:
            a=m.load(m.ROOT/'delivery/joint_fit'/f'{raw:03d}'/'render_float.npz')
            b=m.load(m.ROOT/'delivery/joint_fit_reloaded'/f'{raw:03d}'/'render_float.npz')
            rows.append(dict(raw=raw,maximum={k:float(abs(a[k]-b[k]).max()) for k in a}))
        passed=all(v<2e-5 for row in rows for v in row['maximum'].values())
        m.write(m.ROOT/'disk_reload_proof.json',dict(passed=passed,fresh_process=True,rows=rows))
        if not passed: raise RuntimeError('Fresh-process compact asset reload mismatch')
    m.write(m.ROOT/f'{args.phase}_resource.json',dict(phase=args.phase,
            peak_allocated_bytes=m.torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=m.torch.cuda.max_memory_reserved(),formal_updates=sum(x['role']=='formal' for x in e.records),
            native_calls=e.native_calls,parameter_gradient_VJP_calls=e.backward_calls,
            GPU_keep_on=True,power_operations=0))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        # No old ROOT is consulted in the exception path.
        traceback.print_exc()
        raise
