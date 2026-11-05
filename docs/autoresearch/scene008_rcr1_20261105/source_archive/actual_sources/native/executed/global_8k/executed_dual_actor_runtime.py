"""Two independently editable actor parameter sets packed in the native dynamic model.

No new renderer, motion model, loss, or SGCM rule. Native row-wise parameters and
Adam remain packed; permanent initial-source IDs preserve actor membership under
the existing parent-aware clone/split/prune. Actor 6 is transferred, never copied
into both branches. The input to transfer is an audited external assignment.
"""
from pathlib import Path
from contextlib import contextmanager
import hashlib, json
import numpy as np
import torch

STATIC_INITIAL_ROWS = 591672
A37_INITIAL_ROWS = 98452
ROW_FIELDS = ('_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity','_t','_scaling_t','_velocity')

def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def install_identity_serialization():
    from scene.gaussian_model import GaussianModel, Dynamic_GaussianModel
    # Identical 18-field schema; base implementation adds/restores existing
    # sgcm_lineage_v1 metadata before loading plain Adam. No dynamic SGCM enabled.
    Dynamic_GaussianModel.capture = GaussianModel.capture
    Dynamic_GaussianModel.restore = GaussianModel.restore

def actor_ids(dynamic):
    ids = getattr(dynamic, '_sgcm_lineage_ids', None)
    if ids is None or ids.shape != (len(dynamic.get_xyz),):
        raise RuntimeError('Dynamic checkpoint lacks persistent per-source ownership')
    if bool(((ids < 0) | (ids >= STATIC_INITIAL_ROWS + A37_INITIAL_ROWS)).any()):
        raise RuntimeError('Unknown dynamic source ID')
    return torch.where(ids < STATIC_INITIAL_ROWS, 6, 37)

def validate_partition(static, dynamic, moved_ids):
    device=dynamic.get_xyz.device
    moved_ids=torch.as_tensor(moved_ids,device=device,dtype=torch.int64)
    allowed=torch.zeros(STATIC_INITIAL_ROWS,dtype=torch.bool,device=device)
    allowed[moved_ids]=True
    s=static._sgcm_lineage_ids; d=dynamic._sgcm_lineage_ids
    assert not bool(allowed[s].any()), 'Transferred a6 source survived in static'
    a6=d<STATIC_INITIAL_ROWS
    assert bool(a6.any()) and bool((~a6).any()), 'An actor lost all support'
    assert bool(allowed[d[a6]].all()), 'Unassigned static source moved into dynamic'
    return dict(static_rows=len(s), a6_dynamic_rows=int(a6.sum()), a37_dynamic_rows=int((~a6).sum()),
                duplicate_a6_source_in_static=False, actor_ids=[6,37])

def transfer_initial_rows(scene, indices, output):
    from torch import nn
    static,dynamic=scene.gaussians,scene.dynamic_gaussians
    assert len(static.get_xyz)==STATIC_INITIAL_ROWS and len(dynamic.get_xyz)==A37_INITIAL_ROWS
    indices=np.asarray(indices,dtype=np.int64)
    assert len(indices)>0 and len(np.unique(indices))==len(indices)
    assert (indices>=0).all() and (indices<STATIC_INITIAL_ROWS).all()
    ids=torch.as_tensor(np.sort(indices),dtype=torch.int64,device=static.get_xyz.device)
    keep=torch.ones(STATIC_INITIAL_ROWS,dtype=torch.bool,device=ids.device);keep[ids]=False
    checks={}
    # Scene has built the original parameters in the original RNG order. Move
    # those values before optimizer creation; do not rerun KNN initialization.
    for key in ROW_FIELDS:
        original_s=getattr(static,key).detach(); original_d=getattr(dynamic,key).detach()
        expected_s=original_s[keep]; moved=original_s[ids]
        merged=torch.cat((original_d,moved),dim=0)
        assert torch.equal(merged[:A37_INITIAL_ROWS],original_d)
        assert torch.equal(merged[A37_INITIAL_ROWS:],moved)
        checks[key]=dict(a37_before_sha256=digest(original_d),a37_after_sha256=digest(merged[:A37_INITIAL_ROWS]),
                        remaining_static_sha256=digest(expected_s),moved_a6_sha256=digest(moved))
        setattr(static,key,nn.Parameter(expected_s.contiguous().requires_grad_(True)))
        setattr(dynamic,key,nn.Parameter(merged.contiguous().requires_grad_(True)))
    static.max_radii2D=static.max_radii2D[keep]
    dynamic.max_radii2D=torch.cat((dynamic.max_radii2D,torch.zeros(len(ids),device=ids.device)))
    static._sgcm_lineage_ids=torch.arange(STATIC_INITIAL_ROWS,device=ids.device,dtype=torch.int64)[keep]
    dynamic._sgcm_lineage_ids=torch.cat((torch.arange(A37_INITIAL_ROWS,device=ids.device,dtype=torch.int64)+STATIC_INITIAL_ROWS,ids))
    static._sgcm_clone_blocked=torch.zeros(len(static.get_xyz),device=ids.device,dtype=torch.bool)
    dynamic._sgcm_clone_blocked=torch.zeros(len(dynamic.get_xyz),device=ids.device,dtype=torch.bool)
    rec=validate_partition(static,dynamic,indices)
    assert rec['static_rows']+rec['a6_dynamic_rows']==STATIC_INITIAL_ROWS
    assert rec['a37_dynamic_rows']==A37_INITIAL_ROWS
    rec.update(status='PASS_EXACT_INITIAL_PARAMETER_TRANSFER',moved_initial_rows=len(indices),
               source_id_a37_offset=STATIC_INITIAL_ROWS,per_field=checks,
               packed_storage='Independent per-row actor parameters, persistent source-to-actor mapping; shared native optimizer and rasterizer')
    Path(output).write_text(json.dumps(rec,indent=2)+'\n')
    return rec

def install_union_masks(a6_body_mask_dir):
    import utils.paper_masks as pm
    original=pm.load_dynamic_mask
    root=Path(a6_body_mask_dir)
    def union(source_path,camera):
        a37=original(source_path,camera)
        a6=pm._read_mask(str(root/(Path(camera.image_path).stem+'.png')),camera.image_width,camera.image_height).to(a37.device)
        # Original a37 appearance supervision takes priority in overlaps. Union
        # is the original dynamic-vs-static objective with two foreground IDs.
        a6=a6*(a37==0)
        return torch.maximum(a37,a6)
    pm.load_dynamic_mask=union
    return original

@contextmanager
def hide_actors(dynamic, hidden_ids):
    ids=actor_ids(dynamic); selected=torch.zeros_like(ids,dtype=torch.bool)
    for actor in hidden_ids:
        if actor not in (6,37):raise ValueError(actor)
        selected|=ids==actor
    original=dynamic._opacity.detach().clone()
    try:
        with torch.no_grad():dynamic._opacity[selected]=-100.0
        yield
    finally:
        with torch.no_grad():dynamic._opacity.copy_(original)
        assert torch.equal(dynamic._opacity,original), 'Opacity not exactly restored'

def render_without_a37(camera, static, dynamic, args, background, env_map):
    from gaussian_renderer import render
    with hide_actors(dynamic,[37]):
        result=render(camera,static,dynamic,args,background,env_map=env_map,is_training=False)
        # Return tensors as produced by the native joint-depth render. No image
        # compositing or exclusion of actor 6 is performed.
    return result
