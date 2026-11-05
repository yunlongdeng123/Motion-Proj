import json
from full_pvd import RUN
from scene_runtime import Runtime

rt=Runtime();d=rt.raw(113);rows=rt.n.actor_kwargs(d['cam'],delete_a37=False)
record=dict(native_a6_rows=int(rt.n.keep_a6.sum()),new_asset_rows=2048,
    native_renderer_active_SH_degree=int(rt.n.static.active_sh_degree),
    native_dynamic_SH_degree=int(rt.n.dynamic.active_sh_degree),
    native_SH_shape=None if rows['shs'] is None else list(rows['shs'].shape),
    native_precomputed=rows['colors_precomp'] is not None,new_asset_material_degree=0,
    interpretation='Observed RGB gap does not isolate geometry: primitive density, spatial basis, and SH material capacity also differ')
assert rt.modelunchanged()
(RUN/'capacity_audit.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
