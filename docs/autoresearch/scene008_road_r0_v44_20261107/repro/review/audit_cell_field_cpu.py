"""Read-only NPZ/JSON audit of the final observed-road cell material artifact."""
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4194304),b""):h.update(b)
    return h.hexdigest()


def audit_cell_field(run):
    folder=run/"surface/material_cellfit_strip_v1";c=read(folder/"contract.json");p=load(folder/"prepared.npz")
    q=load(folder/"eligibility/qualified_rows.npz");a=load(folder/"material_override.npz")
    formal=read(folder/"formal/receipt.json");smoke=read(folder/"smoke/receipt.json");restore=read(folder/"restore/receipt.json")
    checks=read(folder/"restore/restore_checks.json");export=read(folder/"export_receipt.json")
    expected="70a9f7262977099f875ebd18dd359313232d0662025e6213b799bce41ea90968"
    assert sha(folder/"material_override.npz")==export["sha256"]==expected
    assert all(sha(path)==digest for path,digest in c["immutable_input_sha256"].items())
    assert sha(folder/"prepared.npz")==c["prepared_sha256"] and sha(folder/"supervision_masks.npz")==c["supervision_masks_sha256"]
    held={89,95,101,105,113,121,125,130,137};build=[r for r in range(89,138) if r not in held]
    assert c["BUILD"]==build and c["HELD"]==sorted(held)
    masks=load(folder/"supervision_masks.npz");mask_rows=[];pixels=0
    for raw in build:
        obs=load(run.parent/f"scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz")
        view=load(run.parent/f"scene008_blue_global_road1k_20261105_r1/inputs/views/{raw:03d}.npz")
        mask=obs["road"].astype(bool)&~obs["visible"].any(0)&~obs["visible_reviewed"].any(0)&~obs["box_candidate"].any(0)&~view["other_vehicle_guard"]
        assert np.array_equal(mask,masks[f"raw{raw:03d}"])
        pixels+=int(mask.sum());mask_rows.append(dict(raw=raw,pixels=int(mask.sum()),exact=True))
    eligible=q["observed_mass_by_state"].sum(0)>0
    assert np.array_equal(eligible,q["eligible"]) and np.array_equal(eligible,a["eligible"])
    assert int(eligible.sum())==formal["eligible_cells"]==2171 and int(eligible.sum()*3)==formal["trainable_scalars"]==6513
    assert not eligible[~p["support_active"]].any() and not q["observed_mass_by_state"][:,~p["support_active"]].any()
    assert np.array_equal(a["colours_srgb"][~eligible],p["initial_field_colours_srgb"][~eligible])
    assert np.array_equal(a["colours_srgb"][~p["support_active"]],p["source_colours_srgb"][~p["support_active"]])
    assert np.array_equal(a["initial_field_colours_srgb"],p["initial_field_colours_srgb"])
    assert np.array_equal(a["source_colours_srgb"],p["source_colours_srgb"])
    init=load(c["initial_material"]);assert np.array_equal(init["colours_srgb"],p["initial_field_colours_srgb"])
    low=p["interval_low"][eligible];high=p["interval_high"][eligible];initial=p["initial_field_colours_srgb"][eligible].astype(float)
    normalized=np.clip((initial-low)/(high-low),1e-7,1-1e-7)
    x=np.log(normalized/(1-normalized))+a["raw_logit_delta"].astype(float)
    rebuilt=(low+(high-low)/(1+np.exp(-x))).astype(np.float32)
    colour_diff=float(np.abs(rebuilt-a["colours_srgb"][eligible]).max());assert colour_diff<=1e-7
    max_delta=float(np.abs(a["colours_srgb"].astype(float)-p["source_colours_srgb"]).max());assert max_delta<=.15+1e-7
    assert np.isfinite(a["colours_srgb"]).all() and a["colours_srgb"].min()>=0 and a["colours_srgb"].max()<=1
    assert formal["steps_executed"]==1000 and formal["initial_step"]==0 and formal["final_step"]==1000
    assert [r["step"] for r in formal["records"]]==list(range(1,1001)) and all(r["raw"] in build for r in formal["records"])
    assert all(np.isfinite([r["data_MSE"],r["anchor"],r["texture"],r["gradient_norm"]]).all() for r in formal["records"])
    assert formal["forwards"]==formal["backwards"]==2000 and formal["regularizer_backward_calls"]==1000
    assert smoke["steps_executed"]==32 and restore["steps_executed"]==2 and restore["initial_step"]==32 and restore["final_step"]==34
    assert all(checks.values())
    assert all(formal[k]==0 for k in ("original_parameter_updates","geometry_updates","opacity_updates"))
    return dict(audit_kind="cellfit1000",material_path=str(folder/"material_override.npz"),material_sha256=expected,asset_sha256=str(a["asset_sha256"]),
        initial_material_sha256=c["initial_material_sha256"],initialization="Matched-strip 11x9 field, not the original wide field",
        field_contract_sha256=sha(folder/"contract.json"),immutable_file_hashes_all_match=True,all40_masks_recomputed_exact=True,mask_checks=mask_rows,
        BUILD40=build,HELD9=sorted(held),supervision_pixels=2*pixels,field_scalar_parameters=6513,eligible_cells=2171,zero_gradient_frozen_cells=12490,
        eligibility_exact_positive_renderer_mass=True,outside_strip_zero_gradient=True,unqualified_colours_bitwise_frozen=True,
        sigmoid_export_reconstruction_maxabs=colour_diff,actual_max_colour_delta=max_delta,source_delta_bound=.15,
        material_optimizer="Adam",material_learning_rate=.01,material_optimizer_steps=1000,smoke_steps_separate=32,restore_disposable_steps_separate=2,
        formal_native_forwards=2000,formal_native_backwards=2000,formal_seconds=formal["seconds"],restore_checks=checks,
        old_model_updates=0,geometry_updates=0,opacity_updates=0,neural_prior=False,hidden_GT_used=False,
        independent_ridge_solution_max_difference=None,field_shared_across_keep_delete=True,
        checkpoint_audit_scope="JSON save/restore receipts and NPZ export independently checked; no Torch checkpoint loaded and no optimization replay performed by this CPU reviewer.",
        caveat="Original V2 source proxy uses all199; HELD9 excludes this material training only. Not a strict blinded-road/prior result.")
