"""Add deterministic row-identity checks and directly inspected PNG findings."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(4194304),b""):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);args=p.parse_args();run=args.run
    path=run/"review/full49_v2_independent_review.json";a=json.loads(path.read_text())
    receipt=json.loads((run/"results/full49_v2/receipt.json").read_text())
    hashes=[]
    for raw in range(89,138):
        r=json.loads((run/f"results/full49_v2/{raw:03d}/metrics.json").read_text())
        hh=[r["states"][s][v]["native_asset_rows_hash"] for s in ("keep","delete") for v in ("baseline","extension")]
        assert len(set(hh))==1
        hashes.append(dict(raw=raw,native_asset_rows_hash=hh[0],all_four_state_variant_hashes_equal=True))
    ext=receipt["runtime"]["extension"]
    asset=Path(ext["path"])
    assert sha(asset)==ext["sha256"]
    with np.load(asset,allow_pickle=False) as f:
        ids=np.flatnonzero(f["active"]&(f["local_xyz"][:,0]>ext["frozen_boundary_local_x_m"]))
    assert len(ids)==ext["rows"] and hashlib.sha256(ids.tobytes()).hexdigest()==ext["selected_ids_sha256"]
    a["independent_checks"]["native_asset_row_identity_across_all_four_variants"]=hashes
    a["independent_checks"]["extension_selected_rows_recomputed_exact"]=dict(rows=len(ids),selected_ids_sha256=ext["selected_ids_sha256"],asset_sha256=ext["sha256"])
    selected=[(109,"delete"),(130,"delete"),(137,"keep")]
    a["visual_review"]=dict(inspected_original_saved_PNGs=[dict(raw=raw,state=state,
        baseline_sha256=sha(run/f"results/full49_v2/{raw:03d}/baseline_{state}.png"),
        extension_sha256=sha(run/f"results/full49_v2/{raw:03d}/extension_{state}.png")) for raw,state in selected],
        selection="Worst observed-road and protected per-frame regressions, used only for review after evaluation.",
        observations=["raw109/delete: extension introduces a visible dark band across the road behind the foreground vehicle compared with baseline.",
                      "raw130/delete: the added road has a clear transverse footprint/colour boundary and darker road patches; the foreground cars remain visually stable.",
                      "raw137/keep: a37 body remains visually stable while the visible road outside it acquires the extension colour change."],
        interpretation="Visual inspection supports the numeric preservation failure; it does not supply hidden-road truth or validate completion accuracy.")
    a["finalized_utc"]=datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(a,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    md=run/"review/full49_v2_review.md"
    note="\n直接查看最差帧PNG：raw109/delete、raw130/delete新增道路有明显暗带及横向足迹/颜色边界；raw137/keep车体保持稳定，道路颜色改变。这与数值保护失败一致，未把视觉补面当隐藏真值验收。另独立重算14661个扩展行的选择及资产SHA，均与运行记录一致；49帧四种状态/变体的车辆资产行hash逐帧一致。\n"
    if "直接查看最差帧PNG" not in md.read_text():md.write_text(md.read_text()+note,encoding="utf-8")
    delivery=dict(schema="R0_FULL49_V2_REVIEW_DELIVERY",status="COMPLETE",verdict=a["verdict"],
        files={str(p.relative_to(run)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (path,md)},
        source_receipt_sha256=a["input_receipt"]["sha256"])
    (run/"review/review_delivery.json").write_text(json.dumps(delivery,indent=2)+"\n")
    print(json.dumps(delivery),flush=True)


if __name__=="__main__":main()
