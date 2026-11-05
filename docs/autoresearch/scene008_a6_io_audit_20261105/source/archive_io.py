"""Archive only this completed audit into the research repository."""
import shutil,json
from pathlib import Path
RUN=Path(__file__).resolve().parent.parent
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
DEST=REPO/'docs/autoresearch/scene008_a6_io_audit_20261105'

def main():
    assert not DEST.exists();shutil.copytree(RUN/'delivery',DEST)
    copy=shutil.copyfile
    copy(RUN/'delivery.tar.gz.sha256',DEST/'delivery.tar.gz.sha256')
    (DEST/'README.md').write_text('''# Scene008 vehicle prior: geometry/material I/O attribution

The complete M2 asset has a broad blue-grey lower body. This audit traces the frozen PVD inputs and outputs, real RGB/LiDAR association, downstream Gaussianization, material fitting and final rendering before considering model training.

![Architecture components](architecture.svg)

## Findings

PVD `car_completion_3199` consumes and emits XYZ only. It never consumes the supplied RGBA hint or produces RGB. Normals, Gaussian scales, peak opacity .995, normal thickness .003m and initial colours are downstream conventions. Coordinate/unit/full-affine roundtrips, fixed200 conditioning, all2048 inverse export, raw JPEG/PNG/RGB correspondence, target reconstruction, camera/time binding, SH0 decoding and optimizer convergence checks pass on the inspected paths. No definite mechanical coordinate or colour-channel bug was found.

M2 has10557/18432 strictly zero RGB-loss columns. Every such point retains the global BUILD median `[.3392157,.4156863,.4892157]`. Temporarily recolouring those points magenta changes none of the27429 BUILD fit predictions, exactly. Below the lowest of all199 associated real LiDAR returns (canonicalz=-.210448m),8178/8407 proposal primitives have zero RGB support. This directly attributes most of the flat lower material to missing appearance supervision and a global fallback, rather than PVD generating grey texture.

The additional150 frames have445 retained sparse RGB hints, all at the roof(z=.576857–.719606m), and no measurements in the lower missing band. Other-vehicle OBB ray exclusion is conservative and may suppress real upper-body observations; existing reviewed masks in inspected other windows belong to other tracks, not a6. Rejected RGB zeros are unknown placeholders and require re-reading original images before reassociation. Multi-camera inspection is limited to raw0/106/198, not all199x5.

## Geometry diagnostics and limits

R0 replays the original200 inputs; R0_repeat repeats identical input and RNG; RY changes only the proper yaw convention about the canonical box centre, including the full affine translation; RV uses200 uniformly selected real reviewed/depth-visible points from raw106, selected by a fixed maximum-BUILD-support rule. Frozen checkpoint, paired original initial noise/RNG,1000 DDPM steps each. No appearance fitting, model training or adoption.

Original PVCNN contains floating-point atomics. Full state/RNG bytes restore exactly, but next-step sampling is not bitwise reproducible. The failed exact smoke is retained. Limited-precision diagnostics add the complete R0 repeat and100-step snapshots. Same-input generated-set symmetric distance is.00551m; RY and RV differ fromR0 by.13645m and.09170m. These are sensitivity diagnostics, not hidden-shape accuracy, a test of true front orientation or statistical significance. Original-input measured-part distance is.06571m; RY.06869m; RV.08090m. RV simultaneously changes sampling, purity and view/time coverage.

Whole-box statistical adaptation, front-direction semantics, raw multi-frame OBB conditioning versus synthetic single-view depth, and point-to-surface Gaussianization remain method assumptions. The audit does not license attributing every remaining error to model weights or starting pretraining.

## Interface correction

`asset/m2_with_io_contract.npz` retains the exact M2 geometry, RGB coefficients and inferred flags, and adds per-point RGB-loss support/source fields plus an explicit geometry/material capability contract. This is an interface clarification, not a claimed visual improvement. raw105/113/130 isolated, keep and delete rendering is bitwise equal to unchangedM2. Nativea6 renderer rows remain zero.

Next: qualify object-specific observations, validate geometry/material recovery using known-surface pseudo-occlusions in several directions, then test an explicit appearance completion branch while preserving observed surfaces. More training of an XYZ-only PVD cannot create an RGB output interface.

## Execution and verification

4000 formal frozen-model sampling steps;80 successful smoke/recheck steps and34 failed exact-smoke steps are separately recorded. Two earlier environment launches failed before sampling because ninja was absent from PATH; the existing dedicated PVD environment and cached extension were reused. Formal sampling lasted about168seconds; no hour-long experiment was started. A reviewed future long-run status gate was fixed and re-smoked; the exact formal source and prior receipt remain archived.

The49-frame provenance rendering preserves M2, native, road and environment. No model training, road experiment or scene-asset adoption. AutoDL remains ON as explicitly requested.

[Interactive report](index.html) · [Training readiness](training_readiness.json) · [I/O asset contract](asset/io_contract.json) · [Interface QA](asset/receipt.json) · [49-frame audit](evidence/current_audit.json) · [Geometry probes](evidence/geometry_summary.json) · [All checks/accounting](evidence/delivery_QA.json)

Byte-exact executed sources, backups and failure logs are retained in `evidence/executed_source_and_logs.tar.gz`. Reproduction needs the existing campaign sibling data layout; standalone asset loading uses `source/sh_asset.py`, `source/full_pvd.py` and the explicit NPZ path. No source checkpoints were overwritten.
''',encoding='utf-8')
    for name in ('EXPERIMENTS.md','RESEARCH_STATUS.md'):
        with (REPO/'docs'/name).open('a',encoding='utf-8') as f:
            f.write('\n\n2026-11-05，Scene008车辆先验输入输出审计：PVD只输出XYZ，M2的10557/18432点无RGB损失支持，低于全199帧实测最低高度的8178/8407点保持全局蓝灰回退色；原图、单位、坐标、颜色解码链未发现确定bug。完成4x1000步冻结几何输入敏感性及同输入复现对照；PVCNN数值非逐位确定，结果不采用。资产新增显式外观来源合同，三帧黑底/街景渲染不变。暂不训练、不启动道路分支、AutoDL保持开机。[报告与证据](autoresearch/scene008_a6_io_audit_20261105/README.md)。\n')
    print(json.dumps(dict(archived=str(DEST))))

if __name__=='__main__':main()
