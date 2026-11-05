"""Archive scoped evidence in the research Git repo, preserving unrelated work."""
import hashlib,json,shutil
from pathlib import Path
from full_pvd import RUN

REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
DEST=REPO/'docs/autoresearch/scene008_a6_complete_asset_20261105'
def main():
    assert not DEST.exists();shutil.copytree(RUN/'delivery',DEST)
    summary=json.loads((RUN/'delivery/summary.json').read_text())['all49']
    def v(key):return summary[key]['mean']
    text=f'''# Scene008: single complete a6 asset, first fixed-geometry control

The native a6 object branch and a partial PVD patch previously competed inside the car. This experiment replaces all100,358 native-a6 rows with one standalone2048-row full PVD asset, shared in both a37-present and a37-deleted scenes. The native asset conditions reliable observed appearance but has zero renderer input rows. Static scene, a37, road1K and the previously saved world road colour field remain frozen.

![Architecture components](architecture.svg)

## Result

The engineering path passes: all49 paired frames, one depth sort, group alpha*T conservation, fixed geometry hashes, exact standalone asset save/reload, and original road1K baseline parity (maxRGB difference5.96e-8). **Quality is not accepted.**

On the same39,308 real observed a6 pixels, isolated nativeRGB MAE is{v('solo_native.a6.rgb_error'):.6f}, complete fitted assetRGB MAE is{v('solo_fitted.a6.rgb_error'):.6f}. On8,930 measured depth points, nativeMAE is{v('solo_native.a6.measured_depth_error'):.6f}m versus{v('solo_fitted.a6.measured_depth_error'):.6f}m. Complete asset coverage on the full fixed core is{v('solo_fitted.a6.visible_coverage'):.6f}; incomplete core coverage alone does not explain the detail gap.

Native uses100,358 primitives and SH3, while the first control uses2048 primitives and SH0. Geometry, spatial resolution and material capacity are therefore not causally isolated. The next contract separates these factors before measured-surface correction; no occlusion-direction-specific clipping is used.

## Method and evidence

- Full original2048 PVD points:200 measured conditioning points plus1848 generated points, all retained. This is not the old de-duplicated1477-point patch.
- Fixed PCA Gaussianization, centres/normals/scales/opacity/trajectory frozen. Shared6144 boundedRGB parameters, BUILD40 real visible a6 pixels only, ridge lambda.001. Fit27,429 eroded pixels; native appearance reference restricted to reliable pixels within the same real mask. Unknown material uses a low-capacity object-colour prior and explicit inferred flags.
- InternalHELD9 is appearance-only: source geometry/full-scene native reference already use scene context, including14 PVD conditioning points from these frames. No independent geometry/generalization claim.
- RealGPU smoke includes nonuniform colour design parity, save/reload and next-stage entry. Fixed-geometry appearance solve took about62s; all49 evaluation about41s. No long neural training or geometry updates.
- Unknown hidden parts are assessed for continuity, ordering and stability; no hiddenRGB accuracy claim. Native static residue remains frozen diagnostic context; external static occlusion is not declared solved.

[49-frame interactive report](index.html) · [Fixed metrics](fixed_metrics.csv) · [Geometry/material capacity audit](evidence/capacity_audit.json) · [Next unified-asset contract, design only](evidence/next_unified_asset_contract.json)

[Standalone complete NPZ](assets/complete_a6_fitted.npz) · [SH0 Gaussian PLY](assets/complete_a6_fitted.ply) · [Independent loader](source/full_pvd.py)

## Old branch closeout

The full-basis clipping/moment branch was stopped and preserved. Its12 engineering smoke cases passed, but formal49 quality evaluation did not finish;34 partial derived NPZ files were integrity-checked, no new asset adopted, S/U arms not run. [Closeout](old_work_closeout/index.html).

AutoDL remains on. This task explicitly does not authorize shutdown.
'''
    (DEST/'README.md').write_text(text,encoding='utf-8')
    for filename in ('EXPERIMENTS.md','RESEARCH_STATUS.md'):
        path=REPO/'docs'/filename
        with path.open('a',encoding='utf-8') as f:
            f.write('\n\n2026-11-05，Scene008旧裁剪分支收尾，转向单一完整a6资产。完整2048 PVD固定几何＋BUILD可见RGB拟合，原生a6退出成像，49帧保存重载和统一遮挡验证通过；画质未验收，native/new黑底RGB .01873/.04951，空间与SH容量差异待独立拆分。AutoDL保持开机。[报告、完整资产与下一合同](autoresearch/scene008_a6_complete_asset_20261105/README.md)。\n')
    print(json.dumps(dict(archived=str(DEST),files=sum(p.is_file() for p in DEST.rglob('*')))))

if __name__=='__main__':main()
