"""Preserve the four pilots; prepare a separate, correctly isolated-body control.

The first raw-body pilot only reweighted visible union alpha. It is retained as
that control and must not be described as restoring the native isolated loss.
"""
from pathlib import Path
import ast, hashlib, json, shutil

OUT = Path(__file__).resolve().parent


def main():
    source = OUT / 'implementation'
    target = OUT / 'isolated_body_implementation'
    if target.exists():
        raise FileExistsError('preserve an existing implementation; no overwrite')
    shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
    p = target / 'train.py'
    old = p.read_text()
    needle = "'dynamic_alpha': package['dynamic_visible_alpha'], 'dynamic_mask': union,"
    assert old.count(needle) == 1
    new = old.replace(needle, "'dynamic_alpha': package['dynamic_alpha'], 'dynamic_mask': union,")
    # Expose the actual loss domain in the saved policy, including resume checks.
    marker = "raw_body_opacity_weight=args.raw_body_opacity_weight)"
    assert new.count(marker) == 1
    new = new.replace(marker, "raw_body_opacity_weight=args.raw_body_opacity_weight, raw_body_opacity_domain='isolated_all_instances_raw_union_body_opacity')")
    ast.parse(new)
    p.write_text(new)
    assert "'dynamic_alpha': package['dynamic_visible_alpha']" not in new
    receipt = dict(
        status='PREPARED_CORRECTED_ISOLATED_CONTROL_NOT_YET_TRAINED',
        old_control='five_raw_body',
        old_control_actual_domain='full_scene_visible_union_alpha',
        old_control_invalid_claim='restored native isolated body alpha',
        preserved_all_old_pilot_checkpoints=True,
        corrected_control='five_raw_body_isolated',
        maximum_new_pilot_updates=300,
        corrected_domain='isolated_all_instances_raw_union_body_opacity',
        raw_body_weight=100., identity_weight=100.,
        source_sha256=hashlib.sha256(old.encode()).hexdigest(),
        corrected_sha256=hashlib.sha256(new.encode()).hexdigest(),
        changed_algorithm_tensor="state.dynamic_alpha: package.dynamic_visible_alpha -> package.dynamic_alpha",
        metadata_change='record the actual domain in training_policy',
        other_training_behavior_unchanged=True,
        original_GN11_training_affected=False,
        human_verdict=None,
    )
    (OUT / 'isolated_control_contract.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
