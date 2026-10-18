"""Static source-path audit: do not confuse a configured term with a GPU pass."""
from pathlib import Path
import ast, hashlib, json

W = Path(__file__).parent


def main():
    train = (W / 'source/train.py').read_text(encoding='utf-8')
    native = (W / 'source/gn11_train.py').read_text(encoding='utf-8')
    upstream = (W / 'upstream/train.py').read_text(encoding='utf-8')
    renderer = (W / 'source/renderer.py').read_text(encoding='utf-8')
    calls = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else ''
             for n in ast.walk(ast.parse(train)) if isinstance(n, ast.Call)]
    assert not set(calls) & {'densify_and_prune', 'clone_points', 'prune_points', 'accumulate_densification_stats'}
    assert "policy['hybrid_lambda_dynamic_opacity'] = 0.0" in train
    assert "'dynamic_alpha': package['dynamic_visible_alpha']" in train
    assert '"dynamic_alpha": render_pkg["dynamic_alpha"]' in native
    assert 'loss_convergence_to_one * 100' in upstream
    assert 'package["dynamic_alpha_raw"] = package["dynamic_alpha"]' in renderer
    terms = [
        dict(item='full-scene RGB', GN11='active .8 L1+.2 DSSIM', adapter='same formula and all-scene domain', classification='inherited', gradient='pending actual component receipt'),
        dict(item='isolated dynamic RGB', GN11='5× photometric on binary union, environment fill during training', adapter='same isolated dynamic branch times union, weight5', classification='inherited intended call path', gradient='pending actual component receipt'),
        dict(item='static RGB', GN11='photometric on observed complement of the selected binary union', adapter='same formula; old GN11 had only3 while five-stage had5', classification='confirmed stage-input confound; matched5 fixes domain consistency', limitation='hidden region still has no paired clean road target'),
        dict(item='isolated raw dynamic body alpha', GN11='100× full HxW MSE(isolated raw alpha, binary union)', upstream='100× MSE in active lambda_opacity_entropy branch', adapter='weight0 and state points to visible union', classification='constraint replacement, not equivalent inheritance', correction='separate five_raw_body_isolated uses package.dynamic_alpha; awaiting GPU'),
        dict(item='visible ID alpha', GN11='not a native per-ID term', adapter='100× mean of K known per-ID full HxW MSE; whole scene depth sorting', classification='intentional rigid identity adaptation', normalization='3→5 makes each coefficient100/3→100/5; separate166.6667 control'),
        dict(item='static duration / velocity / NPS', GN11='compose_hybrid_loss with combined-alpha legacy maps and static-alpha NPS features', adapter='same composer, denominator conventions and current native static velocity', classification='active source path, gradients not yet certified', limitation='a nonzero configured NPS does not certify static edit cleanliness'),
        dict(item='LiDAR / sky / smoothness / combined-opacity entropy', GN11='original composer and decay', adapter='same composer and native functions', classification='inherited intended formulas', gradient='pending actual component receipt'),
        dict(item='native clone/split, opacity and big-point pruning', GN11='densification_and_optimization actually invoked; static/dynamic thresholds and lifetime split rules', adapter='no topology call in actual train loop; buffer/core methods alone are insufficient', classification='unresolved fidelity gap blocks full-native claim', next='close or explicitly bound representation-specific adaptation before formal instance phase'),
        dict(item='static LR', GN11='native scheduler', adapter='new-objective static Adam reset; xyz/t rescheduled and then multiplied .1', classification='intentional adapter policy; claimed10× bug retracted', proof='cpu_policy_audit.json actual caller4nodes passed'),
        dict(item='environment', GN11='original native optimizer active until configured cutoff', adapter='preserve_joint_background_v2 freezes environment parameters and skips optimizer', classification='intentional stage policy', restore='full environment/Adam/RNG proof pending new audit'),
        dict(item='GT rigid poses and local OBB boundary', GN11='native dynamic PVG representation', adapter='fixed genuine track transform, no pose refinement; soft local boundary weight1', classification='intentional representation adaptation', units='normalized squared coordinates; scale .1 implies .01×m², not mislabeled meters'),
        dict(item='restore', GN11='native model/Adam/RNG/paired sampler/SGCM checkpoint', adapter='model+static/instance Adam+environment+sampler+all RNG; actual golden verifier', classification='full runtime checks scheduled', next='actual unmodified update-body continuous vs checkpoint-restored next update; isolated disposable copies only'),
    ]
    receipt = dict(status='STATIC_CALL_PATHS_AUDITED_GPU_COMPONENT_GRADIENTS_PENDING',
        topology_call_absent_in_actual_adapter=True,
        first_four_pilot_body_metric_actual_domain='full_scene_visible_union_alpha',
        five_raw_body_original_restoration_claim_retracted=True,
        raw_body_correction='isolated_body_implementation/train.py separate, leaves all old source/checkpoints untouched',
        terms=terms, source_hashes={n:hashlib.sha256((W/'source'/n).read_bytes()).hexdigest() for n in ['train.py','renderer.py','hybrid30k_losses.py','runtime.py','rigid_model.py','gn11_train.py']},
        human_verdict=None, all_engineering_problems_excluded=False, parameter_updates=0)
    (W / 'constraint_paths.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(receipt['status'])


if __name__ == '__main__':
    main()
