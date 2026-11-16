"""Learn a fast input-view utility from training-only denoising outcomes.

The inference API consumes only a finite candidate set and visible-input
features.  It never receives an image target or invokes diffusion search.
"""
import itertools
import json
from pathlib import Path

import numpy as np


def _features(candidate, names):
    value = candidate['features']
    if isinstance(value, dict):
        result = np.asarray([value[name] for name in names], np.float64)
    else:
        assert list(candidate.get('feature_names', names)) == list(names)
        result = np.asarray(value, np.float64)
    if result.shape != (len(names),) or not np.isfinite(result).all():
        raise ValueError('candidate features must be finite and match the frozen schema')
    return result


def _valid(candidate):
    return bool(candidate.get('hard_valid', True) and candidate.get('rank_eligible', candidate.get('hard_valid',
                candidate.get('feasible', candidate.get('valid', True)))))


def _cost(candidate):
    return float(candidate.get('pixel_cost', candidate.get('estimated_pixel_cost', candidate.get('pixels', 0))))


def fit_utility(groups, feature_names, l2=1., oracle_loss_key='oracle_loss'):
    """Pairwise ridge: source difficulty cancels within each pseudo-hole.

    All groups must be marked train. The caller computes oracle losses after
    generation on independently observed, masked road pixels. Those losses are
    training labels and are absent from `choose_view`.
    """
    if not groups or not feature_names:
        raise ValueError('nonempty training groups and feature schema are required')
    if l2 <= 0:
        raise ValueError('a fixed positive ridge penalty is required')
    rows = []
    for group in groups:
        if group.get('split') != 'train':
            raise ValueError('input-utility fitting accepts training groups only')
        for c in group['candidates']:
            if _valid(c):
                rows.append(_features(c, feature_names))
    matrix = np.stack(rows)
    mean, std = matrix.mean(0), matrix.std(0)
    scale = np.where(std > 1e-8, std, 1.)
    x, y, records = [], [], []
    for group in groups:
        candidates = [c for c in group['candidates'] if _valid(c)]
        for left, right in itertools.combinations(candidates, 2):
            a, b = float(left[oracle_loss_key]), float(right[oracle_loss_key])
            if not np.isfinite([a, b]).all():
                raise ValueError('training oracle losses must be finite')
            # Higher utility corresponds to lower reconstruction+compute loss.
            difference = (_features(left, feature_names)-_features(right, feature_names))/scale
            x.append(difference)
            y.append(b-a)
            records.append(dict(group=group['id'], left=left['id'], right=right['id']))
    if not x:
        raise ValueError('at least one training target needs two feasible candidates')
    x, y = np.stack(x), np.asarray(y)
    weights = np.linalg.solve(x.T@x + float(l2)*np.eye(x.shape[1]), x.T@y)
    prediction = x@weights
    model = dict(schema='road_input_utility.v1', feature_names=list(feature_names),
                 feature_mean=mean.tolist(), feature_scale=scale.tolist(), weights=weights.tolist(),
                 fit='training-only within-target pairwise ridge', l2=float(l2),
                 oracle_loss_key=oracle_loss_key, training_groups=len(groups), training_pairs=len(y),
                 training_pair_rmse=float(np.sqrt(np.mean((prediction-y)**2))),
                 inference_uses_targets=False, inference_diffusion_search=False)
    receipt = dict(training_groups=len(groups), training_candidates=len(rows), pairs=len(y),
                   non_tied_pairs=int((np.abs(y)>1e-9).sum()),
                   pair_order_accuracy=float(((prediction*y)>0)[np.abs(y)>1e-9].mean())
                   if (np.abs(y)>1e-9).any() else None,
                   model_schema=model['schema'], fixed_l2=l2,
                   no_eval_labels_used=True, pair_records=records)
    return model, receipt


def choose_view(candidates, model=None):
    """Choose one already-planned view; construction is O(candidate count).

    Raw candidates may include oracle measurements in a training artifact; this
    function explicitly copies and reads only IDs, feasibility, costs and input
    features. It does not inspect any oracle or target field.
    """
    valid = [c for c in candidates if _valid(c)]
    if not valid:
        raise ValueError('no candidate meets target/context/geometry constraints')
    if model is None:
        quality = [c for c in valid if c.get('quality_feasible', c.get('feasible', True))]
        if quality:
            selected = min(quality, key=lambda c: (_cost(c), str(c['id'])))
            policy = 'fastest-quality-feasible'
        else:
            selected = min(valid, key=lambda c: (float(c.get('constraint_deficit', 0)), _cost(c), str(c['id'])))
            policy = 'hard-valid-minimum-quality-deficit'
        return selected, dict(policy=policy, selected_id=selected['id'], target_used=False)
    names = model['feature_names']
    mean, scale, weights = [np.asarray(model[k], np.float64)
                            for k in ['feature_mean', 'feature_scale', 'weights']]
    values = []
    for c in valid:
        score = float(((_features(c, names)-mean)/scale)@weights)
        values.append((score, -_cost(c), str(c['id']), c))
    # Tie ordering is deterministic; candidate IDs never enter the learned score.
    values.sort(key=lambda row: (-row[0], -row[1], row[2]))
    chosen = values[0][3]
    return chosen, dict(policy='learned-input-utility', selected_id=chosen['id'],
                        quality_feasible=bool(chosen.get('quality_feasible', chosen.get('feasible', True))),
                        candidate_scores=[dict(id=row[2], score=row[0]) for row in values],
                        target_used=False, diffusion_search=False)


def evaluate_selection(groups, model):
    """Offline evaluation labels stay outside the actual selector API."""
    results = []
    for group in groups:
        options = [c for c in group['candidates'] if _valid(c)]
        selected, _ = choose_view(options, model)
        fastest, _ = choose_view(options)
        best = min(options, key=lambda c: (c['oracle_loss'], str(c['id'])))
        results.append(dict(id=group['id'], split=group.get('split'), selected=selected['id'],
                            fastest=fastest['id'], oracle=best['id'],
                            selected_loss=float(selected['oracle_loss']),
                            fastest_loss=float(fastest['oracle_loss']), oracle_loss=float(best['oracle_loss']),
                            regret=float(selected['oracle_loss']-best['oracle_loss']),
                            selected_pixel_cost=_cost(selected)))
    return dict(groups=len(results), rows=results,
                selected_mean_loss=float(np.mean([r['selected_loss'] for r in results])),
                fastest_mean_loss=float(np.mean([r['fastest_loss'] for r in results])),
                oracle_mean_loss=float(np.mean([r['oracle_loss'] for r in results])),
                mean_regret=float(np.mean([r['regret'] for r in results])))


def save_model(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2))
