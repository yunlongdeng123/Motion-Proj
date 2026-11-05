"""Fail a formal continuation at 1K when fixed pilot rays fail basic mechanism checks."""
import json


def check_1k(new, control_path, output_path):
    old = json.loads(control_path.read_text())['pooled']
    new = new['pooled']
    if any(new[key]['count'] != item['count'] for key, item in old.items()):
        raise ValueError('1K acceptance requires identical fixed domains')
    def n(key):
        return new[key]['mean']
    checks = {
        'static_front_below_union_control': n('a6_static_front') < old['a6_static_front']['mean'],
        'static_front_at_most_0_10': n('a6_static_front') <= .10,
        'a6_after_delete_contribution_at_least_0_70': n('a6_delete_w') >= .70,
        'a37_at_most_0_20_on_visible_a6': n('a6_wrong_w37') <= .20,
        'a6_at_most_0_10_on_visible_a37': n('a37_wrong_w6') <= .10,
        'a37_strict_occlusion_contribution_at_least_0_70': n('strict_a37_w') >= .70,
        'a6_isolated_and_deleted_RGB_at_most_0_20': max(n('a6_iso_rgb'), n('a6_delete_rgb')) <= .20,
        'a6_measured_depth_MAE_at_most_5m': n('a6_depth_m') <= 5.,
        'protected_RGB_at_most_0_015_above_control': n('protected_keep_rgb') <= old['protected_keep_rgb']['mean'] + .015,
    }
    payload = {'checks': checks, 'status': 'PASS' if all(checks.values()) else 'FAIL',
               'candidate': new, 'control': old, 'scope': 'gross early failures on raw105/113/130'}
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return payload
