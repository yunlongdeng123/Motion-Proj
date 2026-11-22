"""Record the primary review separately from the immutable independent review."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / 'cpu_review/joint_shared_continuous_v2'


def pin(path):
    return {'path': path.as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}


field = ROOT / 'joint_shared_continuous_v2/final_atlas.npz'
numeric, visual = [json.loads((QA / name).read_text(encoding='utf-8')) for name in ('numeric_acceptance.json', 'visual_review.json')]
expected = '2eaf1569a97fb8ee6907696dfa28db91bb5f3072f74d37491007c87012e2c647'
assert pin(field)['sha256'] == numeric['final_atlas_sha256'] == visual['final_atlas_sha256'] == expected
assert numeric['decision'] == visual['decision'] == 'PASS'
assert all(numeric['checks'].values())
review = {
    'schema': 'R9.root_shared_input_review.v1', 'complete': True,
    'recorded_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'decision': 'PASS', 'material_internal_decision': 'PASS',
    'root_concur': True, 'student_fit_authorized': True,
    'whole_scene_restoration_accepted': False,
    'final_atlas_sha256': expected, 'field': pin(field),
    'numeric_acceptance': pin(QA / 'numeric_acceptance.json'),
    'independent_visual_review': pin(QA / 'visual_review.json'),
    'frames': [86, 110, 134, 179, 198],
    'reviewed_images': [pin(QA / f'{raw:03d}_material_only.png') for raw in (86,110,134,179,198)] + [
        pin(QA / 'five_view_v1_v2_material_contactsheet.png'), pin(QA / 'world_atlas_seed_final.png')],
    'observations': [
        'Root inspected the five original-resolution surface material images, world atlas and v1/v2 comparison. The v1-like stripes have subsided without a display-domain change.',
        'Shared-input colour/seam acceptance only. This is not acceptance of a future Gaussian asset or whole-scene restoration.',
        'Existing unsupported geometry pixels remain classified separately; the old hard-composite rim is not Gaussian coverage evidence.'
    ],
    'required_next_gates': [
        'Actual five-view protected-pixel colour-contribution guard before any RGB update',
        'Discarded short native smoke covering save, optimizer/RNG restore, evaluation and independent fresh loading',
        'Fresh formal400 paired joint road/canonical-A20 RGB fit with persistent export and five-view independent parity'
    ],
    'constraints': {'road_and_actor_centers_fixed': True, 'all_geometry_fixed': True,
                    'scale_and_opacity_fixed': True, 'camera_and_rigid_tracks_fixed': True,
                    'native_context_and_existing_guards_fixed': True, 'new_diffusion_calls': 0},
    'review_GPU_forward': 0, 'review_Gaussian_updates': 0
}
out = QA / 'root_shared_input_review.json'
assert not out.exists(), 'Do not overwrite a recorded review'
out.write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(pin(out)))
