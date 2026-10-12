"""Small CPU arithmetic/source receipt. No new depth test, source change or training."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
WORK = AUDIT.parent
ROOT = WORK.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    matrix_path = HERE / 'transform_matrix_readonly_copy.txt'
    matrix = [[float(v) for v in line.split()] for line in matrix_path.read_text().splitlines()]
    norms = [math.sqrt(sum(matrix[i][j] ** 2 for i in range(3))) for j in range(3)]
    assert all(abs(s - .1) < 1e-14 for s in norms)
    config = json.loads((AUDIT / 'long_pair_30k_32000/materialized_plan/baseline_effective_config.yaml').read_text())
    assert config['depth_blend_mode'] == 0 and config['paper_lambda_depth'] == .1 and config['env_map_res'] == 1024
    baseline = WORK / 'continuation_diagnostic/original30k_to32000'
    first = json.loads((baseline / 'step_losses.jsonl').read_text().splitlines()[0])
    depth_term = first['losses']['loss_paper_depth']
    area, s, eps = 480 * 320, .1, 1e-5
    examples = []
    for alpha in (1., .5):
        lidar_z, foreground_z = s * 10, s * 20
        rendered = 1 / (alpha / foreground_z + (1 - alpha) / 900)
        residual = abs(1 / (lidar_z + eps) - 1 / (rendered + eps))
        examples.append({'kind': 'arithmetic_illustration_not_observed_pixel', 'lidar_physical_m': 10, 'foreground_physical_m': 20, 'alpha': alpha, 'lidar_z_scene_units': lidar_z, 'foreground_z_scene_units': foreground_z, 'harmonic_rendered_depth_scene_units': rendered, 'harmonic_rendered_physical_m': rendered / s, 'single_valid_pixel_inverse_depth_residual_scene_units_inverse': residual, 'single_valid_pixel_loss_divided_by_HW': residual / area, 'single_valid_pixel_weighted_depth_contribution': .1 * residual / area})
    source_paths = ['scene/waymo_loader2.py', 'utils/camera_utils.py', 'train.py', 'utils/paper_losses.py', 'gaussian_renderer/__init__.py']
    source_sha = {p: sha(AUDIT / 'source_frozen' / p) for p in source_paths}
    paper_text = ROOT / 'work/paper_audit/DrivingEditor.txt'
    assert 'respectively represent the inverse of' in paper_text.read_text(encoding='utf-8')
    result = {'status': 'CPU_SOURCE_AND_UNITS_FOLLOWUP_COMPLETE', 'training_or_GPU_started': False, 'source_config_checkpoint_modified': False, 'new_gradient_tests_run': False, 'paper': {'equation': 16, 'equation_printed_page': 1702, 'equation_pdf_page_one_based': 7, 'cross_page_definition_printed_page': 1703, 'definition_pdf_page_one_based': 8, 'text_definition_lines': [503, 504, 505, 506], 'definition': 'D_LiDAR and D_image are explicitly the inverse of sparse LiDAR depth and rendered depth; h,w are spatial dimensions.', 'equation_normalizer': '1/(h*w)', 'depth_units_sparse_validity_eps_alpha_normalization_harmonic900': 'Not specified by the cited equation/prose.'}, 'claims_separated': {'operator_value_and_gradient': 'Previous CPU operator and grad checks only certify the implemented function; not complete fidelity to author pipeline.', 'inverse_vs_metric_depth': 'Inverse representation itself is supported by the explicit cross-page definition; metric-depth difference is not a certified bug.', 'normalizer': 'Full image H*W matches Eq16.', 'implementation_choices_not_author_certified': ['Exclude LiDAR zero/nonfinite pixels from residuals', 'eps=1e-5 in scene-depth units', 'Normalize raw weighted depth by alpha.clamp_min(1e-5)', 'Shared-environment depth with harmonic blend to fixed sky_depth=900', 'Scene normalization scale and lambda_depth=.1']}, 'actual_configuration': {'data_normalization_scale': s, 'source_matrix_column_norms': norms, 'matrix_SHA256': sha(matrix_path), 'matrix_remote_path': '/root/autodl-tmp/drivingeditor_repro/data/waymo_paper_scenes/scene0017_000_048/transform_matrix.txt', 'rendered_HW': [320, 480], 'depth_blend_mode': config['depth_blend_mode'], 'env_map_res': config['env_map_res'], 'lambda_depth': config['paper_lambda_depth'], 'epsilon_in_scene_length_units': eps, 'sky_depth_in_scene_length_units': 900}, 'units': {'basis': 'waymo_loader2.py535 normalizes poses; line559 scales original camera point cloud by same factor; camera_utils.py117-133 stores that scaled camera z as LiDAR depth.', 'physical_unit_assumption': 'Waymo input point/camera distances are meters.', 'conversion': 'z_scene=.1*z_m; inverse(z_scene)=10*inverse(z_m), ignoring epsilon; epsilon1e-5_scene is1e-4 meters.', 'unweighted_loss_dimension': 'Inverse scene length; H*W normalization is dimensionless.', 'sky900_physical_equivalent_m': 900 / s, 'coefficient_status': 'lambda=.1 is an explicit reproduction choice; paper does not publish these unit/weight conventions.'}, 'actual_B1_first_step_scalar': {'iteration': first['iteration'], 'camera_id': first['camera_id'], 'unweighted_inverse_depth_loss_per_full_HW_scene_units_inverse': depth_term, 'weighted_depth_addition_to_original_loss': config['paper_lambda_depth'] * depth_term, 'inverse_metric_depth_equivalent_with_corresponding_epsilon_m_inverse': s * depth_term, 'source_path': str(baseline / 'step_losses.jsonl'), 'source_SHA256': sha(baseline / 'step_losses.jsonl')}, 'illustrative_values': examples, 'source_SHA256': source_sha, 'code_locations': {'waymo_loader2.py': [535, 559], 'camera_utils.py': [117, 133], 'train.py': [359, 360, 370, 371, 399, 405], 'paper_losses.py': [77, 81, 85, 86, 87]}, 'V_algorithm_unchanged': True}
    (HERE / 'depth_definition_followup_cpu.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'status': result['status'], 'actual_scale': s, 'actual_B1_loss_depth': depth_term, 'weighted_depth': config['paper_lambda_depth'] * depth_term, 'sky900_equivalent_m': 900 / s}))


if __name__ == '__main__':
    main()
