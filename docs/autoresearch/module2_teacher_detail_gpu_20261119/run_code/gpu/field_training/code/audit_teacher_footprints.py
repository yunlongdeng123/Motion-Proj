"""冻结教师像素的真实 J_uv 足迹统计；只读，不选择滤波超参数。"""
from pathlib import Path
import json
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np

Q = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
P = Q.parent/'road_multiscale_material_20261118_r1'
QUANTILES = [0, .05, .5, .9, .95, .99, 1]


def stats(J):
    covariance = np.einsum('nij,nkj->nik', J, J)/12
    uv_axis_std = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0))
    image_axis_norm = np.sqrt(np.sum(J*J, axis=1)/12)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    principal_std = np.sqrt(np.maximum(eigenvalues, 0))
    angle = np.arctan2(vectors[:, 1, 1], vectors[:, 0, 1])
    weight = eigenvalues[:, 1]-eigenvalues[:, 0]
    c = np.sum(weight*np.exp(2j*angle))/max(float(weight.sum()), 1e-30)
    return dict(pixels=len(J), sigma_uv_axis_quantiles=np.quantile(uv_axis_std, QUANTILES, axis=0).tolist(),
        image_xy_box_axis_vector_norm_quantiles=np.quantile(image_axis_norm, QUANTILES, axis=0).tolist(),
        sigma_eigen_minor_major_quantiles=np.quantile(principal_std, QUANTILES, axis=0).tolist(),
        covariance_uv_mean=covariance.mean(0).tolist(),
        principal_axis_anisotropy_quantiles=np.quantile(principal_std[:, 1]/np.maximum(principal_std[:, 0], 1e-12), QUANTILES).tolist(),
        major_angle_from_u_deg_axial_weighted=float(np.angle(c)*90/np.pi)%180,
        axial_resultant_length=float(abs(c)),
        sigma_u_over_v_median=float(np.median(uv_axis_std[:, 0]/np.maximum(uv_axis_std[:, 1], 1e-12))))


def main():
    start = time.perf_counter()
    output = Q/'gpu_20261119/field_training/teacher_footprint_audit_r1'
    output.mkdir(parents=True, exist_ok=False)
    result = dict(formula='Sigma_UV = J_uv * I_pixel/12 * J_uv^T',
        J_axes='output chart(u,v), input image(x,y)',
        units='chart texels; nominal texel 0.02m, actual chart need not be global affine',
        quantiles=QUANTILES, axis_order=dict(sigma_uv=['u', 'v'], eigen=['minor', 'major'], image=['x', 'y']),
        teacher_sampling='one 480x320 target pixel; native raw teacher was conservatively4x4 downsampled',
        native_calls=0, optimizer_updates=0, filter_applied=False, rows={})
    union = {key: [] for key in ('A', 'C', 'S', 'fullS')}
    for raw in (86, 110):
        with np.load(Q/'pixel_render/maps'/('%03d.npz'%raw), allow_pickle=False) as z:
            J = z['pixel_uv_jacobian']; valid=z['canonical_mesh_hit_valid']
        with np.load(P/'stage0_native_r1'/('%03d_targets.npz'%raw), allow_pickle=False) as z:
            masks = {key:z[key].astype(bool) for key in union}
        sparse = {}
        for key, mask in masks.items():
            if np.any(mask&~valid) or not np.isfinite(J[mask]).all():
                raise ValueError('真实监督域缺 canonical J: %s %s'%(raw,key))
            values = J[mask]; union[key].append(values)
            result['rows'][str(raw)+'_'+key] = stats(values)
            sparse[key+'_pixel_yx'] = np.stack(np.nonzero(mask), -1)
            sparse[key+'_J_uv'] = values
        np.savez_compressed(output/('%03d_samples.npz'%raw), **sparse)
    for key, values in union.items():
        result['rows']['both_'+key] = stats(np.concatenate(values))
    result['seconds'] = time.perf_counter()-start
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(output/'results.json'), seconds=result['seconds'],
        summary={key:dict(sigma_uv_p50=value['sigma_uv_axis_quantiles'][2],
            eig_p50=value['sigma_eigen_minor_major_quantiles'][2], angle=value['major_angle_from_u_deg_axial_weighted'])
            for key,value in result['rows'].items() if key.startswith('both_')})))


if __name__ == '__main__':
    main()
