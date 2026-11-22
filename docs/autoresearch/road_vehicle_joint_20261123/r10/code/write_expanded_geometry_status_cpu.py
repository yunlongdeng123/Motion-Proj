"""Write compact, evidence-bound negative geometry status. No fitting or activation."""
import argparse,json
from pathlib import Path
from diagnose_nonplanar_inference_cpu import read,write,pin


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);a=p.parse_args()
    R=a.campaign_root/'road_underbody_local_20261123_r10';D=R/'audit/expanded_curvature_loo_v1'
    selection_path=D/'training_only_CV_selection.json';fold_path=D/'training_fold_original_height_audit.json'
    selection,fold=read(selection_path),read(fold_path)
    assert len(selection['models'])==9 and len(fold['models'])==9
    assert not any(x['all8_LOO_actual_requested_height_pass'] for x in fold['models'])
    assert not (D/'expanded_heldout_measurements.npz').exists()
    matched_path=R/'audit/matched_spatial_height_cpu_v1/summary.json'
    assert matched_path.exists()
    status=dict(schema='R10.expanded_geometry_first_family_status.v1',complete=True,decision='FAIL',
        scope='Expanded geometry source qualification, not appearance or Gaussian optimization',
        candidate_geometry=pin(R/'audit/local_surface_expanded_candidate_v1/expanded_candidate_geometry.npz'),
        candidate_integrity_audit=pin(R/'audit/local_surface_expanded_candidate_v1/independent_audit_v1/independent_geometry_audit.json'),
        actual_scope_before_rejection_audit=pin(R/'audit/local_surface_expanded_candidate_v1/actual_scope_residual_v1/actual_scope_residual_diagnostic.json'),
        training_only_selection=pin(selection_path),original_point_fold_audit=pin(fold_path),
        matched_spatial_height_evidence=pin(matched_path),
        original_official_training_points=3276,actual_requested_original_points=1426,training_medoids=1289,
        best_mean_model=selection['best_mean_model'],one_SE_preferred_model=selection['one_SE_preferred_model'],
        all9_fail_fixed_per_frame_actual_height_bands=True,
        fixed_absolute_height_median_max_m=.04,fixed_absolute_height_p95_max_m=.10,
        one_SE_does_not_authorize_physically_failing_model=True,
        expanded_heldout_inputs_opened=False,heldout_not_used_for_model_or_lambda_selection=True,
        geometry_accepted_for_zero_step=False,new_renderable_asset_authorized=False,
        first_family_selection_and_failure_preserved=True,
        next_diagnostic='Independent TOP pixel-pose / timestamp / sensor-to-ego-to-world height-chain audit before fitting more flexible static heightfield',
        no_additional_heightfield_fit=True,old_geometry_cameras_A20_appearance_unchanged=True,
        GPU_forward=0,backward=0,updates=0)
    target=D/'expanded_geometry_status.json';assert not target.exists();write(target,status)
    (D/'architecture_components.svg').write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="250" viewBox="0 0 1100 250">
<rect width="1100" height="250" fill="#111820"/><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#62c8ce"/></marker></defs>
<g fill="#1c2b38" stroke="#547b90"><rect x="20" y="60" width="220" height="130" rx="8"/><rect x="295" y="60" width="230" height="130" rx="8"/><rect x="580" y="60" width="225" height="130" rx="8"/><rect x="860" y="60" width="220" height="130" rx="8"/></g>
<g fill="#edf4f8" font-family="Arial,sans-serif" font-size="18" text-anchor="middle"><text x="130" y="95">8 official-road frames</text><text x="130" y="126">requested XY + 0.4m</text><text x="130" y="157">3276 untrimmed points</text><text x="410" y="95">Training-only LOO</text><text x="410" y="126">plane / quadratic ridge</text><text x="410" y="157">9 models, one-SE</text><text x="692" y="95">Fixed per-frame audit</text><text x="692" y="126">median 4cm / p95 10cm</text><text x="692" y="157">all nine fail</text><text x="970" y="95">Candidate retained</text><text x="970" y="126">same-XY height conflict</text><text x="970" y="157">world-height audit next</text></g>
<g stroke="#62c8ce" stroke-width="2" marker-end="url(#a)"><path d="M240 125H288"/><path d="M525 125H573"/><path d="M805 125H853"/></g>
<text x="550" y="225" text-anchor="middle" fill="#adbdca" font-family="Arial,sans-serif" font-size="16">No heldout tuning, source fitting, Gaussian update, or geometry activation.</text></svg>''',encoding='utf-8')
    plane=next(x for x in selection['models'] if x['name']=='plane');best=next(x for x in selection['models'] if x['name']==selection['best_mean_model'])
    text=f'''# 扩域几何候选：第一轮模型族未通过

1844几何候选完整性通过，但扩展范围的物理高度资格未通过，不批准零步资产初始化。新目录保留3276个原始官方ROAD18/LANE19训练点、1289个0.1m medoid、全部9模型和8个LOO折，不修改原b5df或1844文件。

![组件](architecture_components.svg)

训练范围由五视图筛选前允许编辑区域的世界footprint及既定0.2m环确定，再加0.4m小context。八帧原始点数为204、252、334、472、498、510、506、500；只按finite与XY范围取点，不按Z、模型残差、OBB或照片语义预先裁点。实际请求区域内1426点的原分母全部保留。

原模型族和选择规则保持：斜面、8种曲率正则二次曲面，Huber 3cm与IRLS8，按frame留一并评价该frame全部原始点，one-SE偏好低曲率。斜面LOO MAE={plane['mean_fold_MAE_m']:.8f}m；均值最优二次λ=.1为{best['mean_fold_MAE_m']:.8f}m，改善约1.01mm，小于折间标准误约14mm，one-SE仍选斜面。

独立逐点审计使用绝对高度median≤4cm、p95≤10cm。9模型全部不能在所有实际请求frame中通过。斜面LOO实际raw60 median/p95为13.067/14.793cm，raw70为6.460/8.149cm；最优二次仍约13.044/14.688cm、6.560/8.284cm。低平均MAE不能替代逐帧物理门。

同XY≤10cm的跨frame官方点核查还发现raw60相对后续frame高约15.5cm，raw70约8–9.5cm，说明冲突不只是宽padding或空间取样位置不同。下一步应独立核对TOP pixel-pose、时间戳与sensor→ego→world高度链，先不让更高自由度静态曲面吸收帧间冲突。

五张1152×768几何context已独立目视，无明显翻折或粗坐标错误。照片中的原A9是目标场景待移除车辆，网格位于其后方；这类context不证明被遮道路的实际高度。扩展asset有66.5%顶点超出原477拟合点凸包，条件bootstrap SD不包含模型误差。原几何/K/车辆/来源图未改变，当前未打开新expanded heldout3输入，无新增GPU或更新。

关键收据：`expanded_geometry_status.json`、`training_only_CV_selection.json`、`training_fold_original_height_audit.json`。第一轮模型族失败保留，候选与失败不升级为整场景恢复成功。
'''
    (D/'README.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(complete=True,status=pin(target))),flush=True)


if __name__=='__main__':main()
