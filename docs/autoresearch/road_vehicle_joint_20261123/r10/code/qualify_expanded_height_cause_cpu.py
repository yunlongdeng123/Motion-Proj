"""Explicit interpretation correction; preserve all original status, data and SHA."""
import argparse,json
from pathlib import Path
from diagnose_nonplanar_inference_cpu import read,write,pin


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);a=p.parse_args()
    R=a.campaign_root/'road_underbody_local_20261123_r10';D=R/'audit/expanded_curvature_loo_v1'
    early=D/'expanded_geometry_status.json';original=read(early)
    target=D/'height_cause_qualification_v2.json';assert not target.exists()
    chain=R/'audit/anomalous_raw_point_chain_v1/point_chain_summary.json';same=R/'audit/same_frame_surface_layers_v1/same_frame_surface_summary.json'
    value=dict(schema='R10.expanded_height_cause_qualification.v2',complete=True,
        decision='PHYSICAL_GEOMETRY_FAIL_CAUSE_UNDETERMINED',original_status_preserved=pin(early),
        numerical_model_selection_and_failures_unchanged=True,geometry_accepted_for_zero_step=False,
        correction='The first interpretation called same-XY10cm height differences frame bias too early. Different curb/top/bottom surfaces can lie within10cm XY; nearestXY alone does not distinguish physical layering from pose bias.',
        independent_same_frame_layers=pin(same),independent_record_TOP_chain=pin(chain),
        confirmed=['All9 plane/quadratic models fail original per-frame actual-scope height bands',
            'Original raw60/70 and3 comparison-frame bin redecodes/timestamps/poses match their exact records',
            'Independent float64 TOP world arithmetic differs from exported world by <=.000668m XYZ and <=8.7e-6m Z',
            'Same-frame10cm ROAD/LANE pairs do not show8cm steps in the sampled points'],
        not_yet_determined=['Whether cross-frame matched points lie on the same physical road surface',
            'Whether original official CP and independent Brown image positions support curb or asphalt qualification',
            'Physical accuracy of the dataset-provided per-pixel / frame poses, not just arithmetic consistency'],
        next_checks='Official Front CP vs Brown overlays, original-photo surface qualification, same-frame layer context; coordinate/time physical diagnosis only if those establish same asphalt',
        holdout_not_used_for_new_model_selection=True,new_heightfield_fit=False,old_files_or_RGB_or_geometry_modified=False,
        GPU_forward=0,backward=0,updates=0)
    write(target,value)
    (D/'README_cause_pending.md').write_text('''# 扩域几何首轮未通过，高度差原因待核

新增范围的9个斜面/二次曲面模型均未通过逐帧高度门，1844候选不批准零步资产。原选择、原始点、失败收据和全部SHA保持；本页修订原因解释，不修订数值结论。

![组件](architecture_components.svg)

此前将世界XY≤10cm的跨帧高度差直接称为“帧偏置”过早。curb上沿和下沿也可能相差数cm XY，因此这类近邻证据本身不能排除真实多层路面。

同帧原官方ROAD18/LANE19点在10cm范围内未发现8cm阶跃；同一context的官方CURB17和SIDEWALK22为0，但稀疏点中没有这些标签不能直接排除curb。世界分布图和PCA法线保留在`../same_frame_surface_layers_v1/`，所有原点的label由TOP编码精确回查。

raw60/70及110/145/165的原record CRC、全部6列.bin重解与frame/obs时间绑定均核对一致。独立float64传感器极坐标、extrinsic与SciPy欧拉旋转直接生成TOP worldXYZ，最大XYZ差0.668mm，最大Z差8.7微米，排除了本次导出浮点roundtrip产生15cm的解释；这不证明数据集pose物理精度无偏。原记录官方Front CP已另行导出，用于与Brown投影和原照片对照。

当前原因仍为待定，不能以曲面加自由度、颜色或尺度拟合隐藏冲突。既有source准备可作为颜色诊断保存，当前不新增GPU、不修改原车辆/道路/相机，不使用expanded heldout3选择模型。
''',encoding='utf-8')
    print(json.dumps(dict(complete=True,receipt=pin(target))),flush=True)


if __name__=='__main__':main()
