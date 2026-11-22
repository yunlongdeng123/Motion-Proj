"""Small evidence index and architecture for the limited inference geometry control."""
import argparse,json,hashlib
from pathlib import Path


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();R=a.root;D=R/'audit/local_surface_candidate_v2'
    review=read(D/'independent_limited_geometry_review.json');candidate=read(D/'candidate_geometry_receipt.json')
    evidence=['local_surface_geometry_v1/certificate.json','local_surface_geometry_v1/support_partition.json',
        'local_surface_inference_v1_replay1/inference_statistics.json','physical_crosscheck_v1/crosscheck_receipt.json',
        'physical_crosscheck_v1/01_pointcloud_coordinate_and_time.json','physical_crosscheck_v1/02_P01_P02_independent_physical_references.json',
        'physical_crosscheck_v1/03_parent_actual_geometry_normals.json','official_ground_geometry_v1/official_ground_point_receipt.json',
        'official_ground_profile_v1/official_profile_statistics.json','ground_contact_prior_v1/ground_contact_prior_audit.json',
        'ground_contact_prior_raw_GT_v2/ground_contact_prior_audit.json','curvature_ridge_loo_v1/training_only_CV_selection.json',
        'curvature_ridge_loo_v1/curvature_ridge_diagnostic.json','local_surface_candidate_v2/candidate_geometry.npz',
        'local_surface_candidate_v2/candidate_geometry_receipt.json','local_surface_candidate_v2/independent_limited_geometry_review.json',
        'local_surface_candidate_v2/five_view_geometry_v1/five_view_geometry_receipt.json']
    index=dict(schema='R10.local_geometry_evidence_index.v1',complete=True,current_geometry_sha256=review['candidate_geometry']['sha256'],
        independent_limited_decision=review['decision'],evidence=[pin(R/'audit'/x) for x in evidence],
        local_copy_paths_are_resolved_by_relative_audit_suffix=True,
        retained_remote_attempt='audit/local_surface_inference_v1 (partial before missing projection dependency); v1_replay1 complete; no GPU in either',
        original_assets_or_old_QA_overwritten=False,full_restoration_PASS=False,CPU_only=True,GPU_forward=0)
    (D/'local_geometry_evidence_index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="310" viewBox="0 0 1120 310"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#8ea6ba"/></marker></defs><rect width="1120" height="310" rx="14" fill="#111923"/><g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif" font-size="17" fill="#e7edf3"><text x="28" y="34" font-size="21">局部推断道路几何：输入、证据与限定零步对照</text><g fill="#1e3042" stroke="#587c96"><rect x="26" y="64" width="198" height="92" rx="10"/><rect x="262" y="64" width="220" height="92" rx="10"/><rect x="520" y="64" width="238" height="92" rx="10"/><rect x="796" y="64" width="294" height="92" rx="10"/></g><text x="43" y="94">原TFRecord / TOP语义</text><text x="43" y="122">ROAD + LANE实测XYZ</text><text x="280" y="94">训练帧留一选择</text><text x="280" y="122">曲率正则 / 局部斜面</text><text x="538" y="94">2cm世界格与cell中心</text><text x="538" y="122">紧凑重建区 + 0.2m环</text><text x="815" y="94">物理留出 / 不确定性 / 五视角</text><text x="815" y="122">仅零步模型推断几何意见</text><g stroke="#8ea6ba" stroke-width="2" marker-end="url(#a)"><path d="M224 110H258"/><path d="M482 110H516"/><path d="M758 110H792"/><path d="M939 156V202"/></g><g fill="#26362f" stroke="#6d9d82"><rect x="26" y="205" width="732" height="72" rx="10"/><rect x="796" y="205" width="294" height="72" rx="10"/></g><text x="43" y="235">GT车辆底面与原A20低点分位：只作接地交叉核验</text><text x="43" y="260">保留strict 837与失败候选；实测 / 模型推断标签分开</text><text x="815" y="235">Root批准 + 独立共享外观验收</text><text x="815" y="260">→ 原生联合零步对照</text></g></svg>'''
    (D/'architecture_components.svg').write_text(svg,encoding='utf-8')
    text=f'''# 局部道路模型推断几何：限定零步对照

独立意见为 `{review['decision']}`，绑定候选 `b5df7c8cb2b99723e1fabf3b2ca4c958c88003d38f4f528bb6dd8f14e97a011a`。候选可进入独立共享外观验收和原生联合零步对照；Root批准另存。全道路几何、蓝灰残影全覆盖和最终恢复质量仍单独验收。

![architecture](architecture_components.svg)

## 点云坐标与物理参考

原TFRecord的86/110/137重解码六列与既有bin逐字节一致；pose、timestamp与观测绑定一致。TOP逐像素运动补偿后已转同帧ego，世界坐标由同帧ego_world恢复。第六列是TOP return/row/column编码，非TOP为-1。P01/P02旧plane点与原点云world重载距离为0，独立局部profile对旧物理plane四角误差最大0.77/1.12cm。

父道路真实XYZ三角法线也存在较大边界差，不能只把stored normal当物理几何。原严格证书837格均位于旧parent已支持范围，新洞实测资格仍0；旧证书和未正则推断候选完整保留。

## 官方道路点与模型选择

官方proto核ROAD=18、LANE_MARKER=19；八个训练帧共1531个局部官方道路点，TOP两return编码、有效/NLZ行与语义逐行精确对齐。原OBB和photo-road0只保留为独立标记，未默认删除官方道路点。其中156点在扩张OBB内，209点在原照片非road0位置；没有点直接落入旧parent缺口。

训练帧LOO对plane与固定曲率正则网格评估全部留出训练点，不裁分母。one-SE规则选择局部斜面，之后才读取134/179/198三帧物理留出检查；三帧高度p95约2.53/3.58/3.70cm，法线p95约13.78/9.92/12.90度。GT底面没有加入拟合。

## 候选范围与接地检查

Vertex上下文为151×334、2cm全局整数UV，共50434个有限假设点。Cell为150×333，中心在UV+.5解析求值。实际asset仅{candidate['cell_asset_count']}个cell：紧凑重建{candidate['cell_reconstruction_count']}，0.2m过渡环{candidate['cell_blend_ring_count']}，面积约{candidate['cell_asset_surface_area_xy_m2']:.3f}m²；全部asset cell是模型推断，未冒称洞内LiDAR观测。条件bootstrap高度标准差在asset内中位约1.8cm，属于模型敏感性，非标定概率。

原GT A20底面对相邻官方road的八训练帧中位偏差约-2.87cm，A9约-1.02cm。候选道路在86/110的A20底面中心上方4.01/4.81cm；原ab4高opacity活跃Gaussian均值z1%在GT底面上方约3.98cm。这支持进入接地零步对照，不能把box底面或Trellis极小Z当稠密地面GT。v1的obs维度严格相等谓词false保留；v2直接使用原TFRecord GT pose/dimensions，另记微米级派生尺寸差。

## 五视角与目标覆盖

五张1152×768原尺寸几何投影已目视核验，无明显翻折或粗坐标错配。固定红块357/357、692/692和紧凑残影1923/1923、3288/3288像素均在asset范围且不属body/K；原蓝灰与全部band只部分覆盖，完整分母和未覆盖计数保留。原照片A9遮挡区的网格仅说明目标地面的投影位置，照片只作context，不能当新的道路RGB监督。

原NPZ的renderable=false保持候选验收状态。限定意见不修改该文件；后续证书以同SHA和cell_asset_mask授权，仍需Root、共享外观及真实原生联合画面各自门控。

关键入口：`candidate_geometry.npz`、`independent_limited_geometry_review.json`、`local_geometry_evidence_index.json`、`five_view_geometry_v1/`。本目录准备和审计全部CPU，新增GPU前向/反向/更新均0。
'''
    (D/'README.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(complete=True,index=pin(D/'local_geometry_evidence_index.json'),README=pin(D/'README.md'),architecture=pin(D/'architecture_components.svg'))))


if __name__=='__main__':main()
