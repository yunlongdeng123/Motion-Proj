"""Create a small report/source overlay, excluding matrices and in-flight data."""
import hashlib,json,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parents[1]/'outputs/scene008_vehicle_appearance/diagnosis'
README='''**Scene008 a6 插入成像诊断与观测锚定连续表面**

task/run：`scene008_a6_insert_artifacts_20261105`。本轮定位旧版本已存在的删 a37 黑斑，并修复 projective chart 的深度连续性与生成前景 RGBA 接口。PVD 与预训练模型训练更新为零；背景源参数保留，原 M2、旧表面和诊断控制均保留。候选不自动采用，人工 verdict 未代填。

[静态交互报告与实际图](delivery/index.html) · [完整固定指标及尾部审查](compact_evidence/evidence/surface/review.json) · [紧凑源码与证据清单](compact_evidence/MANIFEST.json)

![architecture](compact_evidence/figures/surface/architecture.png)

黑斑诊断使用同 raw105/113/130 相机和原 raster，保持几何、排序、alpha，仅逐组中和 RGB，分解 static、road、asset 与环境透射项；没有预设物理阴影。raw105 圈定黑核293像素的主要暗色载体为 inherited static，raw113对应暗核主要由 road 基色携带。road1K 的 road RGB/几何/opacity 原本就冻结，不能说这段1K训练学入了阴影。

raw105 原圈定核的 static 条件 camera-z=8.96477m、road=9.22039m，static 相对现有道路局部切平面约 .05757m；同域 keep-a37 条件深度5.42965m、50个实际 a37 LiDAR点均值5.42220m。载体几何接近道路而非当前车尾牌照深度，不能据此判定历史出身、语义身份或物理 cast shadow。24-feature 安全累积前后 RGB/alpha/depth_sum 逐字相同。[来源与几何证据](compact_evidence/evidence/shadow/source_audit.json)、[条件几何v2](compact_evidence/evidence/shadow/conditional_geometry_v2.json)。

连续 surface 以 reference106 UV 上 tensor cubic B-spline 表示 camera-z。控制间距由 source PVD / BUILD实测物理采样 pitch 自动确定（.09022m，约11.862HRpx，30×24控制点），没有网格搜索。prior sigma=.20m、合法可见 BUILD40 LiDAR sigma=.05m、二阶控制差分 sigma=.05m；三项分别按样本数归一，固定5次 Huber IRLS。每 HR像素只取 frontmost 实测样本，与 prior 偏差>.75m 的684点标为关联不确定，不强行拟合；5063点用于锚定。真实几何沿原 reference rays 更新，轨迹与投影UV不自由调整，kernel按新z维持 .6HRpx。[完整深度合同](compact_evidence/evidence/surface/depth_receipt.json)。

前景合同来自固定 strongv2 的单次 SAM2 分割，只 binary-fill 内孔、不膨胀，并 union 真106已知域。保留42536 chart行与2778原 source行，总45314。非真实 chart opacity=.99×sampled_alpha，真106=.99，零alpha非真实点不出生；RGB使用前景 premultiplied/unpremultiplied 采样，不按亮度删黑轮胎，不继承旧 formal 的粗糙深度补偿色。几何后验确实改变；后续1000颜色拟合才冻结新几何。

颜色拟合保留原394206个 BUILD40 HR像素及raw顺序，逐字等同旧拟合域；没有用新alpha筛掉难目标，全部目标alpha>.9804。执行32步真实 smoke、保存恢复及下一阶段插入，正式1000步；两段各2次 disposable resume-nextstep校验，合计1036次颜色更新计算，正式导出仍1000。25095个零贡献列精确保留 matte 初值。canonical资产位于[导出NPZ](delivery/data/surface_fit/combined_v1/observed_fit/formal/complete_a6.npz)，可由原 Runtime 的同一 insertion入口加载。[正式回执](compact_evidence/evidence/surface/formal_receipt.json)、[预算](compact_evidence/evidence/surface/compute_ledger.json)。

同一 canonical 资产保存 native480 与固定解析面积补偿两种 viewport，HR1920仅黑底细节、补偿逐字恒等。旧 projective 同时做 native/area 对照。负 FoV、编译 rasterizer 与相机轨迹保持原样。固定49帧的 RGB、真实 LiDAR 以及 keep/delete 状态 a37/road/outside 成像变化均已实际测量；分母来自原观测，未按新 alpha 缩小。

|固定内部比较|旧 projective|新连续 surface + matte|
|---|---:|---:|
|HR all49 straight RGB MAE|.042121|.041305|
|HR HELD9 straight RGB MAE|.040365|.039359|
|native480 all49 black RGB MAE|.065308|.072399|
|native480 all49 LiDAR z MAE (m)|.317099|.201782|
|双方 area480 all49 black RGB MAE|.048629|.047182|
|双方 area480 all49 LiDAR z MAE (m)|.270509|.168410|

HR和深度有所改善，native480 RGB反而退化；面积补偿会使新车 visible joint alpha 均值 .998828→.976360。因此 viewport 诊断与几何评价分开，不凭清晰度自动采用。原始 [49帧汇总](compact_evidence/evidence/surface/summary.json)、[HR汇总](compact_evidence/evidence/surface/HR_summary.json) 保留全部阈值计数和分母；initial HR只覆盖4个预览视角，不误当all49。

深度相邻边 erode1 p99=.237219→.027087m、>10cm边5684→0，但移动>.6m的点有2210/42536 (5.196%)；最大前移2.70124m、最大锚点残差2.62045m仍保留。这些点多数为高alpha内部区域，不只是低alpha边缘。matched out-of-box总1231→1221，chart1075→1065；没有硬边界或重调sigma掩盖尾部。[尾部证据](compact_evidence/evidence/surface/tail_audit.json)。

源参数冻结不表示最终图像不变。native新旧 keep-a37 RGB最大变化.015596，0/536128像素>.02；keep-road有6/1075281像素>.02；keep-outside有6043/6312448像素>.02、最大.723011，delete-outside有9004像素>.02。变化的RGB、alpha、depth及实际测量误差均报告，没有以同参数代替成像验收。删除状态的旧a37像素不当隐藏道路真值。

HELD9未进入本次颜色 objective，但raw113等此前已用于薄片、球核、表面缺陷诊断和内部方法开发。这些数值不构成独立未触碰泛化测试；生成图也不是隐藏表面真值。原数值回执保留，[角色说明](compact_evidence/evidence/surface/evaluation_roles.json) 修正历史final-only标记的解释。

SurfaceLOD 使用同一已冻结连续 surface S 与正式 canonical RGBA纹理 F，按投影面积 Jacobian 自动选择采样间距，固定mip过滤 premultiplied RGBA后再unpremultiply；没有新增纹理、几何拟合或参数搜索。保留2778个原 source行；camera改变采样surfel的数量、mean与sigma，不能说渲染primitive几何逐字相同；持续 surface / texture 定义与 canonical NPZ不改。实际49帧完成55.27秒，必要覆盖读出13.92秒。raw106/113/130 的du为4.00/4.49/6.18。

|native480固定49指标|旧 native|新 native|新 SurfaceLOD|
|---|---:|---:|---:|
|black RGB MAE，39308真实visible像素|.065308|.072399|.054946|
|LiDAR z MAE，8930点(m)|.317099|.201782|.178226|
|joint keep asset alpha*T均值|.998814|.998828|.995630|
|joint visible alpha<.9数量|56|60|93|
|solo visible alpha<.9数量|0|4|24|
|solo visible alpha<.5数量|0|0|4|
|预先erode1 interior alpha<.9，33331像素|0|0|1|

全49不能声称无新孔：LOD对newnative在joint visible新增33个alpha<.9，预先interior新增1个（raw91，joint alpha=.841326）；visible的4个solo alpha<.5分别在raw89/90/91/92。零覆盖≤1e-6均为0。reviewed边界域45418像素的LOD joint alpha<.5为1173，另域报告，不与真实visible混为一谈。未为这些计数修改方法或补洞。

LOD相对旧native的keep RGB变化：a37为0/536128像素>.02；road为16/1075281；outside为6580/6312448，最大.723011。相对新native分别0、17、1890，outside最大.377866。这是改善候选的内部证据，不能声称其他成像逐字不变，也没有取消2.7m几何尾部的后续质量门槛。[LOD49完整汇总](delivery/data/scale/surface_lod_v1/evaluation49/summary.json)与[覆盖率原始读出](delivery/data/scale/surface_lod_v1/coverage49/summary.json)保留同分母细节；若静态报告复制布局不同，可按publication_manifest定位。

本轮只对raw105/106/113/130保存完整RGB PNG及crop replay，其余45帧保留真实指标，没有完整49帧RGB视频，未额外重放生成视频。含raw113下方旧暗斑的同一宽crop用于before/after，不能用只裁车的画面声称黑斑已消除。

源执行目录：`/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_insert_artifacts_20261105`；GPU Python=`/root/autodl-tmp/envs/drivingeditor/bin/python`。源码副本在`source/`，保留原固定路径合同，运行仍依赖既有 frozen controls Runtime 与原 observations，不是独立scene离线包。大CSR矩阵、checkpoint和全float重放保留在run及本地原始取回包，未加入本紧凑Git材料。failure_ledger_refs=[]，failure_ledger_delta=none；human_verdict=null。
'''

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    expected='170ccef3351164a1ce8660a07d2f906f8e522fd406c9b0956de4b1eae56a0191'
    assert sha(OUT/'index.html')==expected,'Use the corrected explicit three-view publication'
    assert sha(ROOT/'report/publish_diagnosis.py')=='fde076740c794fe6bd0760fc01815f0b0ca8ab158f300a508dd848643dae2f5e'
    qa=json.loads((OUT/'publication_qa.json').read_text(encoding='utf-8'))
    assert qa['HTML_sha256']==expected and qa['local_link_QA'] and qa['decoded_image_QA']
    dest=ROOT/'publication';dest.mkdir(exist_ok=True);readme=dest/'research_README.md';readme.write_text(README,encoding='utf-8')
    attributes=dest/'research.gitattributes'
    attributes.write_bytes(('# Preserve exact reviewed artifact/source bytes and valid Windows line endings.\n'
        '* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n'
        'delivery/** -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n'
        'source/** -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n'
        'compact_evidence/** -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n').encode('utf-8'))
    items=[(p,'delivery/'+p.relative_to(OUT).as_posix()) for p in OUT.rglob('*') if p.is_file()]
    items += [(p,'source/'+p.relative_to(ROOT).as_posix()) for p in ROOT.rglob('*.py')
        if not any(x in p.parts for x in ['compact_git_v1','before_role_correction','retrieved_bundle_20261105']) and '.before_' not in p.name]
    for name in ['final_result_manifest.json','evaluation_roles.json']:
        items.append((ROOT/'report'/name,'source/report/'+name))
    items.append((ROOT/'review/final_report_review.json','publication_review/final_report_review.json'))
    items.append((ROOT/'report/navigation_source_smoke.json','publication_review/navigation_source_smoke.json'))
    items.append((ROOT/'review/navigation_source_fix_review.json','publication_review/navigation_source_fix_review.json'))
    compact=ROOT/'surface_fit/compact_git_v1'
    items += [(p,'compact_evidence/'+p.relative_to(compact).as_posix()) for p in compact.rglob('*') if p.is_file()]
    items += [(readme,'README.md'),(attributes,'.gitattributes')]
    target=dest/'publication_stage_v1.tgz';rows=[];seen=set();asset='data/surface_fit/combined_v1/observed_fit/formal/complete_a6.npz'
    remote_sources={sha(ROOT/'surface_fit/combined_v1/observed_fit/formal/complete_a6.npz'):'/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_insert_artifacts_20261105/surface_fit/combined_v1/observed_fit/formal/complete_a6.npz',
        sha(ROOT/'surface_fit/masked_depth_v1/posterior.npz'):'/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_insert_artifacts_20261105/surface_fit/masked_depth_v1/posterior.npz'}
    with tarfile.open(target,'w:gz') as tar:
        for source,name in items:
            if name in seen:raise ValueError(name)
            seen.add(name);row=dict(path=name,bytes=source.stat().st_size,sha256=sha(source))
            if name.startswith('delivery/') and row['sha256'] in remote_sources:
                row['remote_origin']=remote_sources[row['sha256']]
            else:tar.add(source,arcname=name,recursive=False)
            rows.append(row)
        manifest=dest/'publication_manifest.json';manifest.write_text(json.dumps(dict(phase='fixed_LOD49_review',files=rows,total_bytes=sum(x['bytes'] for x in rows)),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        tar.add(manifest,arcname='publication_manifest.json',recursive=False)
    print(json.dumps(dict(files=len(rows),bytes=sum(x['bytes'] for x in rows),archive_bytes=target.stat().st_size,archive=str(target),sha256=sha(target))))

if __name__=='__main__':main()
