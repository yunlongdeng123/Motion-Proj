"""发布C2失败结果；保留基线和用户未提交文件，不写入大权重到Git。"""
from pathlib import Path
import json
import shutil

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1')
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
DEST=REPO/'docs/autoresearch/scene008_a37_C2_20261103'

def put(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if src.suffix.lower() in ('.py','.json','.html','.css','.js','.md','.mmd','.svg','.txt'):
        dst.write_text(src.read_text(encoding='utf-8').replace('\r\n','\n'),encoding='utf-8',newline='\n')
    else:shutil.copy2(src,dst)

def main():
    assert (RUN/'formal/receipt.json').exists()
    assert json.loads((RUN/'formal/receipt.json').read_text())['updates']==1000
    assert (RUN/'report/result_report.html').exists()
    assert (RUN/'final_acceptance/summary.json').exists()
    assert not DEST.exists(),DEST
    shutil.copytree(RUN/'report',DEST)
    for src in (RUN/'source').rglob('*'):
        if not src.is_file() or src.is_symlink():continue
        rel=src.relative_to(RUN/'source')
        if any(x in rel.parts for x in ('vendor','python_overlay','torch_extensions','env','__pycache__','weights')):continue
        if src.name in ('aria2_minimal_inputs.txt',):continue
        if src.suffix not in ('.py','.mmd','.txt'):continue
        put(src,DEST/'source'/rel)
    fixed=['experiment_contract.json','settings_pvd.json','formal/contract.json','formal/receipt.json',
           'smoke_pvd32/receipt.json','smoke_pvd_restore34/restore.json','smoke_pvd_restore34/receipt.json',
           'final_acceptance/summary.json','final_acceptance/metrics.json','final_acceptance/restore.json',
           'final_acceptance/freeze_audit.json','final_acceptance/thresholds.json',
           'evidence/readout_oracle/summary.json','evidence/baseline_replay/summary.json',
           'evidence/road_measurement_correction_preformal.json',
           'evidence/actor_contributors_pvd/summary.json','evidence/actor_contributors_pvd/contract.json',
           'inputs/a6_observation_report.md','inputs/pvd_adapter_report.md',
           'inputs/actor_surface_pvd_seed0_receipt.json','inputs/a6_observation_manifest.json',
           'source/prior/prior_execution_receipt.json','source/prior/download_receipt.json',
           'source/prior/proposals/pvd_seed000/proposal_report.json',
           'source/prior/proposals/pvd_seed000/transform_contract.json',
           'source/prior/pvd/normalization_contract.json','source/prior/pvd/deployment_patch.json']
    copied=[];missing=[]
    for rel in fixed:
        src=RUN/rel
        if src.exists():put(src,DEST/'evidence'/rel);copied.append(rel)
        else:missing.append(rel)
    for src in (RUN/'source/prior/proposals').glob('*/prior_observation_fit.png'):
        put(src,DEST/'evidence/prior'/(src.parent.name+'.png'))
    for name in ('actor_surface_pvd_seed0_views.png','a6_measured_coverage.png'):
        if (RUN/'inputs'/name).exists():put(RUN/'inputs'/name,DEST/'evidence'/name)
    status='''# 当前研究状态

2026-11-03完成用户指定C2 MVP：scene008只删除a37、保留a6及其他车辆，直接对比“单a37身份10k＋最终r3道路观测修复1k”。本轮未达到用户要求，不替换原基线。

[C2完整报告、49帧对照和组件图](autoresearch/scene008_a37_C2_20261103/result_report.html)。通用Hunyuan3D-Omni的point/bbox两个控制候选分别出现超宽壳与欠尺寸，保留失败现场；随后仅使用一个任务匹配的PVD汽车局部点云补全先验，seed0、1000 DDPM采样步，200真实点固定、1848点生成，先验训练更新0。官方car/train精确归一化统计来自2458个通过ZIP CRC的原始训练样本，未用目标完整车辆GT或partial bounds推尺度。PVD提案通过受控优化准入，不等于真实形状或最终画质通过。

真实199帧原始RGB/LiDAR支持5976个固定a6观测anchors；PVD提供1527个合法表面patch，其中856个补入实测缺失下部。局部优化仅改变持久canonical有效RGB、缺失面≤3cm法向偏移和90369个静态贡献候选的opacity multiplier；原B10k、道路r3权重、相机、轨迹、全局环境和原时间状态均保持。候选不是归属GT。道路只用固定有效外观和低维几何遮光近似，不宣称真实BRDF/光照分解，没有独立学习阴影层。

正式从初始提案开始完成1000更新，124.92秒；短程保存、新进程完整参数/Adam/RNG/采样器恢复及下一入口均通过。额外优化546步单列，正式方法预算未被烟测续用。总先验采样1100步，先验训练0。49原生视图及全部199恢复输入均属BUILD；101/113/125不是独立测试。

C2不编辑完整图PSNR/SSIM/LPIPS为28.751205/0.917114/0.109048，原Native为30.454978/0.936408/0.086108。a6严格可见域删车MAE由强组合基线0.0308185升至0.0594123；未通过。保守道路域固定排除真实a6轨迹OBB的可能占据投影，双方同参考同分母；三项MAE比为1.00723/1.01120/1.00171，旧完整proxy仍原样报告，不将保留车应占据区域误当道路真值。

最后零更新诊断：三锚新a6独立alpha约0.9997，但删车合成alpha*T仅0.389–0.800，静态层贡献占约0.199–0.611，说明现有旧混合表示仍挡住新表面；同时实际颜色误差高于冻结状态下的乐观可达下界，外观拟合也未充分完成。不能据此否定所有先验或声称全方法不可达。下一研究入口是固定提案与数据，验证局部表示替换/可见性接口及外观拟合，不扩大actor队列。

原基线与所有失败/诊断checkpoint、输入和日志保留。报告与代码按当前分支提交推送，完整新现场及选定PVD权重另存远端与本地交付目录；取回与电源状态以交付回执为准。用户已授权资源用完后关机，必须在全部验收、推送、取回及其他活动检查完成后执行。

quality_accepted=false，human_verdict=null。failure_ledger_refs=[V82-F01,V82-F06]；failure_ledger_delta=更新V82-F06证据，无新增ID。
'''
    (REPO/'docs/RESEARCH_STATUS.md').write_text(status,encoding='utf-8',newline='\n')
    index=REPO/'docs/EXPERIMENTS.md'
    entry='\n- `scene008_a37_C2_20261103_r1` — [冻结PVD补全＋局部优化1k，未超过身份10k＋道路1k基线](autoresearch/scene008_a37_C2_20261103/result_report.html)：a6可见域误差与完整重建均未过门槛，完整状态/49帧/成像阻挡诊断保留。\n'
    old=index.read_text(encoding='utf-8')
    assert 'scene008_a37_C2_20261103_r1' not in old
    index.write_text(old.rstrip()+'\n'+entry,encoding='utf-8',newline='\n')
    failure=REPO/'docs/research_failures/entries/V82-F06.md'
    addition='''

## 2026-11-03：C2冻结汽车补全先验与局部表示替换边界

task/run：`scene008_a37_C2_20261103_r1`。[完整组件、49帧、状态恢复和零更新诊断](../../autoresearch/scene008_a37_C2_20261103/result_report.html)。仅删a37、保留a6，原单a37身份10k＋道路修复1k不重训。PVD预训练汽车partial→full先验固定seed0完成1000 DDPM步，200实际点固定、1848点生成；先验训练0。5976真实anchors固定，合入1527个合法生成patch，局部颜色/≤3cm缺失面偏移/90369个静态贡献候选门控训练1000步，额外工程更新546步另列。

完整不编辑PSNR30.454978→28.751205、SSIM0.936408→0.917114、LPIPS0.086108→0.109048；a6严格可见域删车MAE0.0308185→0.0594123，未通过，不替换基线。只把旧delete后处理常驻启用的压力测试本身也会降质，不能把该压力测试当作原Native重建基线。

raw105/113/130独立新a6表面alpha约0.9997，但与旧场景共同渲染后delete alpha*T为0.80046/0.62938/0.38893，静态层为0.19926/0.37034/0.61070，道路为0。冻结全部几何/opacity/gates，仅用黑白颜色oracle诊断后逐张量恢复；实际颜色L1仍显著高于条件下界，不能将误差全归于遮挡，也不证明完整方法不可达。旧贡献候选约91%保护流量高于拟合流量；候选全删或二值改owner均无安全依据。

PVD形状准入不等于编辑通过。必须区分先验提案、原观测关联、表面成像资格及外观拟合；本轮仅说明当前软门控替换接口与优化没有同时维持完整重建和改善保留车。旧bundle的actor-list与box-list顺序不可靠，本轮已按真实track逐帧绑定a6，不用索引冒充身份。全部评价属BUILD，未知显露车体无独立RGB真值。quality_accepted=false、human_verdict=null，不否定所有先验。
'''
    text=failure.read_text(encoding='utf-8')
    assert 'scene008_a37_C2_20261103_r1' not in text
    failure.write_text(text.rstrip()+addition,encoding='utf-8',newline='\n')
    result=dict(destination=str(DEST),copied_evidence=copied,optional_missing=missing,
                source_parent_untouched=True,checkpoint_files_published_to_git=False)
    (RUN/'publication_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
