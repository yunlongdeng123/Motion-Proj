"""记录代理图像复核与用户更新后的实验范围；不代填人工接受。"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
audit = json.loads((ROOT/'full49_independent_audit.json').read_text(encoding='utf-8'))
assert audit['all_checks_passed'] and audit['raws'] == list(range(89, 138))
review = dict(
    method='v44', run='full49_r1', mode='full49', date='2026-11-07', raws=audit['raws'],
    comparison_sha256=audit['comparison_sha256'], full49_executed=True,
    reviewer='Codex root agent: actual paired images and independent saved-float review',
    engine_status='PASS_FIXED49_FROZEN_ASSET_SHARED_NATIVE_SORT',
    agent_assessment='ROAD_BLUE_CORRECTION_SUPPORTED_WITH_LOCAL_REGRESSION_LIMITS',
    quality_accepted=None, human_verdict=None,
    user_scope_update='银色残片可以留到下一轮再改善，先解决这个蓝紫斑',
    silver_deferred_to_next_round=True,
    continuation_decision='银残片延期后完成49帧；本轮验证道路去蓝，保留当前完整车体，不宣称全部像素无回退。',
    summary='49帧两状态的原观测道路蓝色偏差均逐帧改善；a6整体RGB基本保持、没有新增低覆盖孔洞。少数车体像素及深度存在小幅回退；银色残片按用户要求延期。',
    observations=[
        '固定原观测道路1075281像素：keep绝对蓝色误差0.0165024→0.00818452，delete为0.0143270→0.00845001；两状态49帧的均值和超蓝计数均逐帧下降。原始天空和真实蓝车未被统一去色。',
        '全49帧delete道路proxy RGB MAE为0.01860616→0.00961900（287368像素，下降约48.3%）；proxy不是始终遮挡路面的独立真值。',
        'a6 delete RGB为0.05494726→0.05494506（39308固定像素），实测相机深度MAE为0.1719576m→0.1700301m（8930点）。整体接近不变不代表每帧无回退。',
        'a6 RGB局部最大回退raw137：delete +0.0005382，keep +0.0006829；该帧static在a6可见像素上的贡献升高。深度最大局部回退raw101约+2.03cm（delete）。',
        'a37 keep总体RGB为0.01472669→0.01491615，32/49帧回退，最大raw122 +0.0020273；已复查该帧实图，保留数字与图像，不宣称全部车辆无回退。',
        'a6 delete贡献<0.9像素24→24、<0.5像素4→4；keep <0.9为93→77、<0.5为4→4。已有低覆盖点没有被这轮消除。',
        'raw105引擎盖开发诊断中，真实车辆本身蓝色，整体RGB误差改善，但部分超蓝尾部加重；不把真实蓝色当成道路蓝斑，也不称整车颜色均已修好。',
        '原预登记蓝ROI评价域为空，保留null；有支持背景域补充使用原RGB，明确独立分母，未悄悄改写正式指标。',
        '银框净增亮已用补充3次forward分解，主要为static正增量与环境×T负增量抵消；这是成像来源，不证明训练根因或环境内禀纹理。该问题延期，不改变本轮去蓝评价范围。',
        '两端相同C2/RGBA/SurfaceLOD，keep/delete只物理移除a37；背景是RCR-1成套替换，11×9色场为冻结转移，没有针对新车体重新标定。',
    ],
    images_inspected=[
        'pilot_delete_board.png: 105/113/130/137 delete paired',
        'pilot_keep_board.png: 105/113/130/137 keep paired',
        'raw105_other_vehicle_detail.png: observed/current/v44',
        'full49_key_regressions.jpg: 89 delete,101 delete,122 keep,137 delete paired',
    ],
    independent_generalization_claim=False, new_training_updates=0, asset_updates=0,
    shutdown_performed=False,
)
(ROOT/'full49_visual_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
pilot = json.loads((ROOT/'visual_review.json').read_text(encoding='utf-8'))
pilot['mode'] = 'pilot'
pilot['raws'] = [105, 113, 130, 137]
pilot_audit = json.loads((ROOT/'pilot_independent_audit.json').read_text(encoding='utf-8'))
pilot['comparison_sha256'] = pilot_audit['comparison_sha256']
pilot['agent_assessment'] = 'PILOT_ENGINEERING_PASS_SILVER_DEFERRED_BY_USER'
pilot['summary'] = '四帧工程检查通过，道路去蓝明显；用户随后将银残片列为延期问题，因此继续了独立full49运行。四帧本身不当作49帧结论。'
pilot['continuation_decision'] = '原先暂缓判断保存在visual_review.json；用户更新范围后继续full49，主结果见完整49帧页。'
pilot['observations'][3] = 'raw130、137的银灰残片仍在，按用户要求延期；补充三次同排序delete诊断在指定ROI补齐current分组，净成像层差可加性通过，详见另存分层证据。'
pilot['supersedes'] = 'visual_review.json (historical pilot stop before user silver deferral)'
(ROOT/'pilot_visual_review_updated_scope.json').write_text(json.dumps(pilot, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(ROOT/'full49_visual_review.json')
