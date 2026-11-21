"""从最终合法照片交集诊断及独立审计生成同页摘要。"""
from pathlib import Path
import argparse
import json


def build(dataset, output):
    dataset = Path(dataset)
    read = lambda p: json.loads((dataset / p).read_text(encoding='utf8'))
    fit = read('source_rgb_refit_qualified_r2/receipt.json')
    review = read('reference_audit/source_rgb_refit_qualified_r2_independent_cpu.json')
    ranking = read('source_view_ranking/manifest.json')
    retrieval = read('source_view_ranking/retrieval_audit.json')
    uv = read('reference_audit/photo_pair_uv_consistency.json')
    ledger = read('resource_ledger_cumulative.json')
    if not fit['complete'] or not review['frozen_parameters_passed'] or not review['photo_qualification_passed']:
        raise ValueError('最终源诊断的完成或约束核验未通过')
    totals = ledger['totals']
    if (totals['nativeF'], totals['nativeB'], totals['actual_optimizer_updates'],
        totals['final_formal_optimizer_updates'], totals['nonadopted_or_smoke_restore_updates']) != (1663,1624,812,400,412):
        raise ValueError('权威累计计算口径发生变化')
    before = {r['raw']:r for r in fit['original']}
    after = {r['raw']:r for r in fit['refit']}
    spectral = lambda r: min(r['spectral'], key=lambda s:abs(s['sigma_native_px'] - 5/3))
    queries = [86,110,173]
    hf = '、'.join(f"{raw}：{spectral(before[raw])['HF_cosine']:.3f}→{spectral(after[raw])['HF_cosine']:.3f}" for raw in queries)
    mae = '；'.join(f"{raw}：{before[raw]['RGB_MAE']:.5f}→{after[raw]['RGB_MAE']:.5f}" for raw in queries)
    summary = {
        'complete': True, 'schema':'source_quality_final_summary.v1',
        'adopted_source_diagnostic':'source_rgb_refit_qualified_r2',
        'quality_gate_passed':review['outcome']['source_RGB_refit_quality_gate_passed'],
        'verdict':'本轮现成源高斯搬运未成功；仅颜色修复未获跨视角纹理改善。',
        'findings':[
            '源00帧110的原源高斯实测HF对应度为0.094、HF RMS比为1.934（1920栅格，σ=1.2px）；高频幅值与实拍纹理对应分别统计。',
            '最终诊断仅优化源00核心1598个高斯的4794个颜色参数。位置、法向、尺度、透明度及核验边缘颜色全部冻结；评价域精确交照片合法掩码。',
            f'查询视角86/110/173均为历史建模视角。σ=1.667采集像素时，HF对应度分别为{hf}；RGB MAE分别为{mae}。三个查询视角在所测三档高频核上的对应度均回退。',
            f'照片137/173相机基线约{uv["baseline_137_173_m"]*1000:.2f}mm；基于平面及原高斯中心高度的近似UV重投影也呈较弱的细频对应。照片锐度、频带能量和跨视角结构对应分别核查。',
            '本轮停止新增目标搬运，颜色重拟合资产保留为诊断结果。若继续，应先建立并验证源资产全参数重建的质量。',
        ],
        'ranking_rows':[
            {'patch':r['candidate_id'],'qualified_views':r['photo_legal_views'],
             'max_short_axis_px':r['max_short_core_axis_native_px'],
             'top_frames':[v['raw'] for v in r['best_ranked_photo_views'][:5]]}
            for r in ranking['rows']
        ],
        'retrieval_scope':f"已扫描199帧，对既有三块排序参考视角。源块检索使用五个目标相机覆盖域内的{retrieval['occupied_4cm_cells']}个4cm强实拍格，面积代理{retrieval['occupied_grid_area_proxy_m2']:.4f}m²；17个提案经CPU筛选保留4个，去重保留3个。",
        'reference_semantics':'照片可用性、稠密几何支持和高斯保真分别记录。源00帧137/98分别只有12/30个独立4cm实拍见证格（核心共339格）；本轮源块与固定几何颜色诊断的结论限于该输入范围。',
        'resource_rows':[
            ['原源尺寸核查',8,0,0,0],
            ['旧源RGB试验 · 工程归档',827,812,0,406],
            ['合法域接线失败 · 工程归档',1,0,0,0],
            ['最终源RGB诊断',827,812,400,6],
            ['本轮合计',totals['nativeF'],totals['nativeB'],400,412],
        ],
        'source_actual_optimizer_updates':812,'final_source_formal_updates':400,
        'nonadopted_or_smoke_restore_updates':412,'new_target_migration_updates':0,'new_diffusion_calls':0,
    }
    Path(output).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=build(args.dataset,args.output)
    print(json.dumps({'complete':result['complete'],'adopted':result['adopted_source_diagnostic'],'quality_gate_passed':result['quality_gate_passed']}))
