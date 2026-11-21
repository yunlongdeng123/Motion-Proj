"""Second residual pass delivery; first pass remains independently inspectable."""
from pathlib import Path
import argparse,importlib.util,shutil
W=Path(__file__).resolve().parent
ROOT=W.parents[1]
spec=importlib.util.spec_from_file_location('surface_delivery',ROOT/'work/road_surface_adaptive_20261121/finalizer.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
D.WORK=W
D.F.REMOTE='/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2'
D.F.DEST=D.F.OUT/'surface_residual_r2';D.F.DEST.mkdir(parents=True,exist_ok=True)
D.EXPECTED_PARENT_TOTAL=595029
D.EXPECTED_PARENT_ACTIVE=161264
D.HISTORY_FILENAME='history_surface_firstwave.html'
D.HISTORY_TITLE='查看第一轮局部增密与诊断'
D.COVERAGE_DESCRIPTION='第二轮：只在认证残余中采样，先以实际联合成像核验车辆保护，再训练覆盖400步、外观1200步。覆盖采用纯hinge目标和原硬参数界限；同中心点不追加，首轮新点参数复用单列。'
def cumulative(data,metrics,manifest):
    original=D.F.OUT/'surface_adaptive/completed_r1/bench/metrics.json'
    if original.exists():
        source=D.F.read(original)
        data['cumulative_baseline']=[r for r in source['blackgap_rows']if r['id']=='parent'and r['source']in ['C','S']]
        data['evidence'].append(dict(name='第一波原始评价（累计覆盖起点）',path=original.relative_to(D.F.OUT).as_posix()))
    for name in ['boxed_rgb_summary.json','boxed_rgb_audit.json','diagnostic_summary.json']:
        source=W/name
        if source.exists():
            shutil.copy2(source,D.F.DEST/name)
            data['evidence'].append(dict(name=name,path=(D.F.DEST/name).relative_to(D.F.OUT).as_posix()))
    if (W/'diagnostic_summary.json').exists():
        data['color_diagnostic']=D.F.read(W/'diagnostic_summary.json')
        target=D.F.DEST/'color_diagnostic';target.mkdir(exist_ok=True)
        archive=W/'numeric_color/numeric_color_packet.tar.gz'
        if archive.exists():
            receipt=D.F.read(W/'numeric_color/completed.json')
            if D.F.hashlib.sha256(archive.read_bytes()).hexdigest()!=receipt['bundle_sha256']:raise ValueError('numeric packet SHA mismatch')
            D.F.safe_unpack(archive,target)
        for name in ['numeric_summary.json','numeric_receipt.json','mask_inclusion_audit.json','precision_qualification.json','objective_identity.json','compute_ledger.json']:
            source=W/'receipts/color_optimality'/name
            if not (target/name).exists()and source.exists():shutil.copy2(source,target/name)
            if (target/name).exists():
                data['evidence'].append(dict(name=name,path=(target/name).relative_to(D.F.OUT).as_posix()))
    catalog_spec=importlib.util.spec_from_file_location('unified_report_catalog',W/'unified_catalog.py')
    catalog=importlib.util.module_from_spec(catalog_spec);catalog_spec.loader.exec_module(catalog)
    audit=catalog.attach(data,D.F.OUT)
    D.F.write(D.F.DEST/'catalog_validation.json',audit)
D.EXTRA_DATA_FN=cumulative
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--mark-pending',action='store_true');a=p.parse_args()
    if a.mark_pending:
        history=D.F.OUT/D.HISTORY_FILENAME
        if not history.exists():shutil.copy2(D.F.OUT/'index.html',history)
        banner='<section style="border:1px solid #77d6be;background:#17352f;padding:16px;border-radius:10px"><strong>第二轮认证残余修复已安排</strong><p>真实联合保护门通过13个旧锚点与1071个新位置；同中心候选不追加。下方保留已完成的第一轮结果。</p><a href="surface_residual_r2/run_status.json">第二轮队列状态</a></section>'
        (D.F.OUT/'index.html').write_text(history.read_text('utf-8').replace('<main>','<main>'+banner,1),'utf-8')
    else:D.F.main(a.once)
