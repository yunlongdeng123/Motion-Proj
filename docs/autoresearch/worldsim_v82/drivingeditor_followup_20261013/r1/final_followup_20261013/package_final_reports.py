"""Package existing real report assets without model payloads or nested ZIPs."""
from pathlib import Path
import json,zipfile,datetime
W=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-12/w');O=W/'outputs'
folders=['drivingeditor_four_arm_30k','drivingeditor_GMC_GN11_30k','drivingeditor_red_GN11_30k']
review=json.loads((O/folders[-1]/'engineering-visual-review.json').read_text(encoding='utf-8-sig'))
assert review['review_complete'] is True and review['human_verdict'] is None
png=[len(list((O/f/'assets').rglob('*.png'))) for f in folders]
assert png==[1764,441,441],png
paths=[p for f in folders for p in (O/f).rglob('*') if p.is_file() and p.suffix not in {'.zip','.pth','.tar','.gz'}]
paths += [O/n for n in ['DrivingEditor_final_report.html','DrivingEditor_final_summary.json','DrivingEditor_resource_and_speed.png','DrivingEditor_resource_and_speed.pdf']]
paths += [p for p in [O/'DrivingEditor_final_browser_QA.json'] if p.exists()]
dest=O/'DrivingEditor_final_reports.zip';assert not dest.exists()
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in paths:z.write(p,p.relative_to(O).as_posix())
with zipfile.ZipFile(dest) as z:
    # Decompression verifies the portable archive entries; no new fingerprints.
    assert z.testzip() is None
    counts=[sum(n.startswith(f+'/assets/') and n.endswith('.png') for n in z.namelist()) for f in folders]
    assert counts==png
    for f in folders:assert f+'/index.html' in z.namelist()
    assert 'DrivingEditor_final_report.html' in z.namelist()
receipt={'status':'REAL_REPORTS_OFFLINE_PACKAGE_VERIFIED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'path':str(dest),'bytes':dest.stat().st_size,'files':len(paths),'asset_PNG_counts':dict(zip(folders,png)),'total_assets_PNG':sum(png),
 'archive_decompression_passed':True,'complete_models_included':False,'remote_original_models_preserved':True,'human_GN11_verdict':None}
(O/'DrivingEditor_offline_package_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=True))
