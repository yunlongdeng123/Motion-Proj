import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]/'outputs/shadow_sequence_v2'
rows=json.loads((root/'final49_v1/per_frame.json').read_text())
for r in sorted([r for r in rows if r['final']['deleted_target_shadow_area_clean_MAE'] is not None],key=lambda r:r['final']['deleted_target_shadow_area_clean_MAE'],reverse=True)[:10]:
    print(r['frame'],r['deleted_target_associated_shadow_pseudo_samples'],*[round(r[v]['deleted_target_shadow_area_clean_MAE'],5) for v in ['original','deferred','final']])
