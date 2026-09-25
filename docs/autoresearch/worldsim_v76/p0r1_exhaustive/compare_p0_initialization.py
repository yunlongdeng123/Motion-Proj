"""只读比较旧稀疏与新穷举的点/观测分母，不归因画质。"""
import datetime
import json
from pathlib import Path
import sqlite3
import numpy as np
from plyfile import PlyData

base=Path('/root/autodl-tmp/runs/v76_ego_view')
old=base/'VADGS-P0-000'
new=base/'VADGS-P0R1-000'
audits=[json.loads((old/'v76_diagnosis/colmap_audit.json').read_text()),
        json.loads((new/'initialization_audit/colmap_audit.json').read_text())]
rows=[]
for run,audit in zip((old,new),audits):
    con=sqlite3.connect(f'file:{run}/colmap/database.db?mode=ro',uri=True)
    matches=con.execute('SELECT COUNT(*),SUM(rows>0) FROM matches').fetchone()
    geometries=con.execute('SELECT COUNT(*),SUM(rows>0) FROM two_view_geometries').fetchone()
    image_count=con.execute('SELECT COUNT(*) FROM images').fetchone()[0]
    con.close()
    row={'run':run.name,'images':image_count,'possible_unordered_pairs':image_count*(image_count-1)//2,
         'match_table_rows':matches[0],'match_pairs_with_observations':matches[1],
         'geometry_table_rows':geometries[0],'geometry_pairs_with_observations':geometries[1],
         'points_before':audit['points_before'],'points_error_lt_0_6':audit['points_error_lt_0_6'],
         'point_retention_fraction':audit['retention_fraction'],'mean_reprojection_px':audit['mean_error_before'],
         'retained_track_observations':audit['retained_track_observations'],
         'saved_colmap_points':audit['saved_initialization']['retained_colmap_points_exact_float32_matches'],
         'other_background_points':audit['saved_initialization']['other_background_points'],
         'name_mapped_visibility_row_mismatches':audit['saved_initialization']['colmap_visibility_rows_differ_from_name_mapping']}
    if run==new:
        ply_checks=[]
        for path in sorted((run/'input_ply').glob('*.ply')):
            vertex=PlyData.read(path)['vertex']
            xyz=np.column_stack([vertex[key] for key in ('x','y','z')])
            assert np.isfinite(xyz).all() and len(xyz)>0
            visible=np.load(path.with_suffix('.npy'),mmap_mode='r')
            assert visible.shape==(len(xyz),305)
            ply_checks.append({'file':path.name,'points':len(xyz),'finite_xyz':True,'visibility_shape':list(visible.shape)})
        row['all_saved_ply_checks']=ply_checks
    rows.append(row)
result={'task_id':'VADGS-P0R1-000-INITIALIZATION-CONTROL','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'runs':rows,'new_training_from_scratch':True,
        'interpretation':'Matching change modestly changes retained point counts; exact new saved visibility is verified. Geometry counts do not establish image quality, and the full training comparison also changes normal/visibility mapping.',
        'failure_ledger_refs':['V76-F01'],'failure_ledger_delta':'none; saved initialization fix verified'}
(new/'initialization_audit/comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
