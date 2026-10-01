from collections import Counter
from admit_data import select

def test_scene_rotation_and_cap_preserve_rare_types():
    rows=[]
    for s in range(23):
        for j in range(6):rows.append({'case_id':f'M{s:02}_{j}','scene':f's{s:02}','type':'dense_actors' if j==0 and s==0 else 'single_actor' if j==0 else 'background'})
    out=select(rows);c=Counter(r['scene'] for r in out)
    assert len(out)==50 and len(c)==23 and max(c.values())<=3
    assert {'dense_actors','single_actor','background'}<={r['type'] for r in out}
