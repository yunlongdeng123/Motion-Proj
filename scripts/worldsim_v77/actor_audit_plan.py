import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--work-root',type=pathlib.Path,default=pathlib.Path.cwd()/'work')
B=p.parse_args().work_root.resolve()
plan={}
for s,step in [('scene_0230',5),('scene_0255',10)]:
    sel=json.loads((B/'v77_blender_input'/s/'selection.json').read_text())
    rows=[]
    for f in range(0,step*10,step):
        candidates=[]
        for c,v in sel['camera_streams'].items():
            box=v['boxes'][f]
            if box is not None:
                # Favor boxes fully inside the image over edge-truncated boxes.
                complete=box[0]>3 and box[1]>3 and box[2]<684 and box[3]<380
                candidates.append(((box[2]-box[0])*(box[3]-box[1])*(1 if complete else .65),int(c),box))
        _,cam,box=max(candidates)
        rows.append({'frame':f,'camera':cam,'box':box})
    plan[s]={'actor_id':sel['actor'],'yaw_correction_deg':180 if s=='scene_0230' else 0,'frames':rows}
(B/'v77_actor_audit'/'view_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(plan,indent=2))
