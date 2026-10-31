from pathlib import Path
import argparse,html,json,math,re

SCENE='14705303724557273004_3105_000_3125_000'
START='<!-- r4-candidate-preview-start -->'
END='<!-- r4-candidate-preview-end -->'
MARKER='<details class="advanced-controls"><summary>高级几何与语义设置</summary>'
COLORS=('#66d1ff','#f9bd62','#c49aff')

def esc(v): return html.escape(str(v),quote=True)
def make_panel(data, raw):
    if data.get('schema')!='r4_readonly_candidate_preview_v1': raise ValueError('Unexpected preview schema')
    if data.get('scene_ancestor')!='waymo_context_'+SCENE or data.get('scene_id')!='real_first_scene_native30k': raise ValueError('Preview does not match scene 1')
    if data.get('read_only') is not True or data.get('edited') is not False: raise ValueError('Preview must be read-only and unedited')
    groups=data['groups']; points=data['points']
    if not isinstance(groups,list) or len(groups)!=3 or not isinstance(points,list) or len(points)>100: raise ValueError('Expected three groups and at most 100 points')
    ids=[g['group_id'] for g in groups]
    if len(set(ids))!=3: raise ValueError('Duplicate groups')
    rows={g['group_id']:set(g['original_static_rows']) for g in groups}
    if any(len(rows[g['group_id']])!=len(g['original_static_rows']) for g in groups): raise ValueError('Duplicate rows within group')
    allrows=set().union(*rows.values())
    if sum(map(len,rows.values()))!=len(allrows) or len(allrows)!=data['candidate_count']: raise ValueError('Candidate count or group membership inconsistent')
    if data.get('preview_count')!=len(points): raise ValueError('Preview count inconsistent')
    verified=(data.get('coordinate_verification',{}).get('status')=='coordinates_match_existing_native_export' and data.get('coordinate_system')=='original_world_metres' and data.get('checkpoint_namespace','').endswith(':static_original_row'))
    ref='raw'+esc(raw); counts=''.join(f'<li><span style="color:{COLORS[i]}">●</span> 来源组 {i+1}：{len(rows[g["group_id"]]):,} 行</li>' for i,g in enumerate(groups))
    panel=f'{START}<details class="candidate-preview"><summary>本轮候选点</summary><p>固定参考帧：{ref}。这是只读候选预览，不是编辑结果，也不代表归属判定。</p><ul>{counts}</ul><p>候选行总数：{len(allrows):,}</p>'
    if verified:
        samples=[];seen=set()
        for point in points:
            gid=point['group_id']; row=point['original_static_row']; xyz=point['reference_world_xyz']
            if gid not in rows or row not in rows[gid] or row in seen or len(xyz)!=3 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in xyz): raise ValueError('Invalid preview point')
            seen.add(row);samples.append((gid,row,float(xyz[0]),float(xyz[1])))
        if samples:
            xs=[r[2] for r in samples];ys=[r[3] for r in samples]
            xmin,xmax=min(xs),max(xs);ymin,ymax=min(ys),max(ys)
            scale=min(280/max(xmax-xmin,1e-9),160/max(ymax-ymin,1e-9))
            xmid=(xmin+xmax)/2;ymid=(ymin+ymax)/2
            circles=''.join(f'<circle cx="{160+(x-xmid)*scale:.2f}" cy="{100-(y-ymid)*scale:.2f}" r="3" fill="{COLORS[ids.index(gid)]}"><title>来源组 {ids.index(gid)+1} · 原始行 ID {esc(row)}</title></circle>' for gid,row,x,y in samples)
            panel+=f'<p>俯视散点：显示 {len(samples)} 个采样点，使用与原生 30K 相同的世界坐标（米）。颜色对应来源组；悬停可查看原始行 ID。</p><svg viewBox="0 0 320 200" role="img" aria-label="候选点俯视散点" style="width:100%;max-width:420px;background:#0b1522;border-radius:6px"><path d="M20 180H300 M20 20V180" stroke="#5a7189" fill="none"/>{circles}</svg>'
        else: panel+='<p>没有可显示的候选采样点。</p>'
    else: panel+='<p>坐标尚未核验，暂不显示空间散点。</p>'
    return panel+'</details>'+END

def install(preview,root):
    data=json.loads(Path(preview).read_text(encoding='utf8')); scene_dir=Path(root)/SCENE
    scene=json.loads((scene_dir/'scene.json').read_text(encoding='utf8'))
    matching=[k for k,v in scene['frames'].items() if str(v['frame_id'])==str(data['reference_frame_id'])]
    if len(matching)!=1: raise ValueError('Reference frame does not map uniquely to this scene')
    panel=make_panel(data, matching[0])
    target=scene_dir/'index.html'; text=target.read_text(encoding='utf8')
    nstart=text.count(START); nend=text.count(END)
    if (nstart,nend)==(0,0):
        if text.count(MARKER)!=1: raise ValueError('Expected a unique advanced-controls marker')
        text=text.replace(MARKER,MARKER+panel,1)
    elif (nstart,nend)==(1,1):
        pattern=re.escape(START)+r'.*?'+re.escape(END)
        text,n=re.subn(pattern,lambda _:panel,text,count=1,flags=re.S)
        if n!=1: raise ValueError('Invalid marker ordering')
    else: raise ValueError('Invalid marker count')
    target.write_text(text,encoding='utf8')
    print('Installed read-only candidate panel; coordinate chart '+('enabled' if '候选点俯视散点' in panel else 'omitted'))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('preview');parser.add_argument('platform_root');args=parser.parse_args();install(args.preview,args.platform_root)
