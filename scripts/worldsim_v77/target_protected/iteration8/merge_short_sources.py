"""新连续RGB全部到位后扩来源，既有候选/评测角色不改写。"""
from pathlib import Path
import sys,copy
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,F,T,read,dump
from assemble_sources import link
S=O/'short_sources/factory'
def main():
    if (O/'short_merge.json').exists():print('already merged');return
    new=read(S/'source_manifest.json')['clips'];ctx={c['source_id']:c for c in read(S/'source_context.json')['clips']};m=read(F/'source_manifest.json');cm=read(F/'source_context.json')
    used={tuple(f['filename'] for f in c['frames']) for c in m['clips']};count=max(int(c['source_id'][1:]) for c in m['clips'] if c['source_id'].startswith('N'));rows=[];missing=[]
    for c in new:
        if tuple(f['filename'] for f in c['frames']) in used:continue
        oldsid=c['source_id'];co=ctx[oldsid];required=[f['filename'] for f in c['frames']]+[co['frames'][i]['sensors']['LIDAR_TOP']['filename'] for i in [0,3,6]]
        if any(not (S/'rgb'/n).is_file() for n in required):missing.append({'source_id':oldsid,'files':[n for n in required if not (S/'rgb'/n).is_file()]});continue
        count+=1;sid=f'N{count:03}';c=copy.deepcopy(c);co=copy.deepcopy(co);c['source_id']=sid;co['source_id']=sid
        c['reuse_provenance']={'factory':str(S),'source_id':oldsid,'slice':[0,10],'source_protocol':'new actual-window bracket visibility/geometry, no masks yet'}
        m['clips'].append(c);cm['clips'].append(co)
        for n in required:link(S/'rgb'/n,F/'rgb'/n)
        rows.append({'source_id':sid,'original_source_id':oldsid,'scene':c['scene'],'camera':c['camera'],'actors':len(c['actors'])})
    dump(F/'source_manifest.json',m);dump(F/'source_context.json',cm);dump(O/'short_merge.json',{'sources':rows,'missing':missing,'scene_count':len({r['scene'] for r in rows}),'staging_only':True});print('MERGED_SHORT',len(rows),'scenes',len({r['scene'] for r in rows}),'missing',len(missing),flush=True)
if __name__=='__main__':main()
