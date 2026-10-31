"""逐帧独立采样实际NPZ；绝不从全片voxel集合按raw反筛。"""
import json,pathlib,sys,numpy as np
BASE=pathlib.Path(__file__).resolve().parent
def run(root):
    if str(root).startswith('waymo-'):
        cfg=json.loads((BASE/'scenes.json').read_text(encoding='utf8'))
        scene=next(x for x in cfg['scenes'] if x['scene_key']==str(root));root=BASE/'generated'/'scenes'/scene['context_id']
    root=pathlib.Path(root);media=json.loads((root/'media.json').read_text(encoding='utf8'));context=media['context_id'];total=0;observations=0
    for track,actor in media['actors'].items():
        name='actor_'+actor['alias'];data=np.load(root/'lidar'/(name+'.npz'));raw=data['raw_frame'];xyz=data['actor_local_xyz'];rgb=data['rgb'];source=data['source_cloud_row'];frames={}
        for t in actor['trajectory']:
            r=t['raw'];lo,hi=np.searchsorted(raw,[r,r+1]);points=xyz[lo:hi]
            if len(points):
                _,keep=np.unique(np.floor(points/.06).astype(np.int32),axis=0,return_index=True);keep.sort()
                if len(keep)>512:keep=keep[np.linspace(0,len(keep)-1,512,dtype=int)]
            else:keep=np.empty(0,dtype=int)
            ix=keep+lo
            frames[str(r)]=dict(xyz=xyz[ix].round(4).tolist(),rgb=rgb[ix].tolist(),source_cloud_row=source[ix].astype(int).tolist(),
                measured_points_before_voxel=int(hi-lo),display_points=len(ix),voxel_m=.06,display_cap=512)
            total+=len(ix);observations+=1
            if hi>lo:assert len(ix)>0
        path=root/'lidar'/(name+'_frames.js')
        path.write_text('window.ACTOR_LIDAR_FRAMES=window.ACTOR_LIDAR_FRAMES||{};window.ACTOR_LIDAR_FRAMES['+json.dumps(context+'|'+track)+']='+json.dumps(frames,separators=(',',':'))+';',encoding='utf8')
        actor['lidar']['frame_data_js']=f'scenes/{context}/lidar/{name}_frames.js'
        actor['lidar']['frame_display_contract']='per-frame independent 6cm voxel, deterministic cap512; absent label frame omitted, measured zero kept explicit'
    (root/'media.json').write_text(json.dumps(media,ensure_ascii=False,separators=(',',':')),encoding='utf8')
    (root/'media.js').write_text('window.ACTOR_MEDIA=window.ACTOR_MEDIA||{};window.ACTOR_MEDIA['+json.dumps(context)+']='+json.dumps(media,ensure_ascii=False,separators=(',',':'))+';',encoding='utf8')
    summary=dict(schema='independent_frame_point_views_v1',scene=media['scene_key'],actors=len(media['actors']),label_observations=observations,
                 display_points=total,per_frame_cap=512,nonzero_measured_frames_never_empty=True,aggregate_subset_not_reused=True)
    (root/'frame_point_views.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if __name__=='__main__':run(sys.argv[1])
