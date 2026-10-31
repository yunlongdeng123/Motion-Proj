"""检查完整帧分母、实际编码帧数、别名和画面边界；不把文件存在当播放成功。"""
import json,pathlib,subprocess,sys,time
BASE=pathlib.Path(__file__).resolve().parent
def probe(path):
    p=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,avg_frame_rate,codec_name,pix_fmt,duration','-of','json',str(path)],capture_output=True,text=True,check=True)
    return json.loads(p.stdout)['streams'][0]
def audit(key):
    cfg=json.loads((BASE/'scenes.json').read_text(encoding='utf8'));scene=next(s for s in cfg['scenes'] if s['scene_key']==key)
    root=BASE/'generated/scenes'/scene['context_id'];media=json.loads((root/'media.json').read_text(encoding='utf8'));manifest=json.loads((root/'manifest.json').read_text())
    n=scene['expected_frames'];assert len(media['frames'])==n==manifest['raw_frame_count']
    assert [f['raw'] for f in media['frames']]==list(range(n))
    ts=[f['timestamp_micros'] for f in media['frames']];assert all(a<b for a,b in zip(ts,ts[1:]))
    assert set(media['actors'])==set(scene['actors'])
    assert all(a['alias']==scene['actors'][track]['alias'] for track,a in media['actors'].items())
    assert len(list((root/'frames').glob('*.jpg')))==n
    videos={}
    for name in ('front_rgb.mp4','front_boxes.mp4'):
        p=probe(root/name);assert int(p['nb_frames'])==n and [p['width'],p['height']]==media['imageWH']
        assert p['codec_name']=='h264' and p['pix_fmt']=='yuv420p' and p['avg_frame_rate']=='10/1'
        videos[name]=dict(**p,bytes=(root/name).stat().st_size)
    edges=0;front=0;hidden=0;iou=[]
    for f in media['frames']:
        for b in f['boxes']:
            if not b['official_FRONT_visible']:assert not b['edges'];hidden+=1
            else:front+=1
            for e in b['edges']:
                assert len(e)==4 and all(0<=v<media['imageWH'][i%2] for i,v in enumerate(e));edges+=1
            c=b.get('projection_comparison')
            if c and c['IoU'] is not None:iou.append(c['IoU'])
    for track,a in media['actors'].items():
        assert (root/'lidar'/('actor_'+a['alias']+'.npz')).exists()
        assert (BASE/'generated'/a['lidar']['data_js']).exists()
        assert (BASE/'generated'/a['lidar']['frame_data_js']).exists()
    return dict(scene_key=key,context_id=scene['context_id'],status='passed',raw_frames=n,actors=len(media['actors']),
        measured_box_rows=manifest['measured_box_rows'],source_timestamp_start=ts[0],source_timestamp_end=ts[-1],
        source_time_gaps_micros_min=min(b-a for a,b in zip(ts,ts[1:])),source_time_gaps_micros_max=max(b-a for a,b in zip(ts,ts[1:])),
        official_FRONT_box_observations=front,non_FRONT_observations_without_edges=hidden,finite_clipped_edges=edges,videos=videos,
        source='actual ffprobe H264 frame count + frozen source expected denominator + alias/metadata bounds',
        limitation='NumPy projection is provided-image-pose approximation, not official rolling shutter parity')
def main():
    cfg=json.loads((BASE/'scenes.json').read_text(encoding='utf8'));done={};start=time.monotonic()
    while len(done)<len(cfg['scenes']):
        for s in cfg['scenes']:
            key=s['scene_key'];root=BASE/'generated/scenes'/s['context_id']
            if key in done or not (root/'frame_point_views.json').exists():continue
            done[key]=audit(key)
            (BASE/'video_inventory_audit.json').write_text(json.dumps(dict(schema='actor_complete_sequence_video_audit_v1',
                completed_scenes=len(done),scene_denominator=len(cfg['scenes']),actor_denominator=sum(len(s['actors']) for s in cfg['scenes']),
                rows=list(done.values())),indent=2),encoding='utf8')
            print(json.dumps(dict(scene=key,status='passed')),flush=True)
        if time.monotonic()-start>7200:raise TimeoutError('media did not all finish')
        if len(done)<len(cfg['scenes']):time.sleep(10)
    print(json.dumps(dict(status='complete',scenes=len(done),raw_frames=sum(x['raw_frames'] for x in done.values()),actors=sum(x['actors'] for x in done.values()))),flush=True)
if __name__=='__main__':main()
