"""只读取预先冻结开发样例的首237帧；不下载/查看保留集，不改变时间窗口。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import av
import numpy as np
from PIL import Image
from huggingface_hub import get_hf_file_metadata,hf_hub_url,get_token,hf_hub_download
from flashdreams.infra.runner_io import resize_rgb_image
from catalog_samples import RUN,DEST

BASE=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-NATIVE-COHORT-01/20260920-r1')

def prefix(plan,item,out,name):
    local=DEST/item['path']
    if local.exists() and local.stat().st_size==item['bytes']:
        source=str(local);mode='complete_local_asset'
    else:
        meta=get_hf_file_metadata(hf_hub_url(plan['repo_id'],item['path'],repo_type='dataset',revision=plan['revision']),token=get_token())
        assert meta.size==item['bytes']
        source=meta.location;mode='remote_HTTP_Range_prefix_only'
    options={'http_proxy':os.environ['HTTPS_PROXY'],'rw_timeout':'60000000'} if mode.startswith('remote') else {}
    timestamps=[];began=time.monotonic()
    arr=np.lib.format.open_memmap(out/f'{name}.npy',mode='w+',dtype=np.uint8,shape=(237,704,1280,3))
    with av.open(source,options=options) as c:
        stream=c.streams.video[0];assert float(stream.average_rate)==30
        metadata={'path':item['path'],'full_source_bytes':item['bytes'],'read_mode':mode,
                  'source_wh':[stream.width,stream.height],'source_frames':stream.frames,
                  'fps':str(stream.average_rate),'time_base':str(stream.time_base)}
        for i,frame in enumerate(c.decode(video=0)):
            timestamps.append(float(frame.pts*frame.time_base))
            rgb=frame.to_ndarray(format='rgb24')
            arr[i]=resize_rgb_image(rgb,pixel_height=704,pixel_width=1280,interpolation='area')
            if i%60==0:print(json.dumps({'scene':out.name,'stream':name,'decoded_frames':i+1}),flush=True)
            if i==236:break
    assert len(timestamps)==237, len(timestamps)
    assert np.max(np.abs(np.diff(timestamps)-1/30))<1e-6
    arr.flush()
    metadata.update(prefix_frames=237,first_pts_s=timestamps[0],last_pts_s=timestamps[-1],wall_s=time.monotonic()-began)
    return metadata,np.asarray(timestamps)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene-index',type=int,required=True);a=p.parse_args()
    plan=json.loads((RUN/'samples_cohort_manifest.json').read_text())
    development=[s for s in plan['selected'] if s['role']=='development']
    assert 0<=a.scene_index<len(development)
    scene=development[a.scene_index];out=BASE/scene['scene_uuid']
    assert not out.exists(), '拒绝覆盖已读取样例；失败须记录原因后另行处理'
    out.mkdir(parents=True)
    result={'task_id':'WS-V75-NATIVE-COHORT-01','run_id':'20260920-r1','scene_uuid':scene['scene_uuid'],
            'source_revision':plan['revision'],'role':'frozen_native_development',
            'status':'started','human_verdict':None,'world_model_generation_calls':0}
    try:
        for filename in ['first_frame.png','prompt.txt']:
            entry=next(f for f in scene['files'] if Path(f['path']).name==filename)
            local=hf_hub_download(plan['repo_id'],entry['path'],repo_type='dataset',revision=plan['revision'],local_dir=str(DEST))
            assert Path(local).stat().st_size==entry['bytes']
            shutil.copyfile(local,out/filename)
        initial=np.asarray(Image.open(out/'first_frame.png').convert('RGB'))
        result['source_initial_shape']=list(initial.shape)
        initial=resize_rgb_image(initial,pixel_height=704,pixel_width=1280,interpolation='area')
        Image.fromarray(initial).save(out/'initial_rgb.png')
        hd=next(f for f in scene['files'] if f['path'].endswith('_hdmap.mp4'))
        rgb=next(f for f in scene['files'] if f['path'].endswith('.mp4') and not f['path'].endswith('_hdmap.mp4'))
        result['reference_video'],times=prefix(plan,rgb,out,'reference')
        result['condition_video'],cond_times=prefix(plan,hd,out,'conditions')
        assert np.max(np.abs(times-cond_times))<1e-6,'两路PTS不同，停止'
        reference=np.load(out/'reference.npy',mmap_mode='r')
        diffs=[float(np.mean(np.abs(reference[i].astype(float)-initial))) for i in range(60)]
        result['initial_image_alignment']={'prefix_mae':diffs,'best_frame_first60':int(np.argmin(diffs)),
                                           'frame0_mae':diffs[0],'criterion':'minimum difference at frame0 and MAE<=8'}
        assert int(np.argmin(diffs))==0 and diffs[0]<=8,'原始初帧与视频开头不匹配，不自动调整窗口'
        result.update(status='passed',frames=237,fps=30,input_wh=[1280,704],
                      preprocessing='official flashdreams.infra.runner_io.resize_rgb_image; INTER_AREA for all RGB/conditions; no crop',
                      condition_source='official supplied hdmap video; no rerender or synthesized state',
                      reference_pose_and_actor_tracks_available=False,full_source_videos_downloaded=False,
                      claim_boundary='原生条件视频基线；尚不能在三维actor状态上作相同接口的自然重建干预')
        (out/'render_result.json').write_text(json.dumps({'status':'passed','renderer':'official supplied pre-rendered hdmap video','rerendered':False,'frames':237},indent=2)+'\n')
    except BaseException as exc:
        result.update(status='failed_stopped',error_type=type(exc).__name__)
        raise RuntimeError(type(exc).__name__) from None
    finally:
        (out/'input_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['initial_image_alignment']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
