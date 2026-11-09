"""只取回删除/保留原图、显示视频和轻量运行凭据，完整浮点及模型留在远端。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
import numpy as np
import imageio_ffmpeg

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val065_delete30k_20261109_r1')


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    done=read(ROOT/'completed.json')
    assert done['iteration']==30000 and done['frames']==199
    assert done['source_native_checkpoint_unchanged'] and done['native_gaussian_updates']==0
    out=ROOT/'viewer_delivery_r1';out.mkdir(exist_ok=False);delivery=out/'delivery';delivery.mkdir()
    records=[];alpha_max=0.
    for name,dest in [('delete_white','delete_white'),('final_keep','keep')]:
        source=ROOT/name;prov=read(source/'provenance.json')
        assert prov['status']=='COMPLETE' and prov['iteration']==30000
        assert prov['persistent_check_before'] and prov['persistent_check_after']
        assert [r['frame_key'] for r in prov['frames']]==list(range(199))
        assert list(prov['resolved_query']['delete_actor_ids'])==([10] if name=='delete_white' else [])
        folder=delivery/dest;folder.mkdir()
        for row in prov['frames']:
            raw=row['frame_key'];src=source/row['files']['rgb_png'];target=folder/f'{raw}.png'
            shutil.copyfile(src,target)
            assert sha(src)==sha(target)
            if name=='delete_white':
                with np.load(source/row['files']['float_npz']) as z:
                    ids=list(z['owner_alpha_actor_ids']);i=ids.index(10)
                    value=float(abs(z['owner_alpha_T'][i]).max());alpha_max=max(alpha_max,value)
        ff=imageio_ffmpeg.get_ffmpeg_exe()
        movie=delivery/('delete_white_30k.mp4' if dest=='delete_white' else 'keep_30k.mp4')
        args=[ff,'-v','error','-framerate','10','-start_number','0','-i',str(folder/'%d.png'),
              '-frames:v','199','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
              '-movflags','+faststart',str(movie)]
        subprocess.run(args,check=True)
        subprocess.run([ff,'-v','error','-i',str(movie),'-f','null','-'],check=True)
    assert alpha_max<1e-6
    proof=delivery/'proof';proof.mkdir()
    for name in ['registration.json','completed.json','native_parity_check.json','roundtrip.json','final_manifest.json',
                 'final_keep/provenance.json','delete_white/provenance.json','ownership/receipt.json',
                 'world_road/receipt.json','gate32/receipt.json','gate1000/receipt.json','gate1000/restore.json']:
        p=ROOT/name
        if p.is_file():shutil.copyfile(p,proof/name.replace('/','_'))
    write(proof/'display_video_policy.json',dict(fps=10,frames=199,render_wh=[480,320],
          pngs='byte-preserving original render exports',mp4='H264 CRF18 for playback only; compare PNGs for pixels',
          target_owned_alpha_max_after_deletion=alpha_max,visual_edit_quality_accepted=None))
    for p in sorted(delivery.rglob('*')):
        if p.is_file():records.append(dict(path=p.relative_to(delivery).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    write(delivery/'manifest.json',dict(schema='v44.delete30k.delivery.v1',scene='065',frames=199,
          iteration=30000,target_track='NTjq-OhZ_97OMJ4xwJAxXg',files=records))
    archive=out/'delete30k_viewer.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in sorted(delivery.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(delivery).as_posix(),recursive=False)
    # Small independent transport parts avoid a single stalled large SSH stream.
    parts=out/'parts';parts.mkdir()
    sizes=[]
    with archive.open('rb') as stream:
        while True:
            data=stream.read(17*1024*1024)
            if not data:break
            p=parts/f'frames.part{len(sizes)}';p.write_bytes(data);sizes.append(len(data))
    receipt=dict(status='PACKAGED',archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
          parts=sizes,pngs=398,videos=2,files=len(records)+1,source_modified=False,gpu_used=False,unix=time.time())
    write(out/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
