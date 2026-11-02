"""Guarded native-R2 queue after the exact zero-update historical replay."""
import json,subprocess,sys,threading,time
from pathlib import Path
RUN=Path(__file__).resolve().parent.parent

def main():
    replay=json.loads((RUN/'legacy_a5_replay/replay_receipt.json').read_text())
    assert replay['reproduced'] and replay['optimizer_updates']==0
    selection=json.loads((RUN/'selection/selection_receipt.json').read_text());assert selection['parent_unchanged']
    done=threading.Event();peak=[0]
    def monitor():
        with (RUN/'gpu_memory_samples.jsonl').open('x') as log:
            while not done.is_set():
                r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
                if r.returncode==0:
                    peak[0]=max(peak[0],int(r.stdout.split(',')[0]));log.write(json.dumps(dict(time=time.time(),values=r.stdout.strip()))+'\n');log.flush()
                done.wait(1)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();start=time.time()
    try:
        command=[sys.executable,str(RUN/'source/native_r2_runtime.py'),'--steps','1000','--output',str(RUN/'formal_native')]
        with (RUN/'native_train.log').open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Native R2 failed: see native_train.log')
        import imageio_ffmpeg
        frames=RUN/'formal_native/all49';video=RUN/'native_r2_delete5_before_after49.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-n','-framerate','10','-start_number','89',
            '-i',str(frames/'%03d_baseline_delete5.png'),'-framerate','10','-start_number','89','-i',str(frames/'%03d_repaired_delete5.png'),
            '-filter_complex','hstack=inputs=2','-frames:v','49','-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
        reader=imageio_ffmpeg.read_frames(str(video));meta=next(reader);count=sum(1 for _ in reader);assert count==49
        (RUN/'completion.json').write_text(json.dumps(dict(formal_updates=1000,legacy_replay_updates=0,smoke_updates=0,
            formal_completed=True,frames_decoded=count,seconds=time.time()-start,NVML_peak_MiB=peak[0],
            quality_accepted=False,human_verdict=None,needs_visual_review=True),indent=2))
    finally:done.set();watcher.join(timeout=5)

if __name__=='__main__':main()
