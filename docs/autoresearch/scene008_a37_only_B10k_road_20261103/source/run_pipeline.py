from pathlib import Path
import subprocess,sys,json
R=Path(__file__).resolve().parent.parent
try:
    for label,script in [('prepare','prepare_road37.py'),('queue','run_road_queue.py')]:
        with (R/(label+'.log')).open('x') as f:subprocess.run([sys.executable,str(R/'source'/script)],stdout=f,stderr=subprocess.STDOUT,check=True)
except Exception as exc:
    (R/'pipeline_error.json').write_text(json.dumps(dict(error=repr(exc)))+'\n');raise
