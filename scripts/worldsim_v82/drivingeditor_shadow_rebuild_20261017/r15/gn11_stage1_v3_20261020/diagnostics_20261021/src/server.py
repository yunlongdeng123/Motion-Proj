"""Loopback-only static viewer and optional, serialized SSH CPU ray bridge."""
import argparse
import atexit
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json
from pathlib import Path
import queue
import subprocess
import threading
from urllib.parse import urlsplit, parse_qs

REMOTE = 'CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 /root/autodl-tmp/envs/drivingeditor/bin/python /root/autodl-tmp/gsdiag_cpu_20261021_r1/backend.py --stdio'
GPU_REMOTE='PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/envs/drivingeditor/bin/python /root/autodl-tmp/gsdiag_gpu_20261021_r2/gpu_backend.py --stdio'
class Bridge:
    def __init__(self,command=REMOTE): self.process=None; self.lines=queue.Queue(); self.lock=threading.Lock(); self.command=command
    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired: self.process.kill()
        self.process=None
    def query(self, data):
        with self.lock:
            if not self.process or self.process.poll() is not None:
                self.close(); self.lines=queue.Queue()
                self.process=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','wm-vgpu-1011',self.command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
                proc=self.process; lines=self.lines
                def reader():
                    for line in proc.stdout: lines.put(line)
                    lines.put(None)
                threading.Thread(target=reader,daemon=True).start()
                threading.Thread(target=lambda: [None for _ in proc.stderr],daemon=True).start()
            self.process.stdin.write(json.dumps(data)+'\n'); self.process.stdin.flush()
            try: line=self.lines.get(timeout=240)
            except queue.Empty:
                self.close(); raise RuntimeError('CPU reference timed out; own SSH worker stopped, no research jobs touched')
            if line is None:
                self.close(); raise RuntimeError('SSH CPU worker disconnected (check AutoDL CPU memory and connectivity)')
            return json.loads(line)

bridge=Bridge(); atexit.register(bridge.close)
gpu_bridge=Bridge(GPU_REMOTE);atexit.register(gpu_bridge.close)
class Handler(SimpleHTTPRequestHandler):
    remote_enabled=False
    gpu_enabled=False
    def json_result(self,value,code=200):
        body=json.dumps(value,allow_nan=False).encode()
        self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self):
        host=self.headers.get('Host','');origin=self.headers.get('Origin')
        if (self.path!='/api/gpu' or host not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
            or self.headers.get('Sec-Fetch-Site')=='cross-site' or (origin and origin!='http://'+host)
            or self.headers.get('Content-Type','').split(';')[0]!='application/json'):
            self.send_error(403);return
        try:
            if not self.gpu_enabled:raise ValueError('GPU bridge disabled; run start_gpu.ps1')
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=50000:raise ValueError('invalid payload size')
            data=json.loads(self.rfile.read(size));self.validate_gpu(data)
            value=gpu_bridge.query(data);self.json_result(value,503 if value.get('error') else 200)
        except Exception as e:self.json_result(dict(error=str(e),optimizer_updates=0),503)
    @staticmethod
    def validate_gpu(data):
        if not isinstance(data,dict) or set(data)-{'action','arm','raw','x','y','mode','ids','view'}:raise ValueError('unknown fields')
        if data.get('arm') not in ('full','real_only','prior_only','ownership_only','gn11_v1'):raise ValueError('arm')
        if data.get('arm')=='gn11_v1' and data.get('mode')=='road_only':raise ValueError('V1 has no dedicated road branch')
        if data.get('mode','deleted_five') not in ('deleted_five','joint','dynamic','road_only','ordinary'):raise ValueError('mode')
        if any(type(data.get(k)) is not int for k in ('raw','x','y')):raise ValueError('integer query required')
        if not(89<=data['raw']<=137 and 0<=data['x']<480 and 0<=data['y']<320):raise ValueError('bounds')
        if data.get('action','trace') not in ('trace','counterfactual','render_view'):raise ValueError('action')
        if data.get('action')=='counterfactual':
            ids=data.get('ids')
            if not isinstance(ids,list) or not 1<=len(ids)<=200 or any(not isinstance(i,str) or len(i)>120 for i in ids):raise ValueError('select1..200 bound IDs')
    def end_headers(self):
        self.send_header('Cross-Origin-Resource-Policy','same-origin')
        self.send_header('X-Content-Type-Options','nosniff')
        super().end_headers()
    def do_GET(self):
        url=urlsplit(self.path)
        expected_hosts=(f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
        if self.headers.get('Host','') not in expected_hosts:
            self.send_error(403);return
        if self.headers.get('Sec-Fetch-Site')=='cross-site' and self.headers.get('Sec-Fetch-Dest')!='document':
            self.send_error(403);return
        if url.path=='/favicon.ico':
            self.send_response(204);self.end_headers();return
        if url.path=='/api/health':
            body=json.dumps(dict(app='Gaussian Evidence Lab',version=2,remote_cpu_enabled=self.remote_enabled,gpu_enabled=self.gpu_enabled)).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
        if url.path!='/api/ray': return super().do_GET()
        host=self.headers.get('Host','')
        origin=self.headers.get('Origin')
        if (host not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
            or self.headers.get('Sec-Fetch-Site')=='cross-site'
            or (origin and origin not in ('http://'+host,))):
            self.send_error(403); return
        try:
            if not self.remote_enabled: raise ValueError('CPU bridge is disabled; start with --remote-cpu')
            q=parse_qs(url.query)
            if set(q) not in ({'arm','raw','x','y'},{'arm','raw','x','y','mode'}) or any(len(v)!=1 for v in q.values()): raise ValueError('invalid query fields')
            data={k:v[0] for k,v in q.items()}
            if data['arm'] not in ('full','real_only','prior_only','ownership_only'): raise ValueError('invalid arm')
            if data.get('mode','deleted_five') not in ('deleted_five','joint','dynamic','road_only','ordinary'): raise ValueError('invalid render mode')
            for k in ('raw','x','y'): data[k]=int(data[k])
            if not (89<=data['raw']<=137 and 0<=data['x']<480 and 0<=data['y']<320): raise ValueError('out of bounds')
            value=bridge.query(data); body=json.dumps(value,allow_nan=False).encode()
            self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        except Exception as e:
            body=json.dumps({'error':str(e),'gpu_used':False}).encode()
            self.send_response(503);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8767);p.add_argument('--remote-cpu',action='store_true');p.add_argument('--remote-gpu',action='store_true');a=p.parse_args()
    Handler.remote_enabled=a.remote_cpu;Handler.gpu_enabled=a.remote_gpu
    root=Path(__file__).resolve().parent
    print(f'Gaussian Evidence Lab: http://127.0.0.1:{a.port}/ (SSH CPU bridge {a.remote_cpu})',flush=True)
    with ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,directory=str(root))) as server:
        try: server.serve_forever()
        except KeyboardInterrupt: pass
    bridge.close()
    gpu_bridge.close()
