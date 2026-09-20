"""独立环境内准备官方 nuScenes 规划器；不训练，不启动推理。"""
import json, os, subprocess, time, urllib.request, shutil
from pathlib import Path
from packaging.utils import parse_wheel_filename

R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-SPARSE-NATIVE-01/20260915-r1')
B=Path('/root/autodl-tmp/external/worldsim_simimpact/SparseDrive')
V=Path('/root/autodl-tmp/envs/sparsedrive-impact')
R.mkdir(parents=True,exist_ok=True);(R/'assets').mkdir(exist_ok=True)
os.environ.update(HTTP_PROXY='http://127.0.0.1:41841',HTTPS_PROXY='http://127.0.0.1:41841',MAX_JOBS='4',CUDA_HOME='/usr/local/cuda-11.8',TORCH_CUDA_ARCH_LIST='8.6')
os.environ['NO_PROXY']=os.environ.get('NO_PROXY','')+',mirrors.aliyun.com,.tuna.tsinghua.edu.cn,pypi.org,files.pythonhosted.org'
os.environ.update(PIP_INDEX_URL='https://pypi.tuna.tsinghua.edu.cn/simple',PIP_EXTRA_INDEX_URL='')
def run(cmd,cwd=None):
    print('RUN',cmd,flush=True);subprocess.run(cmd,cwd=cwd,check=True)
def download(url,path):
    if path.exists():return
    run(['curl','--fail','--location','--retry','3','--connect-timeout','30','--output',str(path)+'.partial',url])
    Path(str(path)+'.partial').rename(path)
def pypi_wheel(package,version):
    metadata=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json',timeout=30))
    candidates=[a for a in metadata['urls'] if a['filename'].endswith('x86_64.whl') and ('cp39-cp39-' in a['filename'] or 'py3-none-' in a['filename']) and 'manylinux' in a['filename']]
    assert candidates,package
    # 同版本有重打包 build tag 时，按标准 wheel 的最高 build 选择。
    a=max(candidates,key=lambda x:parse_wheel_filename(x['filename'])[2]);p=R/'assets'/a['filename']
    if not p.exists() and package=='torch':
        partial=Path('/tmp/pip-unpack-0vxx2qtx')/a['filename']
        if partial.exists():shutil.copyfile(partial,p)
    if not p.exists() or p.stat().st_size!=a['size'] or Path(str(p)+'.aria2').exists():
        run(['aria2c','--continue=true','--all-proxy=http://127.0.0.1:41841','--max-connection-per-server=8','--split=8','--min-split-size=1M','--file-allocation=none','--auto-file-renaming=false','--summary-interval=30','--console-log-level=warn','--dir',str(p.parent),'--out',p.name,a['url']])
    assert p.stat().st_size==a['size']
    return str(p)
reg={'task_id':'WS-SIM-SPARSE-NATIVE-01','run_id':'20260915-r1','seed':20260915,'created_unix':time.time(),
 'source':'https://github.com/swc-17/SparseDrive','revision':subprocess.check_output(['git','-C',str(B),'rev-parse','HEAD'],text=True).strip(),
 'model':'Official SparseDrive-S stage2 nuScenes checkpoint; not SparseDriveV2 NAVSIM weights',
 'purpose':'Establish a native-dataset real-camera planning baseline after fixed-command and one route-oracle TransFuser baselines failed the fresh-scene admission.',
 'scope':'Runtime and official checkpoint preparation only. A separate frozen input protocol must precede any model forwards. No training, no fit, no reconstructed sensor claim.',
 'runtime_plan':'Isolated Python3.9 venv inherits utility packages from drivestudio; local Torch1.13.1/cu117 and torchvision0.14.1, MMCV1.7.1 and FlashAttention2.3.2 binaries. Official quick start uses Torch1.13.0/cu116; minor runtime difference disclosed.',
 'information_contract':'Six current cameras, calibration, causal temporal memory and native ego status. Official high-level route command is extra GT-derived route information; future actors/trajectory numbers evaluator-only.',
 'failure_ledger_refs':['V74-H2-F18','V74-H2-F21'],'failure_ledger_delta':'pending','human_verdict':None,'completed':False}
if not (R/'registration.json').exists():(R/'registration.json').write_text(json.dumps(reg,indent=2))
download('https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth',R/'assets/sparsedrive_stage2.pth')
if not V.exists():run(['/root/autodl-tmp/envs/drivestudio/bin/python','-m','venv','--system-site-packages',str(V)])
pip=[str(V/'bin/python'),'-m','pip','install','--no-deps','--disable-pip-version-check']
packages=[('torch','1.13.1'),('torchvision','0.14.1'),('nvidia-cuda-runtime-cu11','11.7.99'),('nvidia-cudnn-cu11','8.5.0.96'),('nvidia-cublas-cu11','11.10.3.66'),('nvidia-cuda-nvrtc-cu11','11.7.99')]
run(pip+[pypi_wheel(package,version) for package,version in packages])
run(pip+['mmcv-full==1.7.1','-f','https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html','--only-binary=:all:'])
run(pip+['mmdet==2.28.2','yapf==0.33.0','motmetrics==1.1.3','prettytable==3.7.0','einops==0.6.1'])
release=json.load(urllib.request.urlopen('https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.3.2'))
assets=[a for a in release['assets'] if 'cu117torch1.13cxx11abiFALSE-cp39-cp39-linux_x86_64.whl' in a['name']]
assert len(assets)==1,[a['name'] for a in release['assets']]
a=assets[0];p=R/'assets'/a['name'];download(a['browser_download_url'],p);run(pip+[str(p)])
run([str(V/'bin/python'),'setup.py','build_ext','--inplace'],B/'projects/mmdet3d_plugin/ops')
reg=json.loads((R/'registration.json').read_text());reg.update(runtime_prepared=True,prepared_unix=time.time(),runtime_download_correction='Slow download.pytorch.org combined CUDA wheel was terminated before installation. Aliyun returned IP ACL403; TUNA was also slow. Resume the TUNA partial PyPI wheel via official files.pythonhosted.org with aria2 eight connections. All six PyPI releases are version-pinned; no model change.')
(R/'registration.json').write_text(json.dumps(reg,indent=2));print('RUNTIME_PREPARED',flush=True)
