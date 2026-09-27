"""保留首窗零产物的启动错误，只恢复已验证的单卡启动配置。"""
import shutil
from repair_common import *
state=read(ROOT/'drive_state.json');assert state['state']=='failed' and state['completed']==[] and 'OutOfMemoryError' in state['error']
backup=ROOT/'attempts/01_missing_sequential_cfg';backup.mkdir(parents=True,exist_ok=False)
for name in ['drive_state.json','drive.log','scene_0230/precise']:
 src=ROOT/name;dest=backup/pathlib.Path(name).name;shutil.move(str(src),str(dest))
dump(backup/'correction.json',{'cause':'新入口未带已有DRIVEEDITOR_SEQUENTIAL_CFG=1及PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128；恢复旧已验证单卡配置。','output_frames':0,'model_seed_resolution_unchanged':True,'failure_kind':'engineering_startup','human_verdict':None})
