from pathlib import Path
import json,subprocess
p=Path('outputs/V43_R4_FiveCase_GPU_20261101');d=json.loads((p/'review/data.js').read_text(encoding='utf-8')[len('window.FIVE_DATA='):-1]);r='/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/r4_fivecase_gpu_20261101_r1'
for c in d:
 f=c['frames'][24]['frame_id'];q=p/'mechanisms'/c['case']/(f+'_observed.png')
 subprocess.run(['scp','-o','BatchMode=yes',f'wm-vgpu-1011:{r}/mechanisms_gpu/{c["case"]}/observed/{f}.png',str(q)],check=True)
print('5 observed frames retrieved')
