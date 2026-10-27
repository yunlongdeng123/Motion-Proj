from pathlib import Path
import subprocess,sys
B=Path(__file__).resolve().parents[2];W=B/'work/persistent_world_20261027'
p=W/'scene_viewer.js';s=p.read_text(encoding='utf8');s=s.replace("String(f.maps).includes('v43_badcase_maps')","/v43_(badcase|review)_maps/.test(String(f.maps))")
s=s.replace("['alpha','dynamic_weight','depth_m',...", "['alpha','dynamic_weight',...(String(f.maps).includes('v43_review_maps')?[]:['depth_m']),...")
if 'function selectFromHash()' not in s:
    start=s.index(' const params=new URLSearchParams(location.hash.replace(');end=s.index('\n',start);old=s[start:end]
    s=s[:start]+" selectFromHash();window.addEventListener('hashchange',()=>{selectFromHash();show();});"+s[end:]
    insert=s.index('async function init(){');s=s[:insert]+'function selectFromHash(){'+old.strip()+'}\n'+s[insert:]
p.write_text(s,encoding='utf8')
p=W/'scene_viewer.html';s=p.read_text(encoding='utf8').replace('v43-unknown-cpu-5','v43-native-review-6');p.write_text(s,encoding='utf8')
subprocess.run([sys.executable,str(Path(__file__).with_name('patch_viewer.py'))],check=True)
subprocess.run([sys.executable,str(Path(__file__).with_name('build_gpu_review_report.py'))],check=True)
