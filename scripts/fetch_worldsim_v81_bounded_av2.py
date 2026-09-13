"""仅取得冻结八日志的中心及±0.5s RGB/参考扫描，不下载整日志。"""
import json,os,shutil,concurrent.futures,xml.etree.ElementTree as ET
from pathlib import Path
import requests
O=Path('/root/autodl-tmp/runs/worldsim_v81/WS-V81-CLOSE-01');reg=json.loads((O/'registration.json').read_text());out=Path('/root/autodl-tmp/data/worldsim_v81_close_av2');out.mkdir(exist_ok=True)
base='https://argoverse.s3.amazonaws.com';ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
def listing(prefix):
 keys=[];token=None
 while True:
  params={'list-type':2,'prefix':prefix}
  if token:params['continuation-token']=token
  r=requests.get(base,params=params,timeout=40);r.raise_for_status();doc=ET.fromstring(r.content);keys.extend(x.text for x in doc.findall('s:Contents/s:Key',ns));token=doc.findtext('s:NextContinuationToken',namespaces=ns)
  if not token:return keys
def get(key):
 dest=out/Path(key).relative_to('datasets/av2/sensor/val');old=Path('/root/autodl-tmp/data/av2/sensor/val')/Path(key).relative_to('datasets/av2/sensor/val');dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():return {'key':key,'status':'EXISTS','bytes':dest.stat().st_size}
 if old.exists():shutil.copy2(old,dest);return {'key':key,'status':'LOCAL_COPY','bytes':dest.stat().st_size}
 r=requests.get(base+'/'+key,timeout=50);r.raise_for_status();dest.write_bytes(r.content);return {'key':key,'status':'DOWNLOADED','bytes':len(r.content)}
def one(log):
 prefix=f'datasets/av2/sensor/val/{log}/';local=Path('/root/autodl-tmp/data/av2/sensor/val')/log/'sensors/lidar';ts=sorted(int(p.stem) for p in local.glob('*.feather'));t=ts[len(ts)//2] if ts else None
 keys=listing(prefix);lidar=sorted(k for k in keys if '/sensors/lidar/' in k);lts=[int(Path(k).stem) for k in lidar]
 if t is None:t=lts[len(lts)//2]
 ti=min(range(len(lts)),key=lambda i:abs(lts[i]-t));picked=lidar[max(0,ti-2):ti+3];cameras={}
 for cam in ['ring_front_center','ring_front_left','ring_side_left','ring_rear_left','ring_rear_right','ring_side_right','ring_front_right']:
  imgs=[k for k in keys if '/'+cam+'/' in k and k.endswith('.jpg')];chosen=[min(imgs,key=lambda k:abs(int(Path(k).stem)-q)) for q in [t-500000000,t,t+500000000]];picked+=chosen;cameras[cam]=chosen
 picked += [prefix+f for f in ['city_SE3_egovehicle.feather','annotations.feather','calibration/egovehicle_SE3_sensor.feather','calibration/intrinsics.feather']]
 manifest={'log':log,'target_lidar_timestamp_ns':t,'lidar_keys':lidar[max(0,ti-2):ti+3],'camera_keys':cameras,'root':str(out/log),'role':'NEW_TO_V81_DISCOVERY','selection':'registered middle locally available lidar; no RGB/model selection'}
 (O/f'av2_source_{log}.json').write_text(json.dumps(manifest,indent=2))
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:records=list(ex.map(get,list(dict.fromkeys(picked))))
 manifest.update(status='READY',files=len(records),bytes=sum(r['bytes'] for r in records));print(json.dumps(manifest),flush=True);return manifest
results=[]
for log in reg['new_sources']['logs']:
 try:results.append(one(log))
 except Exception as e:results.append({'log':log,'status':'DATA_UNAVAILABLE','error':str(e)});print(json.dumps(results[-1]),flush=True)
 (O/'new_source_acquisition.json').write_text(json.dumps(results,indent=2))
