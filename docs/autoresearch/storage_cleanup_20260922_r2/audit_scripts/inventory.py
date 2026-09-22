import os,json,collections,pathlib,time
R=pathlib.Path('/root/autodl-tmp')
A=R/'cleanup_manifests/20260922-r2'; A.mkdir(parents=True,exist_ok=True)
roots=[R/'data/worldsim_v4',R/'runs',R/'models/worldsim_v74_mainfig']
count=0; totals=collections.Counter(); groups=collections.Counter(); names=collections.Counter()
v=os.statvfs(R)
if not (A/'baseline.json').exists(): (A/'baseline.json').write_text(json.dumps({'time':time.time(),'available_bytes':v.f_bavail*v.f_frsize,'used_bytes':(v.f_blocks-v.f_bfree)*v.f_frsize},indent=2))
out=(A/'inventory.jsonl').open('w')
for root in roots:
 for base,dirs,files in os.walk(root,followlinks=False):
  for name in files:
   p=pathlib.Path(base)/name
   if p.is_symlink(): continue
   s=p.stat(); rel=p.relative_to(R); parts=rel.parts
   key='/'.join(parts[:4] if parts[0]=='data' else parts[:3]); totals[key]+=s.st_blocks*512
   if parts[0]=='data': groups['/'.join(parts[:4])+'/'+(parts[5] if len(parts)>5 else name)]+=s.st_blocks*512
   if parts[0]=='runs': names[parts[1]+'/'+(name if s.st_size>16*2**20 else p.suffix)]+=s.st_blocks*512
   if s.st_size>=2**20: out.write(json.dumps({'path':str(p),'bytes':s.st_size,'allocated':s.st_blocks*512,'inode':s.st_ino,'device':s.st_dev,'mtime_ns':s.st_mtime_ns,'links':s.st_nlink})+'\n')
   count+=1
 print(str(root)+' scanned',flush=True)
out.close()
result={'files':count,'largest_groups_GiB':[(k,round(v/2**30,3)) for k,v in totals.most_common(65)],'data_categories_GiB':[(k,round(v/2**30,3)) for k,v in groups.most_common(35)],'run_basenames_GiB':[(k,round(v/2**30,3)) for k,v in names.most_common(45)]}
(A/'inventory-summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
