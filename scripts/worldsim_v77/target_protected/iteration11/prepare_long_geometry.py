"""三扫描已缓存时提前准备全30帧世界约束，独立于RGB压缩包读取。"""
import os,sys,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from long_factory import geometry,O,dump,support
def main(shard,count):
 g=geometry();rows=[]
 for sid in sorted(g.sources)[shard::count]:
  start=time.time();meta=g.prepare(sid)
  if meta['pass']:support(g,sid)
  rows.append({'source_id':sid,'ground_pass':meta['pass'],'obstacle_frames':len(g.obstacles[sid]),'seconds':time.time()-start,'support_recovered':bool(meta.get('support_recovery'))})
  dump(O/f'geometry_state_{shard}.json',{'stage':'running','pid':os.getpid(),'completed':len(rows),'rows':rows});print('GROUND',sid,meta['pass'],flush=True)
 dump(O/f'geometry_state_{shard}.json',{'stage':'complete','pid':os.getpid(),'completed':len(rows),'rows':rows})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--count',type=int,default=4);a=p.parse_args();main(a.shard,a.count)
