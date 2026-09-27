"""真实证据来源与合成合同检查，不把通过合同当视觉成功。"""
from repair_common import *
from PIL import Image
def main():
 rows=[]
 for s in read(ROOT/'registration.json')['scenes']:
  out=ROOT/s['name'];donors=np.load(out/'donors.npz');di=read(out/'donor_index.json')
  for i,f in enumerate(s['source_frames']):
   rgb=np.array(Image.open(out/'rgb'/f'{i:05}.png'));ev=np.array(Image.open(out/'evidence'/f'{i:05}.png'));mask=cv2.imread(str(out/'mask'/f'{i:05}.png'),0)>0;res=cv2.imread(str(out/'residual'/f'{i:05}.png'),0)>0;em=cv2.imread(str(out/'evidence_mask'/f'{i:05}.png'),0)>0
   assert np.array_equal(mask,res|em) and not (res&em).any();assert np.array_equal(rgb[~em],ev[~em])
   p=np.load(out/'provenance'/f'{i:05}.npz');assert np.array_equal(p['target_index'],np.flatnonzero(em))
   for sid in np.unique(p['source_id']):
    take=p['source_id']==sid;idx=p['target_index'][take];pid=p['source_point'][take]
    assert np.array_equal(ev.reshape(-1,3)[idx],donors[f'{sid}_rgb'][pid]);assert (donors[f'{sid}_lidar_distance'][pid]<=.200001).all()
   for sid,oth in zip(p['source_id'],p['second_source']):assert abs(di[sid]['frame']-di[oth]['frame'])>=5
   arms={}
   for arm in ['precise','evidence_first']:
    path=out/arm/f'{i:05}.png'
    if path.exists():
     im=np.array(Image.open(path));assert np.array_equal(im[~mask],rgb[~mask])
     if arm=='evidence_first':assert np.array_equal(im[em],ev[em])
     arms[arm]='outside_edit_unchanged; evidence_preserved'
   rows.append({'scene':s['name'],'frame':f,'provenance_pixels':len(p['target_index']),'arms':arms})
 dump(ROOT/'validation.json',{'frames':len(rows),'passed':True,'evidence_copied_pixels':sum(r['provenance_pixels'] for r in rows),'checks':['mask = accepted evidence disjoint-union residual','每个证据像素等于记录的原始RGB来源','LiDAR支持<=20cm','两个来源相隔>=5帧','mask外逐像素不变','DriveEditor不能覆写已接受证据'],'rows':rows,'human_verdict':None});print(len(rows),'frames passed')
if __name__=='__main__':main()
