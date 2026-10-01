"""实际相机/资产栅格oracle与可读原图证据，避免轴向错误进入整批数据。"""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,F,ParkingGeometry,silhouette,read,dump
from pyquaternion import Quaternion
import numpy as np,cv2
from PIL import Image,ImageDraw
def main():
    geo=ParkingGeometry(F);rows=[];out=O/'silhouette_probe';out.mkdir(exist_ok=True)
    for path in sorted((O/'asset_candidates').glob('N*.json')):
        candidates=read(path)['candidates']
        choices=[p for p in candidates if p['type']!='background']
        if not choices:continue
        p=choices[0];f=geo.sources[p['source_id']]['frames'][5];a=p['frames'][5]['actor'];data=dict(np.load(O/'assets'/(p['asset']+'.npz')))
        m=silhouette(data['vertices'],data['faces'],a,f)
        w,l,h=a['size'];R=Quaternion(a['rotation']).rotation_matrix
        world=data['vertices']*np.array([l,w,h])@R.T+np.array(a['translation']);cam=world@f['_w2c'][:3,:3].T+f['_w2c'][:3,3]
        uv=cam@np.array(f['intrinsics_1024']).T;uv=uv[:,:2]/uv[:,2:]
        polys=np.rint(uv[data['faces']]).astype('int32');oracle=np.zeros((576,1024),'uint8')
        for tri in polys:cv2.fillConvexPoly(oracle,tri,1)
        assert np.array_equal(m,oracle>0),'optimized raster disagrees with full-face oracle'
        with Image.open(F/'rgb'/f['filename']) as im:y=np.asarray(im.resize((1024,576),Image.Resampling.LANCZOS).convert('RGB')).copy()
        x=y.copy();x[m]=[96,110,127];hole=cv2.dilate(m.astype('uint8'),np.ones((7,7),'uint8'))>0;cc=y.copy();cc[hole]=127
        sheet=Image.new('RGB',(1536,768),(10,17,25));d=ImageDraw.Draw(sheet)
        for j,(title,arr) in enumerate([('real Y',y),('diagnostic synthetic A',x),('masked condition',cc)]):
            d.text((j*512+10,8),p['source_id']+' '+title,fill='white');im=Image.fromarray(arr);sheet.paste(im.resize((512,288)),(j*512,32))
            box=np.array(p['frames'][5]['box']).astype(int);box+=np.array([-30,-30,30,30]);box[[0,2]]=box[[0,2]].clip(0,1024);box[[1,3]]=box[[1,3]].clip(0,576);crop=im.crop(tuple(box));crop.thumbnail((505,430));sheet.paste(crop,(j*512+(512-crop.width)//2,330))
        name=p['source_id']+'.jpg';sheet.save(out/name,quality=96)
        rows.append({'source_id':p['source_id'],'scene':p['scene'],'asset':p['asset'],'type':p['type'],'raster_full_face_oracle_exact':True,'mask_pixels':int(m.sum()),'contact':name,'pose':a})
        if len(rows)>=3:break
    dump(out/'results.json',{'cases':rows,'all_exact':all(r['raster_full_face_oracle_exact'] for r in rows)});print(rows,flush=True)
if __name__=='__main__':main()
