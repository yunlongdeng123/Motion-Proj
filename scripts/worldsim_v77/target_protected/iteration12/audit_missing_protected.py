"""B->C拒绝定位：哪些未分割实体的包络碰到model H？只输出证据。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import reveal_factory as f
import numpy as np
from PIL import Image, ImageDraw


def main():
    f.legacy.O = f.O; f.legacy.ROOT = f.ROOT; g = f.legacy.geometry()
    map_api = f.NuScenesMap(dataroot=str(f.ROOT), map_name='boston-seaport'); rows = []
    out = f.O/'rejection_contacts'; out.mkdir(exist_ok=True)
    for sid in ['D_V001', 'N111', 'N149']:
        c = g.sources[sid]; g.prepare(sid); f.legacy.support(g, sid); pm = f.legacy.protections(g, sid)
        asset = f.POLICY['split_shape'][c['source_split']]; size = [1.85,4.5,1.5] if asset=='sedan' else [1.9,4.6,1.7]
        mesh = dict(np.load(f.T/'r8/assets'/f'{asset}.npz')); found = False
        for q in f.centers(g, map_api, c, size):
            if found: break
            for speed in f.POLICY['speeds_mps']:
                tr, why = f.trajectory(g, c, q, speed, asset)
                if tr is None: continue
                for i, (p, frame, obs) in enumerate(zip(tr['frames'], c['frames'], g.obstacles[sid])):
                    a = f.legacy.old.silhouette(mesh['vertices'], mesh['faces'], p['actor'], frame)
                    h = f.prepare_masks(a)[0]['model_mask']
                    for ob in obs:
                        pr = ob['_projection']; tok = ob['instance_token']
                        if pr is None or tok in pm or ob['category'].startswith(('static_object', 'movable_object')): continue
                        bb = pr['box']; x0,y0=np.maximum(np.floor(bb[:2]).astype(int),0);x1,y1=np.minimum(np.ceil(bb[2:]).astype(int),[1024,576])
                        if x1<=x0 or y1<=y0 or h[y0:y1,x0:x1].sum()<=max(12,.01*h.sum()): continue
                        row={'source_id':sid,'frame':i,'instance_token':tok,'category':ob['category'],
                             'GT_box':bb.tolist(),'H_envelope_overlap':int(h[y0:y1,x0:x1].sum()),
                             'silhouette_envelope_overlap':int(a[y0:y1,x0:x1].sum()),
                             'source_actors':[v['instance_token'] for v in c['actors']],
                             'SAM_available':list(pm),'A_depth':p['depth_interval'],
                             'obstacle_depth':[pr['near_depth'],pr['far_depth']],
                             'interpolation_uncertain':ob.get('interpolation_uncertain',False)}
                        rgb=np.asarray(Image.open(f.ROOT/'rgb'/frame['filename']).convert('RGB').resize((1024,576))).copy()
                        rgb[h]=np.rint(rgb[h]*.5+np.array([30,130,250])*.5).astype('uint8');im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
                        draw.rectangle(tuple(bb),outline='yellow',width=3);draw.text((10,10),sid+' '+ob['category']+' H=blue / unchecked box=yellow',fill='white')
                        im.save(out/(sid+'.jpg'),quality=95);rows.append(row);found=True;break
                    if found: break
                if found: break
    f.dump(f.O/'missing_protected_audit.json',{'examples':rows,'inference':'GT包络交叠仅是疑点，需要真实实例mask才能确认；不能直接判身份/分割错'})
    print([(r['source_id'],r['category'],r['H_envelope_overlap'],r['silhouette_envelope_overlap']) for r in rows])


if __name__=='__main__':main()
