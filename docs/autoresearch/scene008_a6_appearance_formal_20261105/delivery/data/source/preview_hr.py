"""Fixed internal temporal check at original input resolution; no fitting."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from highres_fit import *
from scipy.ndimage import binary_erosion
from PIL import ImageDraw

@torch.no_grad()
def main():
    dest=RUN/'highres_evaluation';dest.mkdir(exist_ok=False)
    rt=Runtime();assets={name:SHAsset.from_geometry(path).cuda() for name,path in dict(M2=OLD/'m2/complete_a6.npz',HR=RUN/'highres/formal/complete_a6.npz').items()}
    results=[];start=time.time()
    for raw in HELD:
        d=rt.raw(raw);cam=camera_hr(d['cam'])
        image=np.asarray(Image.open(RAW/'image/image_0'/f'0008{raw:03d}.png').convert('RGB')).astype(np.float32)/255
        low=binary_erosion(d['a6_visible'].cpu().numpy(),iterations=1)
        mask=np.repeat(np.repeat(low,4,axis=0),4,axis=1)
        outputs={};row=dict(raw=raw,pixels=int(mask.sum()),methods={})
        for name,asset in assets.items():
            out=raster(cam,rt.pipe,3,tagged(asset.rows(d['pose'],cam,rt.pipe,16),'asset'),rt.n.bg.new_zeros(3))
            rgb=out['rgb'].permute(1,2,0).cpu().numpy();alpha=out['alpha'][0].cpu().numpy()
            error=np.abs(rgb-image).mean(2);norm=np.abs(rgb/np.maximum(alpha[...,None],1e-5)-image).mean(2)
            row['methods'][name]=dict(RGB_MAE=float(error[mask].mean()),RGB_sum=float(error[mask].sum()),normalized_MAE=float(norm[mask].mean()),normalized_sum=float(norm[mask].sum()))
            outputs[name]=(rgb,alpha)
        assert np.array_equal(outputs['M2'][1],outputs['HR'][1])
        if raw in (105,113,130):
            support=outputs['M2'][1]>.01;y,x=np.where(support)
            crop=(max(0,int(x.min())-20),max(0,int(y.min())-20),min(1920,int(x.max())+21),min(1280,int(y.max())+21))
            strips=[]
            observed=np.zeros_like(image);known=np.repeat(np.repeat(d['a6_visible'].cpu().numpy(),4,axis=0),4,axis=1);observed[known]=image[known]
            for name,a in [('True observed RGB',observed)]+[(n,v[0]) for n,v in outputs.items()]:
                p=Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5)).crop(crop)
                p=p.resize((round(p.width*360/p.height),360),Image.Resampling.LANCZOS)
                q=Image.new('RGB',(p.width,393),(240,245,248));q.paste(p,(0,33));ImageDraw.Draw(q).text((7,9),name,fill=(15,30,45));strips.append(q)
            strip=Image.new('RGB',(sum(p.width for p in strips),393));xx=0
            for p in strips:strip.paste(p,(xx,0));xx+=p.width
            strip.save(dest/f'{raw:03d}_objects.jpg',quality=96)
        results.append(row);rt._data.clear();torch.cuda.empty_cache()
    counts=sum(r['pixels'] for r in results)
    pooled={name:{key:sum(r['methods'][name][s] for r in results)/counts for key,s in [('RGB_MAE','RGB_sum'),('normalized_MAE','normalized_sum')]} for name in assets}
    write(dest/'summary.json',dict(status='PASS',rows=results,pooled=pooled,pixels=counts,HELD=HELD,source='Original1920x1280 RGB on fixed old eroded interiors4x',
        geometry_and_alpha_bitwise_unchanged=True,geometry_context_not_independently_held=True,source_appearance_not_used_for_fit=True,seconds=time.time()-start))
    assert rt.modelunchanged() and all(a.assert_geometry_unchanged() for a in assets.values())
    print(json.dumps(dict(stage='HR_eval',pooled=pooled,pixels=counts)),flush=True)
if __name__=='__main__':main()
