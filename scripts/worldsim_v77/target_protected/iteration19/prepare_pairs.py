"""r52只做同一代表帧的两种监督；生成标签与真实RGB严格分开。"""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image

ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r52')

def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    assets=ROOT/'source';src=np.array(Image.open(assets/'query_f05.jpg').convert('RGB'))
    hole=np.array(Image.open(assets/'hole_f05.png').convert('L'))>0
    gen=np.array(Image.open(assets/'pseudo_generated_v1.png').convert('RGB').resize((1024,576),Image.Resampling.LANCZOS))
    # 只用洞外特征对齐生成候选；最后洞外严格写回原图，不能用整幅重生成图当Y。
    valid=(~cv2.dilate(hole.astype('uint8'),np.ones((31,31),'uint8')).astype(bool)).astype('uint8')*255
    sift=cv2.SIFT_create(nfeatures=4000)
    a,da=sift.detectAndCompute(cv2.cvtColor(gen,cv2.COLOR_RGB2GRAY),valid)
    b,db=sift.detectAndCompute(cv2.cvtColor(src,cv2.COLOR_RGB2GRAY),valid)
    good=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<.7*n.distance]
    assert len(good)>40,'生成图与原图不能可靠对齐'
    pa=np.float32([a[m.queryIdx].pt for m in good]);pb=np.float32([b[m.trainIdx].pt for m in good])
    mat,inlier=cv2.findHomography(pa,pb,cv2.RANSAC,2.0)
    error=np.linalg.norm(cv2.perspectiveTransform(pa[:,None],mat)[:,0]-pb,axis=1)
    inlier=inlier[:,0].astype(bool)
    assert inlier.sum()>40 and np.median(error[inlier])<1.5
    aligned=cv2.warpPerspective(gen,mat,(1024,576),flags=cv2.INTER_LINEAR)
    pseudo=src.copy();pseudo[hole]=aligned[hole]
    # 真实GT对：同帧真实道路上平移同一车辆轮廓洞，Y仍是未改动实拍RGB。
    transform=np.float32([[1,0,-195],[0,1,38]])
    synthetic_hole=cv2.warpAffine(hole.astype('uint8'),transform,(1024,576),flags=cv2.INTER_NEAREST)>0
    assert synthetic_hole.sum()>.98*hole.sum() and not (synthetic_hole&hole).any()
    rows=[]
    for cid,kind,h,y in [('R001_realRGB_proxy','real_rgb_synthetic_occlusion',synthetic_hole,src),
                         ('R001_pseudo_delete','reviewed_pseudo_clean_target',hole,pseudo)]:
        out=ROOT/'pairs'/cid;out.mkdir(parents=True,exist_ok=False)
        x=src.copy() if kind=='reviewed_pseudo_clean_target' else y.copy()
        if kind=='real_rgb_synthetic_occlusion':x[h]=127
        assert np.array_equal(x[~h],y[~h])
        for name,value in [('x',x),('y',y),('hole',h.astype('uint8')*255)]:Image.fromarray(value).save(out/f'{name}.png')
        masked=x.copy();masked[h]=127;Image.fromarray(masked).save(out/'masked.png')
        overlay=src.copy();overlay[h]=(overlay[h]*.45+np.array([30,140,255])*.55).astype('uint8')
        Image.fromarray(overlay).save(out/'overlay.png')
        manifest={'case_id':cid,'source_case':'R001','scene':'scene-0290','camera':'CAM_FRONT','frame_index':5,
          'timestamp':1535729081512404,'supervision_kind':kind,'x':str(out/'x.png'),'y':str(out/'y.png'),
          'hole':str(out/'hole.png'),'qa_pass':False,'qa_record':str(out/'quality_review.json'),
          'target_is_real_RGB':kind=='real_rgb_synthetic_occlusion','human_verdict':None,
          'num_frames':1,'no_temporal_claim':True,'hole_translation_xy':[-195,38] if kind=='real_rgb_synthetic_occlusion' else [0,0],
          'limits':'同布局合成恢复对，不是原始R001真实删除GT' if kind=='real_rgb_synthetic_occlusion' else '用户授权的生成伪标签；拟合成功不证明隐藏真值正确或泛化',
          'outside_hole_pixel_exact':True,'masked_before_resize':True}
        dump(out/'pair.json',manifest);rows.append(manifest)
    Image.fromarray(aligned).save(assets/'pseudo_aligned.png')
    dump(ROOT/'pair_preparation.json',{'pairs':rows,'registration':{'matches':len(good),'inliers':int(inlier.sum()),
         'median_reprojection_px':float(np.median(error[inlier])),'homography':mat.tolist()},
         'GPU_forwards':0,'training_steps':0,'pseudo_generation':'built-in image_gen; prompt saved in source/prompt.txt'})
    print(json.dumps({'pairs':len(rows),'alignment_inliers':int(inlier.sum()),'hole_pixels':int(hole.sum())}))

if __name__=='__main__':main()
