"""拆分用户双视图，保留原始RGB，使用CPU GrabCut准备单车alpha。"""
from pathlib import Path
import json,hashlib
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('C:/Users/DENGYU~1/AppData/Local/Temp/codex-clipboard-e28cae1c-581b-4a45-9dc9-11993cf6ad01.png')
def main():
    out=ROOT/'input';out.mkdir(parents=True,exist_ok=True)
    source=Image.open(SOURCE).convert('RGB');source.save(out/'reference_original.png')
    # 原像素完整包围车辆，未对车辆颜色、比例或细节进行编辑。
    boxes={'rear':(24,314,778,830),'front':(780,312,1530,820)}
    bounds={'rear':(47,333,761,805),'front':(798,331,1514,798)}
    rows=[];views=[]
    for name,box in boxes.items():
        rgb=np.asarray(source.crop(box)).copy();x0,y0,_,_=box
        mask=np.full(rgb.shape[:2],cv2.GC_BGD,np.uint8)
        a,b,c,d=bounds[name];a-=x0;c-=x0;b-=y0;d-=y0
        mask[b:d,a:c]=cv2.GC_PR_FGD
        # 只用车身内部做确定前景；轮胎、镜子和镀铬仍交给原图分割。
        if name=='rear':
            cv2.rectangle(mask,(185-x0,412-y0),(559-x0,480-y0),cv2.GC_FGD,-1)
            cv2.rectangle(mask,(371-x0,519-y0),(720-x0,609-y0),cv2.GC_FGD,-1)
            # 首次GrabCut把近侧后轮暗色橡胶误判为地面，按原图可见轮胎补前景种子。
            cv2.ellipse(mask,(282-x0,710-y0),(47,86),-9,0,360,cv2.GC_FGD,-1)
        else:
            cv2.rectangle(mask,(940-x0,770-y0),(1166-x0,817-y0),cv2.GC_BGD,-1)
            cv2.rectangle(mask,(1044-x0,410-y0),(1300-x0,591-y0),cv2.GC_FGD,-1)
            cv2.rectangle(mask,(918-x0,476-y0),(1084-x0,566-y0),cv2.GC_FGD,-1)
        bg=np.zeros((1,65),np.float64);fg=bg.copy()
        cv2.setNumThreads(2);cv2.setRNGSeed(101)
        cv2.grabCut(rgb,mask,None,bg,fg,7,cv2.GC_INIT_WITH_MASK)
        alpha=np.isin(mask,[cv2.GC_FGD,cv2.GC_PR_FGD])
        rgba=np.dstack([rgb,alpha.astype(np.uint8)*255])
        Image.fromarray(rgb).save(out/f'{name}_rgb.png')
        Image.fromarray(rgba).save(out/f'{name}_rgba.png')
        Image.fromarray(alpha.astype(np.uint8)*255).save(out/f'{name}_alpha.png')
        bgview=np.full_like(rgb,220);bgview[alpha]=rgb[alpha]
        views.extend([Image.fromarray(rgb),Image.fromarray(bgview)])
        rows.append(dict(view=name,crop_native_xyxy=list(box),size=list(rgb.shape[1::-1]),
            alpha_pixels=int(alpha.sum()),source_RGB_max_error=0,
            segmentation='CPU GrabCut, 7 iterations, fixed seed101, conservative box + body interior FG; rear near-tire ellipse corrects dark-rubber omission; front floor seeded BG',
            role='user-provided vehicle-class prior, not observed a20 ground truth'))
    sheet=Image.new('RGB',(1508,1032),(16,23,31))
    for i,im in enumerate(views):sheet.paste(im,((i%2)*754,(i//2)*516))
    sheet.save(out/'conditioning_preview.png')
    (out/'mapping.json').write_text(json.dumps(dict(source=str(SOURCE),source_size=list(source.size),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),views=rows,
        primary='rear',front_role='structural reference only; not fused as multi-image'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(rows))
if __name__=='__main__':main()
