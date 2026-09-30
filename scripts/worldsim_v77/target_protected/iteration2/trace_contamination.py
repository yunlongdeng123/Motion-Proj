"""截图到真实供体的可复核映射；不重新审查旧样本、不修改旧mask。"""
import argparse,json,sys
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_pairs import warp_matrix

def read(p):return json.loads(Path(p).read_text())
def main(root,old):
    pair=read(old/'native_window_factory/synthetic/W029/pair_manifest.json');i=8
    src=next(c for c in read(old/'native_window_factory/source_manifest.json')['clips'] if c['source_id']==pair['donor_source_id']);f=src['frames'][i]
    raw=np.asarray(Image.open(old/'native_window_factory/rgb'/f['filename']).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS))
    mask=np.asarray(Image.open(old/'native_window_factory/segmented'/pair['donor_source_id']/'sam2_raw'/f'{i:05}.png'))>0
    T=warp_matrix(f,pair['frames'][i],pair['contact_fractions'][i]);inverse=cv2.invertAffineTransform(T)
    yy,xx=np.where(mask);box=(max(0,xx.min()-25),max(0,yy.min()-25),min(1024,xx.max()+26),min(576,yy.max()+26))
    checker=np.indices(mask.shape).sum(0)//10%2*35+80;isolated=np.where(mask[...,None],raw,checker[...,None]);overlay=raw.copy();overlay[mask]=np.rint(.6*raw[mask]+.4*np.array([40,240,180]))
    sheet=Image.new('RGB',(1500,650),(18,25,35));draw=ImageDraw.Draw(sheet)
    for j,(label,arr) in enumerate([('Actual donor RGB frame 28',raw),('Original SAM2 mask',overlay),('Isolated donor pixels',isolated.astype(np.uint8))]):
        im=Image.fromarray(arr).crop(box);im.thumbnail((490,585));sheet.paste(im,(j*500,55));draw.text((j*500+10,12),label,fill='white')
    (root/'evidence').mkdir(exist_ok=True);sheet.save(root/'evidence/S045_f28_trace.png')
    # 用户指出的完整截图保留；具体轮廓应在放大RGB上决定负点，不能按框盲目裁剪。
    row={'screenshot_best_match':{'case_id':'W029','frame':8,'source_frame':28,'donor_source_id':'S045'},
        'donor_filename':f['filename'],'donor_instance':src['actors'][0]['instance_token'],'source_crop_xyxy':list(map(int,box)),
        'donor_to_output_affine':T.tolist(),'output_to_donor_affine':inverse.tolist(),
        'human_judgment':'贴纸右下方多出一块，不干净，需要重造；用户2026-09-30明确反馈',
        'ai_score':2,'human_score_numeric':None,'human_verdict':'reject_visible_contamination',
        'cause_scope':'原始SAM2连通轮廓随供体复制；待修正分割，不能靠柔化贴片修复',
        'review_policy':'旧数据不重审；外部AI<2淘汰，=2仅候选；污染供体及衍生品停止复用，原始证据不删'}
    (root/'contamination_trace.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(row,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old',type=Path,required=True);a=p.parse_args();main(a.root,a.old)
