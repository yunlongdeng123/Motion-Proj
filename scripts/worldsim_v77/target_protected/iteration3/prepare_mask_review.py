"""真实来源与SAM2输出并排显示，保留像素证据，不用GT硬裁mask。"""
import json,argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def main(root):
    views=json.loads((root/'multiview/selected_views.json').read_text())['views']
    rows={r['view_id']:r for r in json.loads((root/'multiview/segmentation_state.json').read_text())['completed']}
    out=root/'mask_review';out.mkdir(exist_ok=True);manifest=[]
    for v in views:
        row=rows[v['view_id']];dest=root/'multiview/segmented'/v['view_id']
        rgb=Image.open(root/'multiview/rgb'/v['filename']).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS)
        mask=np.asarray(Image.open(dest/row['mask_files'][row['primary_candidate']]))>0
        a=np.asarray(rgb).copy();a[mask]=(a[mask]*.6+np.array([255,30,50])*.4).astype(np.uint8)
        marked=rgb.copy();ImageDraw.Draw(marked).rectangle(v['box_xyxy'],outline='yellow',width=2)
        box=np.array(v['box_xyxy']);b=[max(0,int(box[0])-25),max(0,int(box[1])-25),min(1024,int(box[2])+25),min(576,int(box[3])+25)]
        parts=[marked,Image.fromarray(a),Image.open(dest/'isolated.png')]
        sheet=Image.new('RGB',(1500,540),(20,25,30));d=ImageDraw.Draw(sheet)
        d.text((10,5),f"{v['view_id']} {v['source_ids']} {v['camera']} | same-instance {v['instance_token']} | SAM2 primary {row['primary_candidate']}",fill='white')
        for i,p in enumerate(parts):
            thumb=p.copy();thumb.thumbnail((495,278));sheet.paste(thumb,(i*500,30))
            crop=p.crop(b);crop.thumbnail((495,215));sheet.paste(crop,(i*500,320))
        path=out/f"{v['view_id']}.jpg";sheet.save(path,quality=96)
        manifest.append({'view_id':v['view_id'],'source_ids':v['source_ids'],'camera':v['camera'],'mechanical_gate':row['mechanical_gate'],'primary_score':row['scores'][row['primary_candidate']],'review_image':path.name,'human_verdict':None})
    (out/'manifest.json').write_text(json.dumps({'views':manifest,'scope':'raw source, red primary mask, isolated pixels; full and enlarged crop'},indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
