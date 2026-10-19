"""Make full-frame contact sheets for actual visual review; never infer a verdict."""
from pathlib import Path
import hashlib,json
from PIL import Image,ImageDraw,ImageFont
W=Path(__file__).parent;D=W/'final_visual_review'
FRAMES=[f'0008{i:03d}' for i in range(89,138)];ACTORS=[6,37,5,26,18]
FONT=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)

def sheet(name,cells,columns):
 rows=(len(cells)+columns-1)//columns
 im=Image.new('RGB',(480*columns,344*rows),(12,20,28));draw=ImageDraw.Draw(im)
 for n,(path,label) in enumerate(cells):
  x=(n%columns)*480;y=(n//columns)*344
  with Image.open(path) as src:
   assert src.size==(480,320);im.paste(src.convert('RGB'),(x,y+24))
  draw.text((x+5,y+2),label,font=FONT,fill='white')
 p=D/(name+'.png');im.save(p)
 return dict(path=str(p.resolve()),source_images=[str(p.resolve()) for p,label in cells],whole_frame=True,source_pixels_resized=False,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
 D.mkdir(exist_ok=True);manifest=[]
 for model,folder in [('A','direct_full49'),('B','all_vehicle_full49')]:
  root=W/folder;r=json.loads((root/'receipt.json').read_text());assert r['frames']==49 and r['parameter_updates']==0
  assert {p.name for p in root.iterdir() if p.is_dir() and p.name.startswith('0008')}==set(FRAMES)
  for k in range(0,49,7):
   cells=[(root/f/(ch+'.png'),model+' '+f+' '+ch) for f in FRAMES[k:k+7] for ch in ['gt','full','dynamic','static','delete_all']]
   manifest.append(sheet(f'{model}_overview_{k//7+1:02d}',cells,5))
 root=W/'direct_full49';r=json.loads((root/'receipt.json').read_text());assert len(r['records'])==245 and set(r['actors'])==set(ACTORS)
 for actor in ACTORS:
  for kind in ['isolated','delete']:
   manifest.append(sheet(f'A_a{actor:02d}_{kind}_49',[(root/f/f'a{actor:02d}_{kind}.png',f+' a'+str(actor)+' '+kind) for f in FRAMES],7))
 for kind in ['isolated','delete']:
  cells=[(root/f/f'a{a:02d}_{kind}.png',f+' a'+str(a)+' '+kind) for f in ['0008089','0008107','0008128','0008129','0008137'] for a in ACTORS]
  manifest.append(sheet('A_'+kind+'_anchors',cells,5))
 (W/'final_visual_review_manifest.json').write_text(json.dumps(dict(status='FULL_FRAME_SHEETS_PREPARED_NOT_YET_VISUALLY_REVIEWED',sheets=manifest,all_sources_full480x320=True,ROI=False,source_pixels_resized=False,training_updates=0,human_verdict=None),indent=2))
 print(json.dumps(dict(sheets=len(manifest),full_frame=True,quality_verdict_not_assigned=True)))
if __name__=='__main__':main()
