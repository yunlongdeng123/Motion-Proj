import json,argparse
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
rows=[json.loads(l) for l in (run/'v81_roi_registry.jsonl').read_text().splitlines()]
rows=[r for r in rows if r['cohort'] in ['C00','C10']]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',11)
for start in range(0,len(rows),16):
    chunk=rows[start:start+16];canvas=Image.new('RGB',(1280,((len(chunk)+3)//4)*218),'white');d=ImageDraw.Draw(canvas)
    for i,r in enumerate(chunk):
        x=i%4*320;y=i//4*218;im=Image.open(run/'crops'/f"{r['roi_id']}.jpg");canvas.paste(im,(x,y+38));d.text((x+4,y+3),f"{start+i}: {r['cohort']} {r['semantic']}\n{r['roi_id']}",fill='black',font=font)
    canvas.save(run/f'figures/high_overlap_spotcheck_{start//16}.png')
(run/'high_overlap_spotcheck_index.json').write_text(json.dumps(rows,indent=2))
