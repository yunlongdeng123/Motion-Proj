"""把实际诊断输出拼成同尺度证据图；不修改像素内容或生成模型图像。"""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'outputs/r15_diagnostic_evidence'
data=json.loads((OUT/'actors/actors.json').read_text(encoding='utf-8'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
items=[('gt','Real RGB'),('full','Full reconstruction'),('isolated_actor','Isolated actor'),('isolated_alpha','Isolated raw alpha'),('delete','Delete + shadows'),
       ('delete_no_shadow','Delete / shadows OFF'),('static_only_no_shadow','Static only / shadows OFF'),('other_actor_contribution','Other actor contribution'),('static_visible_contribution','Visible static contribution'),('competition_depth_gate','Depth competition gate')]
for anchor in data['anchors']:
    folder=OUT/'actors'/anchor['folder']
    canvas=Image.new('RGB',(1250,444),'#f6f8fa')
    draw=ImageDraw.Draw(canvas)
    draw.text((12,8),f"Actor a{anchor['actor']} | raw{anchor['raw']} | 4k frozen checkpoint | same crop",fill='#172738',font=font)
    for i,(name,label) in enumerate(items):
        x=(i%5)*250;y=34+(i//5)*202
        draw.text((x+8,y+3),label,fill='#172738',font=font)
        im=Image.open(folder/(name+'.png')).convert('RGB').crop(anchor['crop'])
        im=ImageOps.contain(im,(234,172),Image.Resampling.NEAREST)
        canvas.paste(im,(x+8+(234-im.width)//2,y+27+(172-im.height)//2))
    canvas.save(OUT/(anchor['folder']+'_components.png'))
print('Built',len(data['anchors']),'component figures')
