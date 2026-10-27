"""49-frame RGB/semantic contact sheets from exact delivered source data."""
from pathlib import Path
import json,base64,zlib
import numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];W=B/'outputs/GS_Scene_Diagnostics_20261027';O=B/'outputs/V43_Ownership_20261027/gpu_v43'
C=np.array([[80,89,104],[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32]],np.uint8)
for entry in json.loads((W/'scenes.json').read_text(encoding='utf8')):
    folder=W/entry['context'];data=json.loads((folder/'scene.json').read_text(encoding='utf8'));raws=sorted(map(int,data['frames']));sheet=Image.new('RGB',(7*280,7*205),(12,20,30))
    for i,raw in enumerate(raws):
        s=(folder/'gpu_semantics'/('frame_%03d.js'%raw)).read_text();packed=json.loads(s[s.index(']=')+2:-1]);decoded=zlib.decompress(base64.b64decode(packed['data']));m=packed['arrays']['semantic_image'];labels=np.frombuffer(decoded[m['offset']:m['offset']+m['bytes']],np.uint8).reshape(320,480)
        rgb=Image.open(folder/data['gt'][str(raw)]).convert('RGB');sem=Image.fromarray(C[labels]);tile=Image.new('RGB',(280,205),(14,24,35));tile.paste(rgb.resize((280,93)),(0,0));tile.paste(sem.resize((280,93)),(0,93));ImageDraw.Draw(tile).text((4,188),'raw%d / RGB + semantic'%raw,fill='white');sheet.paste(tile,((i%7)*280,(i//7)*205))
    sheet.save(O/(entry['context'][:6]+'_semantic_contact49.jpg'),quality=92)
print('3 contacts x 49 frames')
