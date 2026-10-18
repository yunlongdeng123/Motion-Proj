"""Single, inseparably synchronized overview MP4 for each of the five identities."""
from pathlib import Path
import json,subprocess,shutil,re,sys
from PIL import Image,ImageDraw,ImageFont
import numpy as np
W=Path(__file__).parent;O=W.parents[1]/'outputs/identity_comparison_20261018';E=O/'sync_exports';E.mkdir(exist_ok=True)
s=(O/'sync_player.html').read_text(encoding='utf-8');data=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',s,re.S).group(1))
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);title=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
width=1260;height=1140;ff=shutil.which('ffmpeg')
for actor in [6,37,5,26,18]:
    target=E/f'a{actor:02d}_old5_vs_r15.mp4'
    if target.is_file():
        if '--overwrite' not in sys.argv:continue
        assert target.resolve().is_relative_to(E.resolve())
        target.unlink()
    p=subprocess.Popen([ff,'-hide_banner','-loglevel','error','-threads','2','-filter_threads','1','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{width}x{height}','-framerate','10','-i','pipe:0','-an','-c:v','libx264','-threads','2','-g','1','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
    try:
        for raw in range(89,138):
            fid=f'0008{raw:03d}';box=data['rois'][str(actor)][raw-89]
            image=Image.new('RGB',(width,height),(9,19,32));draw=ImageDraw.Draw(image)
            for j,(model,label) in enumerate([('old5','旧 GN11 30K + 五实例 30K'),('r15','原 R15 冻结 4K')]):
                draw.text((10,j*570+4),f'{label}  |  a{actor}  |  raw{raw}',font=title,fill='white')
                src=O/'assets'/model/fid;br=W/'branches'/('branches_'+model)/fid
                files=[(src/('full_no_shadow.png' if model=='r15' else 'full.png'),'删除前 · 整景'),(br/'dynamic.png','删除前 · 动态分支'),(br/'static.png','删除前 · 静态分支'),
                    (src/(f'a{actor:02d}_delete_no_shadow.png' if model=='r15' else f'a{actor:02d}_delete.png'),f'删除 a{actor} · 整景'),
                    (br/f'a{actor:02d}_dynamic_delete.png',f'删除 a{actor} · 动态分支'),(br/'static.png',f'删除 a{actor} · 静态分支')]
                for k,(file,caption) in enumerate(files):
                    x=(k%3)*420;y=j*570+38+(k//3)*266
                    draw.rectangle((x+2,y,x+418,y+262),outline=(58,95,121),width=1)
                    draw.text((x+10,y+3),caption,font=font,fill=(205,230,247))
                    tile=Image.open(file).convert('RGB').crop(box);scale=min(410/tile.width,230/tile.height)
                    tile=tile.resize((round(tile.width*scale),round(tile.height*scale)),Image.Resampling.LANCZOS)
                    image.paste(tile,(x+(420-tile.width)//2,y+28+(230-tile.height)//2))
            p.stdin.write(np.asarray(image).tobytes())
    finally:p.stdin.close()
    if p.wait()!=0:raise RuntimeError('overview encoding failed')
    print(target.name,flush=True)
print('All five synchronized overview videos are complete')
