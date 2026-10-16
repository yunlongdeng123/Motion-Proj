import zipfile,subprocess,shutil,argparse
from pathlib import Path
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--latest',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parents[2];work=root/'work/shadow_sequence_v2';out=root/'outputs/shadow_sequence_v2';ev='final49_unselected_v1' if args.latest else 'final49_contribution_v1'
with zipfile.ZipFile(work/('final_delivery_package_v2.zip' if args.latest else 'final_delivery_package_v1.zip')) as archive:
    for info in archive.infolist():
        base=work if info.filename.startswith('raw_videos/') else out
        dest=base/info.filename;assert dest.resolve().is_relative_to(base.resolve());dest.parent.mkdir(exist_ok=True,parents=True)
        with archive.open(info) as source,dest.open('wb') as target:shutil.copyfileobj(source,target)
ffmpeg=shutil.which('ffmpeg');assert ffmpeg
for src in (work/'raw_videos'/ev).glob('*.mp4'):
    dest=out/'videos'/ev/src.name;dest.parent.mkdir(exist_ok=True,parents=True)
    subprocess.run([ffmpeg,'-v','error','-y','-i',str(src),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
review=out/'reviews'/ev;frames=sorted((review/'comparison_frames').glob('*.png'));assert len(frames)==49
for page in range(4):
    subset=frames[page*14:(page+1)*14];sheet=Image.new('RGB',(960,368*((len(subset)+1)//2)),'white');draw=ImageDraw.Draw(sheet)
    for k,p in enumerate(subset):
        image=Image.open(p);image.thumbnail((480,344));x=k%2*480;y=k//2*368;sheet.paste(image,(x,y+24));draw.text((x+5,y+5),'Raw '+p.stem,fill='black')
    sheet.save(review/('page_%d.jpg'%(page+1)),quality=94)
for p in work.glob('*.py'):shutil.copy2(p,out/'scripts'/p.name)
print('Extracted, encoded and prepared all49 native comparison review')
