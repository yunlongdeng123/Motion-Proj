"""Decode every referenced video and check all actual player modes and frames."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib,json,re,subprocess
import numpy as np
from PIL import Image
W=Path(__file__).resolve().parent
O=W.parent.parent/'outputs/gn11_stage1_v2_20261019'

def main():
    receipt=json.loads((O/'cpu_export_media.json').read_text(encoding='utf-8'));assert receipt['new_videos']==22
    results={};fixtures={};links=0
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.refs=[]
        def handle_starttag(self,tag,attrs):
            self.refs.extend(v for k,v in attrs if k in ['src','href'] and v)
    for page in receipt['pages']:
        text=(O/page).read_text(encoding='utf-8');p=Links();p.feed(text)
        for ref in p.refs:
            if ref!='cpu_export_validation.json':assert (O/ref).is_file(),ref
        links+=len(p.refs)
        for n,s in enumerate(re.findall(r'<script>(.*?)</script>',text,re.S)):
            path=W/(page+f'.generated{n}.js');path.write_text(s,encoding='utf-8')
            subprocess.run(['node','--check',str(path)],check=True,capture_output=True)
        f=subprocess.run(['node',str(W/'check_cpu_export_player.js'),str(O/page)],check=True,capture_output=True,text=True)
        fixtures[page]=json.loads(f.stdout)
    for key,item in receipt['videos'].items():
        path=O/item['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        raw=subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'],check=True,capture_output=True).stdout
        frames=np.frombuffer(raw,dtype=np.uint8).reshape(-1,320,480,3);assert len(frames)==49
        errors=[]
        for n,rgb in enumerate(frames):
            p=W/str(item['source_pattern']).replace('%07d',f'{n+8089:07d}')
            with Image.open(p) as im:source=np.asarray(im.convert('RGB'),dtype=np.int16)
            errors.append(float(np.abs(source-rgb.astype(np.int16)).mean()))
        assert max(errors)<8,(key,max(errors))
        results[key]={'decoded_frames':49,'source_frame_order_verified':True,'max_frame_RGB_MAE_255':max(errors),'mean_RGB_MAE_255':sum(errors)/49}
    assert len(results)==34
    r={'status':'PASSED_NATIVE_FOUR_AND_INDEPENDENT_TWO_FULL_FRAME_MEDIA','videos':results,'decoded_frames':len(results)*49,
      'video_count':len(results),'full_frame_wh':[480,320],'local_links':links,'generated_player_fixtures':fixtures,
      'browser_UI_tested':False,'quality_accepted':False,'human_verdict':None}
    (O/'cpu_export_validation.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in r.items() if k not in ['videos']}))

if __name__=='__main__':main()
