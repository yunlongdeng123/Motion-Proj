"""逐视频解码全部49帧，并验证HTML引用、实际JS控制器与完整画面。"""
from pathlib import Path
from html.parser import HTMLParser
import hashlib,json,re,subprocess,xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

W=Path(__file__).resolve().parent
O=W.parent.parent/'outputs/gn11_stage1_v2_20261019'

def main():
    text=(O/'native30k_compare.html').read_text(encoding='utf-8')
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.refs=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['src','href'] and v:self.refs.append(v)
    parser=Links();parser.feed(text)
    for ref in parser.refs:assert (O/ref).is_file(),ref
    ET.parse(O/'native30k_compare.svg')
    scripts=re.findall(r'<script>(.*?)</script>',text,re.S)
    for i,script in enumerate(scripts):
        p=W/f'native30k_player_{i}.js';p.write_text(script,encoding='utf-8')
        subprocess.run(['node','--check',str(p)],check=True,capture_output=True)
    fixture=subprocess.run(['node',str(W/'check_native30k_player.js'),str(O/'native30k_compare.html')],check=True,capture_output=True,text=True)
    handler=json.loads(fixture.stdout)
    receipt=json.loads((O/'native30k_media.json').read_text(encoding='utf-8'))
    videos={}
    for key,item in receipt['videos'].items():
        path=O/item['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        arm,channel=key.rsplit('_',1)
        decoded=subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'],
            check=True,capture_output=True).stdout
        frames=np.frombuffer(decoded,dtype=np.uint8).reshape(-1,320,480,3);assert len(frames)==49
        errors=[]
        for i,rgb in enumerate(frames):
            with Image.open(W/'native30k_pair'/arm/channel/f'0008{i+89:03d}.png') as im:original=np.asarray(im.convert('RGB'),dtype=np.int16)
            errors.append(float(np.abs(rgb.astype(np.int16)-original).mean()))
        # This checks source/order under lossy encoding, not scene reconstruction quality.
        assert max(errors)<8,(key,max(errors))
        videos[key]={'decoded_frames':49,'width':480,'height':320,'source_frame_order_checked':True,
            'mean_encoding_RGB_MAE_255':sum(errors)/len(errors),'max_frame_encoding_RGB_MAE_255':max(errors)}
    result={'status':'PASSED_NATIVE49_FILES_ALL_FRAME_DECODING_AND_PLAYER_FIXTURE',
        'videos':videos,'all_decoded_frames':392,'full_frame':True,'local_static_links':len(parser.refs),
        'architecture_XML':True,'JS_syntax_checks':len(scripts),'generated_handler_fixture':handler,
        'browser_UI_tested':False,'quality_accepted':False,'human_verdict':None}
    (O/'native30k_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':result['status'],'videos':len(videos),'all_decoded_frames':392,'handler':handler}))

if __name__=='__main__':main()
