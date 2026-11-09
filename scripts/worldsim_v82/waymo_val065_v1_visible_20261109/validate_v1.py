from pathlib import Path
from html.parser import HTMLParser
import json,hashlib,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
CHILD=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a/outputs/GS_Scene_Diagnostics_20261027/15021599536622641101_556_150_576_150')
OUT=BASE/'outputs/waymo_val065_experiment'
class Page(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.paths=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        for k in ['href','src']:
            v=a.get(k)
            if v and not v.startswith(('#','http:','https:','file:','data:')):self.paths.append(v)
def main():
    proof=json.loads((HERE/'evidence/v1_provenance.json').read_text())
    assert proof['formal_training']['actual_additional_iterations']==10000
    assert proof['fresh_start_checkpoint'] is None and proof['delete_actor_ids']==[]
    assert proof['replay']['editing_package_imported'] is False
    assert proof['replay']['semantic_rebinding'] is False and proof['replay']['image_composite'] is False
    assert {v['raw'] for v in proof['replay']['rows']}=={60,108,150,198}
    assert all(v['saved_native_maxabs']==0 for v in proof['replay']['rows'])
    count=0
    for directory in [CHILD,OUT]:
        text=(directory/'index.html').read_text(encoding='utf8');page=Page();page.feed(text)
        assert len(page.ids)==len(set(page.ids));assert all((directory/v).exists() for v in page.paths)
        main_section=text.split('<div class="comparison main-compare">')[1].split('<div class="legend">')[0]
        import re
        assert re.findall(r'<canvas id="([^"]+)"',main_section)==['observed','native','nativeDelete','delete']
        assert 'V1 · 完整重建（10K，无编辑）' in main_section
        assert 'id="keep"' in text.split('id="beforeCompare"',1)[1]
        for row in proof['frames']:
            p=directory/'viewer/native'/str(row['raw'])/'keep.png'
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];count+=1
    replay=Image.open(HERE/'evidence/replay108.png');source=Image.open(CHILD/'viewer/native/108/keep.png');assert np.array_equal(np.array(replay),np.array(source))
    subprocess.run(['node','--check',str(HERE/'viewer.js')],check=True)
    subprocess.run(['node',str(HERE/'test_v1.cjs')],check=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);fig=Image.new('RGB',(1920,360),'#102131');draw=ImageDraw.Draw(fig)
    for j,(label,path) in enumerate([('实拍 · 第108帧','observed/108.png'),('V1 · 完整重建10K · 无编辑','native/108/keep.png'),('A · 按原始编号删除','native/108/delete.png'),('B · 重选元素＋补道路','current/108/delete.png')]):
        draw.text((j*480+10,7),label,font=font,fill='white');fig.paste(Image.open(CHILD/'viewer'/path),(j*480,40))
    fig.save(HERE/'frame108_v1_comparison.png')
    receipt=dict(status='PASS_VISIBLE_V1_10K_FRAME108_AND_ALL199_SOURCES',frames=199,local_and_platform_PNGs_checked=count,
        main_panels=['observed','native','nativeDelete','delete'],folded_B_keep=True,raw108_replay_RGB_maxabs=0,
        pure_V1_replay_raws=[60,108,150,198],new_training_updates=0,existing_formal_training=10000,
        original_direct_renderer=True,editing_package_imported=False,automatic_shutdown=False,human_verdict=None,
        browser_visual_acceptance='未进行；此前file协议被拒绝，未绕过。实际JS经DOM/Canvas替身执行。')
    for p in [HERE/'verification.json',OUT/'v1_visible_verification.json',CHILD/'v1_visible_verification.json']:p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
