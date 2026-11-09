"""校验删除结果并原子替换065场景入口；保留此前全部真实结果。"""
from pathlib import Path,PurePosixPath
from datetime import datetime
from html.parser import HTMLParser
import argparse,hashlib,json,os,re,shutil,subprocess,tarfile
from PIL import Image

HERE=Path(__file__).resolve().parent
PLATFORM=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a/outputs/GS_Scene_Diagnostics_20261027')
CONTEXT='15021599536622641101_556_150_576_150'
CHILD=PLATFORM/CONTEXT
ASSET='native_delete30k_20261109'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Page(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.refs=[]
    def handle_starttag(self,t,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        self.refs.extend(a[k] for k in ['src','href'] if k in a)


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--bytes',type=int,required=True)
    a=p.parse_args();assert a.archive.stat().st_size==a.bytes and sha(a.archive)==a.sha256
    unpack=HERE/'delivery_r1';assert not unpack.exists();unpack.mkdir()
    with tarfile.open(a.archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
        for m in members:
            parts=PurePosixPath(m.name);assert m.isfile() and not parts.is_absolute() and '..' not in parts.parts and '\\' not in m.name and not re.match(r'^[A-Za-z]:',m.name)
        for m in members:
            dest=unpack/m.name;dest.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as source,dest.open('wb') as target:shutil.copyfileobj(source,target)
    mf=json.loads((unpack/'manifest.json').read_text());assert mf['scene']=='065' and mf['frames']==199 and mf['iteration']==30000
    assert mf['target_track']=='NTjq-OhZ_97OMJ4xwJAxXg'
    assert set(names)=={'manifest.json'}|{r['path'] for r in mf['files']}
    for r in mf['files']:
        f=unpack/r['path'];assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']
        if f.suffix=='.png':
            with Image.open(f) as image:assert image.size==(480,320);image.verify()
    movies=[]
    for name in ['delete_white_30k.mp4','keep_30k.mp4']:
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
             '-show_entries','stream=width,height,codec_name,nb_read_frames,avg_frame_rate,duration','-of','json',str(unpack/name)]))['streams'][0]
        assert info['codec_name']=='h264' and info['width']==480 and info['height']==320
        assert info['avg_frame_rate']=='10/1' and int(info['nb_read_frames'])==199
        movies.append(dict(name=name,**info))
    done=json.loads((unpack/'proof/completed.json').read_text());assert done['source_native_checkpoint_unchanged'] and done['native_gaussian_updates']==0
    bundle=HERE/'viewer_bundle_r1';asset=bundle/ASSET
    for r in mf['files']:
        target=asset/r['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(unpack/r['path'],target)
    shutil.copyfile(unpack/'manifest.json',asset/'manifest.json')
    subprocess.run(['node','--check',str(asset/'viewer.js')],check=True)
    subprocess.run(['node',str(HERE/'test_viewer.cjs'),str(bundle)],check=True)
    page=(bundle/'index.html').read_text(encoding='utf8');parser=Page();parser.feed(page)
    assert len(parser.ids)==len(set(parser.ids))
    required=set(re.findall(r"\$\('([^']+)'\)",(asset/'viewer.js').read_text(encoding='utf8')))
    assert required<=set(parser.ids)
    d=json.loads((asset/'data.js').read_text(encoding='utf8').strip().split('=',1)[1][:-1])
    assert d['default_raw']==119 and d['default_continuation_step']==30000
    assert [f['raw'] for f in d['frames']]==list(range(199))
    for row in d['frames']:
        for key in ['observed','native','native20','native30','delete_white','edited_keep']:
            f=asset/row[key] if key in ['delete_white','edited_keep'] else CHILD/ASSET/row[key]
            assert f.is_file(),str(f)
    preserved={p:sha(p) for p in [CHILD/'index.html',CHILD/'native_10k.html',CHILD/'history_previous.html',
         PLATFORM/'12820461091157089924_5202_916_5222_916/index.html']}
    assert 'native_continuation_20261109/viewer.js' in (CHILD/'index.html').read_text(encoding='utf8')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    backup=CHILD/f'index_before_delete30k_{stamp}.html';shutil.copy2(CHILD/'index.html',backup)
    prior=CHILD/'native_continuation.html'
    if prior.exists():assert sha(prior)==sha(backup)
    else:shutil.copy2(backup,prior)
    target=CHILD/ASSET;assert not target.exists();shutil.copytree(asset,target)
    for r in mf['files']:assert sha(target/r['path'])==r['sha256']
    for ref in parser.refs:
        if ':' not in ref and not ref.startswith('#'):assert (CHILD/ref.split('#')[0]).is_file(),ref
    temp=CHILD/'index.delete30k.pending.html';temp.write_text(page,encoding='utf8');os.replace(temp,CHILD/'index.html')
    preview=CHILD/'preview_delete30k_20261109.png';shutil.copyfile(target/'delete_white/119.png',preview)
    root=PLATFORM/'index.html';root_text=root.read_text(encoding='utf8');shutil.copy2(root,PLATFORM/f'index_before_delete30k_{stamp}.html')
    pattern=r'(<article\b[^>]*data-scene="'+CONTEXT+r'"[^>]*>)(.*?)(</article>)'
    def card(m):
        content,n=re.subn(r'(<img src=")[^"]+(" alt=")[^"]*(")',lambda x:x[1]+CONTEXT+'/'+preview.name+x[2]+'30K白车删除预览'+x[3],m[2],count=1);assert n==1
        content,n=re.subn(r'<p>.*?</p>','<p>实拍、10K、30K与白车删除同步对照，可切换编辑场景保留。</p>',content,count=1,flags=re.S);assert n==1
        return m[1]+content+m[3]
    updated,n=re.subn(pattern,card,root_text,count=1,flags=re.S);assert n==1
    root.write_text(updated,encoding='utf8');assert updated.count('href="'+CONTEXT+'/index.html"')==1
    for p,h in preserved.items():
        if p!=CHILD/'index.html':assert sha(p)==h
    assert sha(prior)==preserved[CHILD/'index.html']
    receipt=dict(status='INSTALLED_065_DELETE30K_FOUR_COLUMN_VIEWER',index=str(CHILD/'index.html'),
        frames=199,columns=4,default_raw=119,default_native_step=30000,edit_step=30000,
        pngs_verified=398,movies=movies,archive_sha256=a.sha256,history_preserved=True,scene029_unchanged=True,
        CPU_DOM_checks=19,browser_qa=False,quality_accepted=None)
    (HERE/'install_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
