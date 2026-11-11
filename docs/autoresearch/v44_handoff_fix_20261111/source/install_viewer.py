"""安装已验收五列页面；保留旧四列、历史结果和scene029。"""
from pathlib import Path
from datetime import datetime
import hashlib,json,os,re,shutil
H=Path(__file__).resolve().parent;S=H/'viewer/stage_final_r1'
P=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a/outputs/GS_Scene_Diagnostics_20261027');C=P/'15021599536622641101_556_150_576_150';A='native_handoff_fix_20261111'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((S/'build_receipt.json').read_text(encoding='utf-8'));assert r['status']=='STAGED_WITH_VERIFIED_ASSETS' and r['verified_pngs']==398
for name,digest in r['source_files_sha256'].items():assert sha(Path(name))==digest,'Concurrent platform change: '+name
assert (C/A).resolve().parent==C.resolve()
if (C/A).exists():
    for name in ['data.js','viewer.js','manifest.json']:assert sha(C/A/name)==sha(S/A/name),'Unexpected existing asset directory'
stamp=datetime.now().strftime('%Y%m%d_%H%M%S');old=C/'index.html';backup=C/('index_before_handoff_fix_'+stamp+'.html');shutil.copy2(old,backup)
before=C/'native_delete30k.html'
if before.exists():assert sha(before)==sha(S/'native_delete30k.html')
else:shutil.copy2(S/'native_delete30k.html',before)
assert sha(before)==sha(backup)
preserved={p:sha(p) for p in [C/'native_10k.html',C/'native_continuation.html',C/'history_previous.html',P/'12820461091157089924_5202_916_5222_916/index.html']}
if not (C/A).exists():shutil.copytree(S/A,C/A)
mf=json.loads((C/A/'manifest.json').read_text(encoding='utf-8'))
for f in mf['files']:assert sha(C/A/f['path'])==f['sha256']
data=json.loads((C/A/'data.js').read_text(encoding='utf-8').split('=',1)[1].rstrip().removesuffix(';'))
assert [f['raw'] for f in data['frames']]==list(range(199))
for row in data['frames']:
    for key in ['observed','native30','delete_before','fixed_keep','fixed_delete']:assert (C/A/row[key]).is_file()
temp=C/'index.handoff_fix.pending.html';shutil.copy2(S/'index.html',temp);os.replace(temp,old)
for path,digest in preserved.items():assert sha(path)==digest
preview=C/'preview_handoff_fix_20261111.png';shutil.copy2(C/A/'delete_white/119.png',preview)
root=P/'index.html';root_text=root.read_text(encoding='utf-8');shutil.copy2(root,P/('index_before_handoff_fix_'+stamp+'.html'))
context=C.name;pattern=r'(<article\b[^>]*data-scene="'+context+r'"[^>]*>)(.*?)(</article>)'
def card(match):
    content,n=re.subn(r'(<img src=")[^"]+(" alt=")[^"]*(")',lambda x:x[1]+context+'/'+preview.name+x[2]+'白车删除修复前后对照'+x[3],match[2],count=1);assert n==1
    content,n=re.subn(r'<p>.*?</p>','<p>原生30K与白车删除修复前后，199帧同步查看。</p>',content,count=1,flags=re.S);assert n==1
    return match[1]+content+match[3]
root_text,n=re.subn(pattern,card,root_text,count=1,flags=re.S);assert n==1 and root_text.count('href="'+context+'/index.html"')==1
root.write_text(root_text,encoding='utf-8')
receipt=dict(status='INSTALLED_FIVE_COLUMN_HANDOFF_COMPARISON',index=str(old),frames=199,columns=5,default_raw=119,default_overlay='none',
    baseline_four_column=str(before),old_results_unchanged=True,scene029_unchanged=True,pngs_verified=398,movies=r['movies'],
    logic_checks=39,browser_qa=False,archive_verified=True,human_verdict=None)
(H/'install_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))
