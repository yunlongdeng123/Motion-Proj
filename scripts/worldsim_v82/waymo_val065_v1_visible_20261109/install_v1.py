"""将已完成的纯V1 10K提升为scene065主视图，保持现有A/B解释及结果。"""
from pathlib import Path
import json,re,shutil,datetime,hashlib,tarfile
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
PLATFORM=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a/outputs/GS_Scene_Diagnostics_20261027')
CHILD=PLATFORM/'15021599536622641101_556_150_576_150'
OUT=BASE/'outputs/waymo_val065_experiment'
def main():
    with tarfile.open(HERE/'v1_evidence.tar') as archive:
        for member in archive.getmembers():
            assert member.isfile();p=(HERE/'evidence'/member.name).resolve();assert p.is_relative_to((HERE/'evidence').resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(archive.extractfile(member).read())
    proof=json.loads((HERE/'evidence/v1_provenance.json').read_text())
    assert proof['formal_training']['actual_additional_iterations']==10000 and proof['fresh_start_checkpoint'] is None
    assert not proof['replay']['editing_package_imported'] and all(r['saved_native_maxabs']==0 for r in proof['replay']['rows'])
    for row in proof['frames']:
        for directory in [CHILD,OUT]:assert hashlib.sha256((directory/'viewer/native'/str(row['raw'])/'keep.png').read_bytes()).hexdigest()==row['sha256']
    backup=HERE/'backups'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for name,directory in [('platform',CHILD),('local',OUT)]:
        for rel in ['index.html','viewer/viewer.js','viewer/viewer.css']:
            target=backup/name/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(directory/rel,target)
    page=(CHILD/'index.html').read_text(encoding='utf8')
    match=re.search(r'<article><h2>A · 原生重建，不删车</h2><canvas id="native"[\s\S]*?</article>',page)
    if match:
        article=match.group().replace('<article>','<article class="method-v1">',1).replace('A · 原生重建，不删车','V1 · 完整重建（10K，无编辑）').replace('原生模型训练10,000步后的直接渲染。','完整场景从头训练10,000步，保留所有车辆，直接走原生渲染器。')
        page=page[:match.start()]+page[match.end():]
        page=page.replace('<article class="method-a">',article+'\n<article class="method-a">',1)
    else:
        assert 'class="method-v1"' in page,'expected current readable A/B page'
    page=page.replace('对比两种删除方法','完整重建与两种删除方法对比')
    page=page.replace('A 只按原始编号删除；B 跨帧重选要删除的白车元素，再加入道路。两者使用同一份原生重建，显示同一时刻。','V1 展示从头训练10K、不执行编辑的完整重建。A 按原始编号删车；B 重选白车元素并加入道路。四列显示同一相机与时刻。')
    page=page.replace('查看删除前：A、B各自的保留对照','B 的删除前对照：加入额外处理，不删车')
    page=page.replace('A 的删除结果与左图配对；B 的删除结果与右图配对。B 的保留状态也已经接入道路与静态透明度调整，因此两张保留图可能略有不同。','上方 V1 就是 A 的删除前对照。下方是 B 接入重新归属、道路与静态透明度调整后的保留状态，用于检查额外处理本身造成的变化。')
    page=page.replace('<option value="native">A · 只按原始编号</option>','<option value="native">V1／A · 原生通路</option>')
    if 'data-raw="108"' not in page:page=page.replace('<button data-raw="99">留出中段</button>','<button data-raw="99">留出中段</button><button data-raw="108">V1 对照帧108</button>')
    script=(CHILD/'viewer/viewer.js').read_text(encoding='utf8').replace('主区从左到右：实拍 / A 删除结果 / B 删除结果','主区：实拍 / V1 完整重建10K（无编辑） / A 删除 / B 删除')
    script=script.replace('function select(c,e){const box=',"function select(c,e){const route={native:['native','keep'],nativeDelete:['native','delete'],delete:['current','delete'],keep:['current','keep']}[c.id];if(route){S.attributionRoute=route[0];S.mapState=route[1];$('attributionRoute').value=route[0];$('mapState').value=route[1];}const box=")
    css=(CHILD/'viewer/viewer.css').read_text(encoding='utf8')
    marker='/* V1 10K在主区独立展示 */'
    if marker not in css:
        css+='\n'+marker+'\n.main-compare .method-v1{border-top:3px solid #aebff0}.before-compare .keep-compare{grid-template-columns:minmax(0,720px)}\n@media(min-width:1401px){.main-compare{grid-template-columns:repeat(4,minmax(0,1fr))}}\n@media(min-width:601px) and (max-width:1400px){.main-compare{grid-template-columns:repeat(2,minmax(0,1fr))}.main-compare>article:first-child{grid-column:auto;display:block}.main-compare>article:first-child p{margin-top:8px}}\n'
    (CHILD/'index.html').write_text(page,encoding='utf8')
    local=page.replace('href="../style.css"',f'href="{(PLATFORM/"style.css").as_uri()}"').replace('href="../workspace.css"',f'href="{(PLATFORM/"workspace.css").as_uri()}"').replace('href="../index.html"',f'href="{(PLATFORM/"index.html").as_uri()}"')
    (OUT/'index.html').write_text(local,encoding='utf8')
    for directory in [CHILD,OUT]:
        (directory/'viewer/viewer.js').write_text(script,encoding='utf8');(directory/'viewer/viewer.css').write_text(css,encoding='utf8')
        (directory/'viewer/evidence').mkdir(exist_ok=True)
        shutil.copy2(HERE/'evidence/v1_provenance.json',directory/'viewer/evidence/v1_10k_provenance.json')
    (HERE/'viewer.html').write_text(page,encoding='utf8');(HERE/'viewer.js').write_text(script,encoding='utf8');(HERE/'viewer.css').write_text(css,encoding='utf8')
    (HERE/'installation.json').write_text(json.dumps(dict(status='V1_MAIN_PANEL_INSTALLED',backup=str(backup),main_order=['observed','native','nativeDelete','delete'],folded_keep='keep',frames=199,new_training_updates=0,automatic_shutdown=False),indent=2))
    print('V1_MAIN_PANEL_INSTALLED',backup)
if __name__=='__main__':main()
