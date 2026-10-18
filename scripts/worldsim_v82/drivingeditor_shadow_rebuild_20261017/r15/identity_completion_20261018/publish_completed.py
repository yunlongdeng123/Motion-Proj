"""Publish compact HTML/videos and complete tristate evidence; full PNG matrix stays local/remote."""
from pathlib import Path
import shutil,json,os
W=Path(__file__).parent;P=W.parent/'r15_closure_run/publish_repo'
O=W.parents[1]/'outputs/identity_comparison_20261018'
D=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018')
S=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/identity_completion_20261018')
def long(p):return Path('\\\\?\\'+str(p.resolve())) if os.name=='nt' else p
def cp(src,dst):
    dst=long(dst);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(long(src),dst)
page=(O/'index.html').read_text(encoding='utf-8').split('<h2>逐身份、逐帧检查</h2>')[0]
page=page.replace('<div class="note"><strong>已更新：','<div class="note"><strong>Git 轻量报告：已更新：',1)
page+='<h2>完整49帧视频</h2><p>本页保存全49帧视频和固定锚帧对照。全部原分辨率逐帧 PNG 与交互选择位于用户本地 outputs/identity_comparison_20261018/index.html，原始浮点结果保留远端。旧基线与原R15输入、阶段不同，不作单因果质量归因。</p>'
for actor in [6,37,5,26,18]:
    page+=f'<h3>a{actor}</h3><div class="videogrid">'
    for model in ['old3','old5','r15']:
        if model=='old3' and actor not in [5,26,18]:continue
        for mode in ['isolated','delete']+(['delete_no_shadow'] if model=='r15' else []):
            name=f'{model}_a{actor:02d}_{mode}.mp4'
            page+=f'<figure><video controls loop muted preload="metadata" src="videos/{name}"></video><figcaption>{model} / a{actor} / {mode}</figcaption></figure>'
    page+='</div>'
page+='<p><a href="old5_render_receipt.json">旧5渲染凭据</a> · <a href="tristate/index.html">三维归属与原始支撑投影</a></p></html>'
target=long(P/D/'baseline_complete.html');target.parent.mkdir(parents=True,exist_ok=True);target.write_text(page,encoding='utf-8')
# The published center viewer's return link must resolve to this compact page.
viewer=long(P/D/'tristate/index.html');s=viewer.read_text(encoding='utf-8').replace('href="../index.html"','href="../baseline_complete.html"').replace('返回完整49帧独立／删除对照','返回49帧视频与固定锚帧对照').replace('../../r15_repair_4k_20261018/index.html','../../repair_4k_20261018/index.html');viewer.write_text(s,encoding='utf-8')
for f in (O/'videos').glob('*.mp4'):cp(f,P/D/'videos'/f.name)
for name in ['a06_completed_old5_review.png','a37_completed_old5_review.png','completed_viewer_preview.png','completion_status.json']:
    cp(O/name,P/D/name)
for name in ['review_completed.py','pack_completed.py','publish_completed.py','stage_identity_publish.py','update_completed_docs.py']:
    cp(W/name,P/S/name)
for name in ['queue_status.json','tristate_queue_status.json','train5/status.json']:
    cp(W/'completed/metadata_completed'/name,P/D/'completed'/name)
# Update the already published closure's navigation without replacing its history.
closure=P/D.parent/'closure_20261018/report.html'
if long(closure).is_file():
    s=long(closure).read_text(encoding='utf-8')
    marker='<p class="warn">最新旧5对照：<a href="../identity_20261018/baseline_complete.html">旧GN11 30K＋五实例30K已完成，含五身份视频及a6/a37直接对照</a>；<a href="../identity_20261018/tristate/index.html">旧5／原R15三维归属与真实支撑</a>。本页历史实验与电源记录保留。</p>'
    if marker not in s:s=s.replace('<h1>',marker+'<h1>',1)
    long(closure).write_text(s,encoding='utf-8')
print('compact baseline report, videos and evidence published to checkout')
