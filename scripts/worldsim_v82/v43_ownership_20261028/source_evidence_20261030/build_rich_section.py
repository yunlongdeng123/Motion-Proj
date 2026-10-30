"""Compose same-query richer-observation comparison from saved native outputs."""
from pathlib import Path
import json
import shutil
import html
import re

WORK=Path(__file__).resolve().parent
BASE=WORK.parents[1]
OUT=BASE/'outputs/V43_Ownership_20261027/gpu_v43/source_evidence_20261030'
OLD=BASE/'outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030/evidence'
NEW=WORK/'evidence_reconstruction_r1/evidence'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def copy(p,rel):
    target=OUT/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);return rel


def main():
    receipt=read(NEW/'completion_receipt.json')
    assert receipt['formal_updates']==10000 and receipt['smoke']['complete_restore']
    assets={}
    for frame in ['000000','000008','000015']:
        for edit in ['none','delete','shift','rotate']:
            assets[f'{frame}/{edit}/ref']=copy(OLD/f'dataset_r2/privileged/references/{edit}/shadow_on/rgb/{frame}.png',f'images/rich/ref/{edit}/{frame}.png')
            for arm,root in [('old',OLD/'edit_r3_10k'),('new',NEW/'edit_samequery')]:
                for channel in ['final','white','C','alpha','fill','E']:
                    assets[f'{frame}/{edit}/{arm}/{channel}']=copy(root/f'{edit}/{frame}_{channel}.png',f'images/rich/{arm}/{edit}/{frame}_{channel}.png')
    table=''
    scores={a:{(r['frame'],r['edit']):r for r in read(p)['rows']} for a,p in [('old',OLD/'edit_r3_10k/reference_metrics.json'),('new',NEW/'edit_samequery/reference_metrics.json')]}
    for frame in ['000000','000008','000015']:
        for edit in ['none','delete','shift','rotate']:
            old,new=(scores[a][frame,edit]['regions'] for a in ['old','new'])
            region='full' if edit=='none' else 'newly_exposed_road'
            table+=f'<tr><td>{frame}</td><td>{edit}</td><td>{region}</td><td>{old[region]["pixels"]}</td>'
            for arm in (old,new):
                table+=''.join('<td>%.5f</td>'%arm[region][k] for k in ['raw_mae','clamped_mae'])
            table+='</tr>'
    strat=read(NEW/'stratified_metrics.json')
    summaries={(r['arm'],r['edit'],r['group']):r for r in strat['pixel_weighted_summary']}
    strata_table=''
    for edit in ['delete','shift','rotate']:
        for group in ['old_observed_road','old_unseen_road','old_2plus_frames_road']:
            row=[summaries[a,edit,group] for a in ['M0_original_observations','same_world_richer_observations']]
            strata_table+=f'<tr><td>{edit}</td><td>{group}</td><td>{row[0]["pixels"]}</td>'+''.join('<td>%.5f</td>'%r['metrics']['clamped_mae'] for r in row)+'</tr>'
    cap=receipt['parameter_capacity']
    capacity_table=''.join('<tr><td>'+html.escape(a)+'</td><td>'+str(c['static']['gaussians'])+'</td><td>'+str(c['dynamic']['gaussians'])+'</td><td>'+str(c['environment']['model_parameter_elements'])+'</td></tr>' for a,c in cap.items())
    verdict=(WORK/'evidence_reconstruction_r1/RESULT_INTERPRETATION.html').read_text(encoding='utf8')
    section='''<p>完成真实约10分钟smoke，覆盖原生保存、评价、全部状态恢复与下一阶段；随后从零正式10K。新旧使用相同GN11方法、seed、30K学习率周期，固定旧世界→模型坐标和camera extent。输入仍为合法RGB、稀疏LiDAR、登记的轨迹与mask，独立编辑参考未进入训练。</p>
<p>新轨迹是13个环绕视角＋3个保留旧相机。它同时改变RGB、LiDAR覆盖与合法初始化，且终态容量不同；不能称纯RGB效应或等容量。相机逆矩阵采用float64规范化，三个锚点RGB有极少量重放微差，完整披露在收据。</p>
<div class="controls"><label>同帧 <select id="richFrame"><option>000000</option><option selected>000008</option><option>000015</option></select></label><label>编辑 <select id="richEdit"><option>none</option><option selected>delete</option><option>shift</option><option>rotate</option></select></label><label>分量 <select id="richChannel"><option value="final">最终RGB</option><option>white</option><option>C</option><option>alpha</option><option>fill</option><option>E</option></select></label></div><div id="richGrid" class="grid"></div>
<p>这里两臂均为各自原生动态分支执行编辑，不额外调用M1选点。下方M1来源复查另列，不能把旧M1与新原生结果混作方法比较。</p>__VERDICT__
<details><summary>展开原相机分区误差（完整RGB/原环境）</summary><table><tr><th>帧</th><th>编辑</th><th>分区</th><th>像素数</th><th>旧raw MAE</th><th>旧clamped MAE</th><th>新raw MAE</th><th>新clamped MAE</th></tr>__TABLE__</table></details>
<details open><summary>旧输入已见/未见道路：三个锚点像素加权</summary><table><tr><th>编辑</th><th>分区</th><th>像素数</th><th>旧clamped MAE</th><th>新clamped MAE</th></tr>__STRATA__</table><p class="small">世界点先按原16视图first-hit可见性分组，再与当前编辑的独立road表面相交。跨锚点可能重复世界位置，非面积平均；未知未见部分不是单靠身份必须恢复的道路真值。</p></details>
<details><summary>预算、容量与可复查执行收据</summary><table><tr><th>观测臂</th><th>静态行</th><th>动态行</th><th>环境参数</th></tr>__CAPACITY__</table><p>正式10K不包含smoke __SMOKE__次更新与入口__ENTRY__次更新；smoke训练循环__SECONDS__秒。初始化、保存、评价与正式执行墙钟分开登记。完成3锚点×4编辑及1留出相机×4编辑，不能据此宣称全时序稳定性通过。</p><p><a href="evidence/rich/completion_receipt.json">完整执行/恢复/坐标/容量收据</a> · <a href="evidence/rich/edit_samequery/reference_metrics.json">12组共同query指标</a> · <a href="evidence/rich/stratified_metrics.json">世界可见性分层</a> · <a href="evidence/rich/heldout_samequery_metrics.json">留出4编辑指标</a></p></details>
<script>const richA=__ASSETS__;function showRich(){const f=document.querySelector('#richFrame').value,e=document.querySelector('#richEdit').value,c=document.querySelector('#richChannel').value;const x=[[richA[f+'/'+e+'/ref'],'独立世界编辑参考'],[richA[f+'/'+e+'/old/'+c],'原16观测 · 原生10K · '+c],[richA[f+'/'+e+'/new/'+c],'丰富16观测 · 同方法10K · '+c]];document.querySelector('#richGrid').innerHTML=x.map(([s,t])=>'<figure><img src="'+s+'" alt="'+t+'"><figcaption>'+t+'</figcaption></figure>').join('')}for(const id of ['richFrame','richEdit','richChannel'])document.getElementById(id).onchange=showRich;showRich();</script>'''
    for key,value in dict(VERDICT=verdict,TABLE=table,STRATA=strata_table,CAPACITY=capacity_table,ASSETS=json.dumps(assets),
                          SMOKE=str(receipt['separate_extra_budget']['smoke_updates']),ENTRY=str(receipt['separate_extra_budget']['entry_updates']),
                          SECONDS='%.3f'%receipt['runtime']['smoke_loop_seconds']).items():section=section.replace('__'+key+'__',value)
    extra=WORK/'source_control_r1/FRESH_COMPARISON.html'
    if extra.exists():
        text=extra.read_text(encoding='utf8')
        body=re.search(r'<main>(.*?)</main>',text,re.S).group(1)
        body=body.replace('evidence/source/','evidence/source/source/')
        body=re.sub(r'<a href="[^"]+\.npz">.*?</a> · ', '', body)
        section+='''<style>.fresh-content .gallery{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.fresh-content .card{border:1px solid #c8dbe5;padding:12px;margin:15px 0}.fresh-content .anchor{border:1px solid #c8dbe5;padding:12px}.fresh-content .muted{font-size:13px;color:#566f7d}.fresh-content .barrow{display:grid;grid-template-columns:180px 1fr 170px;gap:10px;align-items:center;margin:15px 0}.fresh-content .bar{height:24px;display:flex}.fresh-content .legend span{display:inline-block;width:12px;height:12px;margin-left:14px;margin-right:4px}.fresh-content svg{width:100%}.fresh-content figcaption{font-size:12px}.fresh-content button{padding:8px}@media(max-width:900px){.fresh-content .gallery{grid-template-columns:repeat(2,minmax(0,1fr))}.fresh-content .barrow{grid-template-columns:1fr}}</style><div class="fresh-content">'''+body+'''<p class="small">全行NPZ在本地source_control_r1/evidence/source/与远端保留；网页版/Git附聚合JSON和SHA清单，不嵌入模型数组。</p></div><script>function mode(white){document.querySelectorAll('.fresh-content img[data-white]').forEach(im=>{im.src=white?im.dataset.white:im.dataset.final;im.parentElement.href=im.src})}</script>'''
    (OUT/'rich_section.html').write_text(section,encoding='utf8')
    (OUT/'rich_assets.json').write_text(json.dumps(assets,indent=2),encoding='utf8')
    print(json.dumps(dict(images=len(assets),native_rows=len(scores['new']),formal_updates=10000)))


if __name__=='__main__':main()
