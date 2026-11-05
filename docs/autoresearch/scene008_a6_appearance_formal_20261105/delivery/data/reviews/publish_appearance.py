"""Publish a local, evidence-driven appearance report; no network or training.

Default curated input: work/a6_appearance_formal/collected_publication.
Copy receipts, summaries, assets and final_gallery there using remote-run relative
paths, then rerun this publisher. Missing files remain explicitly unavailable.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
from urllib.parse import quote
import zipfile

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
WORKSPACE = HERE.parents[2]
DEFAULT_OUTPUT = WORKSPACE/'outputs/scene008_vehicle_appearance/delivery'
HELD = {89,95,101,105,113,121,125,130,137}
SUFFIXES = {'.json','.png','.jpg','.jpeg','.svg','.npz','.py','.txt','.md','.log','.csv'}


def load(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):
        return None


def esc(value):
    return html.escape(str(value), quote=True)


def num(value, digits=6):
    if value is None:
        return '—'
    return f'{value:,.0f}' if isinstance(value,int) else f'{value:.{digits}f}'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def empty(colspan):
    return '<tr><td colspan="'+str(colspan)+'" class="muted">对应receipt / summary尚未取回。</td></tr>'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default=str(WORK/'collected_publication'))
    parser.add_argument('--output',default=str(DEFAULT_OUTPUT))
    args=parser.parse_args()
    source=Path(args.input).resolve();dest=Path(args.output).resolve()
    if source==dest or source in dest.parents:
        raise ValueError('Output must be outside the curated input tree')
    dest.mkdir(parents=True,exist_ok=True)

    def copy(src,rel):
        if not src.is_file():
            return None
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if src.resolve()!=target.resolve():
            shutil.copy2(src,target)
        return target

    if source.is_dir():
        for f in source.rglob('*'):
            if f.is_file() and f.suffix.lower() in SUFFIXES and not any(x.startswith('.') or x=='__pycache__' for x in f.relative_to(source).parts):
                copy(f,Path('data')/f.relative_to(source))
    for name in ('experiment_ledger.json','io_safety_review_final.json','sphere_output_and_fit_review.json',
                 'projective_surface_io_review.json','selection_prompt2_seed3.json','backend_inventory.json'):
        copy(HERE/name,Path('evidence')/name)
    copy(HERE/'candidates_raw106/qa_summary.json',Path('evidence/sd_candidates_qa.json'))
    copy(HERE/'candidates_raw106/model_manifest.json',Path('evidence/sd_model_manifest.json'))
    copy(HERE/'batch_manifest_raw106.json',Path('evidence/sd_batch_manifest.json'))
    for relative in ('backend/frozen_sd15_inpaint.py','backend/publish_appearance.py','backend/review_io_cpu.py',
                     'backend/review_sphere_cpu.py','backend/collect_experiment_ledger_remote.py',
                     'backend/review_projective_surface_cpu.py',
                     'fit_adapter/adapter.py','fit_adapter/bridge_packet.py','highres_fit.py',
                     'final_gallery.py','output_adapter/fit_sphere.py'):
        copy(WORK/relative,Path('code')/relative)

    files=[p for p in dest.rglob('*') if p.is_file() and p.name not in ('index.html','publication_receipt.json','file_manifest.json','evidence_and_code.zip')]

    def find(suffix):
        suffix=suffix.replace('\\','/')
        return next((p for p in sorted(files) if p.as_posix().endswith(suffix)),None)

    def url(path):
        return quote(path.relative_to(dest).as_posix(),safe='/') if path else ''

    def fallback(suffix,src,out):
        p=find(suffix)
        if p:return p
        p=copy(src,Path('figures')/out)
        if p:files.append(p)
        return p

    board=fallback('candidate_board.png',HERE/'candidates_raw106/candidate_board.png','sd_fixed8_board.png')
    strong=fallback('strong_candidate_v2_512.png',WORK/'generation/strong_candidate106_v2_512.png','strong_v2_2D_proposal.png')
    selected=fallback('prompt2_seed3/composed.png',HERE/'candidates_raw106/prompt2_seed3/composed.png','sd_prompt2_seed3_2D.png')
    conditioning=fallback('raw106/input_512.png',HERE/'raw106_input_512.png','build106_conditioning_512.png')
    preview=fallback('highres_evaluation/113_objects.jpg',WORK/'highres_evaluation/113_objects.jpg','existing_HR_render_113.jpg')
    kernel_preview=fallback('output_adapter/raw106_compare.png',WORK/'output_adapter/raw106_compare.png','kernel_output_diagnostic_106.png')

    ledger=load(dest/'evidence/experiment_ledger.json') or {}
    thin_review=load(dest/'evidence/io_safety_review_final.json') or {}
    sphere_review=load(dest/'evidence/sphere_output_and_fit_review.json') or {}
    totals=ledger.get('automatically_counted_totals_at_snapshot',{})
    all_json=[(p,load(p)) for p in sorted(files) if p.suffix=='.json']
    all_json=[(p,j) for p,j in all_json if isinstance(j,dict)]

    def label(path):
        value=path.relative_to(dest).as_posix()
        if 'projective_surface' in value:return '单参考view / projective surface'
        if 'sphere_strong' in value:return 'Sphere + strong'
        if 'sphere_highres' in value:return 'Sphere / HR observed'
        if 'sd_hr_hrfit' in value:return 'Thin + SD（离线baseline）'
        if 'strong_hr_hrfit' in value:return 'Thin + strong'
        if 'strong_m2_hrfit' in value:return 'M2 base + strong（诊断）'
        if 'strong_m2' in value:return '低分辨率M2 + strong（诊断）'
        if '/highres/' in value:return 'Thin / HR observed'
        return value.removeprefix('data/')

    fitrows=[]
    for p,j in all_json:
        if 'formal_generated_colour_steps' in j:
            e=j.get('export',{})
            fitrows.append(f'<tr><td>{esc(label(p))}</td><td>生成候选 → 资产颜色</td><td>{num(j.get("formal_generated_colour_steps"))}</td><td>{num(e.get("updated_columns"))}</td><td>{num(j.get("neural_training_steps"))}</td><td><a href="{url(p)}" download>receipt</a></td></tr>')
        elif 'colour_fit_steps' in j and p.parent.name=='formal':
            fitrows.append(f'<tr><td>{esc(label(p))}</td><td>真实BUILD RGB → 资产颜色</td><td>{num(j.get("colour_fit_steps"))}</td><td>见receipt</td><td>{num(j.get("neural_training_steps"))}</td><td><a href="{url(p)}" download>receipt</a></td></tr>')

    metricrows=[]
    for p,j in all_json:
        pooled=j.get('all49',{})
        if not isinstance(pooled,dict) or 'solo.a6.rgb_error' not in pooled:continue
        rgb=pooled.get('solo.a6.rgb_error',{});depth=pooled.get('solo.a6.measured_depth_error',{})
        a37=pooled.get('scene.keep.preservation.a37.parent_rgb_change',{})
        metricrows.append(f'<tr><td>{esc(label(p))}</td><td>{num(rgb.get("mean"))}<small>n={num(rgb.get("count"))}</small></td><td>{num(depth.get("mean"))}<small>n={num(depth.get("count"))}</small></td><td>{num(a37.get("mean"),8)}</td><td><a href="{url(p)}" download>summary</a></td></tr>')

    held_path=find('sphere_highres/evaluation/held_hr.json');held_j=load(held_path) if held_path else {}
    lidar_path=find('sphere_highres/evaluation/lidar49.json');lidar_j=load(lidar_path) if lidar_path else {}
    kernelrows=[]
    for method,title in (('thin_HR','Thin / HR observed'),('sphere_HR','Sphere / HR observed')):
        h=(held_j or {}).get('pooled',{}).get(method,{})
        l=(lidar_j or {}).get('pooled',{}).get('all49',{}).get(method,{})
        if h or l:
            kernelrows.append(f'<tr><td>{title}</td><td>{num(h.get("RGB_MAE"))}</td><td>{num(h.get("normalized_MAE"))}</td><td>{num(l.get("LiDAR_depth_error",{}).get("mean"))}</td><td>{num(l.get("LiDAR_alpha_coverage",{}).get("mean"))}</td></tr>')

    gallery_sets={}
    for gallery_name in ('final_gallery_formal_surface','final_gallery_surface','final_gallery'):
        present=any(f'/{gallery_name}/' in p.as_posix() for p in files)
        if not present:continue
        receipt_path=find(f'{gallery_name}/receipt.json')
        gallery_receipt=load(receipt_path) if receipt_path else {}
        if isinstance((gallery_receipt or {}).get('columns'),list):
            this_columns=[str(v) for v in gallery_receipt['columns']]
        elif isinstance((gallery_receipt or {}).get('assets'),dict):
            this_columns=['Observed']+list(gallery_receipt['assets'])
        else:
            this_columns=['列名等待本图库receipt；不沿用另一套列名']
        frames={}
        for raw in range(89,138):
            obj=find(f'{gallery_name}/{raw:03d}_objects.jpg');street=find(f'{gallery_name}/{raw:03d}_street.jpg')
            frames[str(raw)]={'objects':url(obj),'street':url(street),'split':'HELD' if raw in HELD else 'BUILD'}
        gallery_sets[gallery_name]={'frames':frames,'columns':this_columns,
            'complete':sum(bool(v['objects'] and v['street']) for v in frames.values()),
            'receipt':url(receipt_path),'street_asset_context':('40BUILD正式颜色拟合后的单参考view表面资产' if gallery_name=='final_gallery_formal_surface' else 'raw106原型单参考view表面资产' if gallery_name=='final_gallery_surface' else 'Sphere外观资产')}
    if not gallery_sets:
        gallery_sets['final_gallery']={'frames':{str(raw):{'objects':'','street':'','split':'HELD' if raw in HELD else 'BUILD'} for raw in range(89,138)},
            'columns':['Observed','M2','HR observed','Thin + strong','Sphere + strong'],'complete':0,'receipt':'','street_asset_context':'待本图库receipt确认'}
    active_gallery=next(iter(gallery_sets))
    active_set=gallery_sets[active_gallery]
    gallery=active_set['frames'];columns=active_set['columns'];available=active_set['complete']
    initial=106 if gallery['106']['objects'] else next((int(k) for k,v in gallery.items() if v['objects']),106)
    column_labels=''.join(f'<span>{esc(v)}</span>' for v in columns)
    gallery_options=''.join(f'<option value="{esc(k)}">{esc(k)} · {len(v["columns"])}列 · {v["complete"]}/49</option>' for k,v in gallery_sets.items())
    projective_files=[p for p in files if 'projective_surface' in p.as_posix()]

    def figure(path,title,caption,css=''):
        image=f'<a href="{url(path)}" target="_blank"><img src="{url(path)}" alt="{esc(title)}" loading="lazy"></a>' if path else '<div class="missing">文件尚未取回</div>'
        return f'<figure class="{css}"><figcaption><strong>{esc(title)}</strong><p>{esc(caption)}</p></figcaption>{image}</figure>'

    def link(path,text):
        return f'<a href="{url(path)}" download>{esc(text)}</a>' if path and path.is_file() else '<span class="muted">待取回</span>'

    projective_receipts=[(p,load(p)) for p in projective_files if p.name=='receipt.json']
    projective_rows=''.join(f'<tr><td>{esc(p.relative_to(dest))}</td><td>{esc((j or {}).get("status","未标记"))}</td><td>{link(p,"receipt")}</td></tr>'
                            for p,j in projective_receipts)
    projective_evals=[(p,j) for p,j in projective_receipts if isinstance((j or {}).get('summary'),dict) and '/evaluation/' in p.as_posix()]
    projective_evals.sort(key=lambda item:'formal_observed/evaluation/' in item[0].as_posix(),reverse=True)
    projective_metric_rows=[]
    if projective_evals:
        ep,ej=projective_evals[0];summ=ej['summary'];is_formal='formal_observed/evaluation/' in ep.as_posix()
        for method,m in summ.get('HELD',{}).items():
            title={'sphere_strong_v2':'Sphere + strong','projective_prototype':'单view表面 · raw106原型','projective_surface':'单view表面 · 40BUILD正式颜色' if is_formal else '单view表面 · raw106原型'}.get(method,method)
            all49=summ.get('all49',{}).get(method,{})
            projective_metric_rows.append(f'<tr><td>{esc(title)}</td><td>{num(m.get("HR_RGB_error"))}</td><td>{num(m.get("HR_normalized_RGB_error"))}</td><td>{num(all49.get("LiDAR_depth_error"))}</td><td>{link(ep,"receipt")}</td></tr>')
    projective_section=('''<section class="section"><h2>单参考view的表面输出试验</h2><p class="section-intro">Projective surface由单参考view的sphere alpha/depth重采样为小球surfels，再转入a6-local随轨迹作统一native深度排序。它是view-conditioned surface output test；点集与PVD共享点对照不同，不能据此声称360°补全或独立几何真值。</p><div class="notice warn">单参考view的反面、遮挡后表面及新视角一致性仍受该参考视图限制。生成颜色仍为材质proposal；本试验不自动替换主表示。</div><div class="table-wrap"><table><thead><tr><th>实际结果文件</th><th>状态</th><th>证据</th></tr></thead><tbody>'''+(projective_rows or empty(3))+'''</tbody></table></div></section>''') if projective_files else ''
    if projective_metric_rows:
        projective_table='<div class="table-wrap"><table><thead><tr><th>表示 / 颜色协议</th><th>9 HELD HR black-RGB MAE</th><th>9 HELD HR normalized-RGB MAE</th><th>all49 LiDAR depth MAE / m</th><th>来源</th></tr></thead><tbody>'+''.join(projective_metric_rows)+'</tbody></table></div><p class="caption">原型以raw106着色；正式版仅增加同一40BUILD真实RGB域的32-step smoke与1000-step颜色拟合，保持原型geometry/opacity与未知列先验颜色，无新生成色1000步。纹理清晰度与测深准确性分别判断。</p>'
        projective_section=projective_section.replace('<div class="table-wrap">',projective_table+'<div class="table-wrap">',1)

    evidence_rows=''.join(f'<li>{link(p,p.relative_to(dest).as_posix())}</li>' for p in sorted(files) if p.suffix=='.json' and (p.parent.name=='evidence' or p.name in ('receipt.json','prepare_receipt.json','summary.json','metrics49.json','held_hr.json','lidar49.json','hosted_generation.json')))
    code_rows=''.join(f'<li>{link(p,p.relative_to(dest).as_posix())}</li>' for p in sorted(files) if p.suffix=='.py' and ('code' in p.relative_to(dest).parts or 'source' in p.relative_to(dest).parts))
    manifest=load(dest/'evidence/sd_batch_manifest.json') or {}
    prompts=list(dict.fromkeys(j['prompt'] for j in manifest.get('jobs',[]) if 'prompt' in j))
    prompt_html=''.join(f'<div class="prompt"><b>SD prompt {i+1}</b><p>{esc(p)}</p></div>' for i,p in enumerate(prompts)) or '<p class="muted">提示词manifest尚未取回。</p>'
    hosted_path=find('hosted_generation/hosted_generation.json')
    hosted=load(hosted_path) if hosted_path else {}
    prompt_html+=''.join(f'<div class="prompt"><b>Hosted call {esc(c.get("version",""))}</b><p>{esc(c.get("prompt",""))}</p></div>' for c in (hosted or {}).get('calls',[]))

    svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="250" viewBox="0 0 1280 250"><defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#49697b"/></marker></defs><rect width="1280" height="250" rx="16" fill="#f2f7f9"/><g font-family="Microsoft YaHei,Segoe UI,Arial,sans-serif" fill="#183a4b"><g stroke="#b9ced8" stroke-width="2"><rect x="20" y="45" width="215" height="105" rx="12" fill="white"/><rect x="265" y="45" width="215" height="105" rx="12" fill="white"/><rect x="510" y="45" width="225" height="105" rx="12" fill="#e3f2ed"/><rect x="765" y="45" width="220" height="105" rx="12" fill="white"/><rect x="1015" y="45" width="245" height="105" rx="12" fill="#e6eef8"/></g><g text-anchor="middle" font-size="20" font-weight="600"><text x="127" y="79">BUILD真实可见RGB</text><text x="372" y="79">同pose图 + mask</text><text x="622" y="79">冻结外观先验</text><text x="875" y="79">共享资产材质拟合</text><text x="1137" y="79">真实3D渲染与验收</text></g><g text-anchor="middle" font-size="16"><text x="127" y="113">a6自身身份参考</text><text x="372" y="113">512候选坐标系</text><text x="622" y="108">SD1.5 / hosted ImageGen</text><text x="622" y="132">输出为2D material proposal</text><text x="875" y="108">仅未知贡献列SH0</text><text x="875" y="132">锁定已观测SH与来源</text><text x="1137" y="108">同一资产 / 49帧</text><text x="1137" y="132">Observed / Thin / Sphere</text></g></g><g stroke="#49697b" stroke-width="2.5" fill="none" marker-end="url(#a)"><path d="M238 97H258"/><path d="M483 97H503"/><path d="M738 97H758"/><path d="M988 97H1008"/><path d="M127 152V192H875V157"/></g><g font-family="Microsoft YaHei,Segoe UI,Arial,sans-serif" font-size="16" fill="#486777"><text x="145" y="184">冻结PVD点、位置与来源；Sphere对照仅先改thickness=sigma</text><text x="26" y="228">1920射线 → pixel-centre affine → 512候选采样；A = W / alpha；生成颜色始终标为推断</text></g></svg>'''
    svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="340" viewBox="0 0 1280 340"><defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#49697b"/></marker></defs><rect width="1280" height="340" rx="16" fill="#f2f7f9"/><g font-family="Microsoft YaHei,Segoe UI,Arial,sans-serif" fill="#183a4b"><g stroke="#b9ced8" stroke-width="2"><rect x="25" y="30" width="230" height="110" rx="12" fill="white"/><rect x="300" y="30" width="280" height="110" rx="12" fill="white"/><rect x="650" y="30" width="245" height="110" rx="12" fill="#e6eef8"/><rect x="950" y="30" width="305" height="110" rx="12" fill="#e3f2ed"/><rect x="25" y="205" width="230" height="105" rx="12" fill="white"/><rect x="300" y="205" width="280" height="105" rx="12" fill="#e3f2ed"/><rect x="650" y="205" width="245" height="105" rx="12" fill="white"/></g><g text-anchor="middle" font-size="21" font-weight="600"><text x="140" y="67">冻结PVD完整几何</text><text x="440" y="67">输出表面 / 材质表示</text><text x="772" y="67">单一持久a6资产</text><text x="1102" y="67">原生3D联合渲染</text><text x="140" y="244">真实BUILD可见RGB</text><text x="440" y="244">冻结图像先验</text><text x="772" y="244">逐资产颜色拟合</text></g><g text-anchor="middle" font-size="16"><text x="140" y="101">点 / 来源 / 先验深度</text><text x="440" y="96">Thin / Sphere核</text><text x="440" y="121">或单参考view表面采样</text><text x="772" y="98">a6-local坐标</text><text x="772" y="122">随同一轨迹运动</text><text x="1102" y="98">实际相机 / 49帧 / 保留或删a37</text><text x="1102" y="122">对象、街景与固定域验收</text><text x="140" y="279">a6自身身份与颜色证据</text><text x="440" y="275">SD1.5 / hosted ImageGen</text><text x="440" y="298">图像仅为2D外观proposal</text><text x="772" y="276">真实域与生成域按各自协议</text><text x="772" y="298">保持该表示的geometry / opacity</text></g></g><g stroke="#49697b" stroke-width="2.5" fill="none" marker-end="url(#a)"><path d="M260 84H293"/><path d="M585 84H643"/><path d="M900 84H943"/><path d="M260 258H293"/><path d="M585 258H643"/><path d="M772 200V147"/><path d="M440 145V174H772V198"/></g><text x="949" y="250" font-family="Microsoft YaHei,Segoe UI,Arial,sans-serif" font-size="16" fill="#486777">单参考表面不是360°几何真值</text><text x="949" y="277" font-family="Microsoft YaHei,Segoe UI,Arial,sans-serif" font-size="16" fill="#486777">不把2D候选直接贴到最终画面</text></svg>'''
    (dest/'architecture.svg').write_text(svg,encoding='utf-8')

    template='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene008 · a6车体外观补全</title><style>
:root{--ink:#173242;--muted:#5b7280;--line:#dce6eb;--bg:#eff4f7;--card:#fff;--accent:#176b65;--soft:#e7f2ef}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:var(--bg);font:16px/1.65 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}a{color:#126f87;text-decoration:none}a:hover{text-decoration:underline}header,main,footer{max-width:1480px;margin:auto;padding:26px 34px}header{padding-top:40px;padding-bottom:12px}.eyebrow{font-size:12px;letter-spacing:2px;text-transform:uppercase;color:var(--accent);font-weight:700}h1{font-size:clamp(28px,3.3vw,45px);line-height:1.22;margin:9px 0 15px;letter-spacing:-1px}h2{font-size:25px;line-height:1.3;margin:0 0 8px}h3{font-size:18px;margin:14px 0 6px}p{margin:8px 0 14px}.lead{max-width:960px;color:var(--muted);font-size:17px}.tags{display:flex;gap:8px;flex-wrap:wrap}.tag{font-size:12px;background:#e1ecef;border-radius:99px;padding:4px 10px;font-weight:600}.tag.green{background:var(--soft);color:#176b50}nav{display:flex;gap:20px;margin-top:22px;flex-wrap:wrap;font-size:14px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0 0}.stat{background:#fff;border:1px solid var(--line);border-radius:14px;padding:17px 20px}.stat b{font-size:29px;display:block;line-height:1.3}.stat span{color:var(--muted);font-size:12px}.section{margin-bottom:24px;padding:25px;background:var(--card);border:1px solid var(--line);border-radius:18px;box-shadow:0 5px 20px #17324204}.section-intro{color:var(--muted);max-width:1100px;margin-bottom:18px}.architecture{width:100%;display:block}.gallery-controls{display:flex;gap:12px;align-items:center;padding:14px 0}.gallery-controls input{flex:1;accent-color:var(--accent)}button{background:#edf4f4;color:var(--ink);border:1px solid #cbdcde;border-radius:8px;padding:8px 13px;font:inherit;cursor:pointer}button:hover{background:#dcebea}.raw-label{min-width:135px;font-size:14px;font-weight:700}.viewer{background:#090c10;border-radius:12px;overflow:hidden;border:1px solid #20333e}.viewer img{display:block;width:100%;height:auto}.viewer[hidden],.missing[hidden]{display:none}.missing{padding:45px 20px;background:#f3f6f8;color:var(--muted);text-align:center;border:1px dashed #c7d8e0;border-radius:10px}.column-labels{display:grid;grid-template-columns:repeat(5,1fr);gap:2px;font-size:12px;font-weight:700;text-align:center;margin:0 0 7px}.column-labels span{padding:7px 4px;background:#edf3f6;border-radius:6px}.caption,.muted{font-size:13px;color:var(--muted)}.caption{margin-top:8px}.notice{border-left:4px solid #3b8a80;background:#f1f8f5;padding:12px 16px;margin:16px 0;font-size:14px}.warn{border-left-color:#bf9343;background:#fbf7ee}.fig-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.fig-grid figure{margin:0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#f7fafb}figure{margin:16px 0}figure img{display:block;max-width:100%;width:100%;height:auto;background:#090c10}figcaption{padding:13px 15px}figcaption p{font-size:13px;color:var(--muted);margin:5px 0 0}.fig-grid img{aspect-ratio:1;object-fit:contain}.wide img{max-height:850px;object-fit:contain}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;margin:15px 0;font-size:14px}th{text-align:left;background:#eff5f6;font-size:12px;color:#496876}th,td{padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top}td small{display:block;font-size:11px;color:var(--muted)}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}details{border-top:1px solid var(--line);padding-top:14px;margin-top:18px}summary{cursor:pointer;font-weight:650}ul{padding-left:20px}li{margin:8px 0;overflow-wrap:anywhere}.downloads{max-height:430px;overflow:auto;font-size:13px}.prompt{padding:13px 16px;background:#f2f6f8;margin:12px 0;border-radius:9px;font-size:13px}.prompt p{margin-bottom:0}code{font:12px/1.6 Consolas,monospace;background:#edf2f5;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere}footer{color:var(--muted);font-size:12px;padding-top:0}@media(max-width:850px){header,main,footer{padding-left:16px;padding-right:16px}.section{padding:18px}.stats{grid-template-columns:1fr 1fr}.fig-grid{grid-template-columns:1fr}.two{grid-template-columns:1fr}.column-labels{font-size:10px}.gallery-controls{gap:6px}.raw-label{min-width:95px}.gallery-controls button{padding:6px 8px}nav{gap:12px}}@media print{body{background:white}.section{break-inside:avoid;box-shadow:none}.gallery-controls,nav,button{display:none}header,main{padding:12px}.stats{grid-template-columns:repeat(4,1fr)}a{color:inherit}}
</style></head><body><header><div class="eyebrow">Scene008 / actor a6 / 2026-11-05</div><h1>车体外观补全与输出核诊断</h1><p class="lead">从a6自身的真实可见上半部出发，比较冻结外观先验接回共享3D资产的结果。PVD与预训练模型权重保持冻结；本页将2D候选、材质拟合、真实3D渲染和核形状对照分别呈现。</p><div class="tags"><span class="tag green">PVD / 预训练权重更新 0</span><span class="tag">BUILD选择 · HELD验收</span><span class="tag">未见下半部无真实RGB标签</span><span class="tag">Sphere为对照，未自动采用</span></div><nav><a href="#renders">49帧真实渲染</a><a href="#candidates">2D候选</a><a href="#kernel">核输出对照</a><a href="#evidence">证据与复现</a></nav><div class="stats"><div class="stat"><b>{{FRAMES}} / 49</b><span>已取回完整对象/街景帧对</span></div><div class="stat"><b>{{SDCOUNT}} + {{SMOKECOUNT}}</b><span>SD正式候选 + 单列smoke</span></div><div class="stat"><b>{{FITS}}</b><span>receipt确认的生成色拟合分支</span></div><div class="stat"><b>{{WEIGHTS}}</b><span>预训练神经网络权重训练更新</span></div></div></header><main>
<section class="section"><h2>输入、外观先验与资产数据流</h2><p class="section-intro">生成图提供未知表面的材质建议；真正的输出仍由同一套3D点与相机渲染。Sphere对照在颜色拟合前仅令厚度等于sigma。</p><img class="architecture" src="architecture.svg" alt="BUILD RGB与冻结PVD经同pose输入、冻结外观先验、未知列材质拟合后进入真实3D渲染的架构组件图"></section>
<section class="section" id="renders"><h2>49帧真实3D渲染</h2><p class="section-intro">主列共享同一原始1920相机和对象裁剪。Observed仅显示真实可见RGB；黑色未知区域不是GT。拖动查看raw089–137；分割标签用于验收展示，不用于重新挑选候选。</p><div class="gallery-controls"><button id="prev" aria-label="上一帧">←</button><input id="frame" type="range" min="89" max="137" step="1" value="{{INITIAL}}" aria-label="原始帧"><button id="next" aria-label="下一帧">→</button><button id="play">播放</button><span class="raw-label" id="rawLabel"></span></div><div class="column-labels"><span>Observed</span><span>M2</span><span>HR observed</span><span>Thin + strong</span><span>Sphere + strong</span></div><div class="viewer" id="objectViewer"><img id="objects" alt="同帧五列真实3D对象渲染"></div><div class="missing" id="objectMissing">该帧真实对象渲染尚未取回。</div><p class="caption" id="objectCaption"></p><h3>同一资产放回街景</h3><div class="viewer" id="streetViewer"><img id="street" alt="同一资产保留与删除a37的原生街景渲染"></div><div class="missing" id="streetMissing">该帧街景渲染尚未取回。</div><p class="caption">街景对照来自原生3D场景渲染：a37保留 / a37删除。道路与环境保持冻结；对象图与街景使用的分辨率以final_gallery receipt为准。</p><div class="notice">外观变化只代表推断的材质补全。更完整的2D形状或更不透明的输出核，不能单独证明隐藏表面正确或达到参考图质量。</div>{{FALLBACK}}</section>
<section class="section" id="candidates"><h2>2D外观候选与冻结选择</h2><p class="section-intro">此处为外观先验的二维输出，尚不能代表完整3D结果。离线SD固定两条prompt × seeds 0–3；根据BUILD形状可用性冻结prompt2_seed3。Hosted strong共两次调用，下面展示v2材质proposal。</p><div class="fig-grid">{{CONDITIONING}}{{SELECTED}}{{STRONG}}</div><details open><summary>固定8个SD候选对照板</summary>{{BOARD}}</details><div class="notice warn">SD composed逐像素恢复有效mask外输入。Strong原始proposal会改变可见像素；接回资产时仅采未知前景射线，并锁住已观测贡献列的SH。两者均没有未见下半部的实测RGB监督。</div></section>
<section class="section" id="kernel"><h2>输出核形状与颜色、深度的折中</h2><p class="section-intro">Thin与Sphere保持同一XYZ、sigma、opacity及来源；Sphere仅令thickness=sigma，然后在完全相同的真实BUILD像素域重新拟合颜色。核形状改变会影响覆盖、alpha及期望深度。本报告不自动采用Sphere。</p>{{KERNEL_PREVIEW}}<div class="table-wrap"><table><thead><tr><th>观测颜色控制</th><th>9 HELD HR black-RGB MAE</th><th>9 HELD HR normalized-RGB MAE</th><th>49帧LiDAR depth MAE / m</th><th>49帧LiDAR alpha</th></tr></thead><tbody>{{KERNELROWS}}</tbody></table></div><p class="caption">HR颜色指标在固定的原始可见域评估，未据alpha改变分母；LiDAR为原始固定实测深度域。这里的HELD仍继承同场景几何上下文，不代表独立几何泛化。{{KERNEL_LINKS}}</p></section>
<section class="section"><h2>实际summary与拟合记录</h2><p class="section-intro">下列值直接从已取回的summary/receipt读取。可见区域指标不能评价从未观测的下半部真实颜色；缺失数据以“—”保留。</p><div class="table-wrap"><table><thead><tr><th>分支</th><th>all49可见a6 RGB MAE</th><th>all49实测depth MAE / m</th><th>保留a37区域RGB变化</th><th>来源</th></tr></thead><tbody>{{METRICROWS}}</tbody></table></div><div class="table-wrap"><table><thead><tr><th>分支</th><th>优化对象</th><th>正式颜色步</th><th>更新primitive列</th><th>神经权重训练步</th><th>证据</th></tr></thead><tbody>{{FITROWS}}</tbody></table></div><p class="caption">CPU/CUDA的逐资产颜色迭代不等于预训练神经权重训练。Smoke、失败尝试和next-step重放独立记入ledger；hosted模型未暴露权重hash，也未建立确定性重放。</p></section>
<section class="section" id="evidence"><h2>可下载证据与复现入口</h2><div class="two"><div><h3>审查与结果JSON</h3><ul class="downloads">{{EVIDENCE}}</ul></div><div><h3>实际脚本</h3><ul class="downloads">{{CODE}}</ul><p class="caption">远端run：<code>/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105</code></p><p><a href="evidence_and_code.zip" download>下载JSON证据与代码包</a> · <a href="file_manifest.json" download>文件SHA256清单</a></p></div></div><details><summary>固定SD prompts与参数路径</summary>{{PROMPTS}}<p class="caption">CFG、步数、seed、IP-Adapter及本地checkpoint hash见sd_model_manifest与sd_batch_manifest。Hosted输出以返回图像文件hash追踪，不将其误称为模型权重hash。</p></details></section>
</main><footer>本地静态报告 · {{TIMESTAMP}} · 证据快照：{{STATUS}}。显示的Sphere是实验对照；最终结论与采用决定由结果审查确认。</footer><script>
const galleries={{GALLERYSETS}};const chooser=document.querySelector('#gallerySet');chooser.value={{ACTIVEGALLERY}};const slider=document.querySelector('#frame');let timer=null;
function showImage(id,viewer,missing,src){const image=document.getElementById(id);document.getElementById(viewer).hidden=!src;document.getElementById(missing).hidden=!!src;if(src){image.onerror=()=>{document.getElementById(viewer).hidden=true;document.getElementById(missing).hidden=false};image.src=src;}}
function update(){const set=galleries[chooser.value],raw=Number(slider.value),d=set.frames[String(raw)];const labels=document.querySelector('#columnLabels');labels.style.setProperty('--column-count',set.columns.length);labels.replaceChildren(...set.columns.map(text=>{const e=document.createElement('span');e.textContent=text;return e}));document.querySelector('#rawLabel').textContent=`raw${String(raw).padStart(3,'0')} · ${d.split}`;showImage('objects','objectViewer','objectMissing',d.objects);showImage('street','streetViewer','streetMissing',d.street);document.querySelector('#objectCaption').textContent=`${chooser.value} · 原始帧${raw} / ${d.split} · ${set.columns.length}列共享裁剪 · 已取回完整帧对${set.complete}/49`;document.querySelector('#streetContext').textContent=`本图库街景使用${set.street_asset_context}，同一资产保留/删除a37；与本套对象图和receipt保持一致。`;}
chooser.addEventListener('change',update);slider.addEventListener('input',update);document.querySelector('#prev').onclick=()=>{slider.value=Math.max(89,Number(slider.value)-1);update()};document.querySelector('#next').onclick=()=>{slider.value=Math.min(137,Number(slider.value)+1);update()};document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.querySelector('#play').textContent='播放';}else{timer=setInterval(()=>{slider.value=Number(slider.value)>=137?89:Number(slider.value)+1;update()},600);document.querySelector('#play').textContent='暂停';}};update();
</script></body></html>'''
    template=template.replace('grid-template-columns:repeat(5,1fr)','grid-template-columns:repeat(var(--column-count),1fr)')
    template=template.replace('Scene008 · a6车体外观补全','冻结PVD的车体外观补全：正式对照')
    template=template.replace('车体外观补全与输出核诊断','冻结PVD的车体外观补全：正式对照')
    old_lead='从a6自身的真实可见上半部出发，比较冻结外观先验接回共享3D资产的结果。PVD与预训练模型权重保持冻结；本页将2D候选、材质拟合、真实3D渲染和核形状对照分别呈现。'
    template=template.replace(old_lead,'完整a6资产已呈现牌照槽、保险杠与轮胎等车体细节。本页比较二维外观候选如何进入实际3D资产，并在同相机、多帧与街景中检查最终效果。PVD与预训练模型权重保持冻结。')
    fixed_labels='<div class="column-labels"><span>Observed</span><span>M2</span><span>HR observed</span><span>Thin + strong</span><span>Sphere + strong</span></div>'
    template=template.replace(fixed_labels,f'<div id="columnLabels" class="column-labels" style="--column-count:{len(columns)}">{{{{COLUMNS}}}}</div>')
    template=template.replace('<div class="gallery-controls">','<label class="caption">独立图库 <select id="gallerySet" style="margin:8px;padding:7px;border:1px solid #cbdcde;border-radius:7px">{{GALLERYOPTIONS}}</select></label><div class="gallery-controls">',1)
    template=template.replace('<h3>同一资产放回街景</h3>','<h3>同一资产放回街景</h3><p id="streetContext" class="caption"></p>')
    template=template.replace('五列','并列')
    template=template.replace('真正的输出仍由同一套3D点与相机渲染。Sphere对照在颜色拟合前仅令厚度等于sigma。','主资产对照由共享PVD点与相机渲染；单参考表面试验另列。Sphere对照在颜色拟合前仅令厚度等于sigma。')
    template=template.replace('<section class="section"><h2>实际summary与拟合记录</h2>','{{PROJECTIVE}}<section class="section"><h2>实际summary与拟合记录</h2>')
    architecture_match=re.search(r'<section class="section"><h2>输入、外观先验与资产数据流</h2>.*?</section>',template,re.S)
    if architecture_match:
        architecture_block=architecture_match.group(0)
        template=template.replace(architecture_block,'',1)
        template=template.replace('{{FALLBACK}}</section>','{{FALLBACK}}</section>'+architecture_block,1)
    if '<div class="stats">' in template:
        prefix,rest=template.split('<div class="stats">',1)
        cards,suffix=rest.split('</div></header>',1)
        stats='<div class="stats">'+cards+'</div>'
        template=prefix+'</header>'+suffix
        template=template.replace('<h2>可下载证据与复现入口</h2>','<h2>可下载证据与复现入口</h2>'+stats,1)
    fallback_html=figure(preview,'已取回的真实渲染预览 · raw113','来源为现有HR观测颜色控制预览，最终并列图库取回后以上方slider为准。','wide') if available==0 and preview else ''
    values={
        'FRAMES':str(available),'SDCOUNT':num(ledger.get('offline_SD15_inference',{}).get('formal_candidates')),
        'SMOKECOUNT':num(ledger.get('offline_SD15_inference',{}).get('smoke_candidates_separate')),
        'FITS':num(totals.get('completed_generated_asset_fit_branches')),
        'WEIGHTS':num(totals.get('all_pretrained_neural_weight_updates')),
        'INITIAL':str(initial),'FALLBACK':fallback_html,
        'COLUMNS':column_labels,'PROJECTIVE':projective_section,
        'CONDITIONING':figure(conditioning,'BUILD106 · 真实可见上半部','灰色区域为未知placeholder；黑底为对象外，未作为生成目标。'),
        'SELECTED':figure(selected,'冻结SD候选 · prompt2_seed3','仅凭BUILD形状可用性选择；候选外像素精确恢复。'),
        'STRONG':figure(strong,'Strong v2 · 2D材质proposal','Hosted生成；颜色与隐藏细节均为推断。'),
        'BOARD':figure(board,'SD固定8候选','两条prompt × 四个seed；未据HELD改选。','wide'),
        'KERNEL_PREVIEW':figure(kernel_preview,'同色材质的核输出诊断 · raw106','此图是拟合前厚度变换对照，不代表自动采用Sphere。','wide') if kernel_preview else '',
        'KERNELROWS':''.join(kernelrows) or empty(5),
        'KERNEL_LINKS':' '.join([link(held_path,'HR固定域JSON'),link(lidar_path,'LiDAR固定域JSON')]),
        'METRICROWS':''.join(metricrows) or empty(5),'FITROWS':''.join(fitrows) or empty(6),
        'EVIDENCE':evidence_rows or '<li>结果文件尚未取回。</li>','CODE':code_rows,
        'PROMPTS':prompt_html,'TIMESTAMP':datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        'STATUS':'49帧完整图库已取回' if available==49 else '布局与已取回证据；图库取回后重新运行发布器',
        'GALLERYSETS':json.dumps(gallery_sets,ensure_ascii=False).replace('</','<\\/'),
        'ACTIVEGALLERY':json.dumps(active_gallery),'GALLERYOPTIONS':gallery_options,
    }
    for key,value in values.items():template=template.replace('{{'+key+'}}',value)
    if '{{' in template:raise RuntimeError('Unresolved report template marker')
    (dest/'index.html').write_text(template,encoding='utf-8')
    with zipfile.ZipFile(dest/'evidence_and_code.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file() and (p.suffix in ('.json','.py')) and p.name not in ('file_manifest.json','publication_receipt.json'):
                z.write(p,p.relative_to(dest).as_posix())
    published=[p for p in dest.rglob('*') if p.is_file() and p.name not in ('file_manifest.json','publication_receipt.json')]
    manifest={'files':[{'path':p.relative_to(dest).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(published)]}
    (dest/'file_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    receipt={'status':'REPORT_PUBLISHED_WITH_AVAILABLE_EVIDENCE','input':str(source),'output':str(dest),
             'gallery_complete_frame_pairs':available,'expected_gallery_frame_pairs':49,
             'gallery_columns':columns,'gallery_column_count':len(columns),
             'active_gallery':active_gallery,
             'independent_galleries':{k:{'columns':v['columns'],'complete_pairs':v['complete'],'receipt':v['receipt']} for k,v in gallery_sets.items()},
             'missing_gallery_raws':[int(k) for k,v in gallery.items() if not(v['objects'] and v['street'])],
             'real_summary_rows':len(metricrows),'real_colour_fit_receipt_rows':len(fitrows),
             'sphere_adoption_claimed':False,'reference_quality_attainment_claimed':False,
             'generated_images_classified_as_material_proposals':True,
             'source_publisher_sha256':sha(Path(__file__)),
             'evidence_ledger_sha256':sha(dest/'evidence/experiment_ledger.json') if (dest/'evidence/experiment_ledger.json').is_file() else None,
             'files':len(published),'network_calls':0,'training_updates':0}
    (dest/'publication_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
