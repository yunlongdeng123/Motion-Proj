"""同步本轮源码/QA，不重新创建或重压缩任何已生成科学证据。"""
import json
import base64
import hashlib
from pathlib import Path
import shutil
from verify import read_js
from build_evidence import js
ROOT=Path(__file__).resolve().parent;OUTPUTS=ROOT.parents[1]/'outputs'
for name in ('GS_Reconstruction_Diagnostics_20261021_Evidence_r1','GS_Reconstruction_Diagnostics_20261021_Evidence_portable_r1'):
    out=OUTPUTS/name
    for source in (ROOT/'src').glob('*'):
        if not source.is_file():continue
        shutil.copy2(source,out/'source/viewer'/source.name)
        if source.name in ('app.js','gpu.js','evidence.js','server.py'):shutil.copy2(source,out/source.name)
    if (out/'portable_receipt.json').exists():
        p=out/'app.js';p.write_text(p.read_text(encoding='utf-8').replace('数值预览alpha 1/65535、深度1cm量化','便携热图4×4采样／8bit；指标来自原始浮点'),encoding='utf-8',newline='\n')
    for folder in ('gpu','tests'):
        shutil.copytree(ROOT/folder,out/'source'/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for filename in ('build_evidence.py','verify_evidence.py','finalize_evidence.py'):
        shutil.copy2(ROOT/filename,out/'source/authoring'/filename)
    for filename in ('browser_evidence_qa.json','evidence_original_asset_audit.json','live_cf_smoke.json'):
        if (ROOT/filename).exists():shutil.copy2(ROOT/filename,out/filename)
    if (ROOT/'geometry_selected_lidar_r1.json').exists():
        refreshed=json.loads((ROOT/'geometry_selected_lidar_r1.json').read_text());index=read_js(out/'data/gpu_index.js')
        assert len(refreshed)==len(index['queries'])==180
        for key,path in index['queries'].items():
            value=read_js(out/path);addition=refreshed[key];value['geometry_evidence']=addition['geometry_evidence']
            for row in value['top']:row['geometry_evidence']=addition['by_id'][row['id']]
            js(out/path,key,value)
        p=out/'EVIDENCE_REPORT.html';text=p.read_text(encoding='utf-8')
        if '所选像素本身的实测LiDAR必须' not in text:
            text=text.replace('实际贡献的完整3σ区间','所选像素本身的实测LiDAR必须与道路z带一致，不能以邻近曝光锚点替代。实际贡献的完整3σ区间')
        p.write_text(text,encoding='utf-8',newline='\n')
    if (ROOT/'live_cf_smoke.json').exists():
        live=json.loads((ROOT/'live_cf_smoke.json').read_text());live['temporal_controls']=[]
        assert live['arm']=='full' and live['pixel']==[171,270] and len(live['removed'])==3
        live['selection_kind']='geometry_conflict_candidates';live['selection_rule']='all3 current-ray conservative geometry-conflict candidates; selected manually then full native rerasterization; NOT automatic repair'
        for k,v in live['images'].items():
            blob=base64.b64decode(v.split(',',1)[1]);filename=hashlib.sha256(blob).hexdigest()+'.webp';path=out/'assets/gpu'/filename
            if not path.exists():path.write_bytes(blob)
            live['images'][k]='assets/gpu/'+filename
        key='cf/full/102/171/270';path='data/counterfactual/full_102_171_270.js';js(out/path,key,live)
        index=read_js(out/'data/gpu_index.js');index['counterfactuals'][key]=path;js(out/'data/gpu_index.js','gpu_index',index)
        p=out/'EVIDENCE_REPORT.html';text=p.read_text(encoding='utf-8');marker='<section id="live-smoke"'
        if marker not in text:
            m=live['scientific_evaluation'];paragraph=f'<section id="live-smoke" class="panel"><h2>额外真实浏览器干预：标签不保证收益</h2><p>在full/raw102/(171,270)，手动筛选三个几何冲突候选并完整重渲染。218个验证世界单元ΔL={m["valid"]["delta_L_valid"]:.6f}，正常175单元中3个RGB误伤；深度MAE约0.214905→0.209175m，RGB却略变差。可见“几何代理冲突”不是保证删除收益的判决，也不能以单一深度改善覆盖颜色误伤。此真实第13次干预离线可查，另36个跨帧控制不变。原模型未改，human_verdict=null。</p></section>'
            text=text.replace('</main>',paragraph+'</main>');p.write_text(text,encoding='utf-8',newline='\n')
    contract=json.loads((out/'data/data_contract.json').read_text())
    contract['evidence_r1']=dict(geometry_labels='conditional covariance + measurement policy; current ray only',
        fixed_world_valid_denominator=True,normal_control=True,V1_native30K_readonly=True,
        V1_recovered_history_not_fresh=True,no_hidden_RGB_GT=True,optimizer_updates=0,human_verdict=None)
    (out/'data/data_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8',newline='\n')
print('synchronized current source, evidence contract and available QA')
