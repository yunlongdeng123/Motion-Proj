"""Restore accepted three-frame R2 as explicit reference; archive legacy diagnostics.

No GPU execution, no image modification, no new result or ownership inference.
The 49-frame empty-initialization experiment remains accessible as an ablation.
"""
from pathlib import Path
import argparse, base64, hashlib, json, re, shutil, subprocess

LOCAL_ROOT = Path('C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a')
BASE = LOCAL_ROOT if (LOCAL_ROOT / 'outputs/GS_Scene_Diagnostics_20261027').is_dir() else Path.cwd()
WORK = Path(__file__).parent
OUT = BASE / 'outputs/GS_Scene_Diagnostics_20261027'
C1 = '14705303724557273004_3105_000_3125_000'
R2 = BASE / 'outputs/V43_Ownership_20261027/gpu_v43/bounded_white_v1'
GID = 'v43_r2_reference'
LABEL = '非R2消融 · 空集合初始化49帧（未采纳）'

def required_inputs(workspace_root):
    base=Path(workspace_root).resolve();out=base/'outputs/GS_Scene_Diagnostics_20261027'
    r2=base/'outputs/V43_Ownership_20261027/gpu_v43/bounded_white_v1'
    return [out/'index.html',out/'viewer.js',out/'scene_target_detection.js',out/C1/'index.html',out/C1/'v43_groups_data.json',out/C1/'target_detection_data.js',r2/'index.html',r2/'native_data.js',*(r2/f'native/{raw:03d}/{name}' for raw in (98,113,114) for name in ('white.png','evidence.npz'))]

def can_repair(workspace_root):
    """No fallback to a different scene or fabricated assets when inputs are absent."""
    return all(path.is_file() for path in required_inputs(workspace_root))

def configure(workspace_root):
    global BASE,OUT,R2
    BASE=Path(workspace_root).resolve();OUT=BASE/'outputs/GS_Scene_Diagnostics_20261027';R2=BASE/'outputs/V43_Ownership_20261027/gpu_v43/bounded_white_v1'
    missing=[str(p) for p in required_inputs(BASE) if not p.is_file()]
    if missing:raise FileNotFoundError('R2 repair requires actual existing inputs; nothing changed. Missing: '+', '.join(missing))

def load(path):
    s=path.read_text(encoding='utf8')
    return json.JSONDecoder().raw_decode(s[s.index('=')+1:])[0]

def dump(v): return json.dumps(v,ensure_ascii=False,separators=(',',':'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,s):
    if path.exists() and path.read_text(encoding='utf8')==s: return
    if path.exists():
        relative=path.relative_to(BASE) if path.is_relative_to(BASE) else Path('source_mirrors')/path.relative_to(WORK)
        dest=WORK/'before'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copy2(path,dest)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(s,encoding='utf8')

def block(s,name,value,anchor,before=False):
    whole=f'<!-- {name}-start -->{value}<!-- {name}-end -->'
    pat=rf'<!-- {re.escape(name)}-start -->.*?<!-- {re.escape(name)}-end -->'
    if re.search(pat,s,re.S):return re.sub(pat,lambda _:whole,s,flags=re.S)
    assert anchor in s,(name,anchor)
    return s.replace(anchor,whole+anchor if before else anchor+whole,1)

RUNTIME=r'''(function(root){
  'use strict';const M=root.R2_REFERENCE_DELIVERY,a=root.SceneViewer;if(!M||!a)return;
  const get=id=>document.getElementById(id),valid=[98,113,114],previousInput=get('frame').oninput;
  get('frame').oninput=function(e){if(S.group===M.group_id){S.raw=valid[Number(this.value)];show();}else previousInput.call(this,e);};
  function update(){const active=S.group===M.group_id;if(!active)return;
    const range=get('frame');range.min=0;range.max=2;range.step=1;range.value=Math.max(0,valid.indexOf(S.raw));range.setAttribute('aria-label','已交付R2参考帧：98、113、114');
    get('frameLabel').textContent='raw'+S.raw+' · R2参考 '+(valid.indexOf(S.raw)+1)+'/3（非连续视频）';
    get('geometryStateInfo').textContent='真实R2原static删行几何：existing_removed_ids ∪ new_unknown_ids；Actor为空。二维局部G＋T白底、域外existing；三维中心不是全局像素保护。';
    get('semanticFilterStatus').textContent='本组仅三帧原始R2参考；不使用空初始化49帧结果，不改owner。'+(root.R2_INHERITED_DELIVERY?.validated?'另有真实R2继承49帧组可切换。':'继续全片段R2尚未交付。');
  }
  const beforeShow=show;show=async function(...args){const result=beforeShow(...args);update();await result;update();};
  const beforeDraw=draw3D;draw3D=function(){beforeDraw();update();};
  Object.assign(a,{show,draw3D});root.R2Reference={manifest:M,update};
  const params=new URLSearchParams(location.hash.replace(/^#workspace[?&]?/,''));
  if(!params.has('group')&&!root.R2_INHERITED_DELIVERY?.validated){S.raw=113;S.channel='rgb';S.cameraView=true;S.cameraReference='current';get('geometryState').value='current';get('semanticAblation').value='all';get('cloudColor').value='branch';setCategory('v43',M.group_id);show();}
  else update();
})(window);
'''

def reference_registration():
    import numpy as np
    data=load(R2/'native_data.js');frames={};geometry={};scene=OUT/C1
    source=json.loads((scene/'v43_groups_data.json').read_text(encoding='utf8'))['geometry']['v43_actor_10000']
    for raw in ('98','113','114'):
        f=data['frames'][raw];n=np.load(R2/f['evidence_path'],allow_pickle=False)
        removed=np.union1d(n['existing_removed_ids'],n['new_unknown_ids']).astype(int).tolist()
        assert removed and min(removed)>=0 and max(removed)<918850
        q=json.loads(json.dumps(source[raw]));target=next(o for o in q['objects'] if o.get('key')=='target')
        q.update(objects=[],static_visible=True,static_suppressed_rows=removed)
        for k in ('static_adapter','semantic_render_filter','target_detection'):q.pop(k,None)
        q['reference_boxes']=[dict(target=True,label='R2原目标参考框',corners_local=target['corners_local'])]
        q['note']='原始三帧R2参考；二维局部白底合成，三维真实行ID筛除。'
        q['r2_reference']=dict(source_raw=int(raw),removed_rows=len(removed),row_namespace='original_static',original_receipt=f['receipt_path'])
        geometry[raw]=q
        rel=f'assets/v43_r2_reference/{int(raw):03d}.png';image=R2/f['images']['white'];dest=scene/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(image,dest)
        assert sha(image)==sha(dest)
        write(dest.with_name(dest.name+'.js'),'window.IMAGE_DATA['+dump(rel)+']='+dump('data:image/png;base64,'+base64.b64encode(dest.read_bytes()).decode())+';\n')
        frames[raw]=dict(channels={'rgb':rel},r2_reference_receipt='../../V43_Ownership_20261027/gpu_v43/bounded_white_v1/'+f['receipt_path'])
    # This manifest describes only the preserved reference. Later full runs have
    # their own manifests and must not be represented as missing by this one.
    manifest=dict(schema='accepted-r2-reference-v1',context=C1,group_id=GID,raws=[98,113,114],full_sequence_status='separate_manifest_if_delivered',images_unchanged=True,training_updates=0)
    code='window.R2_REFERENCE_DELIVERY='+dump(manifest)+';\nwindow.SCENE_DATA.groups=window.SCENE_DATA.groups.filter(g=>g.id!=='+dump(GID)+');\nwindow.SCENE_DATA.groups.push('+dump(dict(id=GID,title='真实R2参考 · 已认可三帧（98/113/114）',category='v43',kind='r2_reference',frames=frames)) +');\nwindow.EDIT_GEOMETRY.groups['+dump(GID)+']='+dump(geometry)+';\n'
    write(scene/'r2_reference_data.js',code);write(OUT/'scene_r2_reference.js',RUNTIME)
    return manifest,geometry

def history_runtime(source):
    if '历史三帧检测（98 / 113 / 114）' in source:return source
    source=source.replace("const panel = add('section', { id: 'targetDetection', className: 'panel' });", "const panel = add('details', { id: 'targetDetection', className: 'panel' });\n    panel.append(add('summary', {}, '历史三帧检测（98 / 113 / 114）· 未通过 · 展开查看'));\n    panel.ontoggle = () => { if (panel.open) evidenceFor(S.raw).catch(() => null).then(() => drawPanels(S.raw, S.epoch)); };")
    source=source.replace("'目标专属检测 · 原图 / 阴影候选 / 同帧原生剥离'", "'历史检测对照 · 仅3帧已执行，非R2、非当前逐帧队列'")
    source=source.replace("'冻结检查点的检测诊断。原static语义", "'此历史实验仅执行raw98/113/114，其余帧未执行且没有后台排队。当前R2参考请使用上方入口。原static语义")
    old="for (const raw of Object.keys(manifest.frames).map(Number).sort((a, b) => a - b)) rawSelect.append(add('option', { value: String(raw) }, 'raw' + raw + (manifest.frames[String(raw)].evidence_path ? '' : ' · pending')));"
    new="for (const raw of Object.keys(manifest.frames).map(Number).filter(raw => manifest.frames[String(raw)].evidence_path).sort((a,b)=>a-b)) rawSelect.append(add('option', { value: String(raw) }, 'raw' + raw));\n    rawSelect.append(add('option', { id: 'targetDetectionMissingRaw', value: '', disabled: true, hidden: true }, '当前帧未执行历史对照'));"
    assert old in source;source=source.replace(old,new)
    source=source.replace("get('targetDetectionRaw').value = String(raw);", "const missingOption=get('targetDetectionMissingRaw');missingOption.hidden=!!frame?.evidence_path;missingOption.value=String(raw);missingOption.textContent='raw'+raw+' · 不在历史三帧内（未执行）';\n    get('targetDetectionRaw').value = String(raw);")
    source=source.replace('pending · 此raw尚无检测证据','历史对照未执行：本实验只有raw98/113/114')
    source=source.replace('pending：此raw像素候选证据未导出','历史对照未执行：此raw没有像素候选证据')
    source=source.replace('pending：此raw尚无已核验原生RGB','历史对照未执行：此raw没有该组原生RGB')
    source=source.replace('pending：此raw检测证据尚未可用','历史对照未执行：此raw检测证据不可用')
    source=source.replace("const result = originalShow(...args), raw = S.raw, epoch = S.epoch;\n    await Promise.allSettled", "const result = originalShow(...args), raw = S.raw, epoch = S.epoch;\n    if (detectionCase()) get('targetDetection').open = true;\n    if (!get('targetDetection').open && !isMode(get('cloudColor').value)) return await result;\n    await Promise.allSettled")
    source=source.replace("else {\n    const raw = S.raw, epoch = S.epoch;\n    evidenceFor(raw)", "else if (get('targetDetection').open) {\n    const raw = S.raw, epoch = S.epoch;\n    evidenceFor(raw)")
    return source

def labels(s):
    pairs=[('V43 局部unknown原生诊断 · 49帧',LABEL),('逐帧局部unknown原生诊断 · 49帧独立重算','非R2消融 · 空集合初始化49帧（未采纳）'),('打开本场景49帧真实白底RGB＋删行几何 →','查看非R2消融49帧（未采纳）→'),('新增147帧独立原生诊断：','历史消融147帧已执行（非R2、未采纳）：'),(' · 49帧局部unknown',' · 非R2消融49帧（未采纳）'),('新增：49帧局部unknown诊断 →','历史非R2消融49帧（未采纳）→'),('147帧时序、邻车边界与坏例报告 →','非R2消融147帧报告（未采纳）→'),('车底、阴影语义已登记，暂缓修复','车底与阴影纳入R2迭代；旧记录保留')]
    for a,b in pairs:s=s.replace(a,b)
    return s

def repair_page(s,scene=False,is_c1=False):
    s=labels(s)
    s=re.sub(r'<!-- recall-ceiling-link-start -->.*?<!-- recall-ceiling-link-end -->', '<!-- recall-ceiling-link-start --><details class="notice"><summary>历史召回上界实验（扩域，未采纳）</summary><p>当前约束为蓝框外禁止新增unknown；该历史实验不满足当前范围约束，仅作贡献来源排查记录。</p></details><!-- recall-ceiling-link-end -->',s,flags=re.S)
    s=s.replace('新实验已暂停；三候选仍有残影。','历史Top3候选均未解决车底残影，未采纳；当前以真实R2为迭代基线。')
    if is_c1:
        report='../../V43_Ownership_20261027/gpu_v43/bounded_white_v1/index.html'
        links=' '.join(f'<a class="button" href="#workspace?group={GID}&amp;raw={r}&amp;channel=rgb">真实R2 · raw{r} ↔ 3D</a>' for r in (98,113,114))
        recovery='<a href="../../V43_Ownership_20261027/gpu_v43/r2_recovery_20261029/index.html">R2恢复与回退原因 →</a>' if (BASE/'outputs/V43_Ownership_20261027/gpu_v43/r2_recovery_20261029/index.html').is_file() else ''
        banner=f'<section class="notice" id="r2Baseline"><strong>当前迭代基线：已认可的真实 R2</strong><p>{links}　<a href="{report}">原三帧R2报告 →</a>　{recovery}</p><p>此处恢复原始R2的真实PNG与同帧删行几何，只覆盖98/113/114三帧。<strong>R2全49帧延伸尚未执行，待GPU。</strong> 已跑49帧为空集合初始化消融，未采纳，不能冒充R2延伸。白底表示局部空洞诊断，不是补景。</p></section>'
        s=block(s,'r2-baseline',banner,'<section id="v43Focus"',before=True)
        s=block(s,'r2-reference-data','<script src="r2_reference_data.js?v=r2-reference-20261029-1"></script>','<script src="../viewer.js',before=True)
        s=block(s,'r2-reference-runtime','<script src="../scene_r2_reference.js?v=r2-reference-20261029-1"></script>','</body>',before=True)
    elif not scene:
        banner=f'<section class="notice"><strong>当前迭代基线：真实R2三帧参考</strong><p><a href="{C1}/index.html#workspace?group={GID}&amp;raw=113&amp;channel=rgb">场景1 · 真实R2 RGB与3D →</a>　<a href="../V43_Ownership_20261027/gpu_v43/bounded_white_v1/index.html">原三帧报告 →</a></p><p>R2全49帧延伸尚未执行，待GPU。下列147帧为空初始化消融，非R2、未采纳；历史三帧检测不代表逐帧任务进度。</p></section>'
        s=block(s,'r2-baseline',banner,'<main id="workspace">')
    s=s.replace('目标专属检测、阴影候选 effect 与同帧三组原生剥离','历史三帧检测与原生剥离（非R2）').replace('缺项为pending','仅raw98/113/114，其他帧未执行、未排队')
    return s

def main(workspace_root=None):
    configure(workspace_root or BASE)
    protected={p:sha(p) for p in [R2/'index.html',R2/'native_data.js',*(R2/f'native/{r:03d}/white.png' for r in (98,113,114))]}
    manifest,geometry=reference_registration()
    runtime=history_runtime((OUT/'scene_target_detection.js').read_text(encoding='utf8'))
    runtime_targets=[OUT/'scene_target_detection.js',OUT/C1/'scene_target_detection.js']
    runtime_targets += [p for p in [BASE/'work/persistent_world_20261027/scene_target_detection.js',BASE/'work/v43_unknown_detection_20261028/scene_target_detection.js',WORK/'scene_target_detection.js'] if p.is_file()]
    for p in dict.fromkeys(runtime_targets):write(p,runtime)
    for scene in OUT.iterdir():
        if not scene.is_dir() or not (scene/'scene.json').exists():continue
        page=scene/'index.html';write(page,repair_page(page.read_text(encoding='utf8'),scene=True,is_c1=scene.name==C1))
        p=scene/'bounded_unknown_data.js'
        if p.exists():write(p,labels(p.read_text(encoding='utf8')))
    write(OUT/'index.html',repair_page((OUT/'index.html').read_text(encoding='utf8')))
    # Persist changed generated labels in the original temporal installer.
    installers=[p for p in [BASE/'work/v43_temporal_multiscene_20261029/install_temporal_ui.py',WORK/'install_temporal_ui.py'] if p.is_file()]
    hook="\n    # Reapply R2/history separation only with the actual complete R2 input assets.\n    import runpy\n    for repair in (Path(__file__).with_name('patch_ui.py'), BASE / 'work/v43_r2_regression_20261029/patch_ui.py'):\n        if repair.is_file():\n            namespace = runpy.run_path(str(repair))\n            if namespace['can_repair'](BASE): namespace['main'](BASE)\n            break\n"
    for installer in dict.fromkeys(installers):
        text=labels(installer.read_text(encoding='utf8'))
        text=re.sub(r"\n    # Reapply R2/history separation.*?(?=    return dict\(schema='bounded-unknown-local-install-qa-v1')",'',text,flags=re.S)
        anchor="    return dict(schema='bounded-unknown-local-install-qa-v1'"
        assert anchor in text;text=text.replace(anchor,hook+anchor,1)
        write(installer,text)
    # Base scene rebuild preserves the repair by replaying it as the last local step.
    builders=[p for p in [BASE/'work/persistent_world_20261027/build_scene_workspaces.py',WORK/'build_scene_workspaces.py'] if p.is_file()]
    for builder in dict.fromkeys(builders):
        text=builder.read_text(encoding='utf8')
        text=re.sub(r'\n# Restore R2 reference and legacy-study separation.*\Z','',text,flags=re.S)
        text+="\n# Restore R2 reference and legacy-study separation only when complete input assets exist.\nimport runpy\nfor repair in (Path(__file__).with_name('patch_ui.py'), BASE/'work/v43_r2_regression_20261029/patch_ui.py'):\n    if repair.is_file():\n        namespace=runpy.run_path(str(repair))\n        if namespace['can_repair'](BASE):namespace['main'](BASE)\n        break\n"
        write(builder,text)
    # A later, fully verified inherited49 delivery takes priority over this older
    # three-anchor repair. Never restore a stale default or pending-status banner.
    import runpy
    inherited_ready=False
    for newer in (Path(__file__).with_name('install_r2_inherited_ui.py'), BASE/'work/v43_r2_regression_20261029/install_r2_inherited_ui.py'):
        if newer.is_file():
            module=runpy.run_path(str(newer))
            inherited_ready=module['can_reapply'](BASE)
            if inherited_ready:module['reapply_installed'](BASE)
            break
    assert all(sha(p)==h for p,h in protected.items())
    node=shutil.which('node') or ('C:/Program Files/nodejs/node.exe' if Path('C:/Program Files/nodejs/node.exe').is_file() else None)
    if node:
        for p in [OUT/'scene_target_detection.js',OUT/'scene_r2_reference.js',OUT/C1/'r2_reference_data.js']:
            subprocess.run([node,'--check',str(p)],check=True)
    legacy=load(OUT/C1/'target_detection_data.js')
    report=dict(legacy_ready_raws=[int(k) for k,v in legacy['frames'].items() if v.get('evidence_path')],legacy_unexecuted_frames=sum(not bool(v.get('evidence_path')) for v in legacy['frames'].values()),bounded_raws=49,r2_raws=manifest['raws'],r2_full_sequence_status='inherited49_complete' if inherited_ready else 'not_delivered',r2_removed_rows={r:len(x['static_suppressed_rows']) for r,x in geometry.items()},r2_original_report_and_png_unchanged=True,new_gpu_execution=False,browser_qa=False)
    write(WORK/'ui_repair_audit.json',json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root',type=Path,default=BASE,help='Workspace that contains outputs/GS_Scene_Diagnostics_20261027; required when deployed elsewhere.')
    parser.add_argument('--check-inputs',action='store_true',help='Read-only input availability check; no substitute or fallback.')
    args=parser.parse_args()
    if args.check_inputs:print(json.dumps({'workspace_root':str(args.workspace_root.resolve()),'ready':can_repair(args.workspace_root),'missing':[str(p) for p in required_inputs(args.workspace_root) if not p.is_file()]},ensure_ascii=False,indent=2))
    else:main(args.workspace_root)
