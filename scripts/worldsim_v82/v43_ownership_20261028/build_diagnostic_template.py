"""Portable shared template, preserving existing per-scene data and experiment versions."""
from pathlib import Path
import argparse,hashlib,json,os
p=argparse.ArgumentParser();p.add_argument('--template-dir',type=Path,default=Path(__file__).parent);p.add_argument('--scene-dir',type=Path,required=True);a=p.parse_args();t=a.template_dir;s=a.scene_dir
for name in ['scene.js','cloud.js','identity.js','edit_geometry.js','foreign_clouds.js','semantic_observations.js']:
    if not (s/name).is_file():raise SystemExit('Missing existing scene evidence: '+str(s/name))
if (s/'identity.js').read_text(encoding='utf8').strip().endswith('null;'):raise SystemExit('Export all original vehicle boxes before building')
runtime=(t/'scene_viewer.js').read_text(encoding='utf8')
for token,name in [('EDIT_GEOMETRY','scene_edit_geometry.js'),('SAMPLING','scene_sampling.js'),('SEMANTIC','scene_semantics.js')]:runtime=runtime.replace('/* '+token+'_RUNTIME */',(t/name).read_text(encoding='utf8'))
runtime=runtime.replace('/* BOUNDARY_RUNTIME */',(t/'scene_boundary_semantics.js').read_text(encoding='utf8'))
assert '/* SEMANTIC_RUNTIME */' not in runtime
def atomic(path,text):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf8');os.replace(str(tmp),str(path))
if not (s/'v43_groups.js').exists():atomic(s/'v43_groups.js','// No completed V43 groups delivered yet.\n')
page=(t/'scene_viewer.html').read_text(encoding='utf8')
if not (s/'gpu_semantics/index.js').exists():
    page=page.replace('<option value="gpu" selected>','<option value="gpu">').replace('<option value="depth">','<option value="depth" selected>')
    start=page.index('<div id="semanticStatus"');end=page.index('</div>',start)+6
    page=page[:start]+'<div id="semanticStatus" class="notice">本场景GPU语义证据尚未交付；当前使用已有CPU深度候选，unknown不等于背景。</div>'+page[end:]
atomic(s.parent/'viewer.js',runtime);atomic(s/'index.html',page)
for src,dest in [('scene_workspace.css','workspace.css'),('style.css','style.css'),('architecture.svg','architecture.svg')]:atomic(s.parent/dest,(t/src).read_text(encoding='utf8'))
print(json.dumps(dict(scene=str(s),runtime_sha256=hashlib.sha256(runtime.encode()).hexdigest(),existing_experiments_preserved=True,all_original_vehicle_boxes_required=True,training_updates=0)))
