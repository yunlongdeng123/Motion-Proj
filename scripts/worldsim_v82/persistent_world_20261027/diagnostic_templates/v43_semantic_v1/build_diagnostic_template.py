"""Portable shared viewer builder. Inputs are existing reviewed scene exports."""
from pathlib import Path
import argparse,hashlib,json,os
p=argparse.ArgumentParser();p.add_argument('--template-dir',type=Path,default=Path(__file__).parent);p.add_argument('--scene-dir',type=Path,required=True);a=p.parse_args();t=a.template_dir;s=a.scene_dir
for name in ['scene.js','cloud.js','identity.js','edit_geometry.js','foreign_clouds.js','semantic_observations.js']:
 if not (s/name).is_file():raise SystemExit('Missing required existing scene evidence: '+str(s/name))
if (s/'identity.js').read_text(encoding='utf8').strip().endswith('null;'):raise SystemExit('All original vehicle boxes must be exported before building this template')
runtime=(t/'scene_viewer.js').read_text(encoding='utf8')
for token,name in [('EDIT_GEOMETRY','scene_edit_geometry.js'),('SAMPLING','scene_sampling.js'),('SEMANTIC','scene_semantics.js')]:runtime=runtime.replace('/* '+token+'_RUNTIME */',(t/name).read_text(encoding='utf8'))
assert '/* SEMANTIC_RUNTIME */' not in runtime
def atomic(path,text):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf8');os.replace(str(tmp),str(path))
atomic(s.parent/'viewer.js',runtime);atomic(s/'index.html',(t/'scene_viewer.html').read_text(encoding='utf8'))
for src,dest in [('scene_workspace.css','workspace.css'),('style.css','style.css')]:atomic(s.parent/dest,(t/src).read_text(encoding='utf8'))
print(json.dumps({'scene':str(s),'shared_runtime_sha256':hashlib.sha256(runtime.encode()).hexdigest(),'training_updates':0,'renderer_owner_modified':False}))
