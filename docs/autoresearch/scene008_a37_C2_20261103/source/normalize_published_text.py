"""仅规范本轮报告/出版副本的行尾，源模型/数组/视频不变。"""
from pathlib import Path
import json

run=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_C2_20261103_r1')
publication=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/docs/autoresearch/scene008_a37_C2_20261103')
changed=[]
for root in (run/'report',publication):
    for path in root.rglob('*'):
        if not path.is_file() or path.suffix not in ('.json','.html','.md','.mmd','.svg','.css','.js','.txt'):continue
        raw=path.read_bytes()
        value=raw.decode('utf-8')
        value='\n'.join(line.rstrip() for line in value.splitlines())+'\n'
        new=value.encode('utf-8')
        if new!=raw:path.write_bytes(new);changed.append(str(path))
receipt=dict(scope='only current report and Git publication text copies',changed=changed,
             source_model_and_array_and_video_unchanged=True)
(run/'text_publication_normalization.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(normalized_files=len(changed))))
