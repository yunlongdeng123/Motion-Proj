from pathlib import Path
import shutil
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1')
src=RUN/'delivery_release_v1';out=RUN/'delivery_release_v2'
if out.exists():raise FileExistsError(out)
out.mkdir()
for name in ('index.html','report_data.json','architecture.svg'):
    shutil.copy2(src/name,out/name)
for name in ('frames','crops','figures','evidence'):
    shutil.copytree(src/name,out/name)
print(out)
