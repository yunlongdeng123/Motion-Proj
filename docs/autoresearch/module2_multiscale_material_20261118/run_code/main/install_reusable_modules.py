from pathlib import Path
import shutil,json

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
dest=R/'scripts/worldsim_v82/drivingeditor_dynamic_edit'
tests=R/'tests/worldsim_v82';tests.mkdir(parents=True,exist_ok=True)
bindings={}
for source,target in [('world_material_field.py','road_multiscale_material.py'),('road_chart_sampling.py','road_material_chart.py'),('road_multiscale_asset.py','road_multiscale_asset.py')]:
    src=P/'code'/source;out=dest/target
    if out.exists() and out.read_bytes()!=src.read_bytes():raise RuntimeError('拒绝覆盖已存在的非本轮模块：'+str(out))
    shutil.copy2(src,out);bindings[str(out.relative_to(R))]=str(src)
for source,target in [('test_world_material_field.py','test_road_multiscale_material.py'),('test_road_multiscale_asset.py','test_road_multiscale_asset.py')]:
    text=(P/'code'/source).read_text()
    if source=='test_world_material_field.py':text=text.replace('from world_material_field import','from drivingeditor_dynamic_edit.road_multiscale_material import')
    out=tests/target
    if out.exists() and out.read_text()!=text:raise RuntimeError('拒绝覆盖其他测试：'+str(out))
    out.write_text(text)
(P/'qa/reusable_module_bindings.json').write_text(json.dumps(dict(opt_in=True,default_scene_modified=False,bindings=bindings),indent=2)+'\n')
print(json.dumps(dict(modules=len(bindings),CPU_test_files=2,default_scene_modified=False)))
