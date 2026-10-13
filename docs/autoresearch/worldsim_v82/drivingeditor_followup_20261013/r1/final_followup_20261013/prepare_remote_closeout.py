from pathlib import Path
import shutil,json
F=Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013')
D=F/'reports/final_closeout_20261013';D.mkdir(exist_ok=True)
for link,target in [('drivingeditor_four_arm_30k','../four_arm_30k'),('drivingeditor_GMC_GN11_30k','../gmc_gn11_30k_saved_cpu_20261013_0910'),('drivingeditor_red_GN11_30k','../red_gn11_30k_saved_cpu_20261013')]:
    p=D/link
    assert (D/target/'index.html').is_file(),target
    if p.is_symlink():assert p.readlink().as_posix()==target
    else:assert not p.exists();p.symlink_to(target,target_is_directory=True)
canonical=F/'tools/DRIVINGEDITOR_FOLLOWUP_20261013.json';backup=canonical.with_suffix('.before_final_closeout_20261013.json')
if not backup.exists():shutil.copy2(canonical,backup)
print(json.dumps({'status':'REMOTE_CLOSEOUT_DIRECTORY_AND_EXISTING_REPORT_LINKS_READY','canonical_backup':str(backup)}))
