"""保存首例轴向错误的共同工程修正；初始登记不可覆盖。"""
from nine_common import *
import datetime,copy
path=ROOT/'engineering_amendment_01.json';assert not path.exists()
amend=dict(task_id='WS-V77-NINE-FULL-20260928',run_id='r1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),type='engineering_coordinate_fix',
    trigger='First local scene0230 canonical GLB dimensions[0.894172,1.989194,0.626646] showed longY after fixedRz-90; treating every Hunyuan output as longY-before-conversion is incorrect.',
    preserved='scene_0230/axis_error_control contains old GLB, views, canonical metadata, initial partial RGBA/depth andlogs. No shader or model regeneration.',
    shared_rule='OBJ coordinate Rx+90 then horizontal vertex PCA longaxis ->X; real reference same as asset source. Renderyaw0 and180 at that source pose, score maskIoU-.25*foreground_RGB_MAE[0,1]; gap<.01 choose0 and markambiguous. Bake selected front/rear yaw once into GLB, sameGTpose/size/height for all30 frames.',
    no_per_scene_tuning=True,scope='All9 apply same algorithm; source-based estimated orientation is BUILD fitting, not manual selection from edited videos or heldout quality. Mask/DriveEditor/Omega/Hunyuan parameters unchanged.',
    limitation='PCA does not certify semantic front or mesh quality; actual source may be occluded and scores ambiguous. Preserve both source controls for review.',
    failure_ledger_refs=['V77-F02'],human_verdict=None)
dump(path,amend);reg=copy.deepcopy(read(ROOT/'registration.json'));reg['engineering_amendments']=['engineering_amendment_01.json'];reg['fixed']['pose']='Horizontal vertex PCA longaxis->X, same observed-source yaw0/180 scoring and bake; GTcenter/size/height same all9; fixed light';reg['resource']['actual_model_sequence']='SAM2 -> Hunyuan9 -> DriveEditor -> frozenOmega; localCPU renders overlapGPU';dump(ROOT/'effective_registration.json',reg);print('AXIS_AMENDMENT_REGISTERED')
