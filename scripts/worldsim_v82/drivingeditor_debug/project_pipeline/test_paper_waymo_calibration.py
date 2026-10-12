"""CPU regression for reordered calibration, camera identity and real projection."""
from pathlib import Path
import importlib.util,json,tempfile,io,struct
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw
from waymo_open_dataset import dataset_pb2

base=Path('/root/autodl-tmp/drivingeditor_repro')
repo=base/'DrivingEditor-paper'
spec=importlib.util.spec_from_file_location('calib',repo/'scene/waymo_calibration.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
scene=base/'data/waymo_paper_scenes/scene0017_000_048'
ids=[f'{17000+i:07d}' for i in range(49)]
calib_path=scene/'calib/0017000.txt'
single_args=SimpleNamespace(is_fisheye=False,long_focus=True,long_cam_num=1,cam_num=4)
assert module.selected_camera_indices(single_args)==[0]
single_args.camera_indices=[0,1,2]
assert module.selected_camera_indices(single_args)==[0,1,2]
for name in ids:
    path=scene/'calib'/f'{name}.txt';lines=path.read_text().splitlines()
    old=[list(map(float,x.split()[1:])) for x in lines]
    k,t=module.read_named_camera_calibration(path,[0])
    assert np.array_equal(k,np.array(old[0]).reshape(1,3,4)[:,:,:3])
    assert np.array_equal(t[:,:3],np.array(old[-5]).reshape(1,3,4))
    assert module.camera_observation_identity(int(name[-3:]),name,[0],0)==(int(name[-3:]),name)
with tempfile.TemporaryDirectory(prefix='drivingeditor-calib-test-') as td:
    p=Path(td)/'reordered.txt';p.write_text('\n'.join(reversed(calib_path.read_text().splitlines())))
    k,t=module.read_named_camera_calibration(calib_path,[0,1,2])
    kr,tr=module.read_named_camera_calibration(p,[0,1,2])
    assert np.array_equal(k,kr) and np.array_equal(t,tr)
    observations=[module.camera_observation_identity(i,name,[0,1,2],j) for i,name in enumerate(ids) for j in range(3)]
    assert len({x[0] for x in observations})==147 and len({x[1] for x in observations})==147
    missing=Path(td)/'missing.txt';missing.write_text('\n'.join(x for x in p.read_text().splitlines() if not x.startswith('Tr_velo_to_cam_2:')))
    try:module.read_named_camera_calibration(missing,[0,1,2])
    except ValueError:pass
    else:raise AssertionError('Missing selected extrinsic must fail explicitly')
manifest=json.loads((scene/'preparation_manifest.json').read_text())
with Path(manifest['raw_path']).open('rb') as stream:
    n=struct.unpack('<Q',stream.read(8))[0];stream.read(4)
    frame=dataset_pb2.Frame();frame.ParseFromString(stream.read(n))
raw=np.fromfile(scene/'velodyne/0017000.bin',dtype=np.float32).reshape(-1,6)
xyz=np.column_stack((raw[:,:3].astype(np.float64),np.ones(len(raw))))
pose=np.array(frame.pose.transform).reshape(4,4)
world=xyz@pose.T
axis=np.eye(4);axis[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]]
panels=[];stats=[]
for j,camera_index in enumerate((0,1,2)):
    calibration=next(c for c in frame.context.camera_calibrations if c.name==camera_index+1)
    image_proto=next(c for c in frame.images if c.name==camera_index+1)
    expected=axis@np.linalg.inv(np.array(calibration.extrinsic.transform).reshape(4,4))
    assert np.allclose(t[j],expected,rtol=1e-11,atol=1e-11)
    expected_k=np.array([[calibration.intrinsic[0],0,calibration.intrinsic[2]],[0,calibration.intrinsic[1],calibration.intrinsic[3]],[0,0,1]])
    assert np.allclose(k[j],expected_k,rtol=1e-11,atol=1e-11)
    direct=(xyz@t[j].T)[:,:3]
    world_to_cam=np.linalg.inv(pose@np.linalg.inv(t[j]))
    via_world=(world@world_to_cam.T)[:,:3]
    error=float(np.abs(via_world-direct).max());assert error<1e-8
    image=Image.open(io.BytesIO(image_proto.image)).convert('RGB')
    front=direct[direct[:,2]>.1];uvh=front@k[j].T;uv=uvh[:,:2]/uvh[:,2:]
    visible=(uv[:,0]>=0)&(uv[:,0]<image.width)&(uv[:,1]>=0)&(uv[:,1]<image.height)
    assert visible.sum()>100
    canvas=image.resize((576,384));draw=ImageDraw.Draw(canvas)
    chosen=np.flatnonzero(visible)[::5]
    for at in chosen:
        u,v=uv[at]*.3;z=min(front[at,2]/60,1);draw.point((u,v),fill=(int(255*(1-z)),int(255*z),60))
    panels.append(canvas)
    stats.append({'camera_index':camera_index,'Waymo_enum':camera_index+1,'front_visible_lidar':int(visible.sum()),'calibration_matches_raw_proto':True,'world_vs_direct_camera_xyz_max_error':error,'image_size':list(image.size),'raw_frame_timestamp':frame.timestamp_micros,'camera_timestamp_seconds':image_proto.pose_timestamp})
sheet=Image.new('RGB',(1728,416),'white');draw=ImageDraw.Draw(sheet)
for j,(panel,s) in enumerate(zip(panels,stats)):
    sheet.paste(panel,(j*576,32));draw.text((j*576+10,10),f'camera{s["camera_index"]} / enum{s["Waymo_enum"]}: {s["front_visible_lidar"]} projected real points',fill='black')
sheet.save(base/'logs/paper_multicamera_reprojection.jpg',quality=95)
result={'status':'passed','synthetic_reordered_calibration':True,'missing_selected_extrinsic_rejected':True,'single_camera_49frame_intrinsics_extrinsics_and_identity_exact_parity':True,'multicamera_observation_uid_and_name_unique':147,'camera_indices':[0,1,2],'camera_choice_source':'Declared reproduction setting; paperTableII specifies3cams but not enumidentities','real_frame_checks':stats,'limitations':['Projection follows author pinhole convention; lensdistortion omitted.','Threecamera RGB/skymasks/dynamicmasks have notyetbeen exported forwhole49window; no fullmulticamGPUtrainingclaim.']}
(base/'logs/paper_multicamera_calibration_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
