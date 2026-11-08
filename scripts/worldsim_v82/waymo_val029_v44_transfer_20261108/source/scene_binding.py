import json
from pathlib import Path
def bind_scene_contract(training,config):
    original=training.Scene;spec=config['followup_scene_spec']
    def checked(*args,**kwargs):
        scene=original(*args,**kwargs);cameras=sorted(scene.getTrainCameras(),key=lambda c:float(c.timestamp))
        identities=[Path(c.image_path).parent.name+'/'+c.image_name for c in cameras]
        assert identities==spec['frame_ids_with_camera']
        assert len(cameras)==len(spec['model_timestamps'])
        assert all(abs(float(c.timestamp)-t)<2e-7 for c,t in zip(cameras,spec['model_timestamps']))
        assert all((c.image_width,c.image_height)==(480,320) for c in cameras)
        assert not scene.getTestCameras(scale=4)
        scene.followup_scene_spec=spec
        (Path(config['model_path'])/'actual_scene_binding.json').write_text(json.dumps(dict(status='PASS_ACTUAL_SCENE_BINDING',frame_ids=identities,timestamps=spec['model_timestamps'],render_wh=[480,320],anchor_pose_frame=spec['anchor_frame']),indent=2))
        return scene
    training.Scene=checked
