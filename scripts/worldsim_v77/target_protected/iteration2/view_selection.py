"""固定整段选择真实供体视角，避免逐帧换车/换视角。未满足角度证据就拒绝。"""
import numpy as np

def select_clip_view(target_angles,reference_tracks,instance_token,max_yaw=12,max_pitch=8):
    target=np.asarray(target_angles,dtype=float);eligible=[]
    for track in reference_tracks:
        if track['instance_token']!=instance_token or track.get('quality_status')!='pass':continue
        angles=np.asarray(track['angles_deg'],dtype=float)
        if angles.shape!=target.shape:continue
        delta=target-angles;delta[:,0]=(delta[:,0]+180)%360-180
        if abs(delta[:,0]).max()>max_yaw or abs(delta[:,1]).max()>max_pitch:continue
        # 一整条真实camera-track承担一条目标camera序列，禁止每帧挑最漂亮的图。
        eligible.append((float(np.mean(delta**2)),track['track_id']))
    return min(eligible)[1] if eligible else None
