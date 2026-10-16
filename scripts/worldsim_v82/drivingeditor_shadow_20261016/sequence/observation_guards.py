import json,itertools
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import ConvexHull,QhullError

def annotation_boxes(annotation,norm,selected):
    pose=np.asarray(annotation['frame_pose_world_from_vehicle']).reshape(4,4);result=[]
    signs=np.asarray(list(itertools.product([-.5,.5],repeat=3)))
    for label in annotation['laser_labels']:
        if label['id'] in selected or label['type'] not in ['TYPE_VEHICLE','TYPE_PEDESTRIAN','TYPE_CYCLIST']:continue
        b=label.get('camera_synced_box') or label['box'];angle=b['heading'];c=np.cos(angle);s=np.sin(angle)
        rot=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        vertices=(signs*np.array([b['length'],b['width'],b['height']]))@rot.T+np.array([b['center_x'],b['center_y'],b['center_z']])
        world=vertices@pose[:3,:3].T+pose[:3,3];model=world@norm[:3,:3].T+norm[:3,3]
        result.append(model.astype('float32'))
    return result

def project_guard(boxes,cam):
    image=Image.new('L',(480,320));draw=ImageDraw.Draw(image)
    for corners in boxes:
        p=np.c_[corners,np.ones(8)]@np.asarray(cam['full_proj_transform'])
        if not (p[:,3]>0).all():continue
        uv=(p[:,:2]/p[:,3:4]+1)*np.array([480,320])/2-.5
        if (uv[:,0].max()<0 or uv[:,0].min()>479 or uv[:,1].max()<0 or uv[:,1].min()>319):continue
        try:hull=uv[ConvexHull(uv).vertices]
        except QhullError:continue
        draw.polygon([tuple(x) for x in hull],fill=255)
    return np.asarray(image)>0
