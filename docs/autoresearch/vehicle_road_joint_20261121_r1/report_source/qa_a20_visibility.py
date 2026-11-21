from pathlib import Path
import itertools
import json
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1')
BANK = Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
MASK = BANK.parent / 'repaired_masks_all_r4'
NATIVE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val065_allframes10k_20261109_r1/inputs/native')
TRACK = 'bzFbWRrSGOgEaXxnuGV0tw'


def project_box(obs, box):
    local = np.array(list(itertools.product([-1, 1], repeat=3))) * np.array(box['dimensions']) / 2
    actor = np.asarray(box['world_from_actor']).reshape(4, 4)
    world = local @ actor[:3, :3].T + actor[:3, 3]
    camera = np.asarray(obs['camera_pose_world']).reshape(4, 4) @ np.asarray(obs['camera_extrinsic']).reshape(4, 4)
    camera_xyz = (world - camera[:3, 3]) @ camera[:3, :3]
    x, y = -camera_xyz[:, 1] / camera_xyz[:, 0], -camera_xyz[:, 2] / camera_xyz[:, 0]
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = obs['camera_intrinsic_Brown']
    radius = x*x + y*y
    radial = 1 + k1*radius + k2*radius**2 + k3*radius**3
    xd = x*radial + 2*p1*x*y + p2*(radius + 2*x*x)
    yd = y*radial + p1*(radius + 2*y*y) + 2*p2*x*y
    uv = np.stack([fx*xd + cx, fy*yd + cy], axis=-1)
    bbox = [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]
    camera_local = (camera[:3, 3] - actor[:3, 3]) @ actor[:3, :3]
    return bbox, camera_local.tolist()


records = []
for raw in range(199):
    obs = json.loads((BANK / 'observations' / f'{raw:03d}.json').read_text())
    box = next((item for item in obs['boxes'] if item['track'] == TRACK), None)
    path = MASK / f'{raw:03d}' / 'a20.png'
    if box is None or not path.exists():
        records.append(dict(raw=raw, visible_pixels=0, lower_half_pixels=0))
        continue
    mask = np.asarray(Image.open(path)) > 0
    bbox, camera_local = project_box(obs, box)
    raw_width, raw_height = obs['camera_size']
    width, height = mask.shape[1], mask.shape[0]
    bbox_mask = [bbox[0]*width/raw_width, bbox[1]*height/raw_height, bbox[2]*width/raw_width, bbox[3]*height/raw_height]
    mid = (bbox_mask[1] + bbox_mask[3]) / 2
    y, x = np.where(mask)
    lower = int(np.count_nonzero(y + .5 >= mid))
    bottom_quarter = int(np.count_nonzero(y + .5 >= bbox_mask[1] + .75*(bbox_mask[3]-bbox_mask[1])))
    stored = np.array([v for v in box['corners2D'] if v is not None])
    stored_aabb = np.r_[stored.min(axis=0), stored.max(axis=0)]
    bbox_area = max(1, (bbox_mask[2]-bbox_mask[0])*(bbox_mask[3]-bbox_mask[1]))
    records.append(dict(raw=raw, visible_pixels=int(mask.sum()), lower_half_pixels=lower,
                        lower_fraction_of_visible=float(lower/max(1,len(y))),
                        lower_fraction_of_lower_box=float(lower/(bbox_area/2)),
                        bottom_quarter_pixels=bottom_quarter, bbox_raw=bbox, bbox_mask=bbox_mask,
                        visible_bbox_mask=None if len(x)==0 else [int(x.min()),int(y.min()),int(x.max()),int(y.max())],
                        camera_position_in_actor=camera_local, lidar_points=int(box['measured_lidar_points']),
                        mask_size=[width,height], image_size=[raw_width,raw_height],
                        stored_bbox_to_raw_projection_max_error=float(np.abs(stored_aabb*2-np.array(bbox)).max()),
                        image=str(NATIVE / obs['image_path']), mask=str(path)))

ranked = sorted(records, key=lambda item:item['lower_half_pixels'], reverse=True)
fixed = [86,110,134,179,198]
selected = sorted(set(fixed + [ranked[i]['raw'] for i in [0,10,20,40,70]]))
tile_width, tile_height = 600, 380
sheet = Image.new('RGB',(tile_width*3,tile_height*((len(selected)+2)//3)),(16,19,25))
draw = ImageDraw.Draw(sheet)
for index, raw in enumerate(selected):
    row=records[raw]
    rgb=Image.open(row['image']).convert('RGB')
    bbox=row['bbox_raw']
    alpha=np.asarray(Image.open(row['mask']).resize(rgb.size,Image.Resampling.NEAREST))>0
    a9path=MASK/f'{raw:03d}'/'a9.png'
    a9=np.asarray(Image.open(a9path).resize(rgb.size,Image.Resampling.NEAREST))>0 if a9path.exists() else np.zeros(alpha.shape,bool)
    array=np.asarray(rgb).copy()
    array[alpha]=(.6*array[alpha]+.4*np.array([10,240,180])).astype(np.uint8)
    array[a9]=(.75*array[a9]+.25*np.array([240,30,30])).astype(np.uint8)
    annotated=Image.fromarray(array)
    d=ImageDraw.Draw(annotated)
    d.rectangle(bbox,outline=(255,230,60),width=3)
    d.line((bbox[0],(bbox[1]+bbox[3])/2,bbox[2],(bbox[1]+bbox[3])/2),fill=(255,230,60),width=2)
    crop=annotated.crop((max(0,int(bbox[0])-100),max(0,int(bbox[1])-55),min(rgb.width,int(bbox[2])+100),min(rgb.height,int(bbox[3])+110)))
    crop.thumbnail((tile_width,tile_height-50))
    x=(index%3)*tile_width;y=(index//3)*tile_height
    sheet.paste(crop,(x,y+50))
    draw.text((x+8,y+7),f"raw{raw:03d} visible={row['visible_pixels']} lower={row['lower_half_pixels']} bottom25={row['bottom_quarter_pixels']}",fill='white')
    draw.text((x+8,y+25),f"lower/visible={row['lower_fraction_of_visible']:.1%}; green=a20 red=a9 yellow=projected box",fill=(210,215,225))

ROOT.mkdir(parents=True,exist_ok=True)
sheet.save(ROOT/'qa_a20_visibility.jpg',quality=92)
summary=dict(target=TRACK,coordinate_contract='Brown-projected raw1920x1280 box; repaired mask480x320. Stored corners2D are raw/2.',
             frames=len(records),with_lower=sum(row['lower_half_pixels']>0 for row in records),
             with_bottom_quarter=sum(row.get('bottom_quarter_pixels',0)>0 for row in records),
             fixed=[records[raw] for raw in fixed],top_lower=ranked[:20],selected_contact_sheet=selected,records=records)
(ROOT/'qa_a20_visibility.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({key:summary[key] for key in ['coordinate_contract','frames','with_lower','with_bottom_quarter','fixed','selected_contact_sheet']},indent=2))
print('TOP',json.dumps([{key:row[key] for key in ['raw','visible_pixels','lower_half_pixels','bottom_quarter_pixels','lower_fraction_of_visible','camera_position_in_actor']} for row in ranked[:20]],indent=2))
