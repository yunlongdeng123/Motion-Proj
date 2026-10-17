"""Read-only diagnostic: exact frozen mask channels, complete boundaries, real LiDAR.

Never calls inference/fusion/training, and never writes the shared input directory.
"""
from pathlib import Path
import hashlib,json,time,zipfile,shutil
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion

R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
D=R/'ownership_rebuild_r14';O=D/'mask_visualization_r2'
T=[5,6,12,18,25,26,37]
COLORS={5:(255,180,40),6:(45,220,255),12:(196,130,255),18:(255,120,185),25:(100,230,80),26:(255,100,65),37:(70,150,255)}
M=json.loads((R/'data_r2/manifest.json').read_text())
O.mkdir(exist_ok=True)
for f in ['frames','arrays','binary','actors','reviews']:(O/f).mkdir(exist_ok=True)
stats=[];oldreviews=[];start=time.monotonic()

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def edge(mask):return mask&~binary_erosion(mask,structure=np.ones((3,3),bool),border_value=0)
def overlay(rgb,masks,unknown=None):
    x=np.asarray(rgb).copy()
    for a in T:
        mask=masks[...,a];c=np.array(COLORS[a]);x[mask]=np.rint(.73*x[mask]+.27*c).astype(np.uint8);x[edge(mask)]=c
    if unknown is not None:
        yy,xx=np.indices(unknown.shape);dot=unknown&((xx+yy)%4==0)
        x[unknown]=np.rint(.83*x[unknown]+.17*np.array([255,0,200])).astype(np.uint8);x[dot]=[255,0,200]
    im=Image.fromarray(x);dr=ImageDraw.Draw(im)
    for a in T:
        ys,xs=np.where(masks[...,a])
        if not len(xs):continue
        text=f'a{a}';lo=(max(0,min(447,int(np.median(xs)))),max(0,int(ys.min())-12));box=dr.textbbox(lo,text)
        dr.rectangle(box,fill=(10,15,23));dr.text(lo,text,fill=COLORS[a])
    return im

def caption(im,text):
    c=Image.new('RGB',(im.width,im.height+22),(12,22,32));c.paste(im,(0,22));ImageDraw.Draw(c).text((6,5),text,fill='white');return c

def save(im,p):im.save(p,optimize=False,compress_level=2)

ordered=sorted(M['frames'],key=lambda r:(r['frame_id']!='0008107',r['frame_id']))
for idx,row in enumerate(ordered):
    fid=row['frame_id'];folder=O/'frames'/fid;folder.mkdir(exist_ok=True)
    p=D/'shared_masks_r14'/(fid+'.npz');digest=sha(p)
    # Eager arrays: NpzFile.__getitem__ decompresses anew on every point lookup.
    with np.load(p) as z:sup={k:z[k] for k in z.files}
    with np.load(R/'data_r2'/row['supervision']) as z:old={k:z[k] for k in z.files}
    with np.load(D/'raw_measurements'/(fid+'.npz')) as z:raw={k:z[k] for k in ['BUILD','QUERY','pixel','in_image','camera_z','candidate_owner']}
    rgb=Image.open(R/'data_r2'/row['rgb']).convert('RGB');masks=sup['masks'];valid=sup['mask_valid'];sam=sup['SAM_prediction'];un=sup['uncertainty']
    # Native 480x320 RGB; no resizing or morphological edits to input channels.
    save(rgb,folder/'rgb.png')
    save(overlay(rgb,old['masks']),folder/'deva.png');save(overlay(rgb,sam),folder/'sam3.png');save(overlay(rgb,masks,un),folder/'fused.png')
    idxmap=np.zeros((320,480,3),np.uint8)
    for a in T:idxmap[masks[...,a]]=COLORS[a]
    idxmap[un]=[255,0,200];save(Image.fromarray(idxmap),folder/'labels.png')
    # Actual allowed same-frame returns only; exact depth visibility recomputed.
    allowed=raw['BUILD'] if row['split']=='train' else raw['QUERY'];ij=raw['pixel'].astype(np.int64)
    ids=np.flatnonzero(allowed&raw['in_image']);bins=ij[ids,1]*480+ij[ids,0];depth=np.full(320*480,np.inf);np.minimum.at(depth,bins,raw['camera_z'][ids]);front=ids[raw['camera_z'][ids]<=depth[bins]+.1]
    lidar=rgb.copy();dr=ImageDraw.Draw(lidar)
    for j in front:
        x,y=map(int,ij[j]);a=int(raw['candidate_owner'][j]);dr.point((x,y),fill=COLORS.get(a,(170,170,170)))
    save(lidar,folder/'lidar.png')
    # Lossless original frozen supervision bundle, all43 channels and validity.
    shutil.copy2(p,O/'arrays'/(fid+'.npz'))
    actual=[]
    for a in T:
        bi=O/'binary'/fid/f'a{a:02d}';bi.mkdir(parents=True,exist_ok=True)
        mask=masks[...,a];va=valid[...,a]
        for name,array in [('deva',old['masks'][...,a]),('sam3',sam[...,a]),('positive',mask),('valid',va),('unknown',~va),('boundary',edge(mask))]:
            save(Image.fromarray(array.astype(np.uint8)*255),bi/(name+'.png'))
        tri=np.where(va,np.where(mask,255,0),127).astype(np.uint8);save(Image.fromarray(tri),bi/'supervision.png')
        source=sup['source'][...,a];src=np.zeros((320,480,3),np.uint8)
        src[source==1]=[255,170,30];src[source==2]=[50,150,255];src[source==3]=[60,220,100];src[~va]=[255,0,200];save(Image.fromarray(src),bi/'source.png')
        support=front[raw['candidate_owner'][front]==a]
        positivehits=int(mask[ij[support,1],ij[support,0]].sum()) if len(support) else 0
        yy,xx=np.where(old['masks'][...,a]|sam[...,a]|mask)
        if len(xx):
            box=[max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(480,int(xx.max())+13),min(320,int(yy.max())+13)]
        elif len(support):
            xy=ij[support];box=[max(0,int(xy[:,0].min())-12),max(0,int(xy[:,1].min())-12),min(480,int(xy[:,0].max())+13),min(320,int(xy[:,1].max())+13)]
        else:box=[0,0,480,320]
        isolated=np.zeros_like(masks);isolated[...,a]=mask
        single=overlay(rgb,isolated,~va);single_lidar=single.copy();draw=ImageDraw.Draw(single_lidar)
        for j in support:
            x,y=map(int,ij[j]);draw.ellipse((x-1,y-1,x+1,y+1),fill='white',outline='black')
        tiles=[(rgb,'RGB'),(overlay(rgb,np.where(np.arange(43)==a,old['masks'],False)),'DEVA'),(overlay(rgb,np.where(np.arange(43)==a,sam,False)),'SAM3'),(single,'Fused / magenta unknown'),(Image.fromarray(tri).convert('RGB'),'255 body / 0 outside / 127 unknown'),(single_lidar,'Real LiDAR box candidates, not body truth')]
        cropw=max(1,box[2]-box[0]);croph=max(1,box[3]-box[1]);scale=min(4,600/cropw,450/croph);size=(max(1,round(cropw*scale)),max(1,round(croph*scale)))
        out=Image.new('RGB',(size[0]*3,(size[1]+24)*2),(12,22,32))
        for j,(im,title) in enumerate(tiles):out.paste(caption(im.crop(box).resize(size,Image.Resampling.NEAREST),title),((j%3)*size[0],(j//3)*(size[1]+24)))
        save(out,O/'actors'/f'{fid}_a{a:02d}.png')
        # Check both entering and leaving transitions at their actual coordinates.
        bottom=mask&~np.concatenate([mask[1:],np.zeros_like(mask[:1])],axis=0)
        top=mask&~np.concatenate([np.zeros_like(mask[:1]),mask[:-1]],axis=0)
        assert np.all(edge(mask)[bottom]) and np.all(edge(mask)[top])
        actual.append(dict(actor=a,positive=int(mask.sum()),valid=int(va.sum()),unknown=int((~va).sum()),DEVA=int(old['masks'][...,a].sum()),SAM3=int(sam[...,a].sum()),complete_boundary=int(edge(mask).sum()),bottom_boundary=int(bottom.sum()),top_boundary=int(top.sum()),frontmost_box_candidate_lidar=int(len(support)),positive_lidar_hits=positivehits,crop_box=box))
    # Four source panels plus their exact solid masks. No arrows/occlusion guesses.
    panels=[caption(rgb,'RGB'),caption(overlay(rgb,old['masks']),'DEVA: complete fill + boundary'),caption(overlay(rgb,sam),'SAM3: complete fill + boundary'),caption(overlay(rgb,masks,un),'Fused: complete fill + boundary; magenta unknown')]
    def colored(ms,unknown=None):
        a=np.zeros((320,480,3),np.uint8)
        for actor in T:a[ms[...,actor]]=COLORS[actor]
        if unknown is not None:a[unknown]=[255,0,200]
        return Image.fromarray(a)
    second=[caption(lidar,'Real LiDAR: frontmost BUILD / QUERY only'),caption(colored(old['masks']),'DEVA: solid mask (7 selected Actors)'),caption(colored(sam),'SAM3: solid mask (7 selected Actors)'),caption(colored(masks,un),'Fused: solid mask / explicit unknown')]
    canvas=Image.new('RGB',(1920,684),(12,22,32))
    for j,im in enumerate(panels+second):canvas.paste(im,((j%4)*480,(j//4)*342))
    save(canvas,O/'reviews'/(fid+'_review.png'))
    assert sha(p)==digest,'Frozen mask data mutated!'
    stats.append(dict(frame=fid,raw=row['source_index'],split=row['split'],shape=[320,480,43],frozen_sha256=digest,unchanged=True,actual_visible_lidar_returns=int(len(front)),actors=actual))
    print(json.dumps(dict(frame=idx+1,total=49,raw=row['source_index'],input_unchanged=True)),flush=True)

stats.sort(key=lambda r:r['frame'])
receipt=dict(status='FULL_MASK_VISUALIZATION_COMPLETE',frames=stats,rendering_only=True,supervision_modified=False,models_rerun=False,training_interrupted=False,fill_alpha=.27,boundary='mask AND NOT 3x3 binary erosion; border_value0; all sides and holes',native_size=[480,320],binary_png_values='positive/valid/unknown 0 or255; supervision body255 outside0 unknown127',LiDAR='actual same-frame permitted BUILD or QUERY; frontmost tolerance0.10m; color identifies OBB candidate, not verified vehicle surface',shown_actors=T,all43channels_in_npz=True,seconds=time.monotonic()-start)
(O/'verification.json').write_text(json.dumps(receipt,indent=2))
archive=D/'mask_visualization_r2.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
    for p in O.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(O))
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,seconds=time.monotonic()-start)),flush=True)
