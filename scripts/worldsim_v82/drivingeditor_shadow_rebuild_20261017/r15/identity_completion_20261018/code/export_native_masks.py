"""导出已缓存的真实SAM3完整提议；与R15融合未知域严格分开。"""
from identity_common import *
from PIL import Image,ImageDraw

def main():
    r=ROOT/'shadow_rebuild_20261017_v3'
    shared=r/'ownership_rebuild_r14/shared_masks_r14'
    prediction=r/'ownership_controls_r12_sam3/shared_sam3_masks_r1/native_predictions'
    manifest=read(R15/'data/manifest.json');review=read(shared/'summary.json')
    dest=TASK/'native_masks_for_review'
    if (dest/'manifest.json').exists():raise FileExistsError(dest)
    dest.mkdir(exist_ok=True);ids=[manifest['actor_ids'][a] for a in NEW_ACTORS];records=[]
    sheets={a:Image.new('RGB',(1540,1050),'#eff4f8') for a in NEW_ACTORS}
    for i,row in enumerate(manifest['frames']):
        fid=row['frame_id'];bundle=np.load(shared/(fid+'.npz'),allow_pickle=False)
        raw=np.load(prediction/(fid+'.npz'),allow_pickle=False)
        masks=bundle['SAM_prediction'][...,NEW_ACTORS]
        image=Image.open(R15/'data'/row['rgb']).convert('RGB');rgb=np.asarray(image)
        frame=[]
        expected={6:11,37:14}
        for j,a in enumerate(NEW_ACTORS):
            native=np.flatnonzero(np.array([np.array_equal(x,masks[...,j]) for x in raw['masks']]))
            # QUERY帧原生预测的tracker编号是独立序列；沿用原逐帧真实LiDAR
            # 身份关联，不把BUILD tracker编号强加到QUERY帧。
            candidates=[int(k) for k in native if row['split']!='train' or int(raw['tracker_ids'][k])==expected[a]]
            assert len(candidates)==1,(fid,a,native)
            n=candidates[0];score=float(raw['scores'][n]);assert score>=.5
            rec=next(v for v in review['frames'][i]['actors'] if v['actor']==a)
            assert review['frames'][i]['frame_id']==fid and rec['SAM_associated']
            assert masks[...,j].any()
            ys,xs=np.where(masks[...,j]);bbox=[max(0,int(xs.min())-8),max(0,int(ys.min())-8),min(480,int(xs.max())+9),min(320,int(ys.max())+9)]
            overlay=rgb.copy();overlay[masks[...,j]]=np.rint(.75*overlay[masks[...,j]]+.25*np.array([30,180,255])).astype(np.uint8)
            tile=Image.new('RGB',(220,150),'white');draw=ImageDraw.Draw(tile)
            draw.text((4,2),f'a{a} / raw{int(fid[-3:])} / SAM tid{int(raw["tracker_ids"][n])}',fill='#172f46')
            im=Image.fromarray(overlay).crop(bbox);im.thumbnail((210,125),Image.Resampling.NEAREST)
            tile.paste(im,((220-im.width)//2,23+(125-im.height)//2))
            sheets[a].paste(tile,((i%7)*220,(i//7)*150))
            frame.append(dict(actor=a,track_id=ids[j],tracker_id=int(raw['tracker_ids'][n]),native_index=n,
                score=score,pixels=int(masks[...,j].sum()),exact_unmodified_native_prediction=True,
                previous_fused_positive=rec['fused_positive'],fused_valid_pixels=rec['valid_pixels']))
        assert not (masks.sum(-1)>1).any()
        np.savez_compressed(dest/(fid+'.npz'),masks=masks,mask_known=np.ones(2,bool),mask_valid=np.ones_like(masks))
        records.append(dict(frame_id=fid,actors=frame,source=str(prediction/(fid+'.npz'))))
    for a,im in sheets.items():im.save(dest/f'a{a}_49frame_identity_review.png')
    write(dest/'manifest.json',dict(actor_ids=ids,provenance=dict(
        role='native SAM3 complete image proposals chosen for new legacy-method owner5 input',
        native_predictions=str(prediction),associations=str(shared/'summary.json'),
        association_RULE='same cached real tracker + prior train-only non-road LiDAR association; exact pixels retained',
        R15_fused_unknown_not_filled_or_modified=True,input_role_differs_from_R15_training_masks=True),
        GT_boxes_used_to_create_masks=False,root_engineering_review='PENDING_49FRAME_VISUAL_CHECK',
        human_verdict=None,frames=records))
    print('EXPORTED_NATIVE_MASKS_VISUAL_REVIEW_REQUIRED '+str(dest))

if __name__=='__main__':main()
