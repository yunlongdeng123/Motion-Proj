"""Exact XY displacement, fixed-radius matches, and native Brown photo crops.

Read-only causal diagnostic. No pose/point/model changes and no Z filtering.
"""
from pathlib import Path
import argparse,csv,hashlib,json,time
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def stat(x):
    x=np.asarray(x);return dict(count=len(x),mean=float(x.mean()),median=float(np.median(x)),p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95)),abs_median=float(np.median(abs(x))),abs_p95=float(np.quantile(abs(x),.95))) if len(x) else dict(count=0)
def brown(x,E,intr):
    c=x@E[:3,:3].T+E[:3,3];z=c[:,2];xy=c[:,:2]/np.maximum(z[:,None],1e-8);u,v=xy.T
    fx,fy,cx,cy,k1,k2,p1,p2,k3=intr;r=u*u+v*v;rad=1+k1*r+k2*r*r+k3*r*r*r
    return np.c_[fx*(u*rad+2*p1*u*v+p2*(r+2*u*u))+cx-.5,fy*(v*rad+p1*(r+2*v*v)+2*p2*u*v)+cy-.5],z
def crop_box(uv,size,side=128):
    cx,cy=np.rint(uv).astype(int);w,h=size
    x0=min(max(cx-side//2,0),max(w-side,0));y0=min(max(cy-side//2,0),max(h-side,0))
    return [int(x0),int(y0),int(min(x0+side,w)),int(min(y0+side,h))]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.time();r=a.root;assert not a.output.exists()
    C=r.parent;B=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
    src=r/'audit/local_surface_expanded_candidate_v1/actual_scope_residual_v1/actual_scope_point_residuals.npz'
    cache=r/'audit/ground_contact_prior_raw_GT_v2/GT_bottom_and_road_evidence.npz'
    with np.load(src) as z:
        xyz=z['original_point_worldXYZ'];raw=z['original_point_raw'];requested=z['original_point_requested_spatial_mask'];residual=z['original_point_signed_plane_minus_ground_m']
    with np.load(cache) as z:
        assert np.array_equal(xyz,z['official_training_road_worldXYZ']) and np.array_equal(raw,z['official_training_road_source_raw'])
        lidar=z['official_training_road_lidar_row_id']
    photos={};obs={};Es={};pins=[];semantics={}
    semantic_path=r/'audit/same_frame_surface_layers_v1/context_semantic_points.npz'
    if semantic_path.is_file():
        with np.load(semantic_path) as z:
            if all(k in z.files for k in ['source_raw','lidar_row_id','semantic']):
                semantics={(int(t),int(i)):int(s) for t,i,s in zip(z['source_raw'],z['lidar_row_id'],z['semantic'])}
                pins.append(pin(semantic_path))
    for rr in [60,70,110,145,165]:
        op=B/f'observations/{rr:03d}.json';obs[rr]=read(op);ip=B/obs[rr]['image_path'];photo=Image.open(ip).convert('RGB');assert photo.size==tuple(obs[rr]['camera_size']);photos[rr]=photo
        cp=C/f'road_multisource_20261122_r1/sources/qualified_observations/{rr:03d}.npz'
        with np.load(cp) as z:Es[rr]=z['world_to_camera']
        pins.extend([pin(op),pin(ip),pin(cp)])
    a.output.mkdir(parents=True);all_stats=[];pair_rows=[];panels=[];bank_checks=[];plot_data=[]
    for target_raw in [60,70]:
        target=np.flatnonzero(requested&(raw==target_raw));target_tree=cKDTree(xyz[target,:2])
        for reference_raw in [110,145,165]:
            reference=np.flatnonzero(raw==reference_raw);dist,nn=cKDTree(xyz[reference,:2]).query(xyz[target,:2]);ok=dist<=.10;ti=target[ok];ri=reference[nn[ok]]
            back=target_tree.query(xyz[ri,:2])[1];mutual=back==np.flatnonzero(ok)
            delta=xyz[ti]-xyz[ri];angle=np.degrees(np.arctan2(delta[:,1],delta[:,0]));uvt,zt=brown(xyz[ti],Es[target_raw],np.asarray(obs[target_raw]['camera_intrinsic_Brown']));uvr,zr=brown(xyz[ri],Es[reference_raw],np.asarray(obs[reference_raw]['camera_intrinsic_Brown']))
            label=f'{target_raw:03d}_to_{reference_raw:03d}';out=a.output/label;out.mkdir()
            pair_records=[]
            for radius in [.02,.05,.10]:
                m=dist[ok]<=radius
                all_stats.append(dict(target_raw=target_raw,reference_raw=reference_raw,XY_radius_m=radius,target_requested_points=len(target),matches=int(m.sum()),mutual_matches=int((m&mutual).sum()),
                    delta_X_world_m=stat(delta[m,0]),delta_Y_world_m=stat(delta[m,1]),target_minus_reference_Z_m=stat(delta[m,2]),
                    mutual_target_minus_reference_Z_m=stat(delta[m&mutual,2]),delta_XY_direction_deg=stat(angle[m]),no_Z_filter=True))
            # Validate selected cached point IDs against the original six-column bank.
            for rr,ix in [(target_raw,ti),(reference_raw,ri)]:
                lp=B/obs[rr]['lidar_path'];cloud=np.memmap(lp,dtype='<f4',mode='r').reshape(-1,6);T=np.asarray(obs[rr]['ego_world']).reshape(4,4)
                assert ((lidar[ix]>=0)&(lidar[ix]<len(cloud))).all();world=cloud[lidar[ix],:3].astype(np.float64)@T[:3,:3].T+T[:3,3]
                error=float(abs(world-xyz[ix]).max(initial=0));assert error<1e-5
                bank_checks.append(dict(pair=label,raw=rr,points=len(ix),bank=pin(lp),worldXYZ_max_abs_m=error,only_selected_rows_read=True))
            overlays={rr:photos[rr].copy() for rr in [target_raw,reference_raw]}
            for k,(i,j) in enumerate(zip(ti,ri)):
                one=dict(pair=label,pair_index=k,target_raw=target_raw,reference_raw=reference_raw,target_cache_row=int(i),reference_cache_row=int(j),target_lidar_row_id=int(lidar[i]),reference_lidar_row_id=int(lidar[j]),
                    target_worldXYZ=xyz[i].tolist(),reference_worldXYZ=xyz[j].tolist(),delta_worldXYZ=delta[k].tolist(),XY_distance_m=float(dist[ok][k]),XY_direction_deg=float(angle[k]),mutual_nearest=bool(mutual[k]),
                    epsilon_02=bool(dist[ok][k]<=.02),epsilon_05=bool(dist[ok][k]<=.05),epsilon_10=True,target_Brown_uv=uvt[k].tolist(),reference_Brown_uv=uvr[k].tolist(),target_camera_depth_m=float(zt[k]),reference_camera_depth_m=float(zr[k]),
                    target_semantic_enum=semantics.get((target_raw,int(lidar[i]))),reference_semantic_enum=semantics.get((reference_raw,int(lidar[j]))))
                pair_records.append(one);pair_rows.append(one)
                colour='#ff665b' if abs(delta[k,2])>=.10 else '#ffd069' if abs(delta[k,2])>=.04 else '#5ce2d5'
                for rr,uv in [(target_raw,uvt[k]),(reference_raw,uvr[k])]:
                    if np.isfinite(uv).all() and 0<=uv[0]<photos[rr].width and 0<=uv[1]<photos[rr].height:
                        draw=ImageDraw.Draw(overlays[rr]);x,y=uv;draw.ellipse((x-4,y-4,x+4,y+4),outline=colour,width=2);draw.text((x+5,y-8),str(k),fill=colour)
            for rr,image in overlays.items():image.save(out/f'raw{rr:03d}_all_pairs_overlay.png')
            # Rank only for display; all pairs and all epsilons remain in tables.
            display=np.argsort(-abs(delta[:,2]))[:12];rows=int(np.ceil(len(display)/3));sheet=Image.new('RGB',(840,36+rows*176),'#101b2a');draw=ImageDraw.Draw(sheet)
            draw.text((8,8),f'{label}: original128px crops; crosshair overlay; largest |dZ| displayed; ALL matches retained',fill='white')
            for pos,k in enumerate(display):
                x=(pos%3)*280;y=36+(pos//3)*176
                draw.text((x+2,y),f'pair{k} XY={dist[ok][k]*100:.1f}cm dZ={delta[k,2]*100:+.1f}cm',fill='white')
                for col,(rr,uv,role) in enumerate([(target_raw,uvt[k],'target'),(reference_raw,uvr[k],'reference')]):
                    if not(np.isfinite(uv).all() and 0<=uv[0]<photos[rr].width and 0<=uv[1]<photos[rr].height):continue
                    box=crop_box(uv,photos[rr].size);crop=photos[rr].crop(box);path=out/f'pair{k:03d}_{role}_original_crop.png';crop.save(path)
                    assert np.array_equal(np.asarray(Image.open(path)),np.asarray(photos[rr])[box[1]:box[3],box[0]:box[2]])
                    marked=crop.copy();dd=ImageDraw.Draw(marked);u,v=uv[0]-box[0],uv[1]-box[1];dd.line((u-7,v,u+7,v),fill='#ff665b');dd.line((u,v-7,u,v+7),fill='#ff665b')
                    sheet.paste(marked,(x+col*138,y+20));pair_records[k][role+'_crop']=dict(path=str(path),bbox_xyxy=box,original_pixels_exact=True,sha256=sha(path))
            sheet.save(out/'original_pixel_crop_pairs_contactsheet.png');panels.append(dict(pair=label,path=str(out/'original_pixel_crop_pairs_contactsheet.png'),sha256=sha(out/'original_pixel_crop_pairs_contactsheet.png')))
            (out/'pairs.json').write_text(json.dumps(pair_records,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            np.savez_compressed(out/'exact_pairs.npz',target_cache_rows=ti,reference_cache_rows=ri,target_lidar_rows=lidar[ti],reference_lidar_rows=lidar[ri],target_worldXYZ=xyz[ti],reference_worldXYZ=xyz[ri],delta_worldXYZ=delta,
                XY_distance_m=dist[ok],XY_direction_deg=angle,target_Brown_uv=uvt,reference_Brown_uv=uvr,mutual_nearest=mutual)
            plot_data.append((label,xyz[ti],xyz[ri],delta))
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
        for ax,(label,t,v,d) in zip(axes.flat,plot_data):
            sc=ax.scatter(d[:,0]*100,d[:,1]*100,c=d[:,2]*100,cmap='coolwarm',vmin=-20,vmax=20,s=34,edgecolors='black',linewidths=.3)
            for radius in [2,5,10]:ax.add_patch(plt.Circle((0,0),radius,fill=False,color='grey',lw=.7))
            ax.set(xlim=(-10.5,10.5),ylim=(-10.5,10.5),aspect='equal',title=label,xlabel='target-reference world X (cm)',ylabel='world Y (cm)');ax.grid(alpha=.2)
        fig.colorbar(sc,ax=axes,shrink=.7,label='target-reference Z (cm)');fig.savefig(a.output/'delta_XY_direction_and_height.png',dpi=160);plt.close(fig)
    except ImportError:pass
    summary=dict(schema='R10.height_pairs_original_Brown_photo_diagnostic.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,
        point_source=pin(src),official_cache=pin(cache),actual_input_pins=pins,fixed_target_requested_denominator=1426,source_raw_target_frames=[60,70],reference_frames=[110,145,165],
        fixed_XY_radii_m=[.02,.05,.10],Z_residual_RGB_not_used_for_matching=True,stats=all_stats,point_bank_replay=bank_checks,contact_sheets=panels,
        per_point_semantic_enum=dict(available=bool(semantics),source=pin(semantic_path) if semantics else None,unresolved_values_are_null=True),
        Brown_projection='immutable original world_to_camera and Brown intrinsics, encoded pixel centres cx-.5/cy-.5; no pinhole resize',
        crops_original_encoded_pixels=True,pose_or_geometry_adjusted=False,frame_deletion=False,heightfield_bias_correction=False,
        causal_interpretation='UNRESOLVED until photo/semantic/same-frame layer review: close XY may still be different curb/upper/lower road surfaces. Height difference alone does not identify pose bias.',
        geometry_approval=False,seconds=time.time()-start)
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,summary=pin(a.output/'summary.json'),contact_sheets=panels,epsilon_stats=[{k:x[k] for k in ['target_raw','reference_raw','XY_radius_m','matches','target_minus_reference_Z_m']} for x in all_stats]),ensure_ascii=False))


if __name__=='__main__':main()
