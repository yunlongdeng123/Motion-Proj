"""Same-frame surface layers/normals from cached official labels; no TFRecord scan."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_nonplanar_inference_cpu import load,read,write,pin,stats,design


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C,out=a.campaign_root,a.output;out.mkdir(parents=True,exist_ok=False);start=time.time();R=C/'road_underbody_local_20261123_r10'
    tp=R/'audit/expanded_curvature_loo_v1/training_points.npz';train=load(tp);X,raw,rowid=train['world_xyz'],train['source_raw'],train['source_lidar_row_id']
    coeff=load(R/'audit/expanded_curvature_loo_v1/all_training_model_coefficients.npz');centre=train['fit_XY_centre'];b=coeff['plane__coefficients']
    gp=R/'audit/local_surface_expanded_candidate_v1/expanded_candidate_geometry.npz';g=load(gp)
    footprint=g['cell_world_xyz'][g['requested_cell_asset_mask'],:2];tree_scope=cKDTree(footprint)
    grid=load(C/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz');M,O=grid['world_from_uv_xy'][:2,:2],grid['world_from_uv_xy'][:2,3];inv=np.linalg.inv(M)
    ep=R/'audit/official_ground_geometry_v1/point_evidence_contract.json';enum=read(ep)['official_label_enum'];ground=[enum['TYPE_ROAD'],enum['TYPE_LANE_MARKER']]
    rangesp=R/'audit/official_ground_geometry_v1/official_TOP_range_labels.npz';ranges=load(rangesp)
    bank=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
    rows=[];payload=[];pairs_payload={};normals=np.full_like(X,np.nan);normal_valid=np.zeros(len(X),bool);normal_angle=np.full(len(X),np.nan);semantic=np.full(len(X),-1,np.int16)
    fig,axs=plt.subplots(4,2,figsize=(14,13),constrained_layout=True)
    for ax,rr in zip(axs.ravel(),np.unique(raw)):
        op=bank/f'observations/{rr:03d}.json';obs=read(op);lp=bank/obs['lidar_path'];cloud=np.fromfile(lp,dtype='<f4').reshape(-1,6)
        lab=np.full(len(cloud),-1,np.int16)
        for ret in [0,1]:
            key=f'{rr:03d}_return{ret}_labels_instance_semantic'
            if key not in ranges:continue
            labels=ranges[key];H,W=labels.shape[:2];take=(cloud[:,5]>=ret*H*W)&(cloud[:,5]<(ret+1)*H*W)
            encoded=cloud[take,5].astype(np.int64)-ret*H*W;lab[take]=labels.reshape(-1,2)[encoded,1]
        mask=raw==rr;ii=np.flatnonzero(mask);semantic[ii]=lab[rowid[ii]];assert np.isin(semantic[ii],ground).all()
        pose=np.asarray(obs['ego_world']).reshape(4,4);world=cloud[:,:3].astype(float)@pose[:3,:3].T+pose[:3,3]
        assert np.array_equal(world[rowid[ii]],X[ii])
        finite=np.isfinite(world).all(-1);dist=np.full(len(world),np.inf);dist[finite]=tree_scope.query(world[finite,:2])[0]
        take=finite&(dist<=.4)
        ids=np.flatnonzero(take);points=world[take];labs=lab[take];uv=(points[:,:2]-O)@inv.T
        payload.append(dict(world_xyz=points,source_raw=np.full(len(points),rr,np.int16),lidar_row_id=ids.astype(np.int32),semantic=labs,TOP_encoded=cloud[ids,5],globalUV=uv))
        P=X[mask];tree=cKDTree(P[:,:2]);step_rows=[]
        for eps in [.02,.05,.10]:
            pairs=tree.query_pairs(eps,output_type='ndarray')
            if len(pairs):
                dz=P[pairs[:,0],2]-P[pairs[:,1],2];dxy=np.linalg.norm(P[pairs[:,0],:2]-P[pairs[:,1],:2],axis=-1)
            else:dz=np.empty(0);dxy=np.empty(0)
            pairs_payload[f'{rr:03d}_eps{round(eps*100):02d}_training_cache_indices']=ii[pairs]
            pairs_payload[f'{rr:03d}_eps{round(eps*100):02d}_signed_deltaZ_m']=dz
            pairs_payload[f'{rr:03d}_eps{round(eps*100):02d}_distanceXY_m']=dxy
            step_rows.append(dict(max_XY_m=eps,pairs=len(pairs),absolute_deltaZ_m=stats(abs(dz)),pairs_abs_dZ_ge0p08m=int((abs(dz)>=.08).sum())))
        # Same prior normal witness rule, independent of model residual or semantic height.
        d,near=tree.query(P[:,:2],k=min(20,len(P)))
        for local,global_i in enumerate(ii):
            choose=near[local][d[local]<=.5]
            if len(choose)<10:continue
            q=P[choose];cov=np.cov(q-q.mean(0),rowvar=False);eig,V=np.linalg.eigh(cov)
            if eig[0]/max(eig.sum(),1e-12)>.05:continue
            n=V[:,0];n=n if n[2]>=0 else -n;normals[global_i]=n;normal_valid[global_i]=True
            normal_angle[global_i]=np.degrees(np.arccos(np.clip(n[2],-1,1)))
        neighbours={}
        for name in ['TYPE_CURB','TYPE_LANE_MARKER','TYPE_SIDEWALK']:
            G=points[labs==enum[name]]
            distance=cKDTree(G[:,:2]).query(P[:,:2])[0] if len(G) else np.full(len(P),np.inf)
            neighbours[name]=dict(context_points=len(G),training_ROAD_or_LANE_nearest_distanceXY_m=stats(distance),
                training_points_within0p10m=int((distance<=.1).sum()),training_points_within0p50m=int((distance<=.5).sum()))
        res=design(P[:,:2],centre)@b-P[:,2];PUV=(P[:,:2]-O)@inv.T
        sc=ax.scatter(PUV[:,0],PUV[:,1],c=-res,vmin=-.05,vmax=.18,cmap='coolwarm',s=8)
        for cls,color,marker in [('TYPE_CURB','black','x'),('TYPE_LANE_MARKER','lime','+'),('TYPE_SIDEWALK','gold','.')]:
            q=uv[labs==enum[cls]]
            if len(q):ax.scatter(q[:,0],q[:,1],s=7,c=color,marker=marker,label=cls.replace('TYPE_',''))
        ax.set_title(f'raw{rr:03d} road/lane={len(P)}; median ground-minus-plane={np.median(-res):+.3f}m')
        ax.set_xlim(g['bounds_uvxy'][[0,2]]);ax.set_ylim(g['bounds_uvxy'][[1,3]]);ax.set_xlabel('global u (2cm/texel)');ax.set_ylabel('global v');ax.grid(alpha=.2)
        if rr==60:ax.legend(fontsize=7,loc='upper right')
        row=dict(raw=int(rr),training_points=len(P),training_semantic_counts={str(int(k)):int(n) for k,n in zip(*np.unique(semantic[ii],return_counts=True))},
            all_context_class_counts={str(int(k)):int(n) for k,n in zip(*np.unique(labs,return_counts=True))},
            same_frame_road_pair_steps=step_rows,valid_local_PCA_normals=int(normal_valid[ii].sum()),
            local_normal_tilt_from_worldZ_deg=stats(normal_angle[ii]),official_surface_neighbours=neighbours,
            original_bin=pin(lp),observation=pin(op),no_Z_filter=True,no_model_residual_filter=True)
        rows.append(row)
    fig.colorbar(sc,ax=axs.ravel().tolist(),label='Official road/lane Z minus new fitted plane Z (m)',shrink=.6)
    fig.suptitle('Same-frame geometry context; curb/sidewalk labels are separate original official classes')
    fig.savefig(out/'same_frame_surface_layers.png',dpi=135);plt.close(fig)
    np.savez_compressed(out/'context_semantic_points.npz',**{key:np.concatenate([x[key] for x in payload]) for key in payload[0]},
        training_cache_worldXYZ=X,training_cache_raw=raw,training_cache_lidar_row_id=rowid,training_cache_semantic=semantic,
        training_cache_PCA_normal_world=normals,training_cache_PCA_normal_valid=normal_valid,training_cache_normal_tilt_deg=normal_angle)
    np.savez_compressed(out/'same_frame_ROAD_step_pairs.npz',**pairs_payload)
    result=dict(schema='R10.same_frame_surface_layer_and_normal_diagnostic.v1',complete=True,
        decision='DIAGNOSTIC_CAUSE_UNDETERMINED',training_points=pin(tp),candidate_scope=pin(gp),
        official_labels=pin(rangesp),official_enum=pin(ep),rows=rows,
        semantic_mapping='Exact original cloud column6 return/row/column -> saved official range label. No class guessed from image/height.',
        ground_classes=ground,other_classes_shown_not_added_to_fit=True,
        lane_marker_is_official_point_semantic_not_vector_lane_geometry=True,
        no_TFRecord_rescan=True,no_heldout_read=True,no_model_fit_or_asset_update=True,
        context_points=pin(out/'context_semantic_points.npz'),pairs=pin(out/'same_frame_ROAD_step_pairs.npz'),figure=pin(out/'same_frame_surface_layers.png'),
        curb_vs_pose_cause_not_resolved_by_nearest_XY_alone=True,GPU_forward=0,backward=0,updates=0,seconds=time.time()-start)
    write(out/'same_frame_surface_summary.json',result)
    print(json.dumps(dict(complete=True,receipt=pin(out/'same_frame_surface_summary.json'),rows=rows)),flush=True)


if __name__=='__main__':main()
