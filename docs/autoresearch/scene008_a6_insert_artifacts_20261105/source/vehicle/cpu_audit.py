"""Read-only CPU diagnosis of saved projective-surface assets and native renders.

Writes only derived diagnostics beside this file. No network, torch, GPU,
asset edits, fitting, new candidates, or final image overlays.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import (binary_erosion, distance_transform_edt, label,
                           map_coordinates, maximum_filter)

HERE=Path(__file__).resolve().parent
WS=HERE.parents[2]
DATA=WS/'outputs/scene008_vehicle_appearance/delivery/data'
WORK=WS/'work/a6_appearance_formal/output_adapter/projective_surface'
PREP=DATA/'output_adapter/projective_surface/prepare_receipt.json'
C0=.28209479177387814


def bank(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def stats(a):
    a=np.asarray(a,np.float64);a=a[np.isfinite(a)]
    if not len(a):return dict(count=0)
    q=np.quantile(a,[0,.5,.9,.99,1])
    return dict(count=len(a),min=q[0],median=q[1],p90=q[2],p99=q[3],max=q[4],mean=float(a.mean()))
def rgb8(a):return np.uint8(np.clip(a,0,1)*255+.5)
def rgb(path):return np.asarray(Image.open(path).convert('RGB')).astype(np.float32)/255
def panel(images,labels,box,path):
    x0,y0,x1,y1=box;w=x1-x0;h=y1-y0
    out=Image.new('RGB',(w*len(images),h+30),(237,243,247));draw=ImageDraw.Draw(out)
    for i,(value,title) in enumerate(zip(images,labels)):
        if value.ndim==2:value=np.repeat(value[...,None],3,2)
        out.paste(Image.fromarray(rgb8(value[y0:y1,x0:x1])),(i*w,30));draw.text((i*w+5,8),title,fill=(10,25,38))
    out.save(path)
def scalar_map(v,mask,limit):
    t=np.clip(v/limit,0,1);out=np.stack((t,.15*(1-t),1-t),-1);out[~mask]=0;return out


def main():
    prep=read(PREP);chart=bank(WORK/'reference_chart.npz')
    proto_path=DATA/'output_adapter/projective_surface/projective_surface_asset.npz'
    formal_path=DATA/'output_adapter/projective_surface/formal_observed/formal/complete_a6.npz'
    proto=bank(proto_path);formal=bank(formal_path)
    assert sha(proto_path)==prep['export']['file_sha256']
    assert sha(formal_path)==sha(WORK/'formal_observed/formal/complete_a6.npz')
    cut=prep['retained_source_rows'];M=prep['chart_rows'];xy=chart['pixel_xy'].astype(int);pixels=chart['pixels']
    assert len(proto['xyz'])==cut+M==50322
    candidate=DATA/'inputs/raw106/strong_candidate_v2_512.png'
    hosted=DATA/'hosted_generation/strong_candidate106_v2_512.png'
    assert sha(candidate)==prep['reference_rgb_proposal_sha256']
    proposal=rgb(candidate)
    fg=np.asarray(Image.open(DATA/'inputs/raw106/object_mask_512.png'))>0
    ip=np.asarray(Image.open(DATA/'inputs/raw106/inpaint_mask_512.png'))>0
    uv=chart['generated_xy'];near=np.clip(np.floor(uv+.5).astype(int),[0,0],[511,511])
    known=chart['color_real'];generated=chart['color_generated'];fallback=chart['color_prior_fallback']
    sampled=np.column_stack([map_coordinates(proposal[...,c],[uv[:,1],uv[:,0]],order=1,mode='nearest',prefilter=False) for c in range(3)])
    colour_sampling_error=float(np.abs(sampled[generated]-chart['color_rgb'][generated]).max())
    # Foreground nearest decides eligibility; RGB is a4-tap bilinear sample.
    foreground_coverage=map_coordinates(fg.astype(float),[uv[:,1],uv[:,0]],order=1,mode='nearest',prefilter=False)
    inpaint_coverage=map_coordinates(ip.astype(float),[uv[:,1],uv[:,0]],order=1,mode='nearest',prefilter=False)
    crosses_foreground=generated&(foreground_coverage<1-1e-10)
    crosses_known=generated&(inpaint_coverage<1-1e-10)
    # Black connected to the canvas border may include valid tires. Counts
    # expose ambiguity; no semantic claim that every dark sample is background.
    dark=proposal.max(2)<16/255
    labs,n=label(dark);outside_ids=np.unique(np.r_[labs[0],labs[-1],labs[:,0],labs[:,-1]])
    exterior_dark=np.isin(labs,outside_ids)&dark
    sampled_dark=sampled.max(1)<16/255
    source_black_connected=exterior_dark[near[:,1],near[:,0]]

    H,W=1280,1920;support=np.zeros((H,W),bool);support.ravel()[pixels]=True
    z=np.full((H,W),np.nan);z.ravel()[pixels]=chart['expected_depth_m']
    interior=binary_erosion(support,iterations=6)
    horiz=support[:,1:]&support[:,:-1];vert=support[1:]&support[:-1]
    dx=np.abs(z[:,1:]-z[:,:-1]);dy=np.abs(z[1:]-z[:-1])
    pairs=np.r_[dx[horiz],dy[vert]]
    interior_pairs=np.r_[dx[interior[:,1:]&interior[:,:-1]],dy[interior[1:]&interior[:-1]]]
    gradient=np.zeros((H,W));gradient[:,1:]=np.nan_to_num(dx);gradient[1:]=np.maximum(gradient[1:],np.nan_to_num(dy))
    gradient[~support]=0
    pose=chart['reference_pose'];view=chart['reference_world_view'];full=chart['reference_full_projection']
    model=chart['local_xyz'].astype(float)@pose[:3,:3].T+pose[:3,3]
    hp=np.c_[model,np.ones(M)]@full
    reproj=((hp[:,:2]/hp[:,3,None]+1)*[W,H]-1)/2
    back_error=float(np.linalg.norm(reproj-xy,axis=1).max())
    depth_error=float(np.abs(model@view[:3,2]+view[3,2]-chart['expected_depth_model']).max())

    pc=np.zeros((H,W,3),np.float32);fc=pc.copy()
    pc.reshape(-1,3)[pixels]=proto['rgb'][cut:];fc.reshape(-1,3)[pixels]=formal['rgb'][cut:]
    delta=np.abs(formal['rgb']-proto['rgb']).mean(1)
    new_dark=(formal['rgb'].max(1)<16/255)&(proto['rgb'].max(1)>=16/255)
    colour_groups={}
    for name,mask in dict(real_seed=known,generated_seed=generated,prior_seed=fallback).items():
        colour_groups[name]=dict(rows=int(mask.sum()),formal_seed_RGB_L1_delta=stats(delta[cut:][mask]),
            initially_dark=int((proto['rgb'][cut:][mask].max(1)<16/255).sum()),
            formally_dark=int((formal['rgb'][cut:][mask].max(1)<16/255).sum()),
            newly_dark_after_observed_fit=int(new_dark[cut:][mask].sum()))
    # The formal/prototype render was produced by the real native rasterizer.
    # PNG alpha is16bit, so normalized-preview errors have quantization limits.
    refs={};normalization={}
    for raw in (106,113,130):
        root=WORK/'formal_observed/evaluation'/f'raw{raw:03d}'
        images={}
        for name in ('projective_prototype','projective_surface'):
            image=rgb(root/name/'rgb_black_hr.png')
            alpha=np.asarray(Image.open(root/name/'alpha16_hr.png'),dtype=np.float64)/65535
            straight=image/np.maximum(alpha[...,None],1e-6)
            mask=alpha>.1;core=binary_erosion(mask,iterations=5);edge=mask&~core
            black=image.max(2)<16/255;opaque=alpha>.99
            normalization[f'{raw}_{name}']=dict(
                mean_alpha_core=float(alpha[core].mean()),
                black_pixels_core_opaque=int((black&core&opaque).sum()),
                black_pixels_edge=int((black&edge).sum()),
                premul_vs_normalized_RGB_L1_core=stats(np.abs(image-straight).mean(2)[core]),
                premul_vs_normalized_RGB_L1_edge=stats(np.abs(image-straight).mean(2)[edge]),
                note='Saved8bitRGB/16bitalpha; approximate display audit, not exact raw float reconstruction')
            images[name]=(image,alpha,straight)
        if raw==106:refs=images
    # Detect whether retained source centres project in front of the replacement
    # chart. This cannot prove visibility: source splat footprints and native
    # sort/occlusion still require the separately requested GPU isolation.
    pmodel=proto['xyz'][:cut].astype(float)@pose[:3,:3].T+pose[:3,3]
    php=np.c_[pmodel,np.ones(cut)]@full
    puv=((php[:,:2]/php[:,3,None]+1)*[W,H]-1)/2
    puvnear=np.floor(puv+.5).astype(int);inside=(puvnear[:,0]>=0)&(puvnear[:,0]<W)&(puvnear[:,1]>=0)&(puvnear[:,1]<H)
    inds=np.flatnonzero(inside);pdepth=(pmodel@view[:3,2]+view[3,2])/prep['meter_to_model']
    on_chart=np.zeros(cut,bool);on_chart[inds]=support[puvnear[inds,1],puvnear[inds,0]]
    dz=np.full(cut,np.nan);ii=np.flatnonzero(on_chart);dz[ii]=pdepth[ii]-z[puvnear[ii,1],puvnear[ii,0]]
    # Colourless/coordinate plots are scientific diagnostics, not a final image
    # paste into a street rendering.
    ys,xs=np.where(support);box=(max(0,xs.min()-20),max(0,ys.min()-20),min(W,xs.max()+21),min(H,ys.max()+21))
    p106=refs['projective_prototype'];f106=refs['projective_surface']
    panel([pc,p106[0],np.clip(p106[2],0,1),fc,f106[0]],
          ['106 point RGB / seed','106 native / proto','106 native / alpha','106 point RGB /fit40','106 native /fit40'],box,HERE/'source106_stages.png')
    depth_rgb=scalar_map(z-np.nanquantile(z,.01),support,3.)
    grad_rgb=scalar_map(gradient,support,.10)
    categories=np.zeros((H,W,3));categories.reshape(-1,3)[pixels[known]]=[0,.8,.1]
    categories.reshape(-1,3)[pixels[generated]]=[.1,.35,1]
    categories.reshape(-1,3)[pixels[fallback]]=[1,.2,.1]
    boundary=np.zeros((H,W,3));boundary.reshape(-1,3)[pixels[crosses_foreground]]=[1,0,0]
    boundary.reshape(-1,3)[pixels[crosses_known]]=[1,1,0]
    panel([depth_rgb,grad_rgb,categories,boundary],['prior expected depth /3m','adjacent dz /0.10m','real green /gen blue','bilinear mixed boundary'],box,HERE/'depth_and_sampling.png')
    result=dict(status='PASS_LOCAL_READ_ONLY_AUDIT',network_calls=0,GPU_calls=0,asset_mutations=0,
        actual_input=dict(candidate_path=str(candidate),candidate_sha256=sha(candidate),receipt_sha256=prep['reference_rgb_proposal_sha256'],
            hosted_candidate_same_bytes=sha(candidate)==sha(hosted),
            SD_board_is_a_different_batch=True,SD_board_path=str(DATA/'candidates_raw106/candidate_board.png'),
            candidate_reference_note='The final chart consumed hosted strongv2; the SD candidate board was not this asset input'),
        backprojection=dict(reprojection_HR_px_max=back_error,camera_z_model_roundtrip_maxabs=depth_error,
            conclusion='Transform/pixel-centre units are internally consistent; this does not validate the prior expected-depth surface'),
        depth_surface=dict(adjacent_HR_depth_difference_m=stats(pairs),interior_adjacent_depth_difference_m=stats(interior_pairs),
            interior_adjacent_pairs_gt5cm=int((interior_pairs>.05).sum()),interior_adjacent_pairs_gt10cm=int((interior_pairs>.1).sum()),
            interior_adjacent_pairs_gt25cm=int((interior_pairs>.25).sum()),
            sigma_local_m=stats(chart['sigma_local_m']),
            note='Raw per-pixel expected-depth means include unresolved layer variation; the pipeline does not reconstruct a connected smooth surface'),
        sampling=dict(exact_bilinear_vs_saved_generated_RGB_maxabs=colour_sampling_error,
            generated_rows=int(generated.sum()),generated_rows_with_bilinear_background_mask_tap=int(crosses_foreground.sum()),
            generated_rows_with_bilinear_known_mask_tap=int(crosses_known.sum()),
            foreground_coverage_on_mixed_taps=stats(foreground_coverage[crosses_foreground]),
            generated_sample_dark_rows=int((generated&sampled_dark).sum()),
            generated_nearest_connected_to_black_canvas=int((generated&source_black_connected).sum()),
            dark_foreground_candidate_pixels=int((dark&fg).sum()),
            outside_connected_dark_foreground_pixels=int((exterior_dark&fg).sum()),
            caveat='Dark-source counts include valid tires/seams; segmentation and isolated rendering are needed for attribution'),
        observed_fit=dict(group_changes=colour_groups,retained_prior_RGB_delta=stats(delta[:cut]),
            initial_vs_formal_geometry_hash_equal=str(proto['geometry_hash'])==str(formal['geometry_hash']),
            note='Prototype seed provenance flags remain initial-source flags; final BUILD participation is in formal_observed/inputs.npz unknown'),
        normalization=normalization,
        retained_prior=dict(rows=cut,centres_project_inside_chart=int(on_chart.sum()),
            centre_depth_minus_chart_m=stats(dz[on_chart]),
            centres_in_front_of_chart_by1cm=int((dz<-.01).sum()),
            caveat='Zero contribution in the original sphere reference does not guarantee zero contribution after replacement; isolate2778 rows using the new asset native operator'),
        saved_diagnostics=['source106_stages.png','depth_and_sampling.png'])
    (HERE/'cpu_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
