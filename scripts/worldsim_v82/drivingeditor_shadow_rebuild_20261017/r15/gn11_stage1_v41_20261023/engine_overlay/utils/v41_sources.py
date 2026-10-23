"""固定多源标签合同；只扩展原R单元，验收来源不参与目标颜色拟合。"""
import numpy as np

EVIDENCE_VERSION = 'V41_FIXED_DENSE_EVIDENCE_R7'
SOURCE_POLICY = dict(version='V41_INCREMENTAL_BLOCKED_CONSENSUS_R7',
    scope='only R5 rejected deletion candidates on unchanged R-active units; preserve every legacy label/color',
    frames=49, block_size=7, validation_blocks=[2,5], boundary_guard_frames=1,
    min_fit_inliers=3, min_fit_fraction=.6, min_fit_span_frames=4,
    min_validation_inliers=2, min_validation_fraction=.6, color_mae_tolerance=.05,
    min_new_component_pixels=9, connectivity=8,
    color='channel median of fit inliers about median of all geometrically valid fit sources',
    validation='fixed disjoint temporal blocks; gate only, never target fitting; target frame always excluded',
    limitation='construction holdout, not independent model test; legacy labels unchanged and may use these frames')


def split_sources(target_index):
    ids=np.arange(49)
    validation=np.isin(ids//7,SOURCE_POLICY['validation_blocks'])
    near=validation.copy()
    near[1:] |= validation[:-1];near[:-1] |= validation[1:]
    fitting=~near
    fitting[target_index]=False;validation[target_index]=False
    return fitting,validation


def median_valid(colors):
    result=np.zeros(colors.shape[1:],np.float32)
    valid=np.isfinite(colors[:,:,0]).any(0)
    if valid.any():result[valid]=np.nanmedian(colors[:,valid],axis=0)
    return result


def source_consensus(colors,target_index):
    """colors[49,N,3], NaN for a geometrically rejected view; no target RGB consumed."""
    colors=np.asarray(colors,np.float32)
    if colors.ndim!=3 or colors.shape[0]!=49 or colors.shape[2]!=3:
        raise ValueError('49 x N x 3 source colors required')
    if np.isfinite(colors[target_index]).any():raise ValueError('target-frame RGB is forbidden')
    fit,val=split_sources(target_index)
    fit_colors=np.where(fit[:,None,None],colors,np.nan)
    valid_fit=np.isfinite(fit_colors[:,:,0])
    center=median_valid(fit_colors)
    inliers=valid_fit & (np.abs(colors-center[None]).mean(2)<=.05)
    target=median_valid(np.where(inliers[:,:,None],colors,np.nan))
    fit_count=valid_fit.sum(0);fit_inliers=inliers.sum(0)
    ids=np.arange(49)[:,None]
    span=np.where(inliers,ids,-1).max(0)-np.where(inliers,ids,49).min(0)
    span=np.maximum(span,0)
    fit_ok=(fit_inliers>=3)&(fit_inliers>=.6*fit_count)&(span>=4)
    val_valid=val[:,None]&np.isfinite(colors[:,:,0])
    val_error=np.abs(colors-target[None]).mean(2)
    agree=val_valid&(val_error<=.05)
    val_count=val_valid.sum(0);val_inliers=agree.sum(0)
    validation_ok=(val_inliers>=2)&(val_inliers>=.6*val_count)
    # Diagnostic mean uses all admissible validation views, not only those agreeing with the fit.
    error_sum=np.where(val_valid,val_error,0).sum(0)
    val_mae=np.divide(error_sum,val_count,out=np.zeros_like(error_sum),where=val_count>0)
    return dict(target=target,fit_source_mask=inliers,validation_source_mask=agree,
        fit_count=fit_count,fit_inliers=fit_inliers,fit_span=span,
        validation_count=val_count,validation_inliers=val_inliers,validation_mae=val_mae,
        fit_ok=fit_ok,validation_ok=validation_ok,accepted=fit_ok&validation_ok)


def first_two(mask):
    ids=np.arange(49)[:,None]
    chosen=np.sort(np.where(mask,ids,49),axis=0)[:2].T
    return np.where(chosen<49,chosen,-1).astype(np.int16)


def validate_label_sources(row,target_index):
    valid=row['deletion_domains'].any(0)
    kinds=row['deletion_label_kind']
    legacy=row['deletion_legacy_valid']
    if kinds.shape!=valid.shape or not np.isin(kinds,[0,1,2]).all():
        raise ValueError('invalid source-kind map')
    if not np.array_equal(kinds>0,valid) or not np.array_equal(kinds==1,legacy):
        raise ValueError('legacy/new source domain mismatch')
    sources=row['deletion_source_ids'][valid]
    if (sources<0).any() or (sources>=49).any() or (sources==target_index).any() or (sources[:,0]==sources[:,1]).any():
        raise ValueError('two distinct other-frame witnesses required')
    new=kinds==2
    if new.any():
        fit,val=split_sources(target_index)
        source=row['deletion_source_ids'][new]; check=row['deletion_validation_source_ids'][new]
        if (check<0).any() or (check>=49).any() or not fit[source].all() or not val[check].all() or (check[:,0]==check[:,1]).any():
            raise ValueError('fitting/validation source leakage')
        if (row['deletion_fit_inliers'][new]<3).any() or (row['deletion_fit_span'][new]<4).any():
            raise ValueError('insufficient fitting support')
        if (row['deletion_validation_inliers'][new]<2).any():raise ValueError('missing validation')
        if (row['deletion_fit_inliers'][new]<.6*row['deletion_fit_count'][new]).any() or (row['deletion_validation_inliers'][new]<.6*row['deletion_validation_count'][new]).any():
            raise ValueError('source consensus fraction failed')
    if row['deletion_target_rgb'][:,~valid].any():raise ValueError('unknown RGB must not become a target')
