"""CPU tail-risk audit of the fixed depth posterior; never changes the asset."""
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import BSpline
from fit_surface import ART,OLD,bank,sha,stats,write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=str(ART/'surface_fit/masked_depth_v1/tail_audit.json'));args=p.parse_args()
    fit=ART/'surface_fit/masked_depth_v1';post=bank(fit/'posterior.npz')
    old=bank(OLD/'output_adapter/projective_surface/projective_surface_asset.npz');chart=bank(OLD/'output_adapter/projective_surface/reference_chart.npz')
    new=bank(ART/'surface_fit/combined_v1/combined_initial.npz');samples=bank(OLD/'output_adapter/sphere_inputs_hr/raw106/operator_samples.npz')
    assert np.array_equal(chart['pixels'],samples['pixels'])
    allowed=post['chart_query_keep'];delta=post['query_depth_m']-post['prior_depth_m'][allowed]
    assert len(delta)==int((new['representation_kind']==1).sum())
    prior_alpha=samples['alpha'][allowed]
    prior_domain=np.zeros((1280,1920),bool);prior_domain.ravel()[chart['pixels']]=True
    final_domain=np.zeros_like(prior_domain);final_domain.ravel()[chart['pixels'][allowed]]=True
    dprior=distance_transform_edt(prior_domain).ravel()[chart['pixels'][allowed]]
    dfinal=distance_transform_edt(final_domain).ravel()[chart['pixels'][allowed]]
    pitch=float(post['physical_pitch_m']);spacing=float(np.diff(post['u_knots_pixels'])[0])
    def domain(mask):
        return dict(count=int(mask.sum()),fraction_of_chart=float(mask.mean()),delta_z_m=stats(delta[mask]),
            prior_alpha=stats(prior_alpha[mask]),old_chart_boundary_distance_HRpx=stats(dprior[mask]),
            matte_boundary_distance_HRpx=stats(dfinal[mask]),
            prior_alpha_le09=int((prior_alpha[mask]<=.9).sum()),prior_alpha_le05=int((prior_alpha[mask]<=.5).sum()),
            matte_boundary_le1HRpx=int((dfinal[mask]<=1).sum()),matte_boundary_within_control_pitch=int((dfinal[mask]<=spacing).sum()),
            old_chart_boundary_within_control_pitch=int((dprior[mask]<=spacing).sum()))
    original=post['query_original_asset_rows'];matched=old['xyz'][original]
    def boxstats(x,stored,box):
        computed=(abs(x)>box/2).any(1);return dict(rows=len(x),computed_out_of_box=int(computed.sum()),stored_out_of_box=int(stored.sum()),
            stored_matches_computed=bool(np.array_equal(computed,stored)),largest_axis_box_excess_m=float(np.maximum(abs(x)-box/2,0).max()))
    added=new['representation_kind']==1
    # Numerical continuity at every simple interior knot, no fit or parameter changes.
    tu=post['u_knots_pixels'];tv=post['v_knots_pixels'];coef=post['control_delta_z_m'].reshape(len(tu)-4,len(tv)-4)
    uv=chart['pixel_xy'][allowed];mid=np.median(uv,axis=0);epsilon=1e-6;continuity={}
    for axis,t,c,other,lo,hi in [('u',tu,coef,tv,uv[:,0].min(),uv[:,0].max()),('v',tv,coef.T,tu,uv[:,1].min(),uv[:,1].max())]:
        val=BSpline(other,np.eye(len(other)-4),3)(mid[1] if axis=='u' else mid[0]);spline=BSpline(t,c@val,3)
        inner=np.unique(t[(t>lo)&(t<hi)])
        continuity[axis]=dict(interior_simple_knots=len(inner),epsilon_HRpx=epsilon,
            max_two_sided_difference={str(d):float(abs(spline(inner+epsilon,nu=d)-spline(inner-epsilon,nu=d)).max()) for d in (0,1,2)})
    residual=post['accepted_posterior_z_m']-post['accepted_measured_z_m'];worst=np.argsort(abs(residual))[-8:][::-1]
    receipt=dict(status='PASS_CPU_TAIL_AUDIT_WITH_REPORTED_RISKS',geometry_changed=True,physical_control_pitch_m=pitch,control_spacing_HRpx=spacing,
        reference_support_points=len(delta),delta_z_m=stats(delta),
        tail_abs_gt06=domain(abs(delta)>.6),tail_forward_lt_minus06=domain(delta<-.6),tail_back_gt06=domain(delta>.6),
        whole_chart=domain(np.ones(len(delta),bool)),
        out_of_box=dict(original_full=boxstats(old['xyz'],old['out_of_box'],old['box']),
            original_matched=boxstats(matched,old['out_of_box'][original],old['box']),
            posterior_matched=boxstats(new['xyz'],new['out_of_box'],new['box']),
            original_chart_matched=boxstats(matched[added],old['out_of_box'][original][added],old['box']),
            posterior_chart_matched=boxstats(new['xyz'][added],new['out_of_box'][added],new['box'])),
        accepted_anchor_residual_m=stats(abs(residual)),
        largest_anchor_residuals=[dict(sample_id=int(post['accepted_sample_ids'][i]),raw=int(post['accepted_sample_raw'][i]),
            reference_uv=post['accepted_sample_uv'][i].tolist(),measured_z_m=float(post['accepted_measured_z_m'][i]),
            prior_z_m=float(post['accepted_prior_z_m'][i]),posterior_z_m=float(post['accepted_posterior_z_m'][i]),
            signed_residual_m=float(residual[i])) for i in worst],
        spline_C2_continuity=continuity,prior_alpha_source='Frozen sphere native raw106 packet matched exact chart pixel IDs',
        source_sha256=sha(__file__),no_sigma_or_boundary_or_geometry_refit=True,
        limitation='Training-anchor residuals and prior-alpha/boundary correlations do not establish true hidden-surface geometry')
    write(args.output,receipt);print(receipt['tail_abs_gt06']);print(receipt['out_of_box'])

if __name__=='__main__':main()
