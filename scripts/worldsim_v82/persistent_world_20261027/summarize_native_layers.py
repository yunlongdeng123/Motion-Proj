"""CPU aggregation, with request-collision strata kept separate."""
import argparse,json
from pathlib import Path
from benchmark_geometry import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);a=p.parse_args();folder=ROOT/'native_layer_isolation_v1'/a.context;d=json.loads((folder/'summary.json').read_text());summaries=[]
    for name in d['cases']:
        case=json.loads((ROOT/'motion_edit_suite_v1'/a.context/(name+'.json')).read_text());lookup={r['raw']:r for r in case['frames']};rows=[r for r in d['rows'] if r['case']==name];strata={}
        for stratum in ['all','no_annotated_vehicle_box_collision','annotated_box_collision_stress']:
            selected=[r for r in rows if r['dynamic_core_pixels'] and (stratum=='all' or bool(lookup[r['raw']]['request_box_collision_tracks'])==(stratum=='annotated_box_collision_stress'))]
            mass=sum(r['isolated_dynamic_alpha_mean']*r['dynamic_core_pixels'] for r in selected);pixels=sum(r['dynamic_core_pixels'] for r in selected)
            strata[stratum]=dict(frames=len(selected),core_frame_pixels=pixels,isolated_alpha_mass=mass,
                                static_layer_visibility_mass_reduction=sum(r['static_visibility_suppressed_alpha_mass']*r['isolated_dynamic_alpha_mean']*r['dynamic_core_pixels'] for r in selected)/mass if mass else None,
                                core_fraction_mixed_visibility_below_half=sum(r['mixed_dynamic_weight_below_half_isolated_fraction']*r['dynamic_core_pixels'] for r in selected)/pixels if pixels else None)
        summaries.append(dict(case=name,request_collision_frames=sum(bool(r['request_box_collision_tracks']) for r in case['frames']),strata=strata))
    result=dict(context=a.context,cases=summaries,diagnostic_domain='isolated dynamic alpha>.1, eroded2px; weighted by isolated alpha mass',
                actual_native_replay_max_difference=d['actual_native_and_saved_composite_replay_max_difference'],master_unchanged=d['master_unchanged'],
                interpretation='Removing the ordinary static layer changes dynamic visibility. This is not a semantic contamination rate; legitimate foreground, glass, and unsupported geometry remain possible.',
                visible_asset_defects_must_be_inspected_separately=True,road_support_or_free_space_certified=False,
                new_training_updates=0,new_edit_case_count=0,quality_accepted=False,human_verdict=None)
    (folder/'cpu_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
