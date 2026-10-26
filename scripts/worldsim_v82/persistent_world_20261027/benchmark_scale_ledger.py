"""Live, input-only case readiness counts; never turn request counts into results."""
import argparse,collections,datetime,json,time
from pathlib import Path
from resource_control import require_start
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
FAMILIES=['ego_speed','ego_lane_change','ego_direction','vehicle_speed','vehicle_lane_change','vehicle_direction','vehicle_insertion','vehicle_removal']

def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        require_start();rows=[];by_split={};counts={k:dict(materialized=0,input_screened=0,reserved_input_screened=0,normal_space_request=0,formal_admitted=0,completed=0) for k in FAMILIES}
        for path in (ROOT/'benchmark_cases_v1/scenes').glob('*/summary.json'):
            summary=json.loads(path.read_text());rows.append(summary)
            s=by_split.setdefault(summary['split'],dict(contexts=0,source_frames=0,request_cases=0,input_screened_cases=0,formal_admitted=0,completed=0));s['contexts']+=1;s['source_frames']+=49;s['request_cases']+=len(summary['families']);s['input_screened_cases']+=summary['input_screened_cases']
            for case in summary['families']:
                v=counts[case['family']];v['materialized']+=1;v['input_screened']+=case['input_screen_eligible'];v['reserved_input_screened']+=case['input_screen_eligible'] and summary['split']=='heldout_reserved';v['normal_space_request']+=case['input_screen_eligible'] and not case['collision_frames']
        def read(p):return json.loads(p.read_text()) if p.exists() else None
        result=dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference=dict(name='HorizonForge supplied paper and supplement',task_families=8,paper_edit_cases=109,display_edit_cases=32,display_video_files=128,not_unique_scene_count=32),execution_contract=dict(minimum_valid_edits=109,working_target_valid_edits=128,original_long_term_goal=240,task_families=8,case_unit='one scene-target-physical request; not frames, seeds, methods, backend stages or checkpoint versions',same_protocol_comparison_complete=False,missing_capacity_action='replenish input cases; never relabel draft or failed cases as completed'),materialized_contexts=len(rows),by_split=by_split,families=counts,input_screened_cases=sum(x['input_screened_cases'] for x in rows),formal_admitted_cases=0,completed_new_eight_task_cases=0,expansion=read(ROOT/'benchmark_expansion_v1/status.json'),segmentation=read(ROOT/'benchmark_mask_preparation_v1/status.json'),remaining_requirements=['instance masks and target association','native base and asset/background QA','novel camera/time/trajectory and asset-bank adapters','destination support / collisions / unknown disocclusion','original native and same ProPainter outputs','case-level failure ownership and quality review'],no_edit_outputs_used_to_select=True,heldout_method_selection=False,physical_route_independence_audited=False,quality_accepted=False,human_verdict=None)
        folder=ROOT/'benchmark_cases_v1';folder.mkdir(exist_ok=True);tmp=folder/'scale_ledger.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(folder/'scale_ledger.json')
        if not a.watch:print(json.dumps({k:v for k,v in result.items() if k not in ['expansion','segmentation']}));return
        time.sleep(30)

if __name__=='__main__':main()
