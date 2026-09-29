"""保存A轨迹和protected监督来源；不把隐藏GT偷偷加入模型条件。"""
import argparse
from pathlib import Path
from geometry_factory import read,dump

def main(root,out):
    sources={c['source_id']:c for c in read(root/'source_manifest.json')['clips']}
    for r in read(out/'synthetic_manifest.json')['clips']:
        cid=r['case_id'];dest=root/'synthetic'/cid;c=sources[r['source_id']];d=sources[r['donor_source_id']]
        dump(dest/'target.json',{'synthetic_actor_id':f'{cid}/A','donor_source_id':r['donor_source_id'],'donor_instance':r['donor_instance'],'donor_scene':r['donor_scene'],'placement_mode':r.get('placement_mode','world_offset'),
          'frames':[{'frame':i,'receiver_timestamp_us':f['timestamp'],'donor_timestamp_us':d['frames'][i]['timestamp'],'box_3d_world':p['actor'],'projection_xyxy':p['box'],'alpha':f'alpha/{i:03}.png','influence':f'influence/{i:03}.png','delete_mask':f'model_hole/{i:03}.png'} for i,(f,p) in enumerate(zip(c['frames'],r['frames']))]})
        actors=[]
        for tok in r['occlusion_fraction']:
            frames=[]
            for i,f in enumerate(c['frames']):
                a=next(a for a in f['actors'] if a['instance_token']==tok)
                frames.append({'frame':i,'timestamp_us':f['timestamp'],'box_3d_world':{k:a[k] for k in ['translation','rotation','size']},'full_mask_supervision_only':f'protected/{i:03}_{tok}.png',
                  'visible_evidence':{'rgb':f'X/{i:03}.png','visible_mask_expression':'protected_mask AND NOT model_hole','model_hole':f'model_hole/{i:03}.png','visible_fraction':r['pixel_metrics'][i]['remaining_protected_fraction'][tok],'reference_hidden_GT_used':False}})
            actors.append({'instance_token':tok,'scene':c['scene'],'camera':c['camera'],'frames':frames,'other_camera_references':'not_collected_in_this_pilot','use_of_full_state':'supervision_and_quality_evaluation_only; no additional network input channel'})
        dump(dest/'protected_actors.json',{'actors':actors,'architecture_changed':False,'SV3D_prior_generated':False})
        dump(dest/'sample.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r1','case_id':cid,'type':r['type'],'receiver_scene':c['scene'],'donor_scene':d['scene'],'receiver_source_id':c['source_id'],'donor_source_id':d['source_id'],
          'frame_count':len(c['frames']),'preview_fps':10,'actual_timestamps_us':[f['timestamp'] for f in c['frames']],'receiver_window':c.get('window_provenance'),'donor_window':d.get('window_provenance'),
          'GT':'Y/*.png = deterministic resize/decode of original nuScenes JPEG; never generated','input':'X/*.png','condition_recipe':'normalize X to [-1,1], set model_hole pixels to zero; no hidden Y input','condition_preview_is_model_output':False,
          'edge_mode':r['edge_mode'],'extra_blur_sigma_px':r['extra_motion_blur_px'],'human_verdict':None,'training_ready':False,'split_policy':'receiver AND donor scenes must both respect split; all current sources are train'} )
    print('PAIR_RECORDS',len(read(out/'synthetic_manifest.json')['clips']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
