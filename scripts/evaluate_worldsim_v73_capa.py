"""官方CAPA窗口内适配接入米制Actor融合；build TTA与跨日志训练分开报告。"""
import argparse,json,logging
from pathlib import Path
import resource,subprocess,sys,time,traceback
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from motion_proj.worldsim_v73.capa_inputs import build_capa_condition
from motion_proj.worldsim_v73.spatial_queries import ActorSpatialQueryDecoder
from motion_proj.worldsim_v73.surface_seeds import farthest_indices,native_surface_points
from train_worldsim_v73_physical_surface import lidar_patches,evaluate_actor_surface


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--capa-root',type=Path,default=Path('/root/autodl-tmp/external/worldsim_v72/capa'))
    parser.add_argument('--weights',type=Path,default=Path('/root/autodl-tmp/models/eas_vggt/vggt'))
    parser.add_argument('--baseline-results',type=Path)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--role',choices=['all','fit','development'],default='all')
    parser.add_argument('--alignment-anchor-chunk',type=int,default=128,help='exact anchor enumeration chunks; zero restores unchunked official allocation')
    args=parser.parse_args()
    out=Path('/root/autodl-tmp/runs/worldsim_v73/WS-V73-M2-CAPA-01')/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        temporary=out/(name+'.tmp')
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        temporary.replace(out/name)
    started=time.monotonic(); torch.set_num_threads(4); torch.manual_seed(7305)
    logging.basicConfig(level=logging.INFO)
    config=yaml.safe_load((args.capa_root/'config/vggt_lora.yaml').read_text())
    config.update(ckpt_path=str(args.weights),n_steps=args.steps,max_bs_infer=None)
    save('manifest.json',{'task_id':'WS-V73-M2-CAPA-01','run_id':args.run_id,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config':config,'native_run':str(args.native_run),'actor_data':str(args.actor_data),
        'baseline_results':str(args.baseline_results) if args.baseline_results else None,'role':args.role,'seed':7305,
        'method':'official CAPA VGGT LoRA protocol; separate reset per build window; canonical native+LiDAR PCA fusion',
        'adaptation_boundary':'TTA uses build image/sensor inputs for each fit or development window; extra-time evaluation rays never enter CAPA loss; no cross-log weight carry-over',
        'execution_overrides':'preserve 378x672 build resolution instead of official 518px resize; enable upstream aggregator checkpointing; official 10% random frames per adaptation step, all 24 jointly for final inference',
        'alignment_anchor_chunk':args.alignment_anchor_chunk,
        'alignment_execution':'same official point subsampling/seed, all anchors and all residuals; chunk anchor enumeration only, official weighted median and global selection retained',
        'alignment':'official CAPA per-image affine scale/shift estimated from build measurements each step, differs from main shared fixed scale',
        'surface_density':'min(1024,build points)+512 native FPS, fixed 0.06m spacing/8 triangles per patch; native-only possible when build LiDAR absent',
        'source_test_read':False,'external_test_read':False,
        'boundary':'CAPA task adaptation and surface conversion, not an exact paper benchmark reproduction; TTA budget separate from shared training'})
    save('status.json',{'status':'running','phase':'load_official_model'})
    try:
        sys.path.insert(0,str(args.capa_root))
        from capa.protocol import CAPAProtocol
        if args.alignment_anchor_chunk:
            from functools import partial
            from capa.utils import alignment
            from motion_proj.worldsim_v73.capa_alignment import chunked_align_depth_affine
            alignment.align_depth_affine=partial(chunked_align_depth_affine,anchor_chunk=args.alignment_anchor_chunk)
        # 保留官方每10步已有loss日志，便于长窗口训练的真实进展记录。
        logging.getLogger('capa').setLevel(logging.DEBUG)
        protocol=CAPAProtocol(config,torch.device('cuda'))
        # 保留完整图像与原标定；不把官方缩放后的像素误配给原始K。
        protocol.model.preprocess_inputs=lambda rgb:rgb.to('cuda')
        original_trainable=protocol.model.get_trainable_params
        def trainable(lr):
            groups=original_trainable(lr)
            # 上游已有非reentrant checkpoint路径；冻结权重仍需传播至早期LoRA。
            protocol.model._base_model.aggregator.train()
            return groups
        protocol.model.get_trainable_params=trainable
        index=json.loads((args.actor_data/'index.json').read_text())
        entries=[row for row in index['cases'] if args.role=='all' or row['role']==args.role]
        save('cohort.json',entries)
        scenes=torch.load(args.native_run/'build_observations.pt',map_location='cpu',weights_only=False)
        scene_ids={row['scene'] for row in entries}
        scenes=[scene for scene in scenes if scene['scene_id'] in scene_ids]
        decoder=ActorSpatialQueryDecoder().cuda().eval()
        rows=[]; windows=[]
        for scene in scenes:
            tick=time.monotonic()
            rgb,condition,mask,counts=build_capa_condition(scene)
            save('status.json',{'status':'running','phase':'build_window_adaptation','scene':scene['scene_id'],
                'windows_done':len(windows),'window_views':len(rgb),'sparse_condition_pixels':sum(counts)})
            depth=protocol.run(rgb,condition,mask)
            torch.save(depth.cpu(),out/(scene['scene_id']+'_adapted_depth.pt'))
            adapted={name:p.detach().cpu() for name,p in protocol.model.model.named_parameters() if p.requires_grad}
            torch.save(adapted,out/(scene['scene_id']+'_lora.pt'))
            window={'scene':scene['scene_id'],'role':scene['role'],'views':len(rgb),
                'condition_pixels_per_view':counts,'trainable_parameters':sum(p.numel() for p in adapted.values()),
                'adaptation_and_save_s':time.monotonic()-tick,
                'cumulative_peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
            windows.append(window); save('windows.json',windows)
            with torch.no_grad():
                for entry in [row for row in entries if row['scene']==scene['scene_id']]:
                    case=torch.load(args.actor_data/entry['file'],map_location='cpu',weights_only=True)
                    points=case['points_actor_m'].cuda()
                    if case['view_indices']:
                        native,per_view=native_surface_points(depth[case['view_indices']],1.,case['camera_from_actor'].cuda(),
                            case['intrinsics'].cuda(),case['size_lwh_m'].cuda())
                    else:
                        native=points.new_empty(0,3); per_view=[]
                    support=native if len(native) else points
                    if len(support):
                        n=min(len(points),decoder.evidence_queries)
                        evidence=points[torch.linspace(0,max(0,len(points)-1),n,device='cuda').long()]
                        centers=torch.cat([evidence,support[farthest_indices(support,len(decoder.coarse))]])
                        surface=lidar_patches(torch.cat([points,native]),decoder,centers=centers)
                    else:
                        surface={'vertices_actor_m':points.new_empty(0,3),'faces':torch.empty(0,3,dtype=torch.long,device='cuda'),
                            'centers_actor_m':points.new_empty(0,3)}
                    metrics=evaluate_actor_surface(surface,case['rays'])
                    row={'actor':entry,'surface_patches':len(surface['centers_actor_m']),'frames':metrics,
                        'seed_support':{'native_candidates':len(native),'per_view_native_support':per_view,
                            'lidar_fallback':not len(native),'prediction_unavailable':not len(support)},
                        'extra_time_usage':'evaluation_only','build_usage':'window_specific_TTA_input'}
                    rows.append(row)
                    torch.save({key:value.cpu() for key,value in surface.items()},out/(entry['scene']+'__'+entry['owner']+'_surface.pt'))
                    save('status.json',{'status':'running','phase':'canonical_surface_evaluation','actors_done':len(rows),
                        'scene':entry['scene'],'owner':entry['owner'],'native_candidates':len(native)})
                    print(json.dumps({'scene':entry['scene'],'owner':entry['owner'],'surface_patches':row['surface_patches']}),flush=True)
                    del case,points,native,support,surface
            del depth,rgb,condition,mask,adapted
        result={'status':'done','final':rows,'windows':windows,'fit_label_times':'build',
            'actors':len(rows),'wall_s':time.monotonic()-started,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'boundary':'existing window population; per-window build-only TTA including development inputs; extra times evaluation only; no new-source confirmation'}
        if args.baseline_results:
            identities={(row['scene'],row['owner']) for row in entries}
            result['baseline']=[row for row in json.loads(args.baseline_results.read_text())
                if (row['actor']['scene'],row['actor']['owner']) in identities]
        save('summary.json',result); save('status.json',{'status':'done'})
    except Exception as exc:
        save('status.json',{'status':'blocked','exception':type(exc).__name__,'message':str(exc),
            'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
        (out/'traceback.txt').write_text(traceback.format_exc()); raise


if __name__=='__main__': main()
