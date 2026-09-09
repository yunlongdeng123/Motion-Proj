"""共享原生几何头/空间查询跨日志训练，开发日志仅评价，冻结前缀驻留CPU。"""
import argparse
import json
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))
from motion_proj.worldsim_v73.native_pyramid import NativeGeometryPyramid
from motion_proj.worldsim_v73.native_data import sample_depth
from motion_proj.worldsim_v73.spatial_queries import ActorSpatialQueryDecoder
from motion_proj.worldsim_v73.surface_seeds import farthest_indices,native_surface_seeds,native_surface_points
from motion_proj.worldsim_v73.surface_readout import closest_surface_points,first_triangle_intersection,direct_free_space_loss
from train_worldsim_v73_physical_surface import lidar_patches,evaluate_actor_surface


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--actor-data',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--mode',choices=['joint','pointwise','lidar_only','native_only'],default='joint')
    parser.add_argument('--query-surface',choices=['patches','shared_mesh'],default='patches')
    parser.add_argument('--mesh-level',type=int,default=3)
    parser.add_argument('--upper-lora',action='store_true',help='重算完整窗口18–23组qkv LoRA；默认保留旧冻结聚合器路径')
    parser.add_argument('--upper-rank',type=int,default=8)
    parser.add_argument('--upper-alpha',type=float,default=8.)
    parser.add_argument('--upper-weights',type=Path,default=Path('/root/autodl-tmp/models/eas_vggt/vggt/model.safetensors'))
    parser.add_argument('--completion-init',choices=['native_surface','lidar_surface'],default='native_surface')
    parser.add_argument('--free-weight',type=float,default=.5)
    parser.add_argument('--free-mode',choices=['range','beam_tube','beam_tube_range'],default='range')
    parser.add_argument('--free-width-m',type=float,default=.03)
    parser.add_argument('--free-resolution',type=int,default=32)
    parser.add_argument('--event-weight',type=float,default=0.)
    parser.add_argument('--event-sigma-m',type=float,default=.2)
    parser.add_argument('--event-cap',type=float,default=28.)
    parser.add_argument('--event-width-m',type=float,default=.03)
    parser.add_argument('--event-resolution',type=int,default=32)
    parser.add_argument('--native-data-weight',type=float,default=0.)
    parser.add_argument('--fit-label-times',choices=['build','all_window'],default='build')
    parser.add_argument('--fit-targets',type=Path,help='独立的fit全轨迹标签目录；只替换fit损失目标，不替换模型输入')
    parser.add_argument('--baseline-results',type=Path,help='复用同一cohort与固定patch算子的既有LiDAR PCA结果')
    parser.add_argument('--initial-results',type=Path,help='输入、模型初始化与seed均相同时复用既有initial表面评价')
    parser.add_argument('--resume-from',type=Path,help='恢复已保存epoch的模型/优化器；写入新的run目录并保留中断现场')
    parser.add_argument('--include-visual-only',action='store_true',
                        help='单独输入cohort实验：纳入零build LiDAR但有相机位姿的Actor；旧对照默认关闭')
    args=parser.parse_args()
    if args.upper_lora and args.mode=='lidar_only':
        parser.error('upper LoRA需要实际视觉几何通路，不能用于lidar_only')
    if args.upper_lora and args.initial_results:
        parser.error('上层适配应评价其实际初始化，不能复用旧最终token的initial')
    if args.query_surface=='shared_mesh' and (args.mode=='native_only' or args.initial_results):
        raise ValueError('共享网格须训练Query并重新评价自身初始化；不能复用旧patch initial')
    if args.include_visual_only and args.initial_results:
        raise ValueError('visual-only训练必须重新评价含新增输入条件的initial，不能复用旧cohort预测')
    label_times='full_track' if args.fit_targets else args.fit_label_times
    task='WS-V73-Q-V2-01' if args.query_surface=='shared_mesh' else 'WS-V73-M2-GLOBAL-ACTOR-01'
    out=Path('/root/autodl-tmp/runs/worldsim_v73')/task/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        tmp=out/(name+'.tmp')
        tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        tmp.replace(out/name)
    started=time.monotonic()
    torch.set_num_threads(6); torch.manual_seed(7304); random.seed(7304)
    save('manifest.json',{'task_id':task,'run_id':args.run_id,'seed':7304,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'shared_parameters':('native DPT, query decoder and upper qkv LoRA across fit logs; no development gradient or optimizer update' if args.upper_lora else
                             'native DPT and query decoder across fit logs; no development gradient or optimizer update'),
        'frozen_prefix':('none used by lidar_only' if args.mode=='lidar_only' else
            'CPU layers 4/11/17; full-window 18-23 recomputed before actor view selection' if args.upper_lora else
            'aggregator 24-view joint tokens, stored CPU; upper aggregation not adapted'),
        'upper_adaptation':('qkv LoRA groups 18-23; shared across FIT logs, never development updates; one full-window forward per actor optimizer step' if args.upper_lora else 'none'),
        'native_data_boundary':'current build Actor LiDAR at calibrated camera pixels, independent of predicted in-box support; no frozen-depth target',
        'surface_supervision_times':label_times,
        'query_surface':args.query_surface,
        'surface_boundary':('shared-vertex ellipsoid deformation with build/native evidence queries; fixed genus-zero topology is a prior, not observed occupancy or a guarantee against self-intersection; identical explicit triangles in training and evaluation'
                            if args.query_surface=='shared_mesh' else 'independent 3x3 patches, no interpatch shared vertices'),
        'input_boundary':'all predictions read build points/images only; extra-time labels may be used only in fit losses, never development updates',
        'visual_only_boundary':'opt-in admission depends only on zero build LiDAR and available calibrated camera poses, never predicted support or target quality; no-camera empty inputs remain unavailable; unobserved regions are not free space',
        'event_boundary':'optional capped geometry-first-surface NLL on owned subset of the same sampled original beams; all owned misses included at cap, direct coverage/free retained; no opacity or target-selected visibility',
        'native_only_boundary':'native_only adapts full original DPT with canonical native+LiDAR PCA fusion; query module supplies fixed patch definition only and is frozen; no-camera/no-gradient presentations are recorded without optimizer step',
        'source_test_read':False,'external_test_read':False,'failure_ledger_refs':['V73-F01','V73-F02','V73-F03','V73-F04','V73-F05','V73-F06']})
    save('status.json',{'status':'running','phase':'load_contexts'})
    try:
        index=json.loads((args.actor_data/'index.json').read_text())
        save('cohort.json',index['cases'])
        cases=[]; unsupported=[]
        for row in index['cases']:
            if 'file' not in row: continue
            case=torch.load(args.actor_data/row['file'],map_location='cpu',weights_only=True)
            case['metadata']['input_status']=row['status']
            visual_only=(args.include_visual_only and args.mode!='lidar_only'
                         and not len(case['points_actor_m']) and bool(case['view_indices']))
            case['metadata']['visual_only_prediction_enabled']=visual_only
            (cases if row['status']=='ready' or visual_only else unsupported).append(case)
        scenes={s['scene_id']:s for s in torch.load(args.native_run/'build_observations.pt',weights_only=False,map_location='cpu',mmap=True)}
        scales=json.loads((args.native_run/'metric_scales.json').read_text())
        pyramids={}; head=None; prefix_cache={}
        upper=None
        if args.upper_lora:
            from motion_proj.worldsim_v73.upper_aggregation import VGGTUpperTail,make_upper_pyramid
            upper=VGGTUpperTail(args.upper_weights,rank=args.upper_rank,alpha=args.upper_alpha).cuda()
            save('upper_adaptation.json',{'config':upper.config,
                'trainable_parameters':sum(p.numel() for p in upper.parameters() if p.requires_grad),
                'frozen_parameters':sum(p.numel() for p in upper.parameters() if not p.requires_grad),
                'input_boundary':'whole window in original order before actor DPT view selection; no cross-step adapted cache'})
        if args.mode!='lidar_only':
            for case in cases:
                if not case['view_indices']:
                    case['native_observations']=[]
                    continue
                owner=case['metadata']['owner']; scene=scenes[case['metadata']['scene']]
                key=(scene['scene_id'],owner)
                if upper is not None:
                    pyramid=make_upper_pyramid(args.native_run,scene,case['view_indices'],head,upper,prefix_cache)
                else:
                    pyramid=NativeGeometryPyramid(args.native_run,scene,case['view_indices'],head=head,
                        token_device='cpu',prefix_cache=prefix_cache)
                head=pyramid.head; pyramids[key]=pyramid
                observations=[]
                for i in case['view_indices']:
                    view=scene['views'][i]
                    owned=torch.tensor([identity==owner for identity in view['owners']],dtype=torch.bool)
                    observations.append({'uv':view['uv'][owned],'z_m':view['z_m'][owned]})
                case['native_observations']=observations
        del scenes
        prefix_views=(sum(len(prefix.images) for prefix in prefix_cache.values()) if upper is not None else len(prefix_cache))
        torch.manual_seed(7304)
        if args.query_surface=='shared_mesh':
            from motion_proj.worldsim_v73.shared_mesh_queries import ActorSharedMeshQueryDecoder
            decoder=ActorSharedMeshQueryDecoder(mesh_level=args.mesh_level).cuda()
        else:
            decoder=ActorSpatialQueryDecoder().cuda()
        if args.mode=='native_only': decoder.requires_grad_(False)
        tube_free=None
        event_objective=None
        if args.free_mode in ['beam_tube','beam_tube_range']:
            from motion_proj.worldsim_v73.surface_visibility import BeamTubeFreeSpaceLoss
            tube_free=BeamTubeFreeSpaceLoss(width_m=args.free_width_m,resolution=args.free_resolution,
                penalty='range' if args.free_mode=='beam_tube_range' else 'coverage')
        if args.event_weight:
            from motion_proj.worldsim_v73.first_event import FirstSurfaceEventLoss
            event_objective=FirstSurfaceEventLoss(width_m=args.event_width_m,resolution=args.event_resolution,
                                                 sigma_m=args.event_sigma_m,cap=args.event_cap)
        parameters=[*(head.parameters() if head is not None else []),
                    *(decoder.parameters() if args.mode!='native_only' else [])]
        if upper is not None: parameters.extend(p for p in upper.parameters() if p.requires_grad)
        optimizer=torch.optim.AdamW(parameters,lr=1e-5)
        resumed=None; start_epoch=0
        if args.resume_from:
            resumed=torch.load(args.resume_from/'latest.pt',map_location='cpu',weights_only=True,mmap=True)
            if bool(resumed['config'].get('include_visual_only',False))!=args.include_visual_only:
                raise ValueError('不能通过resume静默改变输入cohort；visual-only实验应从原M1初始化')
            if resumed['config'].get('query_surface','patches')!=args.query_surface:
                raise ValueError('不能把旧patch checkpoint作为共享网格的resume')
            if bool(resumed['config'].get('upper_lora',False))!=args.upper_lora:
                raise ValueError('resume不能改变上层适配通路，应另行登记新实验')
            if upper is not None:
                if any(resumed['upper_config'][key]!=upper.config[key] for key in ['rank','alpha','targets','weights_file']):
                    raise ValueError('resume必须保留LoRA定义与原始权重来源')
                upper.load_adapter_state_dict(resumed['upper_adapter'])
            if head is not None: head.load_state_dict(resumed['depth_head'])
            decoder.load_state_dict(resumed['query_decoder'])
            optimizer.load_state_dict(resumed['optimizer'])
            start_epoch=int(resumed['epoch'])
            save('resume.json',{'parent_run':str(args.resume_from),'completed_epochs':start_epoch,
                'state':'model and optimizer restored; parent incomplete epoch discarded, original logs untouched',
                'random_state':'restored' if 'random_states' in resumed else 'legacy checkpoint has no RNG state; explicit seed7304 restart, not bitwise continuation',
                'native_change_reference':'resumed epoch checkpoint, not original pretrained initialization'})
        fit=[c for c in cases if c['metadata']['role']=='fit']
        if not fit: raise ValueError('没有可训练fit Actor')
        for case in cases:
            case['lidar_seed']=(case['points_actor_m'][farthest_indices(case['points_actor_m'],len(decoder.coarse))]
                                if len(case['points_actor_m']) else None)
            labels=[r for r in case['rays'] if r['role']=='build' or
                    (case['metadata']['role']=='fit' and args.fit_label_times=='all_window')]
            case['training_rays']=labels
            case['training_points']=(torch.unique(torch.cat([r['points_actor_m'][r['positive_actor']] for r in labels]),dim=0)
                if labels and case['metadata']['role']=='fit' and args.fit_label_times=='all_window' else case['points_actor_m'])
            if args.fit_targets and case['metadata']['role']=='fit':
                target_file=args.fit_targets/(case['metadata']['scene']+'__'+case['metadata']['owner']+'.pt')
                target=torch.load(target_file,map_location='cpu',weights_only=True)
                labels=target['target_rays']
                case['training_rays']=labels
                case['training_points']=target['target_points_actor_m']
            case['extra_time_target_points']=sum(r['owned_points'] for r in labels if r['role']!='build')
        before=head.projects[0].weight.detach().clone() if head is not None else None

        def predict(case,depth_cache=None):
            points=case['points_actor_m'].cuda(); size=case['size_lwh_m'].cuda()
            matrices=case['camera_from_actor'].cuda(); calibration=case['intrinsics'].cuda()
            rect=case.get('valid_image_rect_xyxy')
            camera_weights=case.get('camera_embedding_weights')
            depth=None
            has_views=head is not None and bool(case['view_indices'])
            if has_views and args.mode=='native_only':
                features=None
                depth=pyramids[(case['metadata']['scene'],case['metadata']['owner'])].depth_only(depth_cache)
            elif has_views and (args.completion_init=='native_surface' or args.native_data_weight>0):
                features,depth=pyramids[(case['metadata']['scene'],case['metadata']['owner'])](include_depth=True)
            else:
                features=pyramids[(case['metadata']['scene'],case['metadata']['owner'])]() if has_views else None
            if args.mode=='native_only':
                native,per_view=(native_surface_points(depth,scales[case['metadata']['scene']],matrices,calibration,size,rect)
                                 if depth is not None else (points.new_empty(0,3),[]))
                source=native if len(native) else points
                n=min(len(points),decoder.evidence_queries)
                evidence=points[torch.linspace(0,len(points)-1,n,device='cuda').long()]
                if len(source):
                    centers=torch.cat([evidence,source[farthest_indices(source,len(decoder.coarse))]])
                    result=lidar_patches(torch.cat([points,native]),decoder,centers=centers)
                else:
                    result={'vertices_actor_m':points.new_empty(0,3),
                            'faces':torch.empty(0,3,device='cuda',dtype=torch.long),'centers_actor_m':points.new_empty(0,3)}
                support={'native_candidates':len(native),'per_view_native_support':per_view,
                         'lidar_fallback':not len(native),'initialization':'native_lidar_pca',
                         'surface_gradient':'selected native center positions; per-step PCA neighborhood/orientation held fixed'}
                if not len(source):
                    support.update(lidar_fallback=False,prediction_unavailable=True,reason='no_native_or_lidar_surface_support')
            elif args.completion_init=='native_surface' and depth is not None:
                seeds,support=native_surface_seeds(depth,scales[case['metadata']['scene']],matrices,calibration,
                                                    size,points,len(decoder.coarse),valid_image_rect=rect)
            else:
                seeds=case['lidar_seed'].cuda() if case['lidar_seed'] is not None else None
                support={'lidar_fallback':args.completion_init=='native_surface' and len(points)>0,
                    'coarse_fallback':seeds is None,
                    'native_candidates':0 if args.completion_init=='native_surface' else None,'initialization':'lidar_surface'}
            if args.mode!='native_only':
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    result=decoder(points,size,features,matrices,calibration,case['image_hw'],case['camera_ids'].cuda(),
                        case['time_offsets_s'].cuda(),use_spatial=args.mode!='pointwise',use_visual=features is not None,
                        completion_seeds=seeds,camera_weights=camera_weights,valid_image_rect=rect)
            # 数据梯度不经过预测框内候选筛选，候选消失时仍可拉回原生几何。
            native_loss=points.sum()*0
            native_count=0
            if depth is not None and args.native_data_weight>0:
                terms=[]
                for image,observation in zip(depth,case['native_observations']):
                    if not len(observation['uv']): continue
                    predicted=sample_depth(image,observation['uv'].cuda())*scales[case['metadata']['scene']]
                    target=observation['z_m'].cuda()
                    terms.append(torch.nn.functional.smooth_l1_loss(predicted,target,beta=.2,reduction='sum'))
                    native_count+=len(target)
                if terms: native_loss=torch.stack(terms).sum()/native_count
            support['native_observed_points']=native_count
            support['surface_parameterization']=args.query_surface if args.mode!='native_only' else 'native_lidar_pca'
            support['surface_vertices']=len(result['vertices_actor_m'])
            support['surface_faces']=len(result['faces'])
            support['evidence_context_queries']=len(result.get('context_actor_m',[]))
            support['has_actor_camera_pose']=has_views
            support['input_path']='visual_only' if case['metadata']['visual_only_prediction_enabled'] else 'existing_multimodal_or_lidar'
            support['fallback_reason']=('no_actor_camera_pose' if not has_views else 'predicted_native_support_empty') if support.get('lidar_fallback') else None
            return result,support,native_loss

        @torch.no_grad()
        def evaluate(tag,baseline=False):
            rows=[]; depth_cache={}
            for case in [*cases,*unsupported]:
                if (case['metadata']['input_status']!='ready'
                    and (baseline or not case['metadata']['visual_only_prediction_enabled'])):
                    # 缺少可用输入的Actor明确输出缺失，仍计入真实束miss/覆盖评价。
                    surface={'vertices_actor_m':torch.empty(0,3,device='cuda'),
                        'faces':torch.empty(0,3,dtype=torch.long,device='cuda'),
                        'centers_actor_m':torch.empty(0,3,device='cuda')}
                    support={'prediction_unavailable':True,'reason':case['metadata'].get('reason')}
                elif baseline:
                    surface=lidar_patches(case['points_actor_m'].cuda(),decoder)
                    support={}
                else:
                    surface,support,native_loss=predict(case,depth_cache=depth_cache)
                    support['native_sensor_huber_m']=native_loss.item()
                metrics=evaluate_actor_surface(surface,case['rays'])
                rows.append({'actor':case['metadata'],'surface_patches':len(surface['centers_actor_m']),
                             'surface_parameterization':'lidar_pca_patches' if baseline else support.get('surface_parameterization',args.query_surface),
                             'surface_vertices':len(surface['vertices_actor_m']),'surface_faces':len(surface['faces']),
                             'surface_patches_boundary':'legacy center count; shared_mesh counts vertices, not independent patches',
                             'seed_support':support,'frames':metrics,
                             'extra_time_usage':'training_labels' if case['metadata']['role']=='fit' and (case['metadata']['input_status']=='ready' or case['metadata']['visual_only_prediction_enabled']) and label_times!='build' else 'evaluation_only'})
                if tag=='final':
                    torch.save({k:v.cpu() for k,v in surface.items() if isinstance(v,torch.Tensor)},
                               out/(case['metadata']['owner']+'_surface.pt'))
                del surface
            save(tag+'.json',rows)
            return rows

        save('status.json',{'status':'running','phase':'initial_evaluation','fit_actors':len(fit),'all_actors':len(cases),
            'unsupported_input_actors':len(unsupported),'shared_frozen_prefix_views':prefix_views})
        if args.baseline_results:
            baseline=json.loads(args.baseline_results.read_text())
            save('lidar_baseline',baseline)
        else:
            baseline=evaluate('lidar_baseline',True)
        if args.initial_results:
            initial=json.loads(args.initial_results.read_text())
            for row in initial:
                row['extra_time_usage']='training_labels' if row['actor']['role']=='fit' and row['actor'].get('input_status',row['actor'].get('status'))=='ready' and label_times!='build' else 'evaluation_only'
            save('initial',initial)
        else:
            initial=evaluate('initial')
        history=[]
        if resumed is not None:
            history=[json.loads(line) for line in (args.resume_from/'train.jsonl').read_text().splitlines()]
            history=[row for row in history if row['epoch']<=start_epoch]
            for row in history: row.setdefault('optimizer_step',True)
            (out/'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in history))
            if 'fit_order' in resumed:
                by_owner={case['metadata']['owner']:case for case in fit}
                fit=[by_owner[owner] for owner in resumed['fit_order']]
            else:
                # Legacy Python shuffle order is reproducible; CUDA sample RNG was not saved.
                random.seed(7304)
                for _ in range(start_epoch): random.shuffle(fit)
            if 'random_states' in resumed:
                state=resumed['random_states']; random.setstate(state['python'])
                torch.set_rng_state(state['torch_cpu']); torch.cuda.set_rng_state_all(state['torch_cuda'])
            else:
                torch.manual_seed(7304)
            del resumed
        resumed_steps=len(history)
        for epoch in range(start_epoch,args.epochs):
            random.shuffle(fit)
            for case in fit:
                tick=time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                prediction,support,native_loss=predict(case)
                vertices=prediction['vertices_actor_m'].float(); faces=prediction['faces']
                points=case['training_points'].cuda()
                chosen=points[torch.randperm(len(points),device='cuda')[:1024]]
                coverage_available=bool(len(chosen) and len(faces))
                nearest=closest_surface_points(vertices,faces,chosen) if coverage_available else None
                coverage=(nearest-chosen).norm(dim=-1).mean() if coverage_available else vertices.sum()*0
                build=case['training_rays']
                ranges=torch.cat([r['observed_first_range_m'] for r in build]).cuda() if build else vertices.new_empty(0)
                origins=torch.cat([r['origins_actor_m'] for r in build]).cuda() if build else vertices.new_empty(0,3)
                directions=torch.cat([r['directions_actor'] for r in build]).cuda() if build else vertices.new_empty(0,3)
                ids=torch.randperm(len(ranges),device='cuda')[:512]
                if tube_free is None:
                    depth,_=first_triangle_intersection(vertices,faces,origins[ids],directions[ids])
                    free=direct_free_space_loss(depth,ranges[ids])
                    physical_free=free
                else:
                    free=tube_free(vertices,faces,origins[ids],directions[ids],ranges[ids]) if len(faces) else vertices.sum()*0
                    # 硬首交点只作同语义读数；有限宽度覆盖比例不能冒充米制侵入。
                    with torch.no_grad():
                        depth,_=first_triangle_intersection(vertices,faces,origins[ids],directions[ids])
                        physical_free=direct_free_space_loss(depth,ranges[ids])
                envelope=((vertices.abs()-case['size_lwh_m'].cuda()/2-.25).clamp_min(0).square().mean()
                          if len(vertices) else vertices.sum()*0)
                event=vertices.sum()*0; event_statistics={}
                if event_objective is not None:
                    owned=(torch.cat([r['positive_actor'] for r in build]).cuda()[ids] if build else
                           torch.empty(0,device='cuda',dtype=torch.bool))
                    selected=ids[owned]
                    event,event_statistics=event_objective(vertices,faces,origins[selected],directions[selected],ranges[selected])
                    event_statistics['literal_owned_miss_rays']=(~torch.isfinite(depth[owned])).sum().item()
                    del owned,selected
                loss=coverage+args.free_weight*free+.05*envelope+args.native_data_weight*native_loss+args.event_weight*event
                # 空标签不是零几何误差；没有实际测量目标时不靠框正则做一次假数据更新。
                has_supervision=(len(chosen)>0 or
                    (args.free_weight>0 and len(ids)>0) or
                    (args.native_data_weight>0 and support['native_observed_points']>0) or
                    (args.event_weight>0 and event_statistics.get('count',0)>0))
                has_gradient=bool(has_supervision and loss.requires_grad)
                if has_gradient: loss.backward()
                group_gradients={}
                for name,module in [('native_dpt',head),('query_decoder',decoder),*([('upper_lora',upper)] if upper is not None else [])]:
                    norms=[p.grad.detach().norm() for p in module.parameters() if p.grad is not None] if module is not None else []
                    group_gradients[name]=torch.stack(norms).norm().item() if norms else 0.
                grad=torch.nn.utils.clip_grad_norm_(parameters,1.)
                if not torch.isfinite(grad): raise FloatingPointError('共享几何训练出现非有限梯度')
                output_grad=sum(p.grad.norm().item() for p in head.scratch.output_conv2.parameters() if p.grad is not None) if head is not None else None
                if has_gradient: optimizer.step()
                row={'epoch':epoch+1,'scene':case['metadata']['scene'],'owner':case['metadata']['owner'],
                    'loss':loss.item(),'coverage_m':coverage.item() if coverage_available else None,'free_intrusion_m':physical_free.item(),
                    'optimizer_step':has_gradient,
                    'skip_reason':None if has_gradient else ('no_observed_supervision' if not has_supervision else
                        ('no_surface_support' if not len(faces) else 'no_surface_gradient')),
                    'input_path':support['input_path'],'coarse_fallback':support.get('coarse_fallback',False),
                    'training_target_points':len(points),'sampled_free_rays':len(ids),'surface_faces':len(faces),
                    'coverage_unavailable_reason':None if coverage_available else ('no_target_points' if not len(chosen) else 'no_surface_support'),
                    'free_objective':free.item(),'free_mode':args.free_mode,
                    'free_objective_unit':'coverage_fraction' if args.free_mode=='beam_tube' else 'm',
                    'event_capped_nll':event.item(),'event_weight':args.event_weight,'event_statistics':event_statistics,
                    'gradient_norm_before_clip':grad.item(),'native_output_gradient_after_clip':output_grad,
                    'group_gradient_norms_before_clip':group_gradients,
                    'native_sensor_huber_m':native_loss.item(),'native_observed_points':support['native_observed_points'],
                    'native_candidates':support.get('native_candidates'),
                    'fit_label_times':label_times,'extra_time_target_points':case['extra_time_target_points'],
                    'lidar_fallback':support.get('lidar_fallback',False),'views':len(case['view_indices']),
                    'fallback_reason':support['fallback_reason'],
                    'query_count':len(prediction['centers_actor_m']),'step_s':time.monotonic()-tick,
                    'evidence_context_queries':support['evidence_context_queries'],
                    'query_surface':args.query_surface,'surface_vertices':len(vertices),
                    'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30}
                if upper is not None:
                    key=(case['metadata']['scene'],case['metadata']['owner'])
                    row['upper_context_views']=len(pyramids[key].prefix.images) if key in pyramids else 0
                history.append(row)
                with (out/'train.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
                save('status.json',{'status':'running','phase':'shared_train','elapsed_s':time.monotonic()-started,**row})
                print(json.dumps(row),flush=True)
                del prediction,vertices,faces,loss,coverage,free,physical_free,envelope,event,nearest,depth,points,chosen,origins,directions,ranges,native_loss
            checkpoint={'depth_head':head.state_dict() if head is not None else None,'query_decoder':decoder.state_dict(),
                        'optimizer':optimizer.state_dict(),'epoch':epoch+1,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                        'fit_order':[case['metadata']['owner'] for case in fit],
                        'random_states':{'python':random.getstate(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()}}
            if upper is not None:
                checkpoint.update(upper_config=upper.config,upper_adapter=upper.adapter_state_dict())
            torch.save(checkpoint,out/'latest.tmp.pt'); (out/'latest.tmp.pt').replace(out/'latest.pt')
        save('status.json',{'status':'running','phase':'final_evaluation','epochs':args.epochs,
            'updates':sum(r['optimizer_step'] for r in history),'actor_presentations':len(history),'elapsed_s':time.monotonic()-started})
        final=evaluate('final')
        result={'status':'done','fit_actors':len(fit),'development_actors':len(cases)-len(fit),
            'cohort_actors':len(index['cases']),'unsupported_input_actors':len(unsupported),
            'shared_frozen_prefix_views':prefix_views,
            'fit_label_times':label_times,
            'epochs':args.epochs,'updates':sum(r['optimizer_step'] for r in history),'actor_presentations':len(history),
            'resumed_completed_presentations':resumed_steps,'new_presentations':len(history)-resumed_steps,
            'resume_from':str(args.resume_from) if args.resume_from else None,
            'no_gradient_presentations':sum(not r['optimizer_step'] for r in history),
            'visual_only_actors':{role:sum(c['metadata']['visual_only_prediction_enabled'] and c['metadata']['role']==role for c in cases)
                                  for role in ['fit','development']},
            'no_observed_supervision_presentations':sum(r.get('skip_reason')=='no_observed_supervision' for r in history),
            'mode':args.mode,'completion_initialization':args.completion_init,
            'query_surface':args.query_surface,'mesh_level':args.mesh_level if args.query_surface=='shared_mesh' else None,
            'trainable_parameters':{'native_dpt':sum(p.numel() for p in head.parameters() if p.requires_grad) if head is not None else 0,
                                    'query_decoder':sum(p.numel() for p in decoder.parameters() if p.requires_grad)},
            'native_project_max_change':(head.projects[0].weight.detach()-before).abs().max().item() if head is not None else None,
            'first_step':history[0],'last_step':history[-1],'wall_s':time.monotonic()-started,
            'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            'baseline':baseline,'initial':initial,'final':final,
            'boundary':index.get('selection_boundary','existing log cohort')+'; development no gradient; within-window heldout time, not new-source confirmation',
            'failure_ledger_delta':'pending comparison, risks remain active'}
        if upper is not None:
            result['upper_config']=upper.config
            result['trainable_parameters']['upper_lora']=sum(p.numel() for p in upper.parameters() if p.requires_grad)
            result['upper_lora_B_max_abs']=max(p.detach().abs().max().item() for name,p in upper.named_parameters() if name.endswith('lora_B'))
        save('summary.json',result); save('status.json',{'status':'done'})
        print(json.dumps({k:v for k,v in result.items() if k not in ['baseline','initial','final']}),flush=True)
    except Exception as exc:
        save('status.json',{'status':'blocked','exception':type(exc).__name__,'message':str(exc),
                           'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30})
        (out/'traceback.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__': main()
