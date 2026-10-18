"""CPU：旧法原始LiDAR五实例初始化与完整mask入口；不读取R15已训练参数。"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,shutil,sys
from PIL import Image
from identity_common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mask-root',type=Path,
        help='可选完整HW2真掩码目录；必须有manifest.json声明两个真实track_id及来源')
    parser.add_argument('--retry-masks',action='store_true',help='仅允许从本任务尚未生成几何的mask缺口状态继续')
    args=parser.parse_args()
    if TASK.exists():
        if not args.retry_masks or (TASK/'prepared5').exists() or (TASK/'train5').exists() or read(TASK/'mask_coverage.json')['status'] not in ['MASK_INPUT_INCOMPLETE','READY']:
            raise FileExistsError('拒绝覆盖已有几何或训练任务: '+str(TASK))
    old_spec=read(ROOT/'rigid_specs/scene0008_089_137.json')
    old_effective=read(ROOT/'rigid_runs/scene0008_089_137/full30k/effective_config.json')
    original_preparation=read(Path(old_spec['prepared'])/'preparation_report.json')
    original_reference=old_effective['training_policy']['reference_config']
    actual_reference=read(old_spec['reference_config'])
    ignored={'source_path','model_path'}
    if {k:v for k,v in original_reference.items() if k not in ignored}!={k:v for k,v in actual_reference.items() if k not in ignored}:
        raise ValueError('当前原始reference相对旧30K真实快照改变，先核查，不静默改训练')
    manifest=read(R15/'data/manifest.json');actor_ids=manifest['actor_ids']
    old=np.load(Path(old_spec['prepared'])/'instances.npz',allow_pickle=False)
    ids=old['instance_ids'].tolist()
    assert ids==[actor_ids[a] for a in OLD_ACTORS]
    assert old['mask_known'].all() and len(old['frame_ids'])==49
    expected_new=[actor_ids[a] for a in NEW_ACTORS]
    alt=read(args.mask_root/'manifest.json') if args.mask_root else None
    if alt:
        if alt.get('actor_ids')!=expected_new or not alt.get('provenance'):
            raise ValueError('替代mask必须声明同一真实ID顺序与实际来源')
        if alt.get('GT_boxes_used_to_create_masks') is not False:
            raise ValueError('不得用3D框填造二维body mask')
        if alt.get('root_engineering_review',{}).get('status')!='ALL49_IDENTITY_CHECKED':
            raise ValueError('原生完整提议必须先做49帧真实身份视觉核对')
    labels=[];coverage=[]
    for i,row in enumerate(manifest['frames']):
        fid=row['frame_id'];assert fid==old['frame_ids'][i]
        image=Image.open(Path(old_spec['prepared'])/'instance_masks'/(fid+'.png'))
        a=np.asarray(image.resize((480,320),Image.Resampling.NEAREST))
        original=np.stack([a==k+1 for k in range(3)],-1)
        if args.mask_root:
            sup=np.load(args.mask_root/(fid+'.npz'),allow_pickle=False)
            new=sup['masks'];known=sup['mask_known'];valid=sup['mask_valid']
            source=str(args.mask_root/(fid+'.npz'))
        else:
            sup=np.load(R15/'data'/row['supervision'],allow_pickle=False)
            new=sup['masks'][...,NEW_ACTORS];known=sup['mask_known'][NEW_ACTORS]
            valid=sup['mask_valid'][...,NEW_ACTORS] if 'mask_valid' in sup.files else np.broadcast_to(known,new.shape).copy()
            source=str(R15/'data'/row['supervision'])
        label,record=compose_masks(original,new,known,valid)
        labels.append(label);coverage.append(dict(frame_id=fid,source=source,**record))
    TASK.mkdir(exist_ok=True);(TASK/'code').mkdir(exist_ok=True);(TASK/'infra').mkdir(exist_ok=True)
    if args.retry_masks and (TASK/'mask_coverage.json').exists() and read(TASK/'mask_coverage.json')['status']=='MASK_INPUT_INCOMPLETE':
        shutil.copy2(TASK/'mask_coverage.json',TASK/'mask_coverage_R15_partial.json')
    write(TASK/'mask_coverage.json',dict(status='READY' if all(r['ready'] for r in coverage) else 'MASK_INPUT_INCOMPLETE',
        frames=coverage,old_ids=ids,new_ids=expected_new,unknown_never_used_as_background=True))
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,TASK/'code'/p.name)
    for p in (ROOT/'infra').glob('*.py'):shutil.copy2(p,TASK/'infra'/p.name)
    write(TASK/'old_effective_config.json',old_effective)
    paths=[Path(old_spec['background_checkpoint']),Path(old_spec['environment_checkpoint']),
           ROOT/'rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt',
           R15/'paused_checkpoint_004000_20261017/checkpoint.pt',R15/'main/latest.pt']
    write(TASK/'preserved_inputs.json',dict(files=[stat(p) for p in paths],R15_pause_marker=(R15/'PAUSE_REQUESTED.json').exists()))
    if not all(r['ready'] for r in coverage):
        print('MASK_INPUT_INCOMPLETE: coverage saved; no model training, no unknown-to-background fallback')
        return
    # 复用旧CPU原始测量聚合、着色、尺度初始化和歧义拒绝；不用R15 seed。
    sys.path.insert(0,str(TASK/'infra'))
    from data import prepare_scene
    prepared=TASK/'prepared5'
    np.savetxt(TASK/'normalization.txt',old['normalization'])
    report=prepare_scene(old_spec['source_scene'],prepared,annotations=manifest['annotations'],
        normalization_path=TASK/'normalization.txt',
        instance_ids=ids+expected_new,selection_path=None,
        padding_m=float(original_preparation['box_padding_m']),
        scale_prior_m=float(original_preparation['scale_initializer']['singleton_prior_m']),
        strict_existing_scene=False)  # 已逐帧绑定同一旧49窗口，不沿用原最初三窗口的索引白名单。
    bundle=dict(np.load(prepared/'instances.npz',allow_pickle=False))
    assert bundle['instance_ids'].tolist()==ids+expected_new
    assert np.array_equal(bundle['normalization'],old['normalization'])
    assert np.array_equal(bundle['timestamps'],old['timestamps'])
    bundle['mask_known']=np.ones((49,5),bool)
    bundle['front_visible']=np.array([[int((label==k+1).any()) for k in range(5)] for label in labels],np.int8)
    (prepared/'instance_masks').mkdir(exist_ok=True)
    for fid,label in zip(old['frame_ids'],labels):Image.fromarray(label).save(prepared/'instance_masks'/(fid+'.png'))
    np.savez_compressed(prepared/'instances.npz',**bundle)
    trajectories=read(prepared/'trajectories.json')
    for k,item in enumerate(trajectories['trajectories']):
        for i,row in enumerate(item['frames']):
            row['mask_known']=True
            row['front_visibility']='visible' if (labels[i]==k+1).any() else 'not_visible'
    write(prepared/'trajectories.json',trajectories)
    report.update(full_mask_supervision_ready=True,mask_known_pairs=245,mask_complete_frames=49,
        masks=dict(status='five_ID_full_native_masks',known_pairs=245,unknown_pairs=0,
            old_three_source=str(Path(old_spec['prepared'])/'instance_masks'),
            new_two_source=str(args.mask_root) if args.mask_root else str(R15/'data'),
            GT_boxes_used_to_create_masks=False,full_in_out_supervision=True),
        initialization_role='old raw-LiDAR box aggregation, not R15 filtered/clustered seeds',
        identity_completion=TASK_ID)
    write(prepared/'preparation_report.json',report)
    spec=dict(old_spec);spec.update(prepared=str(prepared),stage='old_method_owner5_30k',
        scope='same physical a5,a26,a18,a6,a37; parked/moving track roles unchanged',
        baseline_extension=dict(task_id=TASK_ID,old_3_ID_baseline_preserved=True,
            background='same existing GN11 30000; never load R15 or old learned instance weights',
            changed_inputs=['selected identities 3 to 5','two actual full-mask sequences'],
            not_identical_to_historical_3_ID_baseline=True))
    write(TASK/'spec5.json',spec)
    from runtime import validate_spec
    _,ready=validate_spec(TASK/'spec5.json');assert ready['full_mask_supervision_ready']
    # 逐ID登记旧3种子是否因新框的歧义拒绝发生变化，不隐瞒输入改变。
    comparisons=[]
    for k in range(3):
        before=old['point_ids']==k;after=bundle['point_ids']==k
        comparisons.append(dict(actor=OLD_ACTORS[k],old_points=int(before.sum()),new_points=int(after.sum()),
            xyz_bitwise_equal=bool(np.array_equal(old['local_xyz'][before],bundle['local_xyz'][after])),
            RGB_bitwise_equal=bool(np.array_equal(old['rgb'][before],bundle['rgb'][after]))))
    write(TASK/'prepared.json',dict(status='CPU_READY_GPU_NOT_STARTED',task_id=TASK_ID,
        actors=ACTORS,track_ids=ids+expected_new,old_seed_comparison=comparisons,
        model_updates=0,source_checkpoint_role='GN11 only',mask_pairs=245,
        parameter_config_source=str(ROOT/'rigid_runs/scene0008_089_137/full30k/effective_config.json')))
    assert 'torch' not in sys.modules
    print('CPU_READY_GPU_NOT_STARTED: '+str(TASK))

if __name__=='__main__':main()
