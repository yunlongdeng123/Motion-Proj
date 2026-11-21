"""从实际导出与训练合同构造报告目录，不复算GPU结果。"""
from pathlib import Path
import json
W=Path(__file__).resolve().parent
O=W.parents[1]/'outputs/vehicle_road_joint_20261121'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')

def assemble():
    c=read(O/'contract.json');met=read(O/'delivery/metrics.json');proof=read(O/'delivery/proofs.json')
    stages=[];frames=c['diagnostic_frames']+c['train_frames'];frames=sorted(set(frames))
    labels={
        'native_before':('接入前 · 原生动态表示','既有重建，无本轮更新'),
        'canonical_init':('对象坐标转换 · 无补全先验','0 步；统一对象表示'),
        'canonical_prior_init':('下半车体先验 · 初值','0 步；补足被遮车体支撑'),
        'vehicle_fit':('车辆单独优化',f"车辆 {c['vehicle_steps']} 步"),
        'joint_fit':('车辆与道路联合优化',f"车辆 {c['vehicle_steps']} 步 + 联合 {c['joint_steps']} 步"),
        'handoff_fixed':('旧 a9 交接 · 严格 49 行','继承联合训练；精确零贡献保护，无新增训练'),
        'handoff_fixed_guarded':('旧 a9 交接 · 75 行','相对联合终态：保留车辆不变，外域最多 1 灰度码')}
    def image_path(folder,name):
        for suffix in ['.webp','.png']:
            p=folder/(name+suffix)
            if p.exists():return p.relative_to(O).as_posix()
        return None
    for sid in labels:
        if not (O/'delivery'/sid).is_dir():
            if sid.startswith('handoff_fixed'):continue
            raise FileNotFoundError('缺少对照阶段 '+sid)
        images={str(raw):{v:image_path(O/'delivery'/sid/f'{raw:03d}',v) for v in ['joint_keep','joint_delete','vehicle_only','road_only']} for raw in frames}
        if not all(all(f.values()) for f in images.values()):raise ValueError('incomplete actual export '+sid)
        for raw in frames:
            for view in ['vehicle_alpha_delete','road_alpha']:
                images[str(raw)][view]=image_path(O/'delivery'/sid/f'{raw:03d}',view)
        name,budget=labels[sid]
        if sid=='vehicle_fit':parameters='车辆颜色、不透明度、受限尺度';supervision='a20 原始 RGB / 自动估计可见掩码';frozen='道路、轨迹、相机、其余场景'
        elif sid=='joint_fit':parameters='车辆颜色 + 道路许可行颜色';supervision='保留场景的车辆 + 删除 a9 后的道路';frozen='覆盖、几何、轨迹、静态背景、环境'
        elif sid.startswith('handoff_fixed'):parameters='核验旧残留归属，无梯度更新';supervision='跨帧实际成像贡献与保留对象保护';frozen='全部已训练参数'
        else:parameters='不训练';supervision='真实观测 / 既有重建 / 显式先验';frozen='全部参数'
        stages.append(dict(id=sid,name=name,budget=budget,images=images,parameters=parameters,supervision=supervision,frozen=frozen))
    stages.sort(key=lambda s:list(labels).index(s['id']))
    refs={}
    for raw in frames:
        f=O/'delivery/references'/f'{raw:03d}'
        refs[str(raw)]={k:image_path(f,k) for k in ['rgb','vehicle_mask','road_legal','vehicle_potential_bbox']}
        refs[str(raw)]['road_teacher']=image_path(f,'road_teacher') if raw in c['train_frames'] else None
    boxes={}
    visibility=W/'qa_a20_visibility.json'
    if visibility.exists():
        for row in read(visibility)['records']:
            if row['raw'] not in frames or 'bbox_raw' not in row:continue
            b=row['bbox_raw'];x0,y0,x1,y1=[float(v)*.6 for v in b]
            boxes[str(row['raw'])]=[max(0,int(x0)-100),max(0,int(y0)-50),min(1152,int(x1)+100),min(768,int(y1)+150)]
    rows=[]
    for r in met['rows']:
        if r['stage'] not in [s['id'] for s in stages]:continue
        h=r.get('vehicle_HF_sigma1') or {};rh=r.get('road_HF_sigma1') or {}
        rows.append(dict(stage=r['stage'],raw=r['raw'],role='vehicle',MAE=r.get('vehicle_rgb_mae'),SSIM=r.get('vehicle_masked_ssim'),HF_cosine=h.get('correlation'),HF_energy_ratio=h.get('energy_ratio'),IoU=r.get('vehicle_alpha_iou')))
        if r.get('road_teacher_mae') is not None:rows.append(dict(stage=r['stage'],raw=r['raw'],role='road',MAE=r.get('road_teacher_mae'),SSIM=None,HF_cosine=rh.get('correlation'),HF_energy_ratio=rh.get('energy_ratio'),IoU=None))
    # 优先使用完整CPU细节bench输出。它保留固定评分域与完整核支持。
    detail=O/'delivery/independent_detail_metrics.json'
    if not detail.exists():detail=O/'delivery/detail_metrics.json'
    if not detail.exists():raise FileNotFoundError('完整固定域细节评价尚未交付：'+str(detail))
    if detail.exists():
        rows=[]
        for r in read(detail)['rows']:
            if r['stage'] not in [s['id'] for s in stages]:continue
            rows.append(dict(stage=r['stage'],raw=r['raw'],role=r['role'],MAE=r.get('RGB_MAE'),SSIM=r.get('masked_SSIM'),HF_cosine=r.get('HF_cosine_1'),HF_energy_ratio=r.get('HF_energy_ratio_1'),edge_F1=r.get('edge_F1'),IoU=r.get('vehicle_alpha_iou'),pixels=r.get('pixels'),HF_core_pixels=r.get('HF_core_pixels_1')))
    expected={(s['id'],raw,role) for s in stages for raw in frames for role in (['vehicle','road'] if raw in c['train_frames'] else ['vehicle'])}
    actual={(r['stage'],r['raw'],r['role']) for r in rows}
    if not expected.issubset(actual):raise ValueError('缺少阶段固定域评价 '+str(expected-actual))
    checks=[dict(name='统一可微三维渲染',passed=proof.get('joint_real_renderer_two_branch_nonzero_all_steps',False),detail='联合阶段车辆与道路均获得梯度并更新'),dict(name='固定几何与轨迹',passed=proof.get('local_xyz_equal',False) and proof.get('trajectory_equal',False),detail='道路覆盖、车辆位置与姿态、相机和轨迹保持固定'),dict(name='原模型参数冻结',passed=proof.get('native_scene_unchanged',False),detail='原始模型、环境与既有静态交接参数保持固定')]
    report=dict(schema='vehicle_road_joint_report.v1',subtitle='a9：前方白色 Mazda 轿车 · a20：后方黑色皮卡 · 1152 × 768',status='已接入对象坐标车辆分支；车辆与道路按分区监督进行联合优化。',stats=[dict(name='训练帧',value=' / '.join(map(str,c['train_frames']))),dict(name='诊断帧',value=' / '.join(map(str,c['diagnostic_frames']))),dict(name='车辆 / 联合更新',value=f"{c['vehicle_steps']} / {c['joint_steps']}"),dict(name='道路真新增高斯（累计）',value='7,239')],stages=stages,frames=frames,default_frame=134,references=refs,crop_boxes=boxes,frame_info={str(raw):('本轮训练视角' if raw in c['train_frames'] else '本轮未参与更新的诊断视角') for raw in frames},method='HUGSIM 的对象坐标刚体高斯与统一渲染用于车辆接入。原始车辆有效高斯转为共享对象资产；被遮下半使用显式几何与车漆先验。车辆损失使用真实观测，道路损失使用对应道路目标；当前投影盒与掩码分区仍有越界足迹带来的梯度串扰。',metric_roles=[dict(id='vehicle',name='a20 真实可见车体'),dict(id='road',name='合法道路教师区域')],metric_columns=[dict(key='MAE',name='RGB MAE ↓'),dict(key='SSIM',name='局部 SSIM ↑'),dict(key='HF_cosine',name='HF 对应度 ↑'),dict(key='HF_energy_ratio',name='HF 能量比（目标 1）'),dict(key='edge_F1',name='边缘 F1 ↑'),dict(key='IoU',name='可见轮廓 IoU ↑'),dict(key='pixels',name='评分像素',integer=True),dict(key='HF_core_pixels',name='HF 有效像素',integer=True)],metrics=rows,metric_note='相同帧、相同固定评分区域对比；HF 为原生像素尺度 σ=1 的有符号细节对应，完整记录另含 σ=2/4。道路教师仅存在于 raw86/110；轮廓 IoU 对比自动估计的观测实例掩码。',checks=checks,proofs=proof,config=c,links=[dict(name='完整固定域细节指标',path='delivery/detail_metrics.json'),dict(name='训练阶段指标记录',path='delivery/metrics.json'),dict(name='实际训练合同',path='contract.json'),dict(name='HUGSIM 方法核对',path='hugsim_review.md'),dict(name='原道路统一报告',path='../road_material_three_rounds_20261121/index.html')],disclaimer='被 a9 持续遮挡的 a20 下半部缺少真实外观监督，其几何与颜色来源已在先验记录中标注；可见域指标不代表隐藏部位真实还原。原始重建模型见过全序列，本轮少帧诊断不属于独立盲测。')
    report['findings']=[
        dict(name='a20 可见车体',attention=True,detail='两个训练帧的颜色拟合改善；三个诊断帧仍有对象表示转换与外观适配偏差。'),
        dict(name='a20 被遮下半部',attention=True,detail='624 个新增高斯提供车体、保险杠与轮胎形状先验；纹理与轮胎外形仍较粗糙。'),
        dict(name='其它车辆局部变化',attention=True,detail='raw86 左侧 a10 轿车对原图 MAE 从原生 0.10684 升至联合训练 0.15847；该项保留为未通过检查。'),
        dict(name='旧 a9 残影',attention=True,detail='75 行安全归属修复保持五帧保留车辆像素不变；主要跨域白块仍残留，当前画质未通过。')]
    report['findings'].append(dict(name='实际贡献隔离',attention=True,detail='物理投影盒已排除；大足迹车辆高斯仍越界进入道路教师域，需重建对象支持并使用实际贡献约束。'))
    report['status']='车辆 200 步＋联合 200 步已完成；当前下半车体、旧残影和局部车辆画质仍需改进。'
    report['status_attention']=True
    report['stats'][-1]=dict(name='a20 新增形状先验高斯',value='624')
    report['links'][0]['path']=detail.relative_to(O).as_posix()
    extras={}
    for name in ['disk_reload_proof.json','handoff_fixed_guarded_disk_reload_proof.json','handoff_guarded_proof.json','handoff_guarded_admission.json','qa_final_cpu.json','qa_teacher_render_overlap.json','implementation_delivery_receipt.json','takeback_download_receipt.json']:
        if (O/name).exists():extras[name]=read(O/name)
    report['proofs']=dict(training=proof,additional=extras)
    if 'handoff_fixed_guarded_disk_reload_proof.json' in extras:
        report['checks'].append(dict(name='磁盘资产独立进程重载',passed=extras['handoff_fixed_guarded_disk_reload_proof.json']['passed'],detail='五帧最大 RGB 差 3.64e-6，道路 RGB / Alpha 逐值一致'))
    if 'handoff_guarded_proof.json' in extras:
        report['checks'].append(dict(name='75 行交接集合保护',passed=extras['handoff_guarded_proof.json']['passed'],detail='相对联合资产，KEEP 与保留车辆区 RGB 不变；外域最大 0.003848'))
    report['links'].extend([
        dict(name='实际贡献隔离检查',path='qa_teacher_render_overlap.json'),
        dict(name='资产恢复方法',path='RESTORE.md'),
        dict(name='已核验资产与完整训练状态',path='../../work/vehicle_road_joint_20261121/takeback/vehicle_road_joint_review_and_asset_r2.tar.gz'),
        dict(name='交付核验记录',path='implementation_delivery_receipt.json')])
    report['links'].extend([dict(name='独立工程与效果检查',path='final_independent_qa.md'),dict(name='下半车体先验来源',path='lower_prior.json')])
    dump(W/'report_data.json',report)
    return W/'report_data.json'
if __name__=='__main__':print(assemble())
