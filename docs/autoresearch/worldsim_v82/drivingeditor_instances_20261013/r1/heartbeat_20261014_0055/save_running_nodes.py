"""Archive genuine 10k/20k CPU audits and a bounded live snapshot."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'work/drivingeditor_resume/repo_active'
WORK=ROOT/'work/instance_edit_smoke_300_r1'
SNAP=WORK/'heartbeat_20261014_0055'
E=REPO/'docs/autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/heartbeat_20261014_0055'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
snapshot=read(SNAP/'snapshot.json')
receipts=[read(WORK/('checkpoint_%06d_CPU_structure_receipt.json'%step)) for step in (10000,20000)]
assert all(r['status']=='CPU_STRUCTURE_FINITE_BINDING_POLICY_PASS' and not r['cuda_initialized'] for r in receipts)
E.mkdir(parents=True,exist_ok=False)
shutil.copy2(SNAP/'snapshot.json',E/'snapshot.json')
for step in (10000,20000):
    name='checkpoint_%06d_CPU_structure_receipt.json'%step
    shutil.copy2(WORK/name,E/name)
shutil.copytree(SNAP/'preview',E/'preview')
for name in ('inspect_live_run.py','save_running_nodes.py'):
    shutil.copy2(ROOT/'work/drivingeditor_instances'/name,E/name)
now=datetime.fromisoformat(snapshot['inspected_utc']).astimezone(timezone(timedelta(hours=8))).isoformat()
step=snapshot['status']['step']; peak=snapshot['sampled_device_peak_bytes']/2**30
for path in (ROOT/'outputs/DrivingEditor_instance_editor_status.json',REPO/'docs/v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.json'):
    d=read(path); d['updated_sgt']=now
    d['queue']['latest_run_status']=snapshot['status']
    d['validation']['checkpoint_CPU_structure_verified_steps']=[10000,20000]
    d['validation']['actual_full_GPU_model_Adam_RNG_restore_verified_steps']=[300]
    d['validation']['checkpoint_GPU_restore_verified_for_10k_20k']=False
    d['running_node_inspection']={'snapshot':snapshot,'node_CPU_receipts':receipts,
        'evidence':str(E.relative_to(REPO)),
        'training_phase_pytorch_peak_externally_available':False,'full_run_memory_peak_final':False}
    write(path,d)
for path in (ROOT/'outputs/DrivingEditor_instance_editor_report.md',REPO/'docs/v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.md'):
    s=path.read_text(encoding='utf-8')
    s=s.replace('2026-10-14T00:10:45.044812+08:00',now).replace('当前快照step1100；10k/20k/30k完整保存',
        '当前快照step%d；10k和20k已保存并通过CPU结构/finite/Adam/RNG/policy检查，30k待完成。10k/20k GPU实际恢复尚未认证'%step)
    s+='\n运行中节点检查：10k/20k各935,060,588B，点数298075实例/613405静态固定，两ID及策略不变；CPU单线程/nice10，无CUDA或优化。累计%d NV采样、%d错误，设备采样峰%.3fGiB仅下界；活跃training阶段allocated/reserved峰须等monitor关闭才能完整读取。10k raw47单帧preview显示车辆比300步更成形，但静态道路灰白补片仍在，仅作预览而非49帧质量结论。\n'%(snapshot['NV_samples'],snapshot['sampling_error_rows'],peak)
    path.write_text(s,encoding='utf-8')
path=REPO/'docs/RESEARCH_STATUS.md'
s=path.read_text(encoding='utf-8').replace('2026-10-14T00:10:45.044812+08:00',now)
s=s.replace('当前证据step1100，10k/20k/30k完整保存',
    '当前证据step%d；10k/20k已保存且CPU结构/finite/Adam/RNG/policy通过，30k待完成；10k/20k GPU实际恢复未认证'%step)
s+='\n本半小时节点证据：[10k/20k检查](autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/heartbeat_20261014_0055/)。%d NV采样0错误，设备采样峰%.3fGiB为运行中下界，活跃training阶段PyTorch峰待monitor关闭。10k单帧预览车体更成形、道路灰白补片保留，不能代替最终49帧审查。最终评价/统一HTML接入源码正在CPU准备，无新GPU或训练阶段。\n'%(snapshot['NV_samples'],peak)
path.write_text(s,encoding='utf-8')
path=REPO/'docs/EXPERIMENTS.md'
s=path.read_text(encoding='utf-8').replace('从同step300续至总30k，固定背景轨迹；未收敛透明车体/道路patch保留',
    '从同step300续至总30k，10k/20k完整状态CPU结构检查通过，固定背景轨迹；照片质量和道路patch保留')
path.write_text(s,encoding='utf-8')
print(json.dumps({'snapshot_step':step,'nodes_CPU_checked':[10000,20000],'device_sample_peak_GiB':peak}))
