"""Create a compact offline AI review package for four ablations plus native A."""
from pathlib import Path
import argparse,datetime,difflib,hashlib,html,json,re,shutil,subprocess,zipfile
from build_native30k_compare import probe,sha
W=Path(__file__).resolve().parent
ROOT=W.parent.parent
O=ROOT/'outputs/gn11_stage1_v2_20261019'
BO=ROOT/'outputs/baseline_engineering_20261018'
BW=W.parent/'baseline_engineering_20261018'
S=W/'ai_review_source_snapshot'
D=W/'ai_review_bundle_v1'
ZIP=O/'gn11_four_ablations_plus_native_baseline_ai_review.zip'
ARMS=['baseline','full','real_only','prior_only','ownership_only']
LABELS={'baseline':'原生五身份GN11一阶段30K基线','full':'完整版：归属＋跨帧＋供体','real_only':'归属＋仅跨帧','prior_only':'归属＋仅供体','ownership_only':'仅归属修复'}

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(rel,value):
    p=D/rel;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
def copy(src,rel):
    assert src.is_file(),src
    dest=D/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists() or dest.read_bytes()!=src.read_bytes():shutil.copyfile(src,dest)
def files(root):return {p.relative_to(root).as_posix():p for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}

def build(code_commit):
    D.mkdir(parents=True,exist_ok=True)
    snapshot=read(S/'manifest.json');runtime=read(W/'ablation_runtime.json');media=read(O/'cpu_export_media.json')
    assert snapshot['GPU_used'] is False and snapshot['training_updates']==0
    baseline=files(S/'engines/baseline');source_map={}
    for name,p in baseline.items():copy(p,'code/base/'+name)
    for arm in ARMS:
        engine=files(S/'engines'/arm);assert set(baseline)<=set(engine)
        changed={k:p for k,p in engine.items() if k not in baseline or sha(p)!=sha(baseline[k])}
        if arm!='baseline':
            declared=read(S/'runs'/arm/'deployed_manifest.json')
            assert set(changed)==set(declared['files'])
            for name,e in declared['files'].items():assert sha(engine[name])==e['candidate_sha256']
            for name in ['train.py','scene/gaussian_model.py','gaussian_renderer/__init__.py']:
                if arm=='full':assert sha(baseline[name])==declared['files'][name]['original_sha256']
            for name,p in changed.items():copy(p,'code/overlays/'+arm+'/'+name)
        source_map[arm]={'engine_source':snapshot['runs'][arm]['engine'],'core_file_count':len(engine),'overlay_files':sorted(changed),
          'files':{k:{'sha256':sha(p),'bytes':p.stat().st_size} for k,p in engine.items()}}
        for name,p in files(S/'runs'/arm).items():copy(p,'evidence/runs/'+arm+'/'+name)
    write('code/source_map.json',source_map)
    for name,p in files(S/'entrypoints').items():copy(p,'code/entrypoints/'+name)
    for name,p in files(S/'binding_tools').items():copy(p,'code/binding_tools/'+name)
    for name in ['build_candidate.py','prepare_real_inputs.py','world_road.py','audit_donor_provenance.py','tests_cpu.py','check_ablation_contracts.py','build_road_ablations.py']:
        copy(W/name,'code/review_tools/'+name)
    for filename,raw in [('LICENSE',None),('LICENSE_gaussian_splatting.md',None)]:
        # Preserve the original archived license bytes separately from normalized overlays.
        with zipfile.ZipFile(BO/'gn11_static_duplicate_20261019/actual_source_bytes.zip') as z:
            p=D/'code/licenses'/filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(filename))
    for arm in ['full','real_only','prior_only','ownership_only']:
        left='baseline' if arm=='full' else 'full';a=files(S/'engines'/left);b=files(S/'engines'/arm);parts=[]
        for name in sorted(set(a)|set(b)):
            if name.startswith('LICENSE'):continue
            x=a[name].read_text(encoding='utf-8').splitlines(True) if name in a else []
            y=b[name].read_text(encoding='utf-8').splitlines(True) if name in b else []
            parts.extend(difflib.unified_diff(x,y,fromfile=left+'/'+name,tofile=arm+'/'+name,n=3))
        write('code/diffs/'+left+'_to_'+arm+'.diff',''.join(parts))
    materializer='''"""Reconstruct review code only; never imports training or calls CUDA."""
from pathlib import Path
import argparse,hashlib,json,shutil
p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['baseline','full','real_only','prior_only','ownership_only'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
if a.output.exists():raise FileExistsError('choose a new empty destination')
shutil.copytree(root/'base',a.output)
if a.experiment!='baseline':shutil.copytree(root/'overlays'/a.experiment,a.output,dirs_exist_ok=True)
expected=json.loads((root/'source_map.json').read_text(encoding='utf-8'))[a.experiment]['files']
for name,e in expected.items():assert hashlib.sha256((a.output/name).read_bytes()).hexdigest()==e['sha256']
print('Verified source files:',len(expected))
'''
    write('code/materialize.py',materializer)
    for name,p in files(S/'frozen_input').items():copy(p,'evidence/input/'+name)
    for name in ['actual_road_query_evidence.npz','actual_road_summary.json','actual_donor_provenance.json','calibration_binding_receipt.json',
      'initialization_protocol_binding.json','actual_gpu_gate_receipt.json','cpu_test_receipt.json','independent_export_status.json',
      'native_static_review.json','cpu_export_validation.json','cpu_export_sources.json','native30k_sources.json']:
        copy(O/name,'evidence/'+name)
    copy(S/'manifest.json','evidence/source_snapshot_manifest.json')
    copy(S/'entrypoints_manifest.json','evidence/entrypoints_manifest.json')
    copy(S/'additional_manifest.json','evidence/additional_sources_manifest.json')
    for arm in ['full','real_only','prior_only','ownership_only']:
        for step in range(5000,30001,5000):
            src=O/f'checkpoint_{step:06d}_audit.json' if arm=='full' else O/'ablations'/f'{arm}_checkpoint_{step:06d}_audit.json'
            copy(src,f'evidence/runs/{arm}/audits/{step:06d}.json')
        if arm!='full':
            copy(O/'ablations'/(arm+'_protocol_receipt.json'),'evidence/runs/'+arm+'/first100_protocol.json')
            write('evidence/runs/'+arm+'/actual_gpu_gate.json',runtime['arms'][arm]['gate'])
    for arm in ['full','real_only']:
        for name,p in files(O/'independent_receipts'/arm).items():copy(p,'evidence/runs/'+arm+'/independent49/'+name)
    for name in ['GN11_recovery_audit.json','GN11_resume_GPU_receipt.json','GN11_resume_sequence_receipt.json','GN11_25k_durability_receipt.json',
      'GN11_timing_fix_GPU_receipt.json','matched5_input_receipt.json']:
        copy(BO/name,'evidence/runs/baseline/'+name)
    copy(BW/'GN11_stage1_full49/receipt.json','evidence/runs/baseline/native49_receipt.json')
    for name in ['receipt.json','motion_feature_receipt.json','native_static_time_summary.json','source_comparison.json']:
        copy(BO/'gn11_static_duplicate_20261019'/name,'evidence/baseline_diagnostics/'+name)
    for name in ['audit_gn11_static_duplicate.py','audit_gn11_static_motion_features.py']:
        copy(BW/name,'code/review_tools/'+name)
    copy(BO/'gn11_static_duplicate_20261019/fixed_camera_time_only.mp4','media/baseline_fixed_camera_time_only.mp4')
    # Complete current five-model videos. Media are copied byte-for-byte, not transcoded.
    mapping={};video_records={};physical_by_sha={}
    for key,item in media['videos'].items():
        src=(O/item['path']).resolve();assert sha(src)==item['sha256']
        preferred='media/'+key+'.mp4'
        rel=physical_by_sha.setdefault(item['sha256'],preferred);copy(src,rel);mapping[item['path']]=rel
        previous=D/preferred
        if preferred!=rel and previous.exists():
            previous.resolve().relative_to(D.resolve());assert sha(previous)==item['sha256'];previous.unlink()
        if item.get('poster'):
            poster='media/'+key+'_raw089.png';copy(O/item['poster'],poster);mapping[item['poster']]=poster
        video_records[key]={**item,'path':rel,'source_path':item['path']}
    for channel in ['delete_all','static_alpha','dynamic_alpha']:
        src=BO/'branch_videos'/('gn11five_'+channel+'.mp4');preferred='media/native_baseline_'+channel+'.mp4'
        rel=physical_by_sha.setdefault(sha(src),preferred);copy(src,rel)
        previous=D/preferred
        if preferred!=rel and previous.exists():
            previous.resolve().relative_to(D.resolve());assert sha(previous)==sha(src);previous.unlink()
        video_records['native_baseline_'+channel]={'path':rel,'sha256':sha(src),'bytes':src.stat().st_size,**probe(src)}
    write('evidence/media.json',{'videos':video_records,'channel_references':37,'unique_physical_videos':len(physical_by_sha),'deduplication':'identical SHA256 only; channel semantics and provenance retained',
      'full_frame_wh':[480,320],'fps':10,'frames_each':49,'native_baseline_extra_channels':True})
    # Package relative, offline players; eliminate workspace-external links.
    for src,name in [('native30k_four_compare.html','compare.html'),('independent_pair_compare.html','independent_pair.html')]:
        text=(O/src).read_text(encoding='utf-8')
        data=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',text,re.S)[1])
        for model in data['models'].values():
            for item in model['videos'].values():
                item['path']=mapping[item['path']]
                if item.get('poster'):item['poster']=mapping[item['poster']]
                item.pop('source_pattern',None)
        text=re.sub(r'(<script type="application/json" id="data">).*?(</script>)',lambda m:m[1]+json.dumps(data,ensure_ascii=False,separators=(',',':'))+m[2],text,count=1,flags=re.S)
        for old,new in [('native30k_four_compare.html','compare.html'),('independent_pair_compare.html','independent_pair.html'),('ablations/architecture.svg','architecture.svg'),
          ('ablations/index.html','index.html'),('native30k_compare.html','compare.html'),('cpu_export_media.json','evidence/media.json'),
          ('cpu_export_sources.json','evidence/cpu_export_sources.json'),('cpu_export_validation.json','evidence/cpu_export_validation.json')]:text=text.replace('"'+old+'"','"'+new+'"')
        write(name,text)
    # Three lossless GT anchor frames suffice alongside the full49 GT video.
    # Remove only this builder's old verified generated GT copies from prepare v1.
    for old in (D/'frames/gt').glob('*.png'):
        if old.stem not in ['0008089','0008129','0008137']:
            old.resolve().relative_to(D.resolve())
            assert sha(old)==sha(W/'native30k_pair/full/gt'/old.name)
            old.unlink()
    for n in range(89,138):
        frame=f'0008{n:03d}'
        if n in [89,129,137]:copy(W/'native30k_pair/full/gt'/(frame+'.png'),'frames/gt/'+frame+'.png')
        copy(W.parent/'identity_completion/metadata/native_masks_for_review'/(frame+'.npz'),'evidence/input/query_masks_a6_a37/'+frame+'.npz')
    write('evidence/input/query_masks_a6_a37/README.md','These are query masks for a6 and a37 only, in that channel order. They are NOT the complete five-actor training union. Shape: 320x480x2, full-frame; mask_known and mask_valid retained.\n')
    for arm in ARMS:
        for n in [89,129,137]:
            frame=f'0008{n:03d}'
            for channel in ['static','dynamic','full']:
                if arm=='baseline':src=BW/'GN11_stage1_full49'/frame/(channel+'.png')
                else:
                    root=W/'native30k_pair' if arm in ['full','real_only'] else W/'cpu_exports/native'
                    src=root/arm/('composite' if channel=='full' else channel)/(frame+'.png')
                copy(src,f'frames/samples/{arm}/raw{n}_{channel}.png')
    for name in ['raw89_original.png','raw89_query_and_donors.png','raw89_donor_filter_preview.png','actual_donor_support.png']:
        copy(O/name,'frames/input/'+name)
    for p in sorted((O/'actual_query_projection').glob('*.png')):copy(p,'frames/input/query_projection/'+p.name)
    weights={'baseline':[0,0,0,0],'full':[1,.1,.1,.01],'real_only':[1,0,.1,0],'prior_only':[0,.1,0,.01],'ownership_only':[0,0,0,0]}
    experiments=[]
    for arm in ARMS:
        cfg=read(S/'runs'/arm/'effective_config.json');run=snapshot['runs'][arm]
        experiments.append({'id':arm,'label':LABELS[arm],'actors':[6,37,5,26,18],'raw_frames':[89,137],'frames':49,'image_wh':[480,320],
          'first_stage_logical_updates':30000,'second_stage_updates':0,'seed':0,'ownership_fix':arm!='baseline',
          'road_weights':dict(zip(['observed_rgb','prior_rgb','observed_depth','prior_depth'],weights[arm])),
          'actual_optimizer_step_counts':run['optimizer_step_counts'],'effective_config':'evidence/runs/'+arm+'/effective_config.json',
          'run':run['run'],'independent49':('completed_zero_update' if arm in ['full','real_only'] else 'existing_native49' if arm=='baseline' else 'pending_GPU'),
          'source_materialization':'code/base + code/overlays/'+arm if arm!='baseline' else 'code/base',
          'native_checkpoint49_complete':True,'all_views_are_training_views':True,'quality_accepted':False})
    write('experiments.json',experiments)
    provenance={'repository':'https://github.com/yunlongdeng123/Motion-Proj','branch':'research/worldsim-v8.2-drivingeditor-debug',
      'code_commit':code_commit,'source_snapshot_base_commit':'67d57b88a10a1340c0e6a4537808f8f4324f2c4c','source_snapshot_collected_utc':snapshot['collected_utc'],
      'review_only_not_a_training_checkpoint_bundle':True,'remote_GPU_used_for_packaging':False,'training_updates_for_packaging':0,
      'excluded':['model weights/checkpoints/Adam tensor files','full raw Waymo/LiDAR dataset','large intermediate FP32 render NPZs','compiled CUDA/third-party binaries','complete training logs','unrelated old3/old5/R15 second-stage/B20-car experiments'],
      'included_NPZ_scope':'frozen road query/provenance, exact49 road targets, and two-actor query masks only; no model tensors',
      'pending':['prior_only independent zero-update delete/alpha','ownership_only independent zero-update delete/alpha','complete multi-channel visual quality review'],
      'human_verdict':None,'quality_accepted':False}
    write('PROVENANCE.json',provenance)
    architecture='''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 330"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#93bbd5"/></marker></defs><rect width="1120" height="330" fill="#122536"/><g fill="#23445b" stroke="#89aabc"><rect x="15" y="105" width="195" height="120" rx="12"/><rect x="275" y="20" width="300" height="80" rx="12"/><rect x="275" y="180" width="300" height="120" rx="12"/><rect x="650" y="180" width="215" height="120" rx="12"/><rect x="920" y="95" width="185" height="140" rx="12"/></g><g fill="none" stroke="#93bbd5" stroke-width="2" marker-end="url(#a)"><path d="M210 135H245V60H275"/><path d="M210 195H245V240H275"/><path d="M575 240H650"/><path d="M575 60H1010V95"/><path d="M865 240H890V195H920"/></g><g fill="white" font-family="Microsoft YaHei,sans-serif" text-anchor="middle" font-size="18"><text x="112" y="145">同五身份 / 49帧</text><text x="112" y="180">初始化协议 / seed0</text><text x="112" y="211">完整480×320</text><text x="425" y="52">原生GN11 · 第一阶段30K</text><text x="425" y="82">无本轮三个改进模块</text><text x="425" y="215">共同静动态归属修复</text><text x="425" y="247">严格静态 / 监督路由</text><text x="425" y="279">3D体积排除＋道路薄层保护</text><text x="757" y="214">跨帧真实 / 供体先验</text><text x="757" y="247">独立道路目标</text><text x="757" y="279">2×2开关 → 四组30K</text><text x="1012" y="133">5个配置对照</text><text x="1012" y="168">完整49帧视频</text><text x="1012" y="202">代码 / 来源 / 凭据</text></g></svg>'''
    write('architecture.svg',architecture)
    table='\n'.join('|'+ '|'.join([x['id'], '开' if x['ownership_fix'] else '关',str(x['road_weights']['observed_rgb']),str(x['road_weights']['prior_rgb']),x['independent49']])+'|' for x in experiments)
    readme=f'''# GN11 四组消融＋原生五身份基线：AI review材料包

共 **4组新实验＋1个原生基线**，均为五身份a6/a37/a5/a26/a18，raw89–137完整49帧，逻辑第一阶段30K，第二阶段0。基线严格对应用户指定页面中的 **五身份GN11第一阶段30K**；不包含其后实例训练或20车B实验。

## 从这里开始

1. 阅读 `REVIEW_TASK.md`，再看 `experiments.json` 的开关和实际完成状态。
2. 打开 `index.html` 或 `compare.html`：五个配置的原生四通道完整49帧，可同步播放/逐帧、四组总览和与原生A比较。
3. `independent_pair.html` 是full与real_only已经完成的独立七通道。其余两组独立删车/alpha仍待GPU，不存在的结果没有补造。
4. 先读 `code/diffs/baseline_to_full.diff`，再读 `full_to_*`。完整原生核心代码只存一份在 `code/base/`，四组实际变更存 `code/overlays/`。`code/source_map.json` 提供每组实际文件hash。
5. `evidence/runs/` 按五组分开：真实运行配置/状态、24个新实验保存节点CPU审计、原生30K评估事件、实际前100条采样日志、已完成的独立评估凭据。

| 配置 | 归属修复 | 真实道路RGB权重 | 供体RGB权重 | 额外独立49状态 |
|---|---|---|---|---|
{table}

深度权重为真实0.1、供体0.01，关闭时为0；没有重新分配或归一化。所有新组的static/dynamic/environment实际更新计数各30000。原生基线的历史status没有本轮新增的显式计数器，以其原生完成事件、模型节点元数据和恢复证据解释，不伪造计数器。

## 目录

- `code/`：原生核心、四组差异、实际入口/运行绑定、离线核查脚本和许可证；不含编译二进制。
- `evidence/`：按run分组的配置、保存/恢复凭据、冻结查询及49个真实道路训练目标、来源诊断和已做验证。
- `frames/`：raw89/129/137的评估GT PNG和五组完整画面样本，以及查询/供体全幅图；全部49帧GT保留在完整视频中。
- `media/`：37路完整49帧视频引用，按文件SHA256去重后29个实际MP4，另加1个基线固定相机时间诊断视频；均保留480×320完整画面，通道名称和来源独立保留。
- `MANIFEST.json`：逐文件SHA256/字节数；`PROVENANCE.json`：Git与范围；`VALIDATION.json`：打包检查。

## 必须区分的边界

- full＝归属修复＋真实曝光＋供体；real_only/prior_only也保留归属修复。ownership_only才是道路目标全关闭组。原生baseline没有本轮三类改进。
- 道路来源类别和目标颜色被冻结：7,787个同世界位置真实确认样本、5,119个异位置供体样本、2,963个未知样本。UNKNOWN从不监督道路；关闭一种来源不会把其区域改成另一种来源。样本计数不是均匀世界面积。
- a6的1,149个raw89查询射线当前无合格真实确认；245供体、904未知。a37的14,720个样本中，raw131–137确认4,765，全49确认7,787；供体4,874、未知2,059。没有确认不等于物理上永不曝光。
- 模型坐标锚帧raw99、道路/SGCM锚帧raw89，二者分开。供体框是干净沥青来源，不是query；白线、黄线、自行车标记边缘已过滤。
- 所有49帧都是训练重建视图，额外3D数据集框用于归属/遮挡约束；不是heldout，也没有隐藏道路GT。旧初始GPU张量快照不可得，只能核对输入/seed/采样协议，不能宣称逐位旧初始化相等。
- 原生基线曾因CUDA计时事件读取中断，保留20K并经明确授权重放丢失5,846步，逻辑终点30K；仅改同步/保存工程行为，没有本轮归属或道路目标。相关历史凭据附在baseline目录。
- 冻结config/bundle的 `execution` 字段是训练前的历史计划标记；当前完成状态以每组真实 `status.json` 和 `experiments.json` 为准。全部原始证据保持原字节，不为排版改写旧状态。
- 消融的effective_config/v2_policy保留冻结完整版目标描述；实际道路权重在各自overlay的v2_runtime_context.py中验证variant后覆盖。请同时核对实际CUDA gate的actual_road_weights和源码常量，不要只把旧配置里的完整版权重当作消融实际执行权重。
- full/real独立恢复模型、Adam、RNG、sampler、SGCM后零更新导出；prior在用户切无卡时中断，0PNG/无完成凭据；owner独立导出尚未启动。原生静态没有改名冒充独立删车。
- 实际全49静态观看仍见残留：full有矩形道路条纹/接缝；real有黄白亮块/蓝边；prior有黑块；owner有车状轮廓。未作干净静态质量通过判断，完整多通道质量复核未完成，human_verdict=null。

## 代码复原（仅复制，不训练）

```sh
python code/materialize.py --experiment full --output /path/to/new_empty_review_dir
```

此工具只重建/核对核心源码。实际入口保留远端路径；本包用于代码和结果review，**不包含全量数据、checkpoint/Adam权重、CUDA扩展和全部外部依赖，不是可直接续训的复现实验环境**。不运行任何入口也可完成静态review。

## Git和完整性

- Repository: https://github.com/yunlongdeng123/Motion-Proj
- Branch: `research/worldsim-v8.2-drivingeditor-debug`
- Code commit: `{code_commit}`
- Remote source snapshot collected UTC: `{snapshot['collected_utc']}`
- 打包仅CPU只读取源，无新GPU/训练更新；内容hash见MANIFEST。媒体复用既有完整文件，没有再裁剪或生成补帧。
'''
    write('README.md',readme)
    review='''# 请独立审查，不要把“工程检查通过”等同于效果通过

## 优先阅读

1. README.md / experiments.json：确认4组＋1基线，不把第二阶段或20车混进比较。
2. code/diffs/baseline_to_full.diff；code/overlays/full/utils/v2_static_partition.py、v2_runtime_context.py、v2_road_objective.py。
3. code/review_tools/world_road.py、prepare_real_inputs.py、audit_donor_provenance.py；evidence/input/bundle.json、targets/、actual_road_query_evidence.npz。
4. 各组effective_config与source_map/deployed_manifest；对照full_to_real/prior/ownership的diff，核对消融是否只改变预期道路开关。
5. compare.html静态→动态→完整重建。raw89/129/137的原始完整PNG在frames/samples，不要仅凭压缩视频或单帧下结论。

## 请回答

- 静态位置、时间opacity、时间宽度在训练/增密/剪枝/恢复/评估的各路径是否真的冻结？有没有通过尺度、SH、错误深度、遮挡顺序等继续解释车辆的通路？
- 车身RGB/depth/正则的监督路由和动态alpha完整mask是否一致？独立环境颜色detach有什么影响？哪些变化不是“仅道路”消融能隔离的？
- 3D框体积内静态opacity约束是否会漏掉框外/错误深度Gaussian，或误伤道路薄层？道路clearance/软边界/归一化单位是否匹配？不要建议将整个body mask的static alpha硬清零。
- Query是否准确反投影到同一个世界位置并以正确坐标投影到后续帧？遮挡、全部车辆、道路支撑、视角、深度/颜色一致性筛选有没有盲点？
- 融合来源为OBSERVED/PRIOR/UNKNOWN：真实曝光区域是否出现错误确认？供体是否把未观测部分变成固定颜色斑块或错误几何？49个冻结targets与query evidence是否一致？
- 道路loss是否仅回传静态，并与归属loss分开定义、仅在一次原生optimizer更新前累积？真实/先验关闭时其梯度是否实际为0？哪些UNKNOWN仍能经其他loss收到梯度？
- 2×2开关允许推断什么？覆盖区域不同、权重不同、单seed、额外3D标注和未知背景GT如何限制因果解释？
- baseline与ownership_only比较的是归属修复组合；不能把组合效果归因到内部某一个约束。原生基线有一次授权20K恢复，是否影响公平性应明确分析。
- 结合视频和完整PNG，分别指出可观测道路重建、先验纹理、未知区域、静态重复解释、动态主体保留、阴影的收益/回退。不要只用loss/alpha/节点完整性判干净。

## 输出期望

先按严重性列出有代码位置和证据的具体问题；区分确认事实、推断、缺失证据。给出最小修复建议和后续验证设计，不擅自增加训练组/预算。请保留两组独立渲染尚未完成这一事实；不存在的delete/alpha不能评成已通过。

## 输入说明

- bundle.json包含每帧world_to_camera、五身份3D框和道路target hash；targets/*.npz是实际冻结训练目标。保留原远端路径作为provenance，包内相对targets路径可读取。
- actual_road_query_evidence.npz为CPU几何/融合/来源证据，没有模型tensor。query_masks_a6_a37只有两车查询mask，不是五车总监督mask。
- frames/gt是原生评估导出的8-bit观测PNG；与独立导出GT的量化方法最大有1/255差异，不能假设跨导出文件hash相同。
- 原始源码/配置均保留。代码含绝对远端路径和外部依赖引用，不应直接运行训练入口。
'''
    write('REVIEW_TASK.md',review)
    rows=''.join(f'<tr><td>{html.escape(x["label"])}</td><td>{"开" if x["ownership_fix"] else "关"}</td><td>{x["road_weights"]["observed_rgb"]} / {x["road_weights"]["prior_rgb"]}</td><td>30K / 0</td><td>{x["independent49"]}</td></tr>' for x in experiments)
    index=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GN11 五配置AI review包</title><style>body{{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;background:#102131;color:#e7f1f8;max-width:1100px;margin:auto;padding:28px}}a{{color:#87d9ff}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #557}}img{{width:100%;height:auto}}.note{{background:#253d50;padding:16px;border-radius:10px}}</style>
<h1>四组30K消融＋原生五身份GN11基线</h1><p class="note"><a href="compare.html"><strong>打开五配置完整49帧同步视频</strong></a> · <a href="independent_pair.html">已完成两组的独立七通道</a></p>
<p>与用户指定基线一致：原生A五身份GN11第一阶段30K，没有本轮跨帧、供体、归属修复。其后实例阶段及20车B不在本包中。四组新实验都已经完成第一阶段30K、第二阶段0；两组独立删车/alpha仍待GPU，质量未通过验收。</p>
<img src="architecture.svg" alt="相同输入进入原生基线，或归属修复加真实与供体2x2开关，形成五配置49帧对照的组件箭头图">
<table><thead><tr><th>配置</th><th>归属修复</th><th>真实/先验RGB权重</th><th>一阶段/二阶段</th><th>独立49状态</th></tr></thead><tbody>{rows}</tbody></table>
<h2>AI阅读入口</h2><p><a href="README.md">材料索引、边界与基线恢复历史</a> · <a href="REVIEW_TASK.md">建议review问题</a> · <a href="experiments.json">机器可读实验表</a> · <a href="code/source_map.json">实际源码hash</a> · <a href="code/diffs/baseline_to_full.diff">基线到完整版diff</a></p>
<p><a href="evidence/native_static_review.json">已实际查看的全49静态结论</a> · <a href="evidence/independent_export_status.json">独立渲染完成/待GPU边界</a> · <a href="evidence/actual_road_summary.json">查询曝光来源计数</a> · <a href="evidence/media.json">37个主对照视频索引</a> · <a href="media/baseline_fixed_camera_time_only.mp4">基线固定相机、仅改变时间</a></p>
<p>所有视频和样本为完整480×320，无ROI；全49属于训练视图，没有干净隐藏道路GT。文件/解码/JS fixture通过不代表浏览器UI或视觉质量通过。</p><p><a href="PROVENANCE.json">Git提交与来源</a> · <a href="MANIFEST.json">逐文件清单</a> · <a href="VALIDATION.json">包内验证</a></p></html>'''
    write('index.html',index)
    assert len(video_records)==37
    estimated=sum(p.stat().st_size for p in D.rglob('*') if p.is_file())
    write('package_plan.json',{'version':1,'content_bytes_before_zip':estimated,'max_archive_bytes':100_000_000,'five_configurations':ARMS,
      'main_video_channel_references':37,'main_unique_physical_videos':len(physical_by_sha),'diagnostic_videos':1,'checkpoint_count':0,'full_training_logs_included':False})
    print(json.dumps({'stage':str(D),'bytes_before_zip':estimated,'main_videos':37,'diagnostic_videos':1,'code_commit':code_commit}))

def archive():
    assert (D/'VALIDATION.json').exists(),'validate before creating archive'
    entries=[{'path':p.relative_to(D).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(D.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']
    write('MANIFEST.json',{'algorithm':'SHA256','files':entries,'file_count':len(entries),'manifest_self_excluded':True})
    temp=ZIP.with_suffix('.zip.partial')
    if temp.exists():raise RuntimeError('preserve previous partial review zip')
    with zipfile.ZipFile(temp,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(D.rglob('*')):
            if p.is_file():z.write(p,'gn11_ai_review/'+p.relative_to(D).as_posix())
    assert temp.stat().st_size<100_000_000,'archive exceeds user size preference'
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        for e in entries:
            raw=z.read('gn11_ai_review/'+e['path']);assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256']
    if ZIP.exists():raise RuntimeError('preserve an existing delivered ZIP; version explicitly instead')
    temp.rename(ZIP)
    receipt={'status':'PACKAGED_HASH_VERIFIED_FOUR_ABLATIONS_PLUS_NATIVE_BASELINE','path':str(ZIP),'archive_bytes':ZIP.stat().st_size,'archive_MB':ZIP.stat().st_size/1e6,
       'sha256':sha(ZIP),'files':len(entries)+1,'code_commit':read(D/'PROVENANCE.json')['code_commit'],'max_bytes':100_000_000,'within_limit':True,
       'all_members_hash_verified':True,'main_video_channel_references':37,'main_unique_physical_videos':29,'diagnostic_videos':1,'checkpoints_included':False,'training_updates':0,'GPU_used':False,
       'independent_pending':['prior_only','ownership_only'],'quality_accepted':False,'human_verdict':None}
    (O/'ai_review_package_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (O/'gn11_ai_review_README.md').write_bytes((D/'README.md').read_bytes())
    print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--code-commit',required=True);p.add_argument('--zip',action='store_true');a=p.parse_args()
    build(a.code_commit)
    if a.zip:archive()
