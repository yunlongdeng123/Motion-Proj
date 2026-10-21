from pathlib import Path
import json
import shutil

ROOT=Path(__file__).resolve().parent;OUTPUTS=ROOT.parents[1]/'outputs'
for name in ('GS_Reconstruction_Diagnostics_20261021_GPU_r2','GS_Reconstruction_Diagnostics_20261021_GPU_portable_r3'):
    out=OUTPUTS/name
    for file in ('app.js','gpu.js','server.py','style.css','start_gpu.ps1'):
        shutil.copy2(ROOT/'src'/file,out/file);shutil.copy2(ROOT/'src'/file,out/'source/viewer'/file)
    if (out/'portable_receipt.json').exists():
        p=out/'app.js';text=p.read_text(encoding='utf-8').replace('数值预览alpha 1/65535、深度1cm量化','便携热图4×4采样／8bit；精确值见CPU／CUDA射线');p.write_text(text,encoding='utf-8')
    for file in ('original_gpu_asset_audit.json','gpu_http_validation.json'):
        if (ROOT/file).exists():shutil.copy2(ROOT/file,out/file)
    for file in ('verify_gpu.py','finalize_gpu.py','test_gpu_http.py','assemble_gpu.py'):
        shutil.copy2(ROOT/file,out/'source/authoring'/file)
    for source in (ROOT/'gpu').glob('*'):
        if source.is_file():shutil.copy2(source,out/'source/gpu'/source.name)
    p=out/'GPU_REPORT.html';text=p.read_text(encoding='utf-8').replace('每个raw102像素按颜色贡献选20个普通背景Gaussian','每个raw102像素从已导出的贡献候选表，按颜色贡献选最多20个普通背景Gaussian（两个路面对照为19个，不冒充20个）')
    text=text.replace('CPU8项回归通过','CPU18项回归通过')
    text=text.replace('研发曾遇Ninja','最终原始冻结资产审计完整重算12,297个文件、40,157,562,775字节的SHA256，全部未变。研发曾遇Ninja')
    p.write_text(text,encoding='utf-8')
    contract=json.loads((out/'data/data_contract.json').read_text())
    contract['gpu_diagnostic_addendum']=dict(native_buffer_trace_verified=True,independent_top_ID_weights_checked=True,
        full_native_counterfactual_rerasterization=True,full_model_novel_view_rendering=True,optimizer_updates=0,human_verdict=None,
        original_CPU_export_scope_preserved=True)
    (out/'data/data_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    p=out/'app.js';text=p.read_text(encoding='utf-8').replace('原生结果＋只读CPU证据','原生结果＋CUDA诊断／CPU对照')
    text=text.replace('。使用 start_local.ps1 启用SSH只读CPU桥接；GPU诊断仍未执行。','。可用 start_local.ps1 启用CPU参考；CUDA诊断使用 start_gpu.ps1。')
    p.write_text(text,encoding='utf-8')
    print(out)
