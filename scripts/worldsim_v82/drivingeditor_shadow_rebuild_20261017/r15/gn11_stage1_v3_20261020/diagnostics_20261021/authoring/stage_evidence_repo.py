"""只暂存本轮源码/小凭据；不把大图片或checkpoint纳入Git。"""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent;out=ROOT/'repo_evidence_stage';out.mkdir(exist_ok=True)
release=ROOT.parents[1]/'outputs/GS_Reconstruction_Diagnostics_20261021_Evidence_r1'
for source,dest in [('src','src'),('gpu','gpu'),('tests','tests')]:
    for file in (ROOT/source).iterdir():
        if file.is_file():
            target=out/dest/file.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
shutil.copy2(ROOT/'v1_binding.json',out/'gpu/v1_binding.json')
for filename in ('build_evidence.py','verify_evidence.py','finalize_evidence.py','stage_evidence_repo.py'):
    target=out/'authoring'/filename;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/filename,target)
for filename in ('v1_receipt.json','evidence_cases.json','evidence_receipt.json','browser_evidence_qa.json','evidence_original_asset_audit.json'):
    shutil.copy2(release/filename,out/filename)
(out/'EVIDENCE_REPORT.html').write_text((release/'EVIDENCE_REPORT.html').read_text(encoding='utf-8').replace('href="index.html"','href="README.html"'),encoding='utf-8',newline='\n')
(out/'README.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>只读诊断源码索引</title><style>body{background:#0b1421;color:#dbe7f7;font:16px/1.8 system-ui;max-width:1000px;margin:40px auto}a{color:#8cc0ff}</style><h1>五实验冻结30K可追责诊断</h1><p><a href="EVIDENCE_REPORT.html">几何判别、有效区验收与V1技术报告／组件箭头图</a> · <a href="REPORT.html">原GPU归因报告</a> · <a href="v1_receipt.json">V1原生回放来源</a> · <a href="browser_evidence_qa.json">浏览器验收</a></p><p>这里只有源码和轻量证据，不附大模型/图片。完整本地成品位于本聊天outputs/GS_Reconstruction_Diagnostics_20261021_Evidence_r1/index.html；便携包GS_Reconstruction_Diagnostics_20261021_Evidence_portable_r1.zip，解压后打开index.html。</p><h2>重现</h2><p>src/为界面/127.0.0.1服务，gpu/为冻结CUDA归因、几何证据和完整重新光栅化。部署在/root/autodl-tmp/gsdiag_gpu_20261021_r2/，保留r1与全部失败现场。gpu_backend.py --stdio按gpu_slot.lock单槽运行；所有新结果写runs独立目录，不改原生engine或checkpoint。迁移前核对固定campaign、V1来源、rasterizer binary和server.py入口，不能覆盖旧输出。V1绑定30K SHA且不注入道路，历史恢复与V3fresh不混称同合同。</p><p>Windows启动start_gpu.ps1，CPU参考仍限四组V3；V1与四组均可实时GPU归因。Ctrl+C结束自己的worker。python -m unittest discover -s tests -q运行CPU回归。authoring/build_evidence.py及finalize_evidence.py按原工作区布局装配新目录；verify_evidence.py检查来源、180条归因、固定分母和49帧，以及ZIP完整性/100MB上限。原始未量化NPZ始终是指标输入，便携图只作显示。</p><p>测量标签是当前射线保守候选，不是自动删点规则；条件z代理不是首交面。非目标身份/深度不足为未知；隐藏道路没有伪造RGB真值。所有操作优化更新0、human_verdict=null、quality_accepted=false。当前阶段/授权只见docs/RESEARCH_STATUS.md；不自动关机或恢复旧心跳。遵守src/LICENSE_gaussian_splatting.md研究许可。</p></html>''',encoding='utf-8',newline='\n')
print(out)
