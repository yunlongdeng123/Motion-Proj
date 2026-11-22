from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(r"C:\Users\dengyunlong\Documents\Codex\2026-11-16\xian")
REPORT = ROOT / "outputs/road_material_three_rounds_20261121/index.html"
WORK = Path(__file__).resolve().parent
WORK.mkdir(parents=True, exist_ok=True)
MARKER = b"<!-- NATIVE_SOURCE_PROVENANCE_CORRECTION_R5_BEGIN -->"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scripts(data: bytes):
    return [
        {"index": i, "tag": m.group(1).decode("utf-8"), "sha256": sha(m.group(2)), "bytes": len(m.group(2))}
        for i, m in enumerate(re.finditer(rb"(<script\b[^>]*>)(.*?)</script>", data, re.S | re.I))
    ]


before = REPORT.read_bytes()
if MARKER in before:
    raise SystemExit("来源纠正组已存在；请在现有组内修改。")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = WORK / f"index.before_native_source_correction_{stamp}.html.bak"
backup.write_bytes(before)

correction = r'''
<!-- NATIVE_SOURCE_PROVENANCE_CORRECTION_R5_BEGIN -->
<style id="native-source-correction-style">
#native-source-correction{border:2px solid #d6a04f;background:#1e2530;box-shadow:0 0 0 4px #d6a04f12;margin:20px 0 24px;scroll-margin-top:20px}
#native-source-correction h2{color:#ffd18a;margin-bottom:8px}
.nsc-verdict{color:#ffe1af;font-weight:650;font-size:16px}
.nsc-label{border-left:4px solid #d6a04f;background:#302b25;color:#ffe1af;padding:10px 12px;margin:12px 0;font-weight:600}
.nsc-label small{display:block;color:#d9c7a9;font-weight:400;margin-top:3px}
.nsc-static-label{border-left-color:#68bca4;background:#19332f;color:#a6e7d6}
.nsc-flow{display:flex;align-items:stretch;gap:10px;margin:14px 0;overflow-x:auto;padding:3px 0 8px}
.nsc-step{flex:1 0 146px;border:1px solid #55718a;border-radius:8px;background:#122534;padding:13px 12px;text-align:center}
.nsc-step strong{display:block;white-space:nowrap;color:#e4f1fb}
.nsc-step small{display:block;color:#aac2d5;font-size:12px;margin-top:4px}
.nsc-arrow{align-self:center;color:#8ed9c6;font-size:23px;flex:none}
#native-source-correction table{white-space:normal}
#native-source-correction td,#native-source-correction th{text-align:left}
#native-source-correction code{overflow-wrap:anywhere;color:#bacfe0;font-size:12px}
#native-source-correction h3{font-size:16px;margin:18px 0 4px}
#native-source-correction-entry{border-color:#d6a04f;color:#ffd18a}
@media(max-width:650px){.nsc-flow{gap:7px}.nsc-step{flex-basis:132px;padding:11px 9px}.nsc-arrow{font-size:20px}.nsc-verdict{font-size:15px}}
</style>
<script id="native-source-correction-script">
/* 仅纠正界面来源口径；既有实验数据与数值保持原样。 */
(() => {
  const q = id => document.getElementById(id);
  if (q('native-source-correction')) return;
  const main = document.querySelector('main');
  if (!main) return;
  const group = document.createElement('section');
  group.id = 'native-source-correction';
  group.setAttribute('aria-labelledby', 'native-source-correction-title');
  group.innerHTML = `<h2 id="native-source-correction-title">来源纠正｜主要00／02源块来自加工道路</h2>
    <p class="nsc-verdict">加工道路对照；不作为原生重建源迁移证据</p>
    <div class="scroll"><table><thead><tr><th>来源</th><th>对应结果</th><th>证据口径</th></tr></thead><tbody>
      <tr><td>已加工道路的00／02路块</td><td>第二轮颜色链路、第三轮多帧拟合、第四轮核心迁移</td><td>加工道路对照</td></tr>
      <tr><td>未编辑原生静态检查点</td><td>第二轮两组独立静态诊断</td><td>原生静态对照；原坐标选块，尚未完成干净源块流程</td></tr>
    </tbody></table></div>
    <h3>正确链路</h3>
    <div class="nsc-flow" role="img" aria-label="未编辑重建，经干净路块选取、独立颜色与透明度核验、越界足迹约束，输出目标道路合成">
      <div class="nsc-step"><strong>未编辑重建</strong><small>原生检查点</small></div><span class="nsc-arrow" aria-hidden="true">→</span>
      <div class="nsc-step"><strong>选块</strong><small>干净路块</small></div><span class="nsc-arrow" aria-hidden="true">→</span>
      <div class="nsc-step"><strong>独立核验</strong><small>颜色与透明度</small></div><span class="nsc-arrow" aria-hidden="true">→</span>
      <div class="nsc-step"><strong>足迹约束</strong><small>处理越界投影</small></div><span class="nsc-arrow" aria-hidden="true">→</span>
      <div class="nsc-step"><strong>目标合成</strong><small>道路输出</small></div>
    </div>
    <p class="muted">从未编辑原生重建重新选取干净源块，再按上述链路核验。原生干净源块的新结果尚未纳入。</p>
    <details><summary>加工源资产记录</summary><p><code>road_surface_residual_20261121_r2/appearance_fit/asset</code></p><p class="muted">原有00／02路块已继承道路加工与外观拟合。</p></details>
    <div id="native-source-correction-results"></div>`;
  main.querySelector('h1').after(group);

  const flag = (id, detail = '') => {
    const panel = q(id);
    if (!panel) return;
    const label = document.createElement('p');
    label.className = 'nsc-label';
    label.dataset.sourceProvenance = 'processed-road';
    label.append(document.createTextNode('加工道路对照；不作为原生重建源迁移证据'));
    if (detail) {
      const small = document.createElement('small');
      small.textContent = detail;
      label.append(small);
    }
    panel.querySelector('h2').after(label);
  };
  flag('source-fidelity-chain', '本标注针对主要00／02路块；两组原生静态对照独立保留。');
  flag('source-multiframe-pair');
  flag('clean-core-transfer');
  if (q('source-fidelity-chain')) q('source-fidelity-chain').querySelector('h2').textContent = '源块质量核查 · 加工道路对照';
  if (q('source-multiframe-pair')) q('source-multiframe-pair').querySelector('h2').textContent = '多帧参考池配对 · 加工道路对照';
  if (q('clean-core-transfer')) q('clean-core-transfer').querySelector('h2').textContent = '三维核心与单块迁移 · 加工道路对照';

  const staticDetails = q('sfc-static-details');
  if (staticDetails) {
    const label = document.createElement('p');
    label.className = 'nsc-label nsc-static-label';
    label.dataset.sourceProvenance = 'native-static';
    label.textContent = '原生静态对照：两组诊断直接来自未编辑静态检查点；按原坐标选块，尚未完成干净源块流程。';
    staticDetails.querySelector('summary').after(label);
  }

  /* 覆盖呈现层文本，保持版本键、路径、数值与JSON原文不变。 */
  const panel = q('clean-core-transfer');
  if (panel) {
    const rewrite = root => {
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);
      for (const node of nodes) {
        if (node.parentElement?.closest('script,style')) continue;
        const text = node.nodeValue
          .replaceAll('原资产颜色', '加工道路·未再次改色')
          .replaceAll('原始颜色', '加工道路·未再次改色')
          .replaceAll('原资产', '加工道路·未再次改色');
        if (text !== node.nodeValue) node.nodeValue = text;
      }
      for (const image of root.querySelectorAll('img[alt]')) {
        const text = image.alt.replaceAll('原资产颜色', '加工道路·未再次改色').replaceAll('原始颜色', '加工道路·未再次改色').replaceAll('原资产', '加工道路·未再次改色');
        if (text !== image.alt) image.alt = text;
      }
    };
    rewrite(panel);
    const observer = new MutationObserver(() => {
      observer.disconnect();
      rewrite(panel);
      observer.observe(panel, {childList:true, subtree:true, characterData:true});
    });
    observer.observe(panel, {childList:true, subtree:true, characterData:true});
  }

  const entry = document.createElement('button');
  entry.id = 'native-source-correction-entry';
  entry.textContent = '来源纠正';
  entry.onclick = () => group.scrollIntoView({block:'start'});
  document.querySelector('nav')?.prepend(entry);
  const fragment = () => {if (location.hash === '#native-source-correction') entry.click();};
  window.addEventListener('hashchange', fragment);
  fragment();
})();
</script>
<!-- NATIVE_SOURCE_PROVENANCE_CORRECTION_R5_END -->
'''

added = correction.replace("\n", "\r\n").encode("utf-8")
REPORT.write_bytes(before + added)
after = REPORT.read_bytes()
old_scripts = scripts(before)
new_scripts = scripts(after)
assert after[: len(before)] == before
assert new_scripts[: len(old_scripts)] == old_scripts
manifest = {
    "report": str(REPORT),
    "backup": str(backup),
    "before_sha256": sha(before),
    "after_sha256": sha(after),
    "original_bytes_preserved_as_exact_prefix": True,
    "all_original_script_blocks_exact": True,
    "original_script_blocks": old_scripts,
    "raw_data_scope": "首个原脚本整体以及全部历史JSON原文；D/r1/r2/r3不作任何修改",
    "correction_group": "native-source-correction",
    "append_only_bytes": len(added),
    "native_static_exception": "R2两组直接来自未编辑静态检查点的独立诊断保持原生对照身份",
}
manifest_path = WORK / "source_correction_integrity.json"
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"report":str(REPORT), "backup":str(backup), "manifest":str(manifest_path), "original_scripts_preserved":len(old_scripts), "added_bytes":len(added)}, ensure_ascii=False))
