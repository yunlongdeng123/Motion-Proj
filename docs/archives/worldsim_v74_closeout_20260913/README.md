# V7.4 分支大资产归档

最终精简：GitHub 实际下载 ZIP 约 **22.2 MB**；同口径本地打包105.25→21.45 MB，减少约80%，跟踪文件展开约53.6 MB（原375.9 MB）；两批共307个资产、原文件322.49 MB已归档。全部代码/配置/Markdown、失败查询索引和V7.4报告图保留。旧下载包不会自动变小，需重新下载当前分支；完整 Git 历史保持，未强推改写。 [测量快照](size_report.json)记录添加本说明前的暂存树；最终ZIP仅多出本说明等小文本。

追加精简完成：两批合计移出 307 个批量/生成资产，原始文件共 322.49 MB；两个完整包均已保存远端和本地副本。V7.4 核心图保留，旧论文与 V7.3 非架构图改为按需恢复。 见[追加归档](LEGACY_MEDIA.md)。下文241个文件是第一批清单。

2026-09-13；`WS-V74-CLOSEOUT-01`。241 个资产共 298,113,778 bytes 从当前分支移出，完整归档 60,337,519 bytes（约60.34 MB）。代码、配置、所有 Markdown、失败卡/查询索引、报告内嵌图与核心摘要保留。原始 runs 不删除。

远端完整包：`/root/autodl-tmp/research_archives/worldsim_v74_closeout_20260913/removed_branch_assets.tar.gz`。
本地完整副本：`C:/Users/dengyunlong/Documents/Codex/2026-09-10/root-autodl-tmp-motion-proj-docs/outputs/v74_closeout/removed_branch_assets.tar.gz`。
归档前提交：`dd759123e9f151ac4f10ef633b4a8316b6c1201b`。本次普通提交不改写共享历史；V73/H1 分支不动。旧提交仍可恢复原文件，完整历史 clone 不会因当前树删除而自动变小；分支 ZIP 和浅克隆会缩小。

[逐文件清单](manifest.json) 记录路径、大小、类别；[纯路径列表](paths.txt) 便于 `rg` 查询。大表原值完整保存，没有用删行后的“摘要”冒充原 JSON。需要某个旧脚本的输入时，先按原相对路径恢复该文件即可。

```bash
# 只恢复所需文件；替换最后一个参数，不必展开整包。
tar -xzf /root/autodl-tmp/research_archives/worldsim_v74_closeout_20260913/removed_branch_assets.tar.gz -C /root/autodl-tmp/motion_proj docs/autoresearch/worldsim_v74_h2/p16/control_rows.json
# 归档包不可用时，可从保留的 Git 历史读取原文件。
git show dd759123e9f151ac4f10ef633b4a8316b6c1201b:docs/autoresearch/worldsim_v74_h2/p16/control_rows.json
# 轻量获取当前分支代码和文档。
git clone --depth 1 --single-branch --branch research/worldsim-v7.4-h2-generative-surface https://github.com/yunlongdeng123/Motion-Proj.git
```

恢复后的大资产由 `.gitignore` 排除，防止再次误提交。历史 failure 分片和固定行号索引不改写；其中的旧路径可在本清单查找。报告文字链接改指本目录对应条目，原路径保留在条目内。下列清单按需读取即可，不必默认全文载入。

<a id="asset-001"></a>

### 001 · HYPOTHESES.jsonl

`docs/autoresearch/worldsim_v6/HYPOTHESES.jsonl`；492,178 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-002"></a>

### 002 · REFLECTIONS.jsonl

`docs/autoresearch/worldsim_v6/REFLECTIONS.jsonl`；150,474 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-003"></a>

### 003 · av2_external20_index.json

`docs/autoresearch/worldsim_v73/coverage/av2_external20_index.json`；683,346 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-004"></a>

### 004 · av2_old_fixed_r1_analysis.json

`docs/autoresearch/worldsim_v73/coverage/av2_old_fixed_r1_analysis.json`；114,317 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-005"></a>

### 005 · av2_old_joint_r5_analysis.json

`docs/autoresearch/worldsim_v73/coverage/av2_old_joint_r5_analysis.json`；159,218 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-006"></a>

### 006 · capa_build_conditions_r1.json

`docs/autoresearch/worldsim_v73/coverage/capa_build_conditions_r1.json`；176,452 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-007"></a>

### 007 · fit_track_payload_inventory_r1.json

`docs/autoresearch/worldsim_v73/coverage/fit_track_payload_inventory_r1.json`；2,822,639 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-008"></a>

### 008 · fit_track_targets_r2_index.json

`docs/autoresearch/worldsim_v73/coverage/fit_track_targets_r2_index.json`；443,681 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-009"></a>

### 009 · log_payload_inventory_r1.json

`docs/autoresearch/worldsim_v73/coverage/log_payload_inventory_r1.json`；260,389 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-010"></a>

### 010 · m1_r2.json

`docs/autoresearch/worldsim_v73/coverage/m1_r2.json`；117,926 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-011"></a>

### 011 · m1_r3.json

`docs/autoresearch/worldsim_v73/coverage/m1_r3.json`；196,681 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-012"></a>

### 012 · V73_FINAL_EXTERNAL20_PAIRS.pdf

`docs/autoresearch/worldsim_v73/final_confirmation/V73_FINAL_EXTERNAL20_PAIRS.pdf`；21,018 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-013"></a>

### 013 · analysis.json

`docs/autoresearch/worldsim_v73/final_confirmation/analysis.json`；7,640,360 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-014"></a>

### 014 · analysis.json

`docs/autoresearch/worldsim_v73/final_scene/analysis.json`；492,926 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-015"></a>

### 015 · summary.json

`docs/autoresearch/worldsim_v73/final_scene/summary.json`；3,015,994 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-016"></a>

### 016 · V73_SURFACE_SUPPORT.pdf

`docs/autoresearch/worldsim_v73/m2/V73_SURFACE_SUPPORT.pdf`；17,983 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-017"></a>

### 017 · V73_ACTOR_TSDF.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_ACTOR_TSDF.pdf`；18,514 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-018"></a>

### 018 · V73_ACTOR_TSDF.png

`docs/autoresearch/worldsim_v73/m2/global/V73_ACTOR_TSDF.png`；127,518 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-019"></a>

### 019 · V73_ADAPOINTR_DENSITY.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_ADAPOINTR_DENSITY.pdf`；29,647 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-020"></a>

### 020 · V73_ADAPOINTR_DENSITY.png

`docs/autoresearch/worldsim_v73/m2/global/V73_ADAPOINTR_DENSITY.png`；213,625 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-021"></a>

### 021 · V73_JOINT_POPULATION_RESULTS.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_JOINT_POPULATION_RESULTS.pdf`；35,457 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-022"></a>

### 022 · V73_JOINT_POPULATION_RESULTS.png

`docs/autoresearch/worldsim_v73/m2/global/V73_JOINT_POPULATION_RESULTS.png`；228,384 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-023"></a>

### 023 · V73_JOINT_R10_PAIRS.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_JOINT_R10_PAIRS.pdf`；19,932 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-024"></a>

### 024 · V73_JOINT_R10_TRAINING.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_JOINT_R10_TRAINING.pdf`；27,127 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-025"></a>

### 025 · V73_JOINT_R12_PAIRS.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_JOINT_R12_PAIRS.pdf`；19,837 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-026"></a>

### 026 · V73_JOINT_R12_TRAINING.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_JOINT_R12_TRAINING.pdf`；27,010 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-027"></a>

### 027 · V73_JOINT_R5_TRAINING.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_JOINT_R5_TRAINING.pdf`；26,475 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-028"></a>

### 028 · V73_JOINT_R5_TRAINING.png

`docs/autoresearch/worldsim_v73/m2/global/V73_JOINT_R5_TRAINING.png`；338,684 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-029"></a>

### 029 · V73_MOVING_QUERY_INTERSECTIONS.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_MOVING_QUERY_INTERSECTIONS.pdf`；460,361 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-030"></a>

### 030 · V73_MOVING_QUERY_INTERSECTIONS_1.png

`docs/autoresearch/worldsim_v73/m2/global/V73_MOVING_QUERY_INTERSECTIONS_1.png`；274,909 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-031"></a>

### 031 · V73_MOVING_QUERY_INTERSECTIONS_2.png

`docs/autoresearch/worldsim_v73/m2/global/V73_MOVING_QUERY_INTERSECTIONS_2.png`；307,630 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-032"></a>

### 032 · V73_MOVING_QUERY_INTERSECTIONS_3.png

`docs/autoresearch/worldsim_v73/m2/global/V73_MOVING_QUERY_INTERSECTIONS_3.png`；278,000 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-033"></a>

### 033 · V73_NATIVE_R14_PAIRS.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_NATIVE_R14_PAIRS.pdf`；19,532 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-034"></a>

### 034 · V73_NATIVE_R14_TRAINING.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_NATIVE_R14_TRAINING.pdf`；26,694 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-035"></a>

### 035 · V73_POPULATION_FREE_TRADEOFFS.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_POPULATION_FREE_TRADEOFFS.pdf`；29,068 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-036"></a>

### 036 · V73_POPULATION_FREE_TRADEOFFS.png

`docs/autoresearch/worldsim_v73/m2/global/V73_POPULATION_FREE_TRADEOFFS.png`；228,720 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-037"></a>

### 037 · V73_QUERY_PROVENANCE.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_QUERY_PROVENANCE.pdf`；28,630 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-038"></a>

### 038 · V73_SURFACE_SUPPORT_MATCHED_R2.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_SURFACE_SUPPORT_MATCHED_R2.pdf`；18,353 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-039"></a>

### 039 · V73_UPPER_ADAPTATION_PROPOSAL.pdf

`docs/autoresearch/worldsim_v73/m2/global/V73_UPPER_ADAPTATION_PROPOSAL.pdf`；15,307 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-040"></a>

### 040 · V73_UPPER_ADAPTATION_PROPOSAL.png

`docs/autoresearch/worldsim_v73/m2/global/V73_UPPER_ADAPTATION_PROPOSAL.png`；159,087 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-041"></a>

### 041 · actor_tsdf_r1_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/actor_tsdf_r1_analysis.json`；4,472,835 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-042"></a>

### 042 · actor_tsdf_r1_construction.json

`docs/autoresearch/worldsim_v73/m2/global/actor_tsdf_r1_construction.json`；629,929 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-043"></a>

### 043 · adapointr_density_r1_summary.json

`docs/autoresearch/worldsim_v73/m2/global/adapointr_density_r1_summary.json`；499,342 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-044"></a>

### 044 · adapointr_r1_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/adapointr_r1_analysis.json`；3,555,862 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-045"></a>

### 045 · adapointr_r1_initial_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/adapointr_r1_initial_analysis.json`；678,309 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-046"></a>

### 046 · adapointr_r2_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/adapointr_r2_analysis.json`；4,992,369 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-047"></a>

### 047 · capa_r2_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/capa_r2_analysis.json`；2,829,821 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-048"></a>

### 048 · empty_fixed_r2_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/empty_fixed_r2_analysis.json`；381,088 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-049"></a>

### 049 · empty_fixed_r2_summary.json

`docs/autoresearch/worldsim_v73/m2/global/empty_fixed_r2_summary.json`；186,773 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-050"></a>

### 050 · fusion_r1_summary.json

`docs/autoresearch/worldsim_v73/m2/global/fusion_r1_summary.json`；129,842 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-051"></a>

### 051 · mechanism_tradeoffs.pdf

`docs/autoresearch/worldsim_v73/m2/global/mechanism_tradeoffs.pdf`；23,419 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-052"></a>

### 052 · mechanism_tradeoffs.png

`docs/autoresearch/worldsim_v73/m2/global/mechanism_tradeoffs.png`；195,108 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-053"></a>

### 053 · population_fusion_r2_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/population_fusion_r2_analysis.json`；1,375,386 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-054"></a>

### 054 · population_fusion_r2_summary.json

`docs/autoresearch/worldsim_v73/m2/global/population_fusion_r2_summary.json`；4,370,365 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-055"></a>

### 055 · population_joint_r10_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/population_joint_r10_analysis.json`；5,339,335 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-056"></a>

### 056 · population_joint_r10_summary.json

`docs/autoresearch/worldsim_v73/m2/global/population_joint_r10_summary.json`；6,889,020 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-057"></a>

### 057 · population_joint_r12_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/population_joint_r12_analysis.json`；4,641,715 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-058"></a>

### 058 · population_joint_r12_summary.json

`docs/autoresearch/worldsim_v73/m2/global/population_joint_r12_summary.json`；6,898,604 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-059"></a>

### 059 · population_joint_r5_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/population_joint_r5_analysis.json`；7,199,431 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-060"></a>

### 060 · population_lidar_r6_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/population_lidar_r6_analysis.json`；2,100,613 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-061"></a>

### 061 · population_lidar_r6_summary.json

`docs/autoresearch/worldsim_v73/m2/global/population_lidar_r6_summary.json`；6,544,045 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-062"></a>

### 062 · population_lidar_r7_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/population_lidar_r7_analysis.json`；3,572,625 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-063"></a>

### 063 · population_lidar_r7_summary.json

`docs/autoresearch/worldsim_v73/m2/global/population_lidar_r7_summary.json`；6,548,249 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-064"></a>

### 064 · population_lidar_r8_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/population_lidar_r8_analysis.json`；3,588,404 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-065"></a>

### 065 · population_lidar_r8_summary.json

`docs/autoresearch/worldsim_v73/m2/global/population_lidar_r8_summary.json`；6,540,676 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-066"></a>

### 066 · population_native_r11_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/population_native_r11_analysis.json`；5,316,844 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-067"></a>

### 067 · population_native_r11_summary.json

`docs/autoresearch/worldsim_v73/m2/global/population_native_r11_summary.json`；6,930,231 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-068"></a>

### 068 · population_native_r14_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/population_native_r14_analysis.json`；3,834,208 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-069"></a>

### 069 · population_native_r14_summary.json

`docs/autoresearch/worldsim_v73/m2/global/population_native_r14_summary.json`；6,984,486 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-070"></a>

### 070 · population_r9_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/population_r9_analysis.json`；3,587,523 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-071"></a>

### 071 · query_provenance_r1_summary.json

`docs/autoresearch/worldsim_v73/m2/global/query_provenance_r1_summary.json`；401,945 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-072"></a>

### 072 · r1_log_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/r1_log_analysis.json`；165,887 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-073"></a>

### 073 · r1_summary.json

`docs/autoresearch/worldsim_v73/m2/global/r1_summary.json`；377,906 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-074"></a>

### 074 · r2_free_attribution.json

`docs/autoresearch/worldsim_v73/m2/global/r2_free_attribution.json`；114,192 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-075"></a>

### 075 · r2_log_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/r2_log_analysis.json`；166,514 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-076"></a>

### 076 · r2_summary.json

`docs/autoresearch/worldsim_v73/m2/global/r2_summary.json`；384,864 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-077"></a>

### 077 · r3_free_attribution.json

`docs/autoresearch/worldsim_v73/m2/global/r3_free_attribution.json`；114,121 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-078"></a>

### 078 · r3_log_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/r3_log_analysis.json`；166,697 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-079"></a>

### 079 · r3_summary.json

`docs/autoresearch/worldsim_v73/m2/global/r3_summary.json`；383,939 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-080"></a>

### 080 · r4_free_attribution.json

`docs/autoresearch/worldsim_v73/m2/global/r4_free_attribution.json`；117,293 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-081"></a>

### 081 · r4_log_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/r4_log_analysis.json`；174,207 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-082"></a>

### 082 · r4_summary.json

`docs/autoresearch/worldsim_v73/m2/global/r4_summary.json`；391,584 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-083"></a>

### 083 · surface_support_r1_summary.json

`docs/autoresearch/worldsim_v73/m2/global/surface_support_r1_summary.json`；599,631 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-084"></a>

### 084 · surface_support_r2_summary.json

`docs/autoresearch/worldsim_v73/m2/global/surface_support_r2_summary.json`；400,074 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-085"></a>

### 085 · visual_only_r13_analysis.json

`docs/autoresearch/worldsim_v73/m2/global/visual_only_r13_analysis.json`；298,658 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-086"></a>

### 086 · visual_only_r13_summary.json

`docs/autoresearch/worldsim_v73/m2/global/visual_only_r13_summary.json`；558,000 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-087"></a>

### 087 · window_population_r2_index.json

`docs/autoresearch/worldsim_v73/m2/global/window_population_r2_index.json`；411,548 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-088"></a>

### 088 · V73_BUILD_FREE_BACKGROUND.pdf

`docs/autoresearch/worldsim_v73/m4/V73_BUILD_FREE_BACKGROUND.pdf`；33,160 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-089"></a>

### 089 · V73_BUILD_FREE_BACKGROUND.png

`docs/autoresearch/worldsim_v73/m4/V73_BUILD_FREE_BACKGROUND.png`；235,398 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-090"></a>

### 090 · V73_SCENE_BACKGROUND_COMPOSITION.pdf

`docs/autoresearch/worldsim_v73/m4/V73_SCENE_BACKGROUND_COMPOSITION.pdf`；22,297 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-091"></a>

### 091 · V73_SCENE_BACKGROUND_COMPOSITION.png

`docs/autoresearch/worldsim_v73/m4/V73_SCENE_BACKGROUND_COMPOSITION.png`；149,575 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-092"></a>

### 092 · V73_TRAJECTORY_EDIT.pdf

`docs/autoresearch/worldsim_v73/m4/V73_TRAJECTORY_EDIT.pdf`；64,780 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-093"></a>

### 093 · V73_TRAJECTORY_EDIT.png

`docs/autoresearch/worldsim_v73/m4/V73_TRAJECTORY_EDIT.png`；280,665 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-094"></a>

### 094 · V73_VDB_BACKGROUND.pdf

`docs/autoresearch/worldsim_v73/m4/V73_VDB_BACKGROUND.pdf`；37,034 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-095"></a>

### 095 · av2_old_scene_data_index.json

`docs/autoresearch/worldsim_v73/m4/av2_old_scene_data_index.json`；1,509,998 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-096"></a>

### 096 · av2_old_scene_r1_summary.json

`docs/autoresearch/worldsim_v73/m4/av2_old_scene_r1_summary.json`；104,788 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-097"></a>

### 097 · av2_old_scene_r2_summary.json

`docs/autoresearch/worldsim_v73/m4/av2_old_scene_r2_summary.json`；104,824 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-098"></a>

### 098 · av2_old_vdb_scene_r4_paired.json

`docs/autoresearch/worldsim_v73/m4/av2_old_vdb_scene_r4_paired.json`；104,158 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-099"></a>

### 099 · av2_old_vdb_scene_r4_summary.json

`docs/autoresearch/worldsim_v73/m4/av2_old_vdb_scene_r4_summary.json`；132,746 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-100"></a>

### 100 · observed_points_r1_summary.json

`docs/autoresearch/worldsim_v73/m4/observed_points_r1_summary.json`；808,177 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-101"></a>

### 101 · scene_composition_r1_summary.json

`docs/autoresearch/worldsim_v73/m4/scene_composition_r1_summary.json`；237,660 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-102"></a>

### 102 · scene_composition_r2_paired.json

`docs/autoresearch/worldsim_v73/m4/scene_composition_r2_paired.json`；114,706 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-103"></a>

### 103 · scene_composition_r2_summary.json

`docs/autoresearch/worldsim_v73/m4/scene_composition_r2_summary.json`；299,790 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-104"></a>

### 104 · scene_composition_r3_summary.json

`docs/autoresearch/worldsim_v73/m4/scene_composition_r3_summary.json`；299,563 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-105"></a>

### 105 · scene_composition_r4_summary.json

`docs/autoresearch/worldsim_v73/m4/scene_composition_r4_summary.json`；531,874 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-106"></a>

### 106 · scene_composition_r5_paired.json

`docs/autoresearch/worldsim_v73/m4/scene_composition_r5_paired.json`；327,702 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-107"></a>

### 107 · scene_composition_r5_summary.json

`docs/autoresearch/worldsim_v73/m4/scene_composition_r5_summary.json`；531,533 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-108"></a>

### 108 · scene_composition_r6_paired.json

`docs/autoresearch/worldsim_v73/m4/scene_composition_r6_paired.json`；111,165 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-109"></a>

### 109 · scene_composition_r6_summary.json

`docs/autoresearch/worldsim_v73/m4/scene_composition_r6_summary.json`；287,740 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-110"></a>

### 110 · scene_composition_r7_paired.json

`docs/autoresearch/worldsim_v73/m4/scene_composition_r7_paired.json`；134,856 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-111"></a>

### 111 · scene_composition_r7_summary.json

`docs/autoresearch/worldsim_v73/m4/scene_composition_r7_summary.json`；287,826 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-112"></a>

### 112 · scene_data_carved_r3_index.json

`docs/autoresearch/worldsim_v73/m4/scene_data_carved_r3_index.json`；313,693 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-113"></a>

### 113 · scene_data_r1_index.json

`docs/autoresearch/worldsim_v73/m4/scene_data_r1_index.json`；301,643 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-114"></a>

### 114 · scene_data_r2_index.json

`docs/autoresearch/worldsim_v73/m4/scene_data_r2_index.json`；303,927 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-115"></a>

### 115 · vdb_background_r4_construction.json

`docs/autoresearch/worldsim_v73/m4/vdb_background_r4_construction.json`；314,216 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-116"></a>

### 116 · vdb_background_r5_construction.json

`docs/autoresearch/worldsim_v73/m4/vdb_background_r5_construction.json`；323,709 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-117"></a>

### 117 · vdb_scene_r8_summary.json

`docs/autoresearch/worldsim_v73/m4/vdb_scene_r8_summary.json`；778,767 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-118"></a>

### 118 · vdb_scene_r9_joint_background_change.json

`docs/autoresearch/worldsim_v73/m4/vdb_scene_r9_joint_background_change.json`；225,687 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-119"></a>

### 119 · vdb_scene_r9_summary.json

`docs/autoresearch/worldsim_v73/m4/vdb_scene_r9_summary.json`；778,718 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-120"></a>

### 120 · vdb_scene_r9_vs_pca_background.json

`docs/autoresearch/worldsim_v73/m4/vdb_scene_r9_vs_pca_background.json`；365,597 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-121"></a>

### 121 · vdb_scene_r9_vs_uncarved.json

`docs/autoresearch/worldsim_v73/m4/vdb_scene_r9_vs_uncarved.json`；350,725 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-122"></a>

### 122 · V73_OPEN_CHARTS_lidar_r3_PAIRS.pdf

`docs/autoresearch/worldsim_v73/open_charts/V73_OPEN_CHARTS_lidar_r3_PAIRS.pdf`；19,320 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-123"></a>

### 123 · V73_OPEN_CHARTS_lidar_r3_TRAINING.pdf

`docs/autoresearch/worldsim_v73/open_charts/V73_OPEN_CHARTS_lidar_r3_TRAINING.pdf`；26,366 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-124"></a>

### 124 · V73_OPEN_CHART_ARCHITECTURE.pdf

`docs/autoresearch/worldsim_v73/open_charts/V73_OPEN_CHART_ARCHITECTURE.pdf`；20,568 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-125"></a>

### 125 · open_charts_lidar_r3_analysis.json

`docs/autoresearch/worldsim_v73/open_charts/open_charts_lidar_r3_analysis.json`；4,371,170 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-126"></a>

### 126 · open_charts_lidar_r3_summary.json

`docs/autoresearch/worldsim_v73/open_charts/open_charts_lidar_r3_summary.json`；7,155,477 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-127"></a>

### 127 · WORLDSIM_V7_3_INTERIM_r4.pdf

`docs/autoresearch/worldsim_v73/paper/WORLDSIM_V7_3_INTERIM_r4.pdf`；371,086 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-128"></a>

### 128 · interim_r1.pdf

`docs/autoresearch/worldsim_v73/paper/interim_r1.pdf`；294,174 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-129"></a>

### 129 · interim_r2.pdf

`docs/autoresearch/worldsim_v73/paper/interim_r2.pdf`；316,734 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-130"></a>

### 130 · interim_r3.pdf

`docs/autoresearch/worldsim_v73/paper/interim_r3.pdf`；344,398 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-131"></a>

### 131 · AdaPoinTr.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/AdaPoinTr.blend`；2,629,372 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-132"></a>

### 132 · AdaPoinTr_detail.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/AdaPoinTr_detail.blend`；1,615,372 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-133"></a>

### 133 · AdaPoinTr_detail.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/AdaPoinTr_detail.png`；1,064,054 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-134"></a>

### 134 · AdaPoinTr_whole.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/AdaPoinTr_whole.png`；1,072,545 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-135"></a>

### 135 · Attraction-r4.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/Attraction-r4.blend`；1,983,932 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-136"></a>

### 136 · Attraction-r4_detail.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/Attraction-r4_detail.blend`；1,552,032 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-137"></a>

### 137 · Attraction-r4_detail.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/Attraction-r4_detail.png`；814,915 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-138"></a>

### 138 · Attraction-r4_whole.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/Attraction-r4_whole.png`；914,176 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-139"></a>

### 139 · First-r6.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/First-r6.blend`；2,181,628 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-140"></a>

### 140 · First-r6_detail.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/First-r6_detail.blend`；1,691,536 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-141"></a>

### 141 · First-r6_detail.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/First-r6_detail.png`；845,713 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-142"></a>

### 142 · First-r6_whole.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/First-r6_whole.png`；906,231 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-143"></a>

### 143 · LiDAR-R8.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/LiDAR-R8.blend`；2,859,804 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-144"></a>

### 144 · LiDAR-R8_detail.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/LiDAR-R8_detail.blend`；1,672,092 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-145"></a>

### 145 · LiDAR-R8_detail.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/LiDAR-R8_detail.png`；978,497 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-146"></a>

### 146 · LiDAR-R8_whole.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/LiDAR-R8_whole.png`；1,080,158 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-147"></a>

### 147 · Open-r3.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/Open-r3.blend`；2,518,044 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-148"></a>

### 148 · Open-r3_detail.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/Open-r3_detail.blend`；2,300,624 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-149"></a>

### 149 · Open-r3_detail.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/Open-r3_detail.png`；1,256,144 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-150"></a>

### 150 · Open-r3_whole.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/Open-r3_whole.png`；986,680 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-151"></a>

### 151 · VGGT-native.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/VGGT-native.blend`；2,759,260 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-152"></a>

### 152 · VGGT-native_detail.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/VGGT-native_detail.blend`；1,831,964 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-153"></a>

### 153 · VGGT-native_detail.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/VGGT-native_detail.png`；1,241,168 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-154"></a>

### 154 · VGGT-native_whole.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/renders/VGGT-native_whole.png`；1,055,854 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-155"></a>

### 155 · summary.json

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T184200Z__saved-surface-oracles-r1/summary.json`；473,375 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-156"></a>

### 156 · Joint-r7.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T190600Z__saved-r7-surface-oracles-r1/renders/Joint-r7.blend`；1,383,980 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-157"></a>

### 157 · Joint-r7_detail.blend

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T190600Z__saved-r7-surface-oracles-r1/renders/Joint-r7_detail.blend`；1,120,208 bytes；可重建 Blender 场景。归档中的成员路径相同。

<a id="asset-158"></a>

### 158 · Joint-r7_detail.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T190600Z__saved-r7-surface-oracles-r1/renders/Joint-r7_detail.png`；882,705 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-159"></a>

### 159 · Joint-r7_whole.png

`docs/autoresearch/worldsim_v73/paper_forensics/20260909T190600Z__saved-r7-surface-oracles-r1/renders/Joint-r7_whole.png`；953,519 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-160"></a>

### 160 · V73_QV2_ARCHITECTURE.pdf

`docs/autoresearch/worldsim_v73/qv2/V73_QV2_ARCHITECTURE.pdf`；22,092 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-161"></a>

### 161 · V73_QV2_ARCHITECTURE.svg

`docs/autoresearch/worldsim_v73/qv2/V73_QV2_ARCHITECTURE.svg`；91,058 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-162"></a>

### 162 · V73_QV2_FIXED_DIAGNOSTICS.pdf

`docs/autoresearch/worldsim_v73/qv2/V73_QV2_FIXED_DIAGNOSTICS.pdf`；23,229 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-163"></a>

### 163 · V73_QV2_LIDAR_SUPPORT.pdf

`docs/autoresearch/worldsim_v73/qv2/V73_QV2_LIDAR_SUPPORT.pdf`；20,557 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-164"></a>

### 164 · V73_QV2_joint_r1_PAIRS.pdf

`docs/autoresearch/worldsim_v73/qv2/V73_QV2_joint_r1_PAIRS.pdf`；20,141 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-165"></a>

### 165 · V73_QV2_joint_r1_TRAINING.pdf

`docs/autoresearch/worldsim_v73/qv2/V73_QV2_joint_r1_TRAINING.pdf`；27,126 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-166"></a>

### 166 · V73_QV2_lidar_r2_PAIRS.pdf

`docs/autoresearch/worldsim_v73/qv2/V73_QV2_lidar_r2_PAIRS.pdf`；18,061 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-167"></a>

### 167 · V73_QV2_lidar_r2_TRAINING.pdf

`docs/autoresearch/worldsim_v73/qv2/V73_QV2_lidar_r2_TRAINING.pdf`；26,324 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-168"></a>

### 168 · summary.json

`docs/autoresearch/worldsim_v73/qv2/mesh_diagnostic_r1/summary.json`；414,019 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-169"></a>

### 169 · shared_mesh_joint_r1_analysis.json

`docs/autoresearch/worldsim_v73/qv2/shared_mesh_joint_r1_analysis.json`；6,143,123 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-170"></a>

### 170 · shared_mesh_joint_r1_summary.json

`docs/autoresearch/worldsim_v73/qv2/shared_mesh_joint_r1_summary.json`；7,359,470 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-171"></a>

### 171 · shared_mesh_lidar_r2_analysis.json

`docs/autoresearch/worldsim_v73/qv2/shared_mesh_lidar_r2_analysis.json`；3,456,567 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-172"></a>

### 172 · shared_mesh_lidar_r2_summary.json

`docs/autoresearch/worldsim_v73/qv2/shared_mesh_lidar_r2_summary.json`；7,114,430 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-173"></a>

### 173 · summary.json

`docs/autoresearch/worldsim_v73/qv2/support_r3/summary.json`；400,184 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-174"></a>

### 174 · summary.json

`docs/autoresearch/worldsim_v73/qv2/support_r4/summary.json`；200,457 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-175"></a>

### 175 · V73_FIRST_SURFACE_R6_PAIRS.pdf

`docs/autoresearch/worldsim_v73/ray_support/V73_FIRST_SURFACE_R6_PAIRS.pdf`；19,549 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-176"></a>

### 176 · V73_FIRST_SURFACE_R6_TRAINING.pdf

`docs/autoresearch/worldsim_v73/ray_support/V73_FIRST_SURFACE_R6_TRAINING.pdf`；26,200 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-177"></a>

### 177 · V73_OPEN_JOINT_COMPONENTS.pdf

`docs/autoresearch/worldsim_v73/ray_support/V73_OPEN_JOINT_COMPONENTS.pdf`；20,596 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-178"></a>

### 178 · V73_OPEN_JOINT_R7_PAIRS.pdf

`docs/autoresearch/worldsim_v73/ray_support/V73_OPEN_JOINT_R7_PAIRS.pdf`；20,686 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-179"></a>

### 179 · V73_OPEN_JOINT_R7_TRAINING.pdf

`docs/autoresearch/worldsim_v73/ray_support/V73_OPEN_JOINT_R7_TRAINING.pdf`；26,139 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-180"></a>

### 180 · V73_RAY_SUPPORT_ARCHITECTURE.pdf

`docs/autoresearch/worldsim_v73/ray_support/V73_RAY_SUPPORT_ARCHITECTURE.pdf`；19,925 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-181"></a>

### 181 · V73_RAY_SUPPORT_R4_PAIRS.pdf

`docs/autoresearch/worldsim_v73/ray_support/V73_RAY_SUPPORT_R4_PAIRS.pdf`；19,907 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-182"></a>

### 182 · V73_RAY_SUPPORT_R4_TRAINING.pdf

`docs/autoresearch/worldsim_v73/ray_support/V73_RAY_SUPPORT_R4_TRAINING.pdf`；26,583 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-183"></a>

### 183 · first_surface_r6_analysis.json

`docs/autoresearch/worldsim_v73/ray_support/first_surface_r6_analysis.json`；5,285,617 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-184"></a>

### 184 · first_surface_r6_summary.json

`docs/autoresearch/worldsim_v73/ray_support/first_surface_r6_summary.json`；7,159,574 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-185"></a>

### 185 · first_surface_r6_training.json

`docs/autoresearch/worldsim_v73/ray_support/first_surface_r6_training.json`；106,607 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-186"></a>

### 186 · open_joint_r7_analysis.json

`docs/autoresearch/worldsim_v73/ray_support/open_joint_r7_analysis.json`；5,302,829 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-187"></a>

### 187 · open_joint_r7_summary.json

`docs/autoresearch/worldsim_v73/ray_support/open_joint_r7_summary.json`；7,402,048 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-188"></a>

### 188 · open_joint_r7_training.json

`docs/autoresearch/worldsim_v73/ray_support/open_joint_r7_training.json`；110,413 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-189"></a>

### 189 · ray_support_r4_analysis.json

`docs/autoresearch/worldsim_v73/ray_support/ray_support_r4_analysis.json`；5,285,450 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-190"></a>

### 190 · ray_support_r4_summary.json

`docs/autoresearch/worldsim_v73/ray_support/ray_support_r4_summary.json`；7,159,499 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-191"></a>

### 191 · summary.json

`docs/autoresearch/worldsim_v73/ray_support/support_r5/summary.json`；400,640 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-192"></a>

### 192 · V73_UPPER_TAIL_INTERFACE.pdf

`docs/autoresearch/worldsim_v73/upper_tail/V73_UPPER_TAIL_INTERFACE.pdf`；15,636 bytes；未内嵌于报告的渲染或导出。归档中的成员路径相同。

<a id="asset-193"></a>

### 193 · real_r1_assets.json

`docs/autoresearch/worldsim_v74/a_wex/real_r1_assets.json`；190,632 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-194"></a>

### 194 · real_solver_status.json

`docs/autoresearch/worldsim_v74/b_rif/real_solver_status.json`；202,050 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-195"></a>

### 195 · real_r1_assets.json

`docs/autoresearch/worldsim_v74/c_dcs/real_r1_assets.json`；202,612 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-196"></a>

### 196 · fit_geometry.json

`docs/autoresearch/worldsim_v74/calibration/fit_geometry.json`；193,483 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-197"></a>

### 197 · manifest.json

`docs/autoresearch/worldsim_v74/evaluation/a_wex/manifest.json`；120,116 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-198"></a>

### 198 · per_actor.json

`docs/autoresearch/worldsim_v74/evaluation/a_wex/per_actor.json`；734,736 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-199"></a>

### 199 · summary.json

`docs/autoresearch/worldsim_v74/evaluation/a_wex/summary.json`；178,205 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-200"></a>

### 200 · manifest.json

`docs/autoresearch/worldsim_v74/evaluation/b_rif/manifest.json`；120,108 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-201"></a>

### 201 · per_actor.json

`docs/autoresearch/worldsim_v74/evaluation/b_rif/per_actor.json`；583,371 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-202"></a>

### 202 · summary.json

`docs/autoresearch/worldsim_v74/evaluation/b_rif/summary.json`；144,612 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-203"></a>

### 203 · manifest.json

`docs/autoresearch/worldsim_v74/evaluation/c_dcs/manifest.json`；120,124 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-204"></a>

### 204 · per_actor.json

`docs/autoresearch/worldsim_v74/evaluation/c_dcs/per_actor.json`；743,809 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-205"></a>

### 205 · summary.json

`docs/autoresearch/worldsim_v74/evaluation/c_dcs/summary.json`；177,437 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-206"></a>

### 206 · partial_AC_paired_log_comparisons.json

`docs/autoresearch/worldsim_v74/evaluation/partial_AC_paired_log_comparisons.json`；624,775 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-207"></a>

### 207 · real_ABC_paired_log_comparisons.json

`docs/autoresearch/worldsim_v74/evaluation/real_ABC_paired_log_comparisons.json`；815,834 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-208"></a>

### 208 · manifest.json

`docs/autoresearch/worldsim_v74/evaluation/references/manifest.json`；120,111 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-209"></a>

### 209 · per_actor.json

`docs/autoresearch/worldsim_v74/evaluation/references/per_actor.json`；571,193 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-210"></a>

### 210 · summary.json

`docs/autoresearch/worldsim_v74/evaluation/references/summary.json`；146,140 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-211"></a>

### 211 · summary.json

`docs/autoresearch/worldsim_v74/event_evidence/a_wex/summary.json`；150,899 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-212"></a>

### 212 · summary.json

`docs/autoresearch/worldsim_v74/event_evidence/b_rif/summary.json`；164,605 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-213"></a>

### 213 · summary.json

`docs/autoresearch/worldsim_v74/event_evidence/c_dcs/summary.json`；171,040 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-214"></a>

### 214 · assets.json

`docs/autoresearch/worldsim_v74/final/a_assets/assets.json`；192,385 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-215"></a>

### 215 · assets.json

`docs/autoresearch/worldsim_v74/final/b_assets/assets.json`；143,403 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-216"></a>

### 216 · assets.json

`docs/autoresearch/worldsim_v74/final/c_assets/assets.json`；202,612 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-217"></a>

### 217 · event_summary.json

`docs/autoresearch/worldsim_v74/final/event_summary.json`；315,526 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-218"></a>

### 218 · results.json

`docs/autoresearch/worldsim_v74/final/expanded_dictionary/results.json`；139,245 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-219"></a>

### 219 · results.json

`docs/autoresearch/worldsim_v74/final/geometric_mechanisms/results.json`；2,435,006 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-220"></a>

### 220 · mechanism_summary.json

`docs/autoresearch/worldsim_v74/final/mechanism_summary.json`；128,445 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-221"></a>

### 221 · manifest.json

`docs/autoresearch/worldsim_v74/final/nksr_evaluation/manifest.json`；120,064 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-222"></a>

### 222 · per_actor.json

`docs/autoresearch/worldsim_v74/final/nksr_evaluation/per_actor.json`；139,015 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-223"></a>

### 223 · paired_log_comparisons.json

`docs/autoresearch/worldsim_v74/final/paired_log_comparisons.json`；853,352 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-224"></a>

### 224 · query_timing.json

`docs/autoresearch/worldsim_v74/final/query_timing.json`；262,654 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-225"></a>

### 225 · probe_cohort.json

`docs/autoresearch/worldsim_v74/p0/probe_cohort.json`；120,434 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-226"></a>

### 226 · storage_plan.json

`docs/autoresearch/worldsim_v74/p0/storage_plan.json`；2,150,788 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-227"></a>

### 227 · storage_result.json

`docs/autoresearch/worldsim_v74/p0/storage_result.json`；787,401 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-228"></a>

### 228 · boundary_diagnostics.json

`docs/autoresearch/worldsim_v74_h2/cpu/boundary_diagnostics.json`；515,220 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-229"></a>

### 229 · results.json

`docs/autoresearch/worldsim_v74_h2/cpu/results.json`；573,585 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-230"></a>

### 230 · results.json

`docs/autoresearch/worldsim_v74_h2/gpu_p1/20260912__C1-ordinary-r1/results.json`；110,871 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-231"></a>

### 231 · results.json

`docs/autoresearch/worldsim_v74_h2/gpu_p1/20260912__C3-beam-r1/results.json`；110,497 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-232"></a>

### 232 · C1_results.json

`docs/autoresearch/worldsim_v74_h2/gpu_p1/20260912__controls-r1/C1_results.json`；439,071 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-233"></a>

### 233 · teacher_results.json

`docs/autoresearch/worldsim_v74_h2/gpu_p1/20260912__fit-teacher-r3/teacher_results.json`；425,259 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-234"></a>

### 234 · interpolations.json

`docs/autoresearch/worldsim_v74_h2/p15/interpolations.json`；124,598 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-235"></a>

### 235 · transitions.json

`docs/autoresearch/worldsim_v74_h2/p15/transitions.json`；109,863 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-236"></a>

### 236 · control_rows.json

`docs/autoresearch/worldsim_v74_h2/p16/control_rows.json`；423,609 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-237"></a>

### 237 · local_control_rows.json

`docs/autoresearch/worldsim_v74_h2/p16/local_control_rows.json`；295,674 bytes；逐对象/逐射线大结果表。归档中的成员路径相同。

<a id="asset-238"></a>

### 238 · main_zh_reading_package.zip

`docs/paper/main_zh_reading_package.zip`；421,118 bytes；可重建阅读包。归档中的成员路径相同。

<a id="asset-239"></a>

### 239 · worldsim_v7_final_main.pdf

`paper/baselines/worldsim_v7_final_main.pdf`；1,926,625 bytes；论文编译产物。归档中的成员路径相同。

<a id="asset-240"></a>

### 240 · main.pdf

`paper/main.pdf`；4,085,703 bytes；论文编译产物。归档中的成员路径相同。

<a id="asset-241"></a>

### 241 · supplement.pdf

`paper/supplement.pdf`；8,159,955 bytes；论文编译产物。归档中的成员路径相同。
