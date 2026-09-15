# 复现入口

服务器原始证据根：`/root/autodl-tmp/runs/worldsim_v74_h2/WS-V74-MAINFIG-01/20260915-first-return-r1`。本包已保存表图、汇总、配置和核心脚本；权重/全部原生预测保留服务器仓库外，避免重复分发数十GB。没有hash检查或新训练。

主前向：worldsim-v81环境运行 `infer_official.py --method vggt|omega512|dvgt1|pi3x`，完成项自动跳过。evaluate_surfaces、evaluate_common6、summarize_results处理冻结输出。不要重跑prepare_inputs以覆盖冻结登记。

第二批：同环境 infer_secondary.py --method dvgt2|dggt。DGGT缓存渲染用render_dggt_cached.py；PATH包含motionproj/bin的ninja，TORCH_EXTENSIONS_DIR=/root/autodl-tmp/cache/worldsim_v74_gsplat。输入位姿4x4、点数组float32；native_render保存RGB/expected-depth/alpha。此处没有物理Gaussian LiDAR响应。

NKSR/NoKSR：envs/nksr-v74/bin/python，NoKSR额外PYTHONPATH=/root/autodl-tmp/envs/noksr-v74-overlay。run_secondary_nksr.py、run_secondary_noksr.py只读secondary/build_inputs；QUERY独立交给evaluate_secondary_native.py。NoKSR采用官方enable_flash=False；完整配置在evidence/noksr_config.yaml。旧NKSR空层级网格兼容修复已有项目patch。不要把输入不足记成成功。

`V74_Main_Paper_Reproduction.zip`包含作图必需的真实RGB/网格/射线与相对目录结构。解压后执行 `python reproduce_figures.py`（需要numpy、matplotlib、Pillow），图写入outputs/V74_Main_Paper_Figures。此过程不需要GPU。报告包含结论边界，英文图注见CAPTIONS.md。
