# 当前研究状态

更新：2026-11-09（Asia/Singapore）。

当前任务`WAYMO-VAL065-V1-VISIBLE-20261109`已完成。用户要求scene065补V1通路10K并给出第108帧入口；核验发现合格的独立原生10K已存在，但最新可读性页面将它折叠在“A的删除前对照”。本次将V1提升为常显独立主列，保留实拍、A来源删除、B重选/道路删除，B的KEEP仍可展开。[第108帧对照、组件图和来源记录](autoresearch/waymo_val065_v1_visible_20261109/index.html)。

复用原生fresh10000（start_checkpoint=null、seed0，LR时钟30000，正式1336.382s），本次新训练更新0。既有604.435s真实生命周期smoke与精确恢复5199工程更新单列。V1只恢复原生静态/动态/环境检查点并调用原渲染器，不导入编辑包、不重绑归属、不插车辆/道路、不拼接图像。60/108/150/198重新执行纯V1，所有float RGB对原V1最大差0；第108帧PNG也逐像素相同。108属于79–118连续40帧留出，不冒充训练视图；V1是同预算参照而非理论最优。

全部199个原生V1 PNG在本地/平台共398文件逐一核对来源。默认保留完整画面，主区四列顺序为实拍/V1完整重建10K（无编辑）/A删除/B删除；新增108跳转。点击V1/A/B画面自动选择相应贡献通路与KEEP/DELETE状态，缺float的时刻仍留空。原A/B说明、B保留图、阶段/归因、诊断记录与其他场景均保留。URL raw108、主列顺序、全时间窗源图、异步拖动和记录逻辑经真实JS的DOM/Canvas替身验证；不是浏览器视觉验收，未绕过此前file协议限制。

新复算/记录位于/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val065_v1_presentation_20261109_r1；原模型waymo_val065_native10k_20261108_r1/formal10k及全部旧结果保持。当前V1与删除图未修改画质，原scene065车形残留/F06结论保持。scene029同街段删红/删后方白SUV两case及其89帧仍独立保留。不使用subagent，不进行电源操作；按用户明确要求GPU保持开机。human_verdict=null，failure_ledger_delta=none。源码/来源与本次提交归档，推送/取回收据在本地交付保存；不从旧队列追加训练。
