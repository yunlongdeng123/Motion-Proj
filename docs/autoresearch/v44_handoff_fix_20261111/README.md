# 通用v44保持域与候选覆盖修复

入口index.html及architecture.svg；实际重跑scene065白色轿车，全部199帧训练无留出。代码为共享v44_handoff_policy / gate_fit / road，不含街段、演员或特定帧分支。18项CPU通用策略检查和真实32步保存/完整恢复/原生CLI出口通过；32+968=1000正式gate更新，高斯与环境0更新，无额外优化；旧采样序列完全相同。原参考21字段及199视角photo权重/参考逐像素相等。

候选45449→76861，旧行丢0；target keep/left全199视角清零；范围外静态贡献已选份额119/198为97.02%/97.73%。初始化logit2、mean retention分母扩展和初始0.880797乘子已记录，初始化图单列。未知无额外photo或opacity=0监督。

结果仍有明显车形残影。实拍支持道路代理MAE .026322→.024741、推断域 .032520→.029968，不是独立无车GT。邻车KEEP变化8.96e-6、目标外8.90e-5；可见道路和白车KEEP误差略升。实际看过119/198及初始化对照，不据loss/覆盖改进判画质成功，human_verdict=null。

代码提交01e029c7已推送。核心源码和测试、driver、CPU评价打包、viewer源在source；正式checkpoint/float/runtime在远端campaign/waymo_val065_handoff_fix_20261111_r1，原30K/旧结果保留。view binding与baseline必须匹配当前完整删除集合，换目标不能重用不对应的旧photo/底图。图表producer现为单目标适配器，多目标策略检查不等同完整多目标输入构造或成功实验。

本地平台15021599536622641101_556_150_576_150/index.html现五列199帧，旧四列native_delete30k.html保留，旧schema不误套新实验，scene029未改。398原PNG和两H264完整视频80MB归档取回、逐项验收；39项DOM替身逻辑与本地静态引用检查通过，browser_qa=false。平台没有添加报告/状态链接。

failure_ledger_refs=[V82-F06]；failure_ledger_delta=updated:V82-F06；不新增失败ID。AutoDL保持开机，不追加其他训练或场景适配。
