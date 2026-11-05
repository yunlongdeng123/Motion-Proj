backgroundJoint1 独立备用包（仅本地准备，源码已冻结）

源：完整复制已冻结 work/native_loss_probe/source，original_source留原样；source只有global_rgb_probe.py与train_loss_probe.py改变，其余25份逐字一致。原full-global包未改。

唯一方法变化：新增1×joint RGB从整图限制为Ownership.terms同一个真实background。collar为两actor visible_reviewed union的5×5 dilation；background=~collar & (dynamic_union==0)。joint prediction与target都先guard成0，再用原photometric_rgb_loss/SSIM；不另按active像素数归一化。静态2、其它loss、增长、阈值、SGCM、采样与LR不改。该背景包括原定义里的天空，并非只取道路。

模式：--loss-probe-mode global_background；metadata名backgroundJoint1；term名joint_background_rgb。control/global仍可显式使用，global数值和梯度保持旧整图项。probe选项不进入cfg。resume必须同probe mode，防止误把full-global接成background控制。

入口：远端独立campaign目录中的run_jointbg.py将运行backgroundJoint1_1k(formal0→1000)，随后backgroundJoint1_restore_1002(完整restore再2更新)。此处没有执行或上传。source/train_loss_probe.py --help可查看直接入口。原100/500/1000完整快照、每100族RGB、RNG-neutral readout和完整resume路径不变。

CPU审核：test_jointbg_cpu.py；结果cpu_jointbg_audit.json。实际原photometric loss与SSIM；其余terms用sentinel检验返回tensor身份，并通过源码逐字/runner AST对比确认原方法未改。1340背景/196排除像素测试中，新增项对pred和target的排除区梯度最大值均0；全背景与整图global完全相等，全前景loss/梯度为0；RNG不变。没有构造原生模型、导入CUDA场景或跑GPU。

这只是监督域单变量1K对照，不能预先称作修复。遮罩只限制新增项的图像梯度，Gaussian共享参数仍可能经背景足迹接到梯度；蓝族是否压住和银灰残影是否改善由真实实验决定。没有加入旧road冻结方案。
