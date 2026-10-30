# 当前研究状态

当前任务：WS-V43-WORLD-GT-PROTOCOL-R5-20261030 / WS-V43-SIM-RECON-M0-20261030。V43 Revision5已更新并保留大模块＋内部小模块主架构，新增世界GT与合法输入隔离图。GT位于独立世界的表面、对象与编辑干预结果，原静态/动态分支和网页语义仅为带噪线索；不假定每颗高斯存在唯一标签。M1/M2称为真值辅助归属/表示修复参考，不能称数学上界。

M0已实际完成：1个程序化普通车/道路/灯光场景，16帧480×320单前向观测和稀疏raycast LiDAR；独立删车/横移/转向参考与1个留出相机。显式使用oracle body/sky masks和ideal tracks，属于登记的传感器控制，未宣称可部署的预测分割管线。完整网格、稠密深度、表面ID与编辑参考未进入初始化和训练。生成器r2已通过；适配器r3修复mask查询半像素取整，K/poses不变，旧r1/r2现场保留。

真实原生GN11整场景训练包含static/dynamic/environment。先做600.159秒训练循环墙钟smoke（扣除评价/保存），6702更新＋2步恢复入口，参数/optimizer/RNG/sampler等完整恢复和实际none/delete渲染通过；另有317步计时探测，均单列。随后从零正式10000步，保留原30000步学习率周期，5K/10K完整保存；正式调用1193.844秒，峰值allocated约5.00GiB、reserved约6.22GiB。16帧/112浮点数组核验通过，3关键帧×4编辑实际完成，与独立参考评分12/12；原生none最大差0，实际训练相机核验通过。1个留出相机渲染完成。

结果：正式中间帧删除整个原动态分支后仍有彩色车形残留与暗区，独立无车参考为连续道路；白底替换环境后残留仍在，说明至少该部分由保留静态高斯贡献。横移后新车与旧残影并存，留出视角仍有边缘/路面伪影。该数据链自然复现可编辑分解的失败，未注入错误高斯或打乱标签；这不是新V43方法改善或泛化证明。没有把任何高斯全部标成完美GT。

下一步：先建立独立表面/深度支持下的条件责任审计，区分mixed、错深度、无对应和缺测；再做M1/M2真值辅助参考与受控修复，验证表示操作的收益；之后扩到3–5机制场景再小规模联合学习和批量造数据。当前新联合模块未训练，真实场景一仍只开发诊断/推理。32samples阴影开关差分含采样噪声，仅诊断代理，不能作为阴影GT或训练标签；须补采样收敛与receiver/effect证据。环境保持分支诊断，已保存CG、alpha、E、实际fill和final/white。

文档：[V43 Revision5](autoresearch/v43_ownership_20261028/gt_protocol_r5_20261030/index.html)。实际报告：[M0原生10K与独立编辑核验](autoresearch/v43_ownership_20261028/sim_mechanism_20261030/index.html)。报告72张配对图、12指标行、完成收据、留出图、链接及浏览器切换均核验；取回183文件17.6MB与远端SHA256相同。旧R4、R2、unknown-white和真实场景诊断不覆盖。

现场：`/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030`。human_verdict=null；failure_ledger_refs=[V82-F01]；failure_ledger_delta=none。当前不启动新的长跑或后台控制器。

**本对话禁止自动AutoDL关机。完成后保留实例；只有用户明确要求才关机。subagent仅gpt-6.1-sol/gpt-6.1-terra。**
