# WorldSim V7.1 M28：Typed Authority and Non-Interference

## 目标

M28不再调整几何、外观或后处理阈值，而是把M8/M18/M21的物理表面、M22的刚体运动、
M26/M27的visual-only层写成一个可执行的权限边界。要证明的不是“后处理后指标没变”，而是：

1. 几何从GT target surface、local plane/scale、frame coverage和literal first-return supervision学习；
2. physical query的函数签名只接受physical field与read-only Actor pose；
3. image-trained visual geometry、SH和opacity处于兄弟类型，不被physical query接受；
4. 因此appearance-only update对physical energy的雅可比结构为零，而不是依赖部署过滤。

## 文献迁移

- Neural Scene Graphs（CVPR 2021）把object transform与object representation解耦：
  <https://openaccess.thecvf.com/content/CVPR2021/html/Ost_Neural_Scene_Graphs_for_Dynamic_Scenes_CVPR_2021_paper.html>
- UniSim（CVPR 2023）分开static background与dynamic actors并在新pose下组合：
  <https://openaccess.thecvf.com/content/CVPR2023/html/Yang_UniSim_A_Neural_Closed-Loop_Sensor_Simulator_CVPR_2023_paper.html>
- EmerNeRF（ICLR 2024）以hybrid static--dynamic field与scene flow建模：
  <https://openreview.net/forum?id=ycv2z8TYur>

上述工作支持object/static/dynamic分层，但M28额外收紧权限：渲染属性不是occupancy，不能反向
修改physical field。不迁移其image-only geometry、scene flow学习或closed-loop safety claim。

## 类型化状态

```text
PhysicalActorField P_i = (Actor ID, GT-supervised canonical centers/scales)
ActorPose          T_it = (Actor ID, read-only R_it/t_it)
VisualActorLayer   V_i = (render centers/scales/rotation, SH, opacity)

Q_phy(x, P_i, T_it)             # 不接受 V_i
R_image(camera, stopgrad(P_i), V_i, T_it, B_vis)
```

physical training保留

\[
\mathcal L_{\rm phys}=\mathcal L_{\rm set}+\mathcal L_{\rm plane}+\mathcal L_{\rm scale}
+\mathcal L_{\rm frame}+\mathcal L_{\rm first-return},
\]

其target均由GT endpoint/ray在Actor canonical frame中构造。visual training只有

\[
\min_\phi\mathcal L_{\rm image}
\bigl(R(\operatorname{sg}(P_i),V_i(\phi),T_{it},B_{\rm vis}),I_{\rm GT}\bigr),
\]

其中`sg`为stop-gradient。

## 命题与证明边界

对M21使用的Actor-local Gaussian energy，

\[
Q_{it}(x;P_i,T_{it})=\log\sum_j\exp\left[
-\frac{\|R_{it}^{\top}(x-t_{it})-c_{ij}\|^2}{2s_{ij}^2}
\right].
\]

由函数依赖集可直接得到，对任意visual-only更新`Delta V_i`，

\[
Q_{it}(x;P_i,T_{it},V_i+\Delta V_i)=Q_{it}(x;P_i,T_{it}),
\qquad \partial Q_{it}/\partial V_i=0.
\]

对任意world rigid transform `g`，还有`Q_{gT}(gx)=Q_T(x)`。这两个结论分别界定appearance
non-interference与SE(3) equivariance。

它们不保证：GT本身完整、learned geometry正确、trajectory正确、background collision完备、
sensor occlusion/no-return正确、photorealism或closed-loop safety。M21 fresh AV2 20/20仍是几何外测的必要边界。

## 最小实现与退出

- 新增`motion_proj/worldsim_v71/authority_contract.py`，实现三个兄弟状态类型与纯physical query；
- 论文Method增加权限分层、non-interference proposition、SE(3) corollary与non-claims；
- 只做`py_compile`和论文完整编译；不跑训练、不读M21 partial quality、不增加阈值、哈希或过度门控。
