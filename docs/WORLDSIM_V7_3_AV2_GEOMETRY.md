# V7.3 AV2逐点时空几何接口

2026-09-08。实现 `motion_proj/worldsim_v73/av2_geometry.py`，仅在旧开发日志02678d04-cc9f-3148-9f95-1ba66347dff9执行真实数据诊断。20条新确认日志的760文件已传输完成（977.74s，s5cmd返回0，下载PID34904退出），没有运行其网络推理或读取其中传感器数值进行方法选择。

依据[官方Sweep接口](https://github.com/argoverse/av2-api/blob/main/src/av2/structures/sweep.py)及[官方SE3读取约定](https://github.com/argoverse/av2-api/blob/main/src/av2/utils/io.py)，每点时间为扫描参考纳秒时间加offset_ns；发布的坐标已补偿到参考时刻ego系。因此世界端点只乘一次参考ego姿态，不能再次按逐点ego变换移动端点；束原点则取逐点时刻的ego姿态与其物理LiDAR外参：

\[
x_w=T_{w\leftarrow ego}(t_0)x_{ego(t_0)},\qquad
o_w(t_i)=T_{w\leftarrow ego}(t_i)o_{sensor,ego},\qquad
d_i=\|x_w-o_w(t_i)\|.
\]

Actor规范坐标用只读轨迹在同一个t_i的逆变换处理端点、原点与方向；相机投影用该相机实际曝光t_v的ego与Actor位姿。`TimedPoses`先用int64减去时间原点，再转相对秒做平移插值和Slerp；不将大绝对纳秒先转float。越界位姿返回独立known掩码，数值占位虽夹到边界但不能作为已知pose使用。Actor标注原本在对应时刻ego系，先转city系再插值；num_interior_pts不参与输入或选择。

旧开发log扫描序号5有94860个原始返回，实际offset为0.615–106.315ms，ego位姿覆盖全部返回。忽略逐点时间、只用参考扫描时刻的两个原点，与逐点原点之差中位0.59106m、p95 1.07697m、最大1.15741m。这是此旧日志的几何诊断，不是全AV2误差估计，也不是模型精度增益。

官方文档/API确认laser_number为0–63、传感器有up/down，但本次检索未找到明确的编号组映射声明。为避免默默假设，诊断在相同原始返回和固定标定下比较两种映射的逐ring仰角离散度：0–31→up时，各ring角度MAD的中位为0.011554°；交换后为0.126340°。据此选0–31为up、32–63为down，记录为旧开发数据支持的推断，代码仍显式暴露lower_ids_sensor参数。没有按新确认日志单独重估映射或按模型质量选传感器约定。

七路图像全部保留，不裁剪视场；当前接口按最长边672缩放并放入672×672画布，内参同步应用缩放/平移。旧日志前相机有效矩形[81,0,590,672]，其他六台[0,81,672,590]；记录实际时间和光轴方向。七台光轴方位约0.20°、45.11°、−44.91°、99.48°、−98.89°、153.31°、−152.94°，不能把nuScenes的六项相机ID直接索引成七项。完整扫描、每ring及每camera结果在 `docs/autoresearch/worldsim_v73/coverage/av2_geometry_development_r1.json`。CPU诊断0.99s、RSS0.22056GiB，没有GPU/网络模型测试。

当前交付是几何底层，尚未完成V7.3 Actor case/build observation导出、七路相机嵌入迁移及padding区域在原生支持生成和局部特征读取中的排除。后续在旧开发日志完成这些接口，再以固定方法处理新确认日志。不能用零填充图像区生成新的几何支持，不能丢弃第七台相机，不能将这些输入诊断当成外部重建评价。F04/F05继续active，整个V7.3未完成。
