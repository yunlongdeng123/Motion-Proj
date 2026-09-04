# WorldSim V7.1 M10：GT 监督的 Oriented Planar Gaussian

## 从 M9 得到的表示结论

M9表明scale-aware ray physics本身有效，但isotropic sphere无法同时表示“沿表面需要宽覆盖”和“沿法向必须薄”。优化将
mean scale从`0.16196m`缩到`0.11227m`，support early下降却丢失约6.1pp hit；这是表示耦合，不是loss不收敛。

## 一手资料迁移

2DGS以oriented planar Gaussian disk内生表示表面，并用ray--splat intersection；PGSR直接render Gaussian plane depth；
SuGaR与DN-Splatter分别强调surface alignment以及depth/normal supervision。M10迁移最小共同结构：

- center (c_j\in\mathbb{R}^3)；
- unit normal (n_j\in\mathbb{S}^2)；
- tangent support radius (s_j)；
- independent normal thickness (h_jll s_j)。

为避免zero-thickness数值不适合collision query，部署/评价采用有限厚度oblate `1σ` ellipsoid，而非无限薄disk。其局部
等值面为

\[
\frac{\|q-(q^\top n)n\|_2^2}{s^2}+\frac{(q^\top n)^2}{h^2}=1,
\qquad q=x-c.
\]

## 模型与监督

从M8初始化4-child center/tangent scale；normal初始为build candidate局部PCA normal，normal residual从0开始；thickness固定
从`0.02m`开始。训练继续使用完整actor-canonical GT：

- symmetric union-set Chamfer与frame-balanced endpoint coverage监督center/coverage；
- 8NN GT plane normal以sign-invariant cosine监督normal；
- 8NN median radius监督tangent scale；
- fixed `0.02m`表面带宽监督normal thickness；
- point first/free-space与anisotropic density first/free-space都在训练图内；
- geometry与combined physics继续symmetric PCGrad。

单seed、6轮、同一593/66 development；不引入image、motion、hazard、time，也不调branch factor/loss/厚度。

## Exact physical audit

对每条GT ray解析求最小正ray--oblate-ellipsoid交点，不以center beam tube近似。冻结M8 oriented initializer（M8 center/
tangent scale + parent normal + `0.02m` thickness）为reference。通过条件：

1. 原五项point physical contract全部通过；
2. hazardous exact ellipsoid-support early相对reference下降至少5%；
3. exact ellipsoid-support hit相对reference下降不超过1 percentage point。

所有children与scale/normal/thickness均输出；不允许按audit删除、mask、缩放。若失败登记`V71-F14`并根据normal error与
support trade-off决定是否停止Gaussian collision路线，不调`σ`倍数恢复。

