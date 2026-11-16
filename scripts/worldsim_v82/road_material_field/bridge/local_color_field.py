"""世界道路纹理的局部颜色锚定。推理API不接收孔洞真值或街段身份。"""
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, distance_transform_edt


def luminance(rgb):
    return rgb @ np.array([.2126,.7152,.0722],np.float32)


def paint_mask(rgb, threshold):
    r,g,b=rgb.transpose(2,0,1)
    white=(luminance(rgb)>threshold)&(np.ptp(rgb,axis=2)<.20)
    # 比训练阶段的亮度漆弱标签更保守地保护褪色黄色漆；一套规则用于全部街段。
    yellow=(r-b>.055)&(g-b>.035)&(r-g<.22)&(luminance(rgb)>.20)
    return white|yellow


def asphalt_mask(rgb, support, context, context_support):
    if context_support.sum()<64:
        raise ValueError('insufficient legal visible road context')
    lm=luminance(context)
    center=float(np.median(lm[context_support]))
    threshold=float(np.clip(center+.09,.35,.85))
    paint=binary_dilation(paint_mask(rgb,threshold),iterations=2)
    gray=luminance(rgb)
    gy,gx=np.gradient(gray)
    grad=np.hypot(gx,gy)
    pool=support&~paint
    if pool.sum()<64:
        raise ValueError('insufficient asphalt after foreground/paint protection')
    lo,hi=np.percentile(gray[pool],[5,95])
    cutoff=max(float(np.percentile(grad[pool],85)),.006)
    return pool&(gray>=lo)&(gray<=hi)&(grad<=cutoff)


def basis(height,width):
    yy,xx=np.mgrid[:height,:width].astype(np.float32)
    xx=(xx+.5)/width*2-1
    yy=(yy+.5)/height*2-1
    return np.stack([np.ones_like(xx),xx,yy],-1)


def robust_plane(rgb, mask, weights=None):
    phi=basis(*mask.shape)
    x,y=phi[mask].astype(np.float64),rgb[mask].astype(np.float64)
    if len(x)<32:
        raise ValueError('too few reliable color anchors')
    w=np.ones(len(x)) if weights is None else weights[mask].astype(np.float64)
    ridge=np.diag([1e-5,2.,2.])
    for _ in range(4):
        c=np.linalg.solve(x.T@(w[:,None]*x)+ridge,x.T@(w[:,None]*y))
        residual=np.linalg.norm(y-x@c,axis=1)
        robust=np.minimum(1.,.025/(residual+1e-6))
        w=robust if weights is None else weights[mask]*robust
    c[1:]=np.clip(c[1:],-.06,.06)
    return c.astype(np.float32)


def anchor_geometry(context, hole, valid, generated=None, reference_only=False):
    """只从合法上下文和当前预测提取低频颜色/置信度，不读取隐藏道路。"""
    if generated is None:
        generated=context
    visible=valid&~hole
    if reference_only:
        visible=valid
    threshold=float(np.clip(np.median(luminance(context)[visible])+.09,.35,.85))
    paint=paint_mask(generated,threshold)
    # 扩散输出中的明显漆线保持颜色；过渡仅作用于漆线邻域，不羽化孔洞边界。
    pw=1-np.maximum(paint.astype(np.float32),gaussian_filter(paint.astype(np.float32),.8))
    if reference_only:
        return {'paint_weight':pw}
    context=context.copy()
    context[hole|~valid]=0  # 隐藏域的占位颜色不参与邻域统计或梯度筛选。
    outside_distance=distance_transform_edt(~hole)
    ring=visible&(outside_distance<=24)&binary_erosion(valid,iterations=1)
    cmask=asphalt_mask(context,visible,context,visible)&ring
    if cmask.sum()<128:
        cmask=asphalt_mask(context,visible,context,visible)
    # 所有合法路块由道路语义和车辆保护区预先限定；这里进一步剔除标线、极暗点和强边缘。
    pmask=asphalt_mask(generated,valid&hole,context,visible)
    weight=np.exp(-np.square(outside_distance/16))
    c=robust_plane(context,cmask,weight)
    p=robust_plane(generated,pmask)
    coefficients=c-p
    coefficients[0]=np.clip(coefficients[0],-.12,.12)
    coefficients[1:]=np.clip(coefficients[1:],-.04,.04)
    stats=np.concatenate([c.ravel(),p.ravel(),coefficients.ravel(),
        np.std(context[cmask],axis=0),np.std(generated[pmask],axis=0),
        [cmask.mean(),pmask.mean(),hole.mean(),paint[hole].mean(),
         float(distance_transform_edt(hole).max()/128)]])
    assert stats.size==38
    return dict(context_plane=c,generated_plane=p,coefficients=coefficients,
        features=stats.astype(np.float32),paint_weight=pw,context_asphalt=cmask,
        generated_asphalt=pmask,threshold=threshold,
        anchor_pixels=int(cmask.sum()),prediction_asphalt_pixels=int(pmask.sum()))


def apply_field(generated, context, hole, valid, geometry, coefficients=None):
    coeff=geometry['coefficients'] if coefficients is None else coefficients
    delta=np.clip(basis(*hole.shape)@coeff,-.12,.12)
    delta*=geometry['paint_weight'][:,:,None]
    corrected=np.clip(generated+delta,0,1)
    output=context.copy()
    output[hole&valid]=corrected[hole&valid]
    return output,delta


def feather_only(generated,context,hole,valid,geometry):
    # 仅两格宽内圈，用合法沥青平面作边缘协调；孔洞中央逐值保持原预测。
    distance=distance_transform_edt(hole)
    a=np.clip((3-distance)/2,0,1)*hole*geometry['paint_weight']
    color=basis(*hole.shape)@geometry['context_plane']
    output=generated.copy()
    output[hole&valid]=(generated*(1-a[:,:,None])+color*a[:,:,None])[hole&valid]
    output[~hole]=context[~hole]
    return np.clip(output,0,1)


def constant_anchor(generated,context,hole,valid,geometry):
    coefficients=np.zeros((3,3),np.float32)
    coefficients[0]=geometry['coefficients'][0]
    return apply_field(generated,context,hole,valid,geometry,coefficients)[0]


def neural_network():
    import torch
    from torch import nn
    class ColorResidualField(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(38,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,9))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, features, prior):
            # 学习残差系数而非重新合成纹理；低阶世界纹理场，输出幅度全局受限。
            residual=.04*torch.tanh(self.net(features)).reshape(-1,3,3)
            return torch.clamp(prior+residual,-.12,.12)
    return ColorResidualField()


def harmonize_world_texture(generated, context, hole, valid, world_from_uv, model=None):
    """通用世界纹理入口：调用方持久保存结果，并让高斯按原世界坐标取色。

    RGB为[0,1]浮点纹理；输入角色不含目标答案、街段ID、车辆ID或查询帧。
    model可为已加载的小型颜色网络；None表示无学习的解析控制。
    原几何变换仅记录、不修改。证据不足时明确拒绝，不凭空补成灰色。
    """
    generated=np.asarray(generated,dtype=np.float32)
    context=np.asarray(context,dtype=np.float32)
    hole,valid=np.asarray(hole,dtype=bool),np.asarray(valid,dtype=bool)
    matrix=np.asarray(world_from_uv,dtype=np.float64)
    if generated.shape!=context.shape or generated.shape!=(*hole.shape,3) or valid.shape!=hole.shape:
        raise ValueError('world texture and masks must share the same grid')
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or np.linalg.norm(np.cross(matrix[:3,0],matrix[:3,1]))<1e-12:
        raise ValueError('a fixed nondegenerate world_from_uv transform is required')
    if not np.isfinite(generated).all() or not np.isfinite(context).all() or min(generated.min(),context.min())<0 or max(generated.max(),context.max())>1:
        raise ValueError('expected finite normalized sRGB appearance, not calibrated material')
    geometry=anchor_geometry(context,hole,valid,generated)
    coefficients=geometry['coefficients']
    if model is not None:
        import torch
        device=next(model.parameters()).device
        with torch.no_grad():
            coefficients=model(torch.from_numpy(geometry['features'][None]).to(device),
                torch.from_numpy(coefficients[None]).to(device)).detach().cpu().numpy()[0]
    output,delta=apply_field(generated,context,hole,valid,geometry,coefficients)
    receipt=dict(schema='world_road_local_color_field.v1',world_from_uv=matrix.tolist(),
        normalized_uv_basis='[1, (u+.5)/width*2-1, (v+.5)/height*2-1]',
        coefficients=coefficients.tolist(),asphalt_anchors=geometry['anchor_pixels'],
        channel_delta_limit=.12,geometry_updated=False,renderer_updated=False,
        target_used=False,semantics='exposure-conditioned sRGB world appearance; generated prior remains unmeasured')
    return output,receipt
