"""CPU数据合同；不依赖模型，不新增任何architecture字段。"""
import numpy as np

def masked_condition_from_x(input_rgb, model_hole):
    """与deletion约定一致：RGB归一化[-1,1]，洞内置0。禁止传GT参数。"""
    x=np.asarray(input_rgb);h=np.asarray(model_hole)
    if x.ndim!=4 or x.shape[-1]!=3 or h.shape!=x.shape[:-1] or x.dtype!=np.uint8 or h.dtype!=bool:
        raise ValueError('Expected uint8 T,H,W,3 input X and bool T,H,W hole')
    cond=x.astype(np.float32)/127.5-1
    cond[h]=0
    return cond

def assert_pair_pixels(gt_rgb,input_rgb,influence,model_hole):
    """X-Y不能超出申报的合成域，完整删除洞必须覆盖全部合成影响域。"""
    y=np.asarray(gt_rgb);x=np.asarray(input_rgb)
    if y.shape!=x.shape or y.dtype!=np.uint8 or x.dtype!=np.uint8:raise ValueError('Mismatched RGB')
    region=np.asarray(influence);hole=np.asarray(model_hole)
    if region.dtype!=bool or hole.dtype!=bool or region.shape!=x.shape[:-1] or hole.shape!=region.shape:raise ValueError('Mismatched masks')
    if np.any(np.any(y!=x,axis=-1)&~region):raise ValueError('undeclared_synthesis_change')
    if np.any(region&~hole):raise ValueError('synthetic_actor_or_edge_outside_deletion_hole')
    return True

def admitted_to_next_stage(geometry_pass,subagent_pass,frame_scores,frame_count):
    """最终人工全检门槛，不把缺分或仅抽帧通过当准入。"""
    if not geometry_pass or not subagent_pass:return False
    if set(frame_scores)!=set(range(frame_count)):return False
    return all(type(frame_scores[i]) is int and frame_scores[i]==1 for i in range(frame_count))
