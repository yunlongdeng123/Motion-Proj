"""训练权重必须完整，不能把推理的成功加载当成训练成功加载。"""
PREFIX='first_stage_model.encoder.'
def complete_training_state(expected,base,encoder):
    required={k for k in expected if k.startswith(PREFIX)}
    if len(required)!=106 or set(encoder)!=required:
        raise ValueError('训练目标encoder的106个权重必须全部来自已验证预训练来源')
    missing=set(expected)-set(base)
    if missing!=required or set(base)-set(expected):
        raise ValueError('基础推理checkpoint除已知encoder外有缺失/额外权重，禁止继续训练')
    merged=dict(base);merged.update(encoder)
    for k,v in expected.items():
        if merged[k].shape!=v.shape:raise ValueError('权重shape不一致: '+k)
    return merged
