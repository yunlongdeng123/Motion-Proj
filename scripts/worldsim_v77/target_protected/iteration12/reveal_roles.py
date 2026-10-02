"""主显露对象与顺带遮到的保留对象分工；所有受影响B仍须有恢复证据。"""
def primary_and_preserved(ratios, process):
    """不通过遮更多邻车来满足显露门槛；每例至少一个真正的显露主任务。"""
    active=sorted(ratios)
    amounts=[t for t in active if max(ratios[t])>=.30 and sum(v>.05 for v in ratios[t])>=3]
    if not amounts:return None,'insufficient_actual_occlusion'
    primary=[t for t in amounts if process['protected'][t]['visibility_transition'] or process['protected'][t]['sweep_over_B']]
    if not primary:return None,'no_reveal_process'
    # 邻车不能因为不是主任务就放宽证据要求；隐藏部分始终未知仍拒绝。
    if any((process['protected'][t]['approx_other_frame_support_mean'] or 0)<.5 for t in active):
        return None,'insufficient_other_frame_evidence'
    return {'reveal_instances':primary,'preserved_incidental_instances':[t for t in active if t not in primary]},None
