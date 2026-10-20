"""固定世界子单元覆盖与先验连通性；不是隐藏纹理真值。"""

from collections import deque
import numpy as np
import torch
import torch.nn.functional as F


def prior_graph(provenance, supported, query, edges):
    """先验上下文也参与传播；UNKNOWN/无曲面支持不能成为桥。"""
    prior = (provenance == 2) & supported
    observed = (provenance == 1) & supported
    eligible = supported[edges].all(1) & (provenance[edges] != 0).all(1) & prior[edges].any(1)
    adjacency = [[] for _ in provenance]
    for a, b in edges[eligible]:
        adjacency[a].append(int(b)); adjacency[b].append(int(a))
    def distance(seeds):
        result = np.full(len(prior), -1, np.int32)
        queue = deque(np.flatnonzero(seeds).tolist())
        result[list(queue)] = 0
        while queue:
            a = queue.popleft()
            for b in adjacency[a]:
                if result[b] < 0:
                    result[b] = result[a]+1; queue.append(b)
        return result
    context_hops = distance(observed & ~query)
    any_hops = distance(observed)
    component = np.full(len(prior), -1, np.int32)
    records = []
    for start in np.flatnonzero(prior):
        if component[start] >= 0:
            continue
        members, anchors, queue = [], set(), deque([int(start)])
        index = len(records); component[start] = index
        while queue:
            a = queue.popleft(); members.append(a)
            for b in adjacency[a]:
                if observed[b]:
                    anchors.add(b)
                elif prior[b] and component[b] < 0:
                    component[b] = index; queue.append(b)
        ids = np.asarray(members)
        records.append(dict(component=index, prior_units=len(members), query_units=int(query[ids].sum()),
            observed_context_boundary_units=sum(not query[a] for a in anchors),
            observed_query_boundary_units=sum(bool(query[a]) for a in anchors),
            anchored_to_context=bool((context_hops[ids] >= 0).any())))
    return dict(edge_valid=eligible, component=component, context_hops=context_hops,
                any_observed_hops=any_hops), records


def fixed_subcells(arrays, metadata, policy):
    """每个0.2m单元上固定3×3位置，局部切平面只用于子单元诊断/先验。"""
    offsets = np.asarray([(x, y) for x in policy['coverage_offsets'] for y in policy['coverage_offsets']])*policy['cell_m']
    anchor = np.asarray(metadata['coordinate_anchor'])
    normal = arrays['normal_world'] @ anchor[:3, :3]
    dz = -(normal[:, None, :2]*offsets[None]).sum(2)/np.maximum(normal[:, None, 2], 1e-6)
    local = np.concatenate((np.broadcast_to(offsets, (*dz.shape, 2)), dz[..., None]), 2)
    world = arrays['world'][:, None]+local @ anchor[:3, :3].T
    transform = np.asarray(metadata['world_to_model'])
    return (world @ transform[:3, :3].T+transform[:3, 3]).astype(np.float32)


def rendered_coverage(alpha, uv, valid, denominator, batch_factor, target=.95):
    from utils.v3_multiview_road import sample_map
    shape = valid.shape
    values = sample_map(alpha, uv.reshape(-1, 2))[:, 0].reshape(shape)
    loss = (F.relu(target-values).square()*valid).sum()/denominator.clamp_min(1)*batch_factor
    return loss, values
