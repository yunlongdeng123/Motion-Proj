"""Complete r5 trajectory state, variable-N Adam migration, and execution accounting.

Runtime contract: capture_model_state(), restore_model_state(state, optimizer),
named_new_parameters(), make_optimizer(cfg), torch, and frozen_assets.
Checkpoints are taken after the optimizer/topology event at an update boundary.
ExecutionLedger is deliberately outside snapshot: a replay has the same trajectory
state while its extra physical update/topology work is still counted separately.
Importing this module does not import torch or initialize CUDA.
"""
from collections import OrderedDict
import copy
import math
import random
import numpy as np

PARAMETER_NAMES = ("xyz", "log_scales", "raw_quaternion", "opacity_logits", "dc")
MODEL_SCHEMA = "scene065.teacher_gaussian.model_state.v1"
CHECKPOINT_SCHEMA = "scene065.teacher_gaussian.complete_checkpoint.v1"
FAIR_GROUPS = [[106, 107, 108, 109, 110], [120], [137], [165]]
TOPOLOGY_KINDS = ("birth", "clone", "split", "prune")


def cpu(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, OrderedDict):
        return OrderedDict((k, cpu(v)) for k, v in value.items())
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return copy.deepcopy(value)


def equal(a, b, rtol=0.0, atol=0.0):
    """Structural equality; tolerances apply only to floating/complex values."""
    import torch
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        x, y = a.detach().cpu(), b.detach().cpu()
        if (a.is_floating_point() or a.is_complex()) and (rtol or atol):
            return bool(torch.allclose(x, y, rtol=rtol, atol=atol, equal_nan=False))
        return bool(torch.equal(x, y))
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.shape != b.shape or a.dtype != b.dtype:
            return False
        if np.issubdtype(a.dtype, np.inexact) and (rtol or atol):
            return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False))
        return bool(np.array_equal(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k], rtol, atol) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y, rtol, atol) for x, y in zip(a, b))
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)
    result = a == b
    return bool(result)


class BalancedSampler:
    """The previous fair four-group schedule, including the exact RNG behavior."""
    def __init__(self, frames, seed, groups=None):
        self.frames = list(frames)
        if groups is None:
            groups = FAIR_GROUPS if sorted(frames) == sorted(x for g in FAIR_GROUPS for x in g) else [[x] for x in frames]
        self.groups = [list(g) for g in groups]
        flat = [x for g in self.groups for x in g]
        assert all(self.groups) and sorted(flat) == sorted(self.frames) and len(flat) == len(set(flat))
        self.rng = random.Random(seed)
        self.queue = []
        self.draws = 0

    def next(self):
        if not self.queue:
            self.queue = list(range(len(self.groups)))
            self.rng.shuffle(self.queue)
        group = self.queue.pop()
        self.draws += 1
        return self.rng.choice(self.groups[group])

    def state(self):
        return dict(frames=self.frames.copy(), groups=copy.deepcopy(self.groups), queue=self.queue.copy(),
                    rng=self.rng.getstate(), draws=self.draws)

    def restore(self, state):
        assert self.frames == state["frames"] and self.groups == state["groups"]
        assert len(set(state["queue"])) == len(state["queue"])
        assert all(0 <= i < len(self.groups) for i in state["queue"]) and state["draws"] >= 0
        self.queue = state["queue"].copy()
        self.rng.setstate(state["rng"])
        self.draws = int(state["draws"])


def _named(rt):
    result = OrderedDict(rt.named_new_parameters())
    assert tuple(result) == PARAMETER_NAMES, "All five parameter groups must have the agreed stable order"
    assert len({id(p) for p in result.values()}) == 5
    n = len(result["xyz"])
    assert n > 0 and all(len(p) == n for p in result.values())
    return result


def _optimizer_layout(named, optimizer):
    names = {id(p): name for name, p in named.items()}
    layout = []
    for group in optimizer.param_groups:
        assert len(group["params"]) == 1, "Use one actual new-field parameter per Adam group"
        parameter = group["params"][0]
        assert id(parameter) in names, "Frozen/native parameters cannot enter this optimizer"
        name = names[id(parameter)]
        assert group.get("name", name) == name
        layout.append([name])
    assert layout == [[name] for name in PARAMETER_NAMES], "Optimizer parameter references/order are stale"
    return layout


def _validate_model(model):
    import torch
    assert model["schema"] == MODEL_SCHEMA
    field = model["field"]
    parameters = field["parameters"]
    assert tuple(parameters) == PARAMETER_NAMES
    n = len(parameters["xyz"])
    assert n > 0
    for name, value in parameters.items():
        assert torch.is_tensor(value) and value.device.type == "cpu" and len(value) == n
        assert value.is_floating_point() and bool(torch.isfinite(value).all()), name
    assert parameters["xyz"].shape == (n, 3)
    assert parameters["log_scales"].shape == (n, 3)
    assert parameters["raw_quaternion"].shape == (n, 4)
    assert parameters["opacity_logits"].numel() == n and parameters["dc"].numel() == n * 3
    for key in ["prior_xyz", "prior_log_scales", "prior_raw_quaternion", "prior_opacity_logits", "prior_dc", "row_ids", "parent_ids", "source_candidate_ids",
                "confidence", "grad_accum", "vis_count", "max_screen_radius"]:
        assert key in field and len(field[key]) == n, f"Missing or stale per-row state: {key}"
    rows = np.asarray(field["row_ids"])
    assert np.issubdtype(rows.dtype, np.integer) and rows.shape == (n,) and len(np.unique(rows)) == n
    assert int(field["next_row_id"]) > int(rows.max())
    assert int(model["candidate_cursor"]) >= 0
    assert "used_candidate_ids" in model and isinstance(model["topology_events"], list)
    return n


def snapshot(rt, optimizer, sampler, step, bindings):
    """Immutable complete trajectory checkpoint, after any append/split/prune."""
    t = rt.torch
    assert int(step) == step and step >= 0
    assert int(getattr(rt, "pending_microbatches", 0)) == 0, "Save only at optimizer boundaries"
    named = _named(rt)
    layout = _optimizer_layout(named, optimizer)
    model = cpu(rt.capture_model_state())
    assert _validate_model(model) == len(named["xyz"])
    assert all(equal(model["field"]["parameters"][k], p) for k, p in named.items())
    gradients = {name: cpu(p.grad) if p.grad is not None else None for name, p in named.items()}
    return dict(schema=CHECKPOINT_SCHEMA, model=model, optimizer=cpu(optimizer.state_dict()),
        optimizer_layout=layout, sampler=cpu(sampler.state()), logical_step=int(step),
        accumulation=dict(pending_microbatches=0, gradients=gradients), scheduler=None,
        python_rng=copy.deepcopy(random.getstate()), numpy_rng=cpu(np.random.get_state()),
        torch_rng=cpu(t.get_rng_state()), cuda_rng=cpu(t.cuda.get_rng_state_all()) if t.cuda.is_initialized() else [],
        bindings=cpu(bindings), frozen_assets=cpu(rt.frozen_assets))


def restore(rt, optimizer, sampler, state, bindings):
    """Reconstruct variable-N Parameters, restore all state, return the bound Adam."""
    t = rt.torch
    assert state["schema"] == CHECKPOINT_SCHEMA and state["scheduler"] is None
    assert equal(state["bindings"], bindings) and equal(state["frozen_assets"], rt.frozen_assets)
    assert state["accumulation"]["pending_microbatches"] == 0
    _validate_model(state["model"])
    # The runtime rebuilds Parameters at the stored N and rebinds this optimizer.
    rebuilt = rt.restore_model_state(cpu(state["model"]), optimizer)
    if rebuilt is not None:
        optimizer = rebuilt
    named = _named(rt)
    assert _optimizer_layout(named, optimizer) == state["optimizer_layout"]
    optimizer.load_state_dict(cpu(state["optimizer"]))
    assert _optimizer_layout(named, optimizer) == state["optimizer_layout"]
    for name, parameter in named.items():
        gradient = state["accumulation"]["gradients"][name]
        assert gradient is None or gradient.shape == parameter.shape
        parameter.grad = None if gradient is None else gradient.detach().to(device=parameter.device, dtype=parameter.dtype).clone()
    sampler.restore(cpu(state["sampler"]))
    random.setstate(copy.deepcopy(state["python_rng"]))
    np.random.set_state(cpu(state["numpy_rng"]))
    t.set_rng_state(cpu(state["torch_rng"]))
    if state["cuda_rng"]:
        assert t.cuda.is_initialized(), "Construct the native GPU runtime before CUDA RNG restore"
        assert len(state["cuda_rng"]) == t.cuda.device_count()
        t.cuda.set_rng_state_all(cpu(state["cuda_rng"]))
    else:
        assert not t.cuda.is_initialized(), "A CPU trajectory cannot silently become a CUDA trajectory"
    # Capturing must be side-effect-free: this checks parameters, moments, buffers,
    # lineage, cursor, all RNGs, hashes and bindings, not only rendered colors.
    assert equal(snapshot(rt, optimizer, sampler, state["logical_step"], bindings), state), "Complete restore differs"
    return optimizer


def remap_adam_rows(optimizer, old_named, new_named, source_rows):
    """Rebind five variable-N groups while migrating survivor Adam moments exactly.

    source_rows[j] is the old index of a surviving row, or -1 for a new row.
    Birth/clone/split rows get zero moments; parent lineage is stored separately.
    Adam's scalar step and all group hyperparameters remain unchanged. This is
    also valid with an all--1 map before loading a complete checkpoint state.
    """
    import torch
    old_named, new_named = OrderedDict(old_named), OrderedDict(new_named)
    assert tuple(old_named) == tuple(new_named) == PARAMETER_NAMES
    _optimizer_layout(old_named, optimizer)
    old_n, new_n = len(old_named["xyz"]), len(new_named["xyz"])
    assert new_n > 0 and all(len(p) == old_n for p in old_named.values()) and all(len(p) == new_n for p in new_named.values())
    rows = torch.as_tensor(source_rows, dtype=torch.long)
    assert rows.ndim == 1 and len(rows) == new_n
    assert int(rows.min()) >= -1 and int(rows.max()) < old_n
    survivors = rows >= 0
    survivor_ids = rows[survivors]
    assert len(torch.unique(survivor_ids)) == len(survivor_ids), "Duplicate parents must be new clone/split rows with zero moments"
    new_states = {}
    for name in PARAMETER_NAMES:
        old, new = old_named[name], new_named[name]
        assert old.shape[1:] == new.shape[1:] and old.dtype == new.dtype
        old_state = optimizer.state.get(old, {})
        mapped = {}
        for key, value in old_state.items():
            if torch.is_tensor(value) and value.shape == old.shape:
                result = torch.zeros_like(new, dtype=value.dtype, device=value.device)
                target_rows = survivors.to(value.device)
                from_rows = rows[survivors].to(value.device)
                result[target_rows] = value[from_rows]
                mapped[key] = result
            elif torch.is_tensor(value):
                assert value.numel() == 1, f"Unsupported non-row Adam state: {key}"
                mapped[key] = value.detach().clone()
            else:
                mapped[key] = copy.deepcopy(value)
        if old_state:
            assert "exp_avg" in mapped and "exp_avg_sq" in mapped and "step" in mapped
        if mapped:
            new_states[new] = mapped
    optimizer.state.clear()
    for group, name in zip(optimizer.param_groups, PARAMETER_NAMES):
        group["params"] = [new_named[name]]
    optimizer.state.update(new_states)
    _optimizer_layout(new_named, optimizer)
    return optimizer


def gradient_stats(rt, require_present=True):
    """Actual finite/nonzero gradients for every group, before zero_grad()."""
    t = rt.torch
    result = {}
    for name, parameter in _named(rt).items():
        gradient = parameter.grad
        if gradient is None:
            assert not require_present, f"No gradient reached parameter group {name}"
            result[name] = dict(present=False, finite=False, nonzero_values=0)
            continue
        value = gradient.detach()
        finite = bool(t.isfinite(value).all())
        assert finite, f"Nonfinite gradient in {name}"
        result[name] = dict(present=True, finite=True, values=value.numel(),
            nonzero_values=int(t.count_nonzero(value)), max_abs=float(value.abs().max()),
            l2=float(t.linalg.vector_norm(value)))
    return result


def parameter_change_stats(before_model, after_model):
    """Measure real raw-parameter changes on surviving immutable row IDs.

    Births and pruning are reported separately and cannot masquerade as a
    learned change to an existing row. Inputs can come from snapshot['model'].
    """
    import torch
    _validate_model(before_model); _validate_model(after_model)
    before, after = before_model["field"], after_model["field"]
    a = np.asarray(before["row_ids"]); b = np.asarray(after["row_ids"])
    _, ai, bi = np.intersect1d(a, b, return_indices=True)
    result = dict(rows_before=len(a), rows_after=len(b), survivor_rows=len(ai),
                  born_rows=len(b) - len(bi), pruned_rows=len(a) - len(ai), groups={})
    for name in PARAMETER_NAMES:
        x, y = before["parameters"][name][ai], after["parameters"][name][bi]
        delta = (y - x).reshape(len(ai), -1) if len(ai) else torch.empty((0, 1))
        finite = bool(torch.isfinite(delta).all())
        assert finite, f"Nonfinite parameter change in {name}"
        result["groups"][name] = dict(finite=True, compared_rows=len(ai),
            changed_rows=int((delta != 0).any(dim=1).sum()), changed_values=int(torch.count_nonzero(delta)),
            max_abs=float(delta.abs().max()) if delta.numel() else 0.0,
            l2=float(torch.linalg.vector_norm(delta)) if delta.numel() else 0.0)
    return result


class OptimizerEventLedger:
    """Physical work ledger; keep outside the numerically replayed trajectory.

    split means produced child rows, prune includes removed split-parent rows.
    When replaying 32, load the already completed probe ledger (unique=32),
    then record_update(32, replay=True, topology=event32).
    """
    def __init__(self, state=None, unique_budget=2000):
        self.data = copy.deepcopy(state) if state is not None else dict(
            schema="scene065.teacher_gaussian.execution_ledger.v1", unique_budget=unique_budget,
            unique_updates=0, engineering_replay_updates=0, actual_optimizer_updates=0,
            topology_unique={k: 0 for k in TOPOLOGY_KINDS}, topology_replay={k: 0 for k in TOPOLOGY_KINDS}, events=[])
        assert self.data["schema"] == "scene065.teacher_gaussian.execution_ledger.v1"
        assert self.data["unique_budget"] == unique_budget
        assert self.data["actual_optimizer_updates"] == self.data["unique_updates"] + self.data["engineering_replay_updates"]

    def record_update(self, step, replay=False, topology=None):
        assert int(step) == step and step >= 1
        step = int(step)
        if replay:
            assert 1 <= step <= self.data["unique_updates"], "Replay cannot create a new logical update"
        else:
            assert step == self.data["unique_updates"] + 1 and step <= self.data["unique_budget"]
        raw_counts = {k: (topology or {}).get(k, 0) for k in TOPOLOGY_KINDS}
        assert all(int(v) == v for v in raw_counts.values())
        counts = {k: int(v) for k, v in raw_counts.items()}
        assert all(v >= 0 for v in counts.values())
        if topology and "rows_before" in topology and "rows_after" in topology:
            assert topology["rows_after"] == topology["rows_before"] + counts["birth"] + counts["clone"] + counts["split"] - counts["prune"]
        if replay:
            self.data["engineering_replay_updates"] += 1
        else:
            self.data["unique_updates"] = step
        self.data["actual_optimizer_updates"] += 1
        destination = self.data["topology_replay" if replay else "topology_unique"]
        for key, value in counts.items():
            destination[key] += value
        self.data["events"].append(dict(step=step, engineering_replay=bool(replay), topology=counts,
            **({k: int(topology[k]) for k in ["rows_before", "rows_after"] if topology and k in topology})))
        return self.state()

    def state(self):
        return copy.deepcopy(self.data)


def linear_torch(x):
    x = x.clamp_min(0)
    return (x / 12.92).where(x <= .04045, ((x + .055) / 1.055).pow(2.4))


def category_loss(pred, target, torch):
    support = target["support"]
    weights = target["weight"][support]
    denominator = weights.sum()
    assert bool(support.any()) and float(denominator) > 0 and bool(torch.isfinite(denominator))
    error = (pred[:, support] - target["rgb"][:, support]).square().mean(0)
    return (error * weights).sum() / denominator
