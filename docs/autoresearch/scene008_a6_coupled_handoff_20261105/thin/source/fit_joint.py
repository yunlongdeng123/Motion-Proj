"""BUILD-only numerical fitting of a frozen measured group's proxy opacities.

This deterministic heuristic jointly fits rendered RGB, measured expected depth,
and old-plus-new alpha*T coverage. It does not change the geometry, groups,
retirement operation, method split, or any learned model parameter. The only
optimized values are the small frozen group's proxy gamma vector.
"""
import gc
import hashlib
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.optimize import lsq_linear
import torch

from coupled import RUN, BUILD, HELD, PRIMARY, Coupled, group_mask, read, write
from run_coupled import evaluate, measure, avg


# Registered before any HELD evaluation. These are gamma multipliers in the
# existing Bundle interface. Since the fixed proxy row opacity is .98, the
# effective new-row opacity is bounded above by .98*.35=.343.
GAMMA_CAP = .35
FINITE_DIFFERENCE = .01
MAX_ITERATIONS = 2
LINE_SEARCH = (1., .5, .25)
RIDGE = .02  # Add RIDGE * sum(gamma**2) to the squared residual objective.
EXPECTED_PROXY_ROWS = (7, 5, 5)
TERM_SCALES = {
    "coverage": .01,
    "rgb": .003,
    "depth": .03,
    "known_outside_group_change": .003,
    "legacy_outside_change": .005,
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_flat(value):
    return value.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)


class BuildResiduals:
    """Cache the immutable BUILD traces, masks, and parent references once.

    A residual block is divided by its registered error scale and by the square
    root of that term's total scalar count across BUILD. RGB counts include all
    three channels. Thus masks retain every relevant pixel without allowing a
    large protection region to dominate just because it contains more pixels.
    """

    def __init__(self, experiment):
        self.experiment = experiment
        self.frames = []
        self.counts = {key: 0 for key in TERM_SCALES}
        for raw in BUILD:
            bundle = experiment.bundle(raw, "black")
            mask = group_mask(experiment, raw)
            baseline = bundle.render(0., 0.)
            assert experiment.tr.measure(raw, baseline) == experiment.prior["black"][raw - 89], raw
            reference, _ = bundle.coverage(0., 0., mask)
            data = experiment.tr.raw(raw)
            known_outside = data["obs"] & ~mask
            legacy_outside = experiment.tr.protection[raw]["outside"]
            group_count = int(mask.sum())
            self.counts["coverage"] += group_count
            self.counts["rgb"] += 3 * group_count
            self.counts["depth"] += group_count
            self.counts["known_outside_group_change"] += 3 * int(known_outside.sum())
            self.counts["legacy_outside_change"] += 3 * int(legacy_outside.sum())
            self.frames.append(dict(
                raw=raw, bundle=bundle, mask=mask, baseline=baseline,
                reference=reference, known_outside=known_outside,
                legacy_outside=legacy_outside,
            ))
        self.normalizers = {
            key: TERM_SCALES[key] * np.sqrt(count)
            for key, count in self.counts.items() if count
        }

    def residual(self, gamma):
        """Render each joint scene exactly; do not approximate point insertions."""
        gamma = np.asarray(gamma, dtype=np.float64)
        gamma_gpu = self.experiment.zero.new_tensor(gamma)
        chunks = []
        term_losses = {key: 0. for key in TERM_SCALES}
        for frame in self.frames:
            bundle, mask = frame["bundle"], frame["mask"]
            output = bundle.render(PRIMARY, gamma_gpu)
            coverage, _ = bundle.coverage(PRIMARY, gamma_gpu, mask)
            data = self.experiment.tr.raw(frame["raw"])
            depth = (output["depth_sum"][0] /
                     output["alpha"][0].clamp_min(1e-5) / self.experiment.tr.scale)
            parent_delta = output["rgb"] - frame["baseline"]["rgb"]
            blocks = {
                "coverage": (coverage - frame["reference"])[mask],
                "rgb": (output["rgb"] - data["rgb"])[:, mask],
                "depth": (depth - data["depth"])[mask],
                "known_outside_group_change": parent_delta[:, frame["known_outside"]],
                "legacy_outside_change": parent_delta[:, frame["legacy_outside"]],
            }
            for name, block in blocks.items():
                if name not in self.normalizers:
                    continue
                values = cpu_flat(block) / self.normalizers[name]
                if not np.isfinite(values).all():
                    raise RuntimeError(f"Nonfinite BUILD {name} residual at raw{frame['raw']}")
                chunks.append(values)
                term_losses[name] += float(values @ values)
        ridge = np.sqrt(RIDGE) * gamma
        chunks.append(ridge)
        term_losses["ridge"] = float(ridge @ ridge)
        result = np.concatenate(chunks)
        return result, term_losses

    def metrics(self, gamma):
        gamma_gpu = self.experiment.zero.new_tensor(np.asarray(gamma, dtype=np.float64))
        rows = []
        for frame in self.frames:
            output = frame["bundle"].render(PRIMARY, gamma_gpu)
            coverage, _ = frame["bundle"].coverage(PRIMARY, gamma_gpu, frame["mask"])
            rows.append(measure(
                self.experiment, frame["raw"], output, frame["mask"], coverage,
                frame["reference"], frame["baseline"],
            ))
        return {key: avg([row[key] for row in rows]) for key in rows[0]}

    def close(self):
        self.frames.clear()


def finite_difference_jacobian(problem, gamma, residual):
    """Central .01 differences internally; one-sided .01 near either bound."""
    jacobian = np.empty((len(residual), len(gamma)), dtype=np.float64)
    schemes = []
    for column, value in enumerate(gamma):
        if value >= FINITE_DIFFERENCE and value <= GAMMA_CAP - FINITE_DIFFERENCE:
            above, below = gamma.copy(), gamma.copy()
            above[column] += FINITE_DIFFERENCE
            below[column] -= FINITE_DIFFERENCE
            plus, _ = problem.residual(above)
            minus, _ = problem.residual(below)
            jacobian[:, column] = (plus - minus) / (2 * FINITE_DIFFERENCE)
            schemes.append("central")
            del plus, minus
        elif value < FINITE_DIFFERENCE:
            above = gamma.copy()
            above[column] += FINITE_DIFFERENCE
            plus, _ = problem.residual(above)
            jacobian[:, column] = (plus - residual) / FINITE_DIFFERENCE
            schemes.append("forward")
            del plus
        else:
            below = gamma.copy()
            below[column] -= FINITE_DIFFERENCE
            minus, _ = problem.residual(below)
            jacobian[:, column] = (residual - minus) / FINITE_DIFFERENCE
            schemes.append("backward")
            del minus
    if not np.isfinite(jacobian).all():
        raise RuntimeError("Nonfinite finite-difference Jacobian")
    return jacobian, schemes


def fit_joint(experiment, rank, ledger_hash):
    root_path = RUN / f"root_group{rank}_fit.json"
    root_hash = sha256(root_path)
    scalar = read(root_path)["choices"][str(PRIMARY)]["gamma"]
    if not np.isscalar(scalar):
        raise ValueError("The registered initialization is a scalar root-fit gamma")
    parameter_count = len(experiment.proxy.anchor)
    initial = np.full(parameter_count, np.clip(float(scalar), 0., GAMMA_CAP), dtype=np.float64)
    zero = np.zeros(parameter_count, dtype=np.float64)
    problem = BuildResiduals(experiment)
    history = []
    try:
        zero_residual, zero_terms = problem.residual(zero)
        zero_loss = float(zero_residual @ zero_residual)
        del zero_residual
        residual, terms = problem.residual(initial)
        initial_loss = float(residual @ residual)
        gamma, current_loss = initial.copy(), initial_loss
        # Follow the registered scalar-root initialization. Keep zero as a
        # separate incumbent so the final exact BUILD loss cannot exceed it.
        if zero_loss <= current_loss:
            best_gamma, best_loss, best_terms, best_origin = zero.copy(), zero_loss, zero_terms, "zero"
        else:
            best_gamma, best_loss, best_terms, best_origin = gamma.copy(), current_loss, terms, "root_seed"
        for iteration in range(MAX_ITERATIONS):
            jacobian, schemes = finite_difference_jacobian(problem, gamma, residual)
            solution = lsq_linear(
                jacobian, -residual, bounds=(-gamma, GAMMA_CAP - gamma),
                method="bvls", lsq_solver="exact", tol=1e-8, max_iter=100,
            )
            del jacobian
            record = dict(
                iteration=iteration + 1, gamma_before=gamma.tolist(),
                loss_before=current_loss, finite_difference_schemes=schemes,
                solver=dict(success=bool(solution.success), status=int(solution.status),
                            message=str(solution.message), cost=float(solution.cost),
                            optimality=float(solution.optimality), iterations=int(solution.nit)),
                trials=[],
            )
            history.append(record)
            if not solution.success or not np.isfinite(solution.x).all():
                record["accepted"] = False
                record["stop_reason"] = "BOUNDED_LINEAR_SOLVE_FAILED"
                break
            step = np.asarray(solution.x, dtype=np.float64)
            record["step"] = step.tolist()
            chosen = None
            for strength in LINE_SEARCH:
                candidate = np.clip(gamma + strength * step, 0., GAMMA_CAP)
                candidate_residual, candidate_terms = problem.residual(candidate)
                loss = float(candidate_residual @ candidate_residual)
                record["trials"].append(dict(
                    strength=strength, gamma=candidate.tolist(), loss=loss,
                    term_losses=candidate_terms,
                ))
                # The search is exact and deterministic. Ties retain the first
                # scale in the registered order [1, .5, .25].
                if chosen is None or loss < chosen["loss"]:
                    chosen = dict(gamma=candidate, residual=candidate_residual,
                                  terms=candidate_terms, loss=loss, strength=strength)
                else:
                    del candidate_residual
            tolerance = max(1e-12, abs(current_loss) * 1e-9)
            if chosen["loss"] >= current_loss - tolerance:
                record["accepted"] = False
                record["stop_reason"] = "NO_EXACT_BUILD_LOSS_DECREASE"
                del chosen
                break
            gamma, residual = chosen["gamma"], chosen["residual"]
            terms, current_loss = chosen["terms"], chosen["loss"]
            record.update(accepted=True, selected_strength=chosen["strength"], loss_after=current_loss)
            if current_loss < best_loss:
                best_gamma, best_loss = gamma.copy(), current_loss
                best_terms, best_origin = terms, f"iteration_{iteration + 1}"
            print("JOINT_ITERATION", rank, iteration + 1, current_loss, gamma.tolist(), flush=True)
        del residual
        selected_metrics = problem.metrics(best_gamma)
        retirement_metrics = problem.metrics(zero)
        errors = [
            key for key in ("rgb", "depth")
            if selected_metrics[key]["mean"] is None
            or retirement_metrics[key]["mean"] is None
            or selected_metrics[key]["mean"] > retirement_metrics[key]["mean"] + 1e-7
        ]
        coverage_error = selected_metrics["coverage_error"]["mean"]
        if coverage_error is None or coverage_error > .01:
            errors.append("coverage")
        if not problem.counts["coverage"]:
            status = "NO_BUILD_MEASUREMENT"
        elif not np.any(best_gamma > 1e-12):
            status = "ABSTAIN"
        else:
            status = "BUILD_REJECT" if errors else "BUILD_PASS"
        choice = dict(
            gamma=best_gamma.tolist(), status=status, rgb_depth_failures=errors,
            coverage_error=coverage_error, build_group_pixels=problem.counts["coverage"],
            metrics=selected_metrics, retirement_metrics=retirement_metrics,
            exact_build_loss=best_loss, selected_origin=best_origin,
        )
        choices = {str(PRIMARY): choice}
        assert best_loss <= zero_loss + max(1e-12, abs(zero_loss) * 1e-9)
        assert sha256(root_path) == root_hash
        write(RUN / f"joint_group{rank}_fit.json", dict(
            status="EXECUTION_PASS", rank=rank, choices=choices,
            build=BUILD, held=HELD, selection_used_heldout=False,
            method="Deterministic bounded joint Gauss-Newton opacity heuristic",
            registered=dict(
                retirement=PRIMARY, gamma_cap=GAMMA_CAP, max_iterations=MAX_ITERATIONS,
                finite_difference=FINITE_DIFFERENCE, interior_difference="central",
                boundary_difference="one-sided", exact_line_search=LINE_SEARCH,
                term_scales=TERM_SCALES, count_normalization="global BUILD sqrt scalar count per term",
                ridge=RIDGE, ridge_definition="0.02 * sum(gamma**2)",
                masks="group_mask; observed pixels outside group_mask; fixed legacy outside mask",
                no_other_retirement_levels_run=True,
            ),
            term_scalar_counts=problem.counts, root_fit_sha256=root_hash,
            ledger_sha256=ledger_hash, proxy_parameters=parameter_count,
            initialization=dict(root_scalar=float(scalar), clipped_vector=initial.tolist(),
                                loss=initial_loss, zero_loss=zero_loss,
                                zero_term_losses=zero_terms),
            final_term_losses=best_terms, history=history, training_updates=0,
            learned_policy=False, geometry_and_groups_modified=False,
        ))
        print("JOINT_FIT", rank, choice, flush=True)
        return choices
    finally:
        problem.close()


def main():
    groups_path = RUN / "groups.json"
    groups_hash = sha256(groups_path)
    count = read(groups_path)["qualified_count"]
    if count != len(EXPECTED_PROXY_ROWS):
        raise ValueError(f"Registered experiment expects three qualified groups, got {count}")
    summaries = []
    ledger_hashes = {}
    for rank, expected_rows in enumerate(EXPECTED_PROXY_ROWS):
        path = RUN / f"group_ledger_{rank}.npz"
        ledger_hashes[str(rank)] = sha256(path)
        with np.load(path, allow_pickle=False) as bank:
            ledger = {key: bank[key] for key in bank.files}
        if len(ledger["xyz"]) != expected_rows:
            raise ValueError(f"Frozen group{rank} expected {expected_rows} proxies, got {len(ledger['xyz'])}")
        experiment = Coupled(ledger)
        try:
            choices = fit_joint(experiment, rank, ledger_hashes[str(rank)])
            summaries.append(evaluate(experiment, rank, choices, prefix="joint_"))
            assert experiment.tr.exp.unchanged()
            assert sha256(path) == ledger_hashes[str(rank)]
            assert sha256(groups_path) == groups_hash
        finally:
            del experiment
            gc.collect()
            torch.cuda.empty_cache()
    write(RUN / "joint_summary.json", dict(
        status="EXECUTION_PASS", groups=count, all_groups_tested=True,
        accepted_groups=[result["rank"] for result in summaries if result["method_accepted"]],
        selection_used_heldout=False, retirement=PRIMARY, training_updates=0,
        groups_sha256=groups_hash, ledger_sha256=ledger_hashes,
        learned_policy=False,
    ))


if __name__ == "__main__":
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
