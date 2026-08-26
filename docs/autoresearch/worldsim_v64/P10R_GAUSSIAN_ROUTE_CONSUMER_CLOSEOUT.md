# P10R Bounded Gaussian Route Consumer Closeout

Date: 2026-08-26

## Outcome

Canonical run: `run://worldsim_v64/WS-V64-P10R-GAUSSIAN-ROUTE-CONSUMER-01/20260826T183000Z__gaussian-route-consumer-s0-r1`

Verdict: `supported_bounded_gaussian_route_exposure`

All 96 cases were consumed over 1,241.4030305234483 metres of logged future route. C0 supported 12,081 route-corridor cells and M0 supported 12,456, yielding 375 additional cells. Thirty-six cases had positive additional support. Both preregistered gates passed.

## Saturation boundary

C0 and M0 each intercepted all 96 route corridors, so M0 added zero newly intercepted cases. The result therefore supports a fine-grained exposure increase only. It does not support a claim that M0 detects additional collision cases. Any later collision-semantic work must use a non-saturated severity or ground-truth contract rather than retune this binary metric.

The run used the frozen two-second, 20-frame future lidar route and 1.5 metre corridor. It read P10G density packages and processed lidar poses only. Target evidence, the risk model, and collision ground truth were not read. GPU wall time was 0.747154364362359 seconds and peak RSS was 0.6907081604003906 GiB.

## Claim boundary

This result is bounded logged-route semantic exposure. It is not physical collision truth, counterfactual route validity, planning utility, closed-loop behavior, or safety. No parameter sweep, hash, checksum, fingerprint, smoke matrix, or regression matrix was used.
