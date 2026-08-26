# P10R2 Exact-Once Route-Aware Confirmation Closeout

Date: 2026-08-27  
Canonical run: `run://worldsim_v64/WS-V64-P10R2-EXACT-ONCE-CONFIRMATION-01/20260826T203000Z__exact-once-confirmation-s3-r1`  
Formal verdict: `supported_exact_once_route_aware_confirmation`

The single frozen run scored all 96 fresh cases. M0 and M1 both realized mean total coverage 0.4749744584, giving exact
mean coverage delta zero. M1's empirical worst10 route hidden-FREE conflict mean was 0.0403132867, below the frozen 0.05
gate. Both preregistered gates passed. The model was not refit and no policy choice occurred during the run. GPU wall time was
11.804068 seconds and peak RSS was 0.884323 GiB.

The comparative result is mixed and must remain explicit. M1 reduced route-selected voxels from 8,117 to 4,971 and absolute
route conflicts from 54 to 20. However, M0's worst10 rate was 0.0391814871, so M1-M0 was +0.0011317996. Pointwise route
failures increased from one to two, and maximum case rate increased from 0.0681818182 to 0.0833333333. Thus the run supports
M1's own bounded fresh observed empirical tail gate, but does not confirm a relative tail-rate improvement over M0.

This result neither rewrites the historical P10T rejection for current M0 nor establishes a population CVaR bound, physical
collision label, planning benefit, closed-loop behavior, or real-world safety. V64-F25 remains active for the sparse-denominator
relative-effect failure. No tuning on this confirmation cohort is allowed.
