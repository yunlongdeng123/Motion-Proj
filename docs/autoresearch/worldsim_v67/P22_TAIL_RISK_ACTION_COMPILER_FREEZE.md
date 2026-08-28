# V6.7 P22 tail-risk action compiler freeze

P22 trains the P20 architecture on six consumed development domains. The existing equal-domain regression, pairwise and soft
bottom-quartile cost losses remain; a fixed 0.25 weight is added for the soft selected rate of actions that visit any hidden-FREE
state. This binary event is a tail-risk proxy, not a CVaR, collision or safety certificate.

The tail compiler and frozen P20 baseline are written before materializing V64 P10R4 scenes
`1084/1081/0462/0820/0534/0598/0527/0668`. Their action lattice targets are untouched by V6.7. Gates require mean-cost
reduction >=0.35, unsafe-rate reduction >=0.20, unsafe-reduction gain over P20 >=0.02 and six non-increasing mean-cost scenes.

One run only; no unsafe weight, architecture, residual, temperature, selected fraction, tail definition or gate sweep.
