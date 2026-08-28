# P34 Heteroscedastic Authority Freeze

P34 follows NeurIPS 2023 heteroscedastic regression but uses a minimal bounded mean/scale head. In light of NeurIPS 2024
criticism of single-network evidential epistemic claims, P34 labels its scale aleatoric only and makes no epistemic, OOD, or
calibrated-interval claim.

The joint budget/H training data and features match P31. Mean is bounded to `+-0.05`; scale to `[0.005,0.10]`; training uses
Gaussian NLL plus a fixed mean Huber anchor. On consumed P10X at `(1/3,1.5s)`, conservative priority is fixed as `mean+1*scale`.
It is compared with the frozen P31 mean compiler under identical budget/group constraints.

Gates: exact total, group coverage `>=0.50`, scale-error Spearman `>=0.15`, reduction delta over P31 mean `>=0.01`, and at
least five non-increasing scenes. No scale bound, uncertainty weight, loss, model, or gate sweep. Action-tail P22/P23 stays
closed. No hash/checksum/fingerprint.
