# Motion-Proj: Dynamics Projection Distillation for Vision-Centric Driving World Models

Version: 2026-06-28
Purpose: CVPR-style research plan + implementation roadmap + compute/resource guide

---

## 0. Executive Summary

This document proposes **Motion-Proj**, a compute-conscious and mathematically defensible alignment framework for vision-centric driving world models.

The central idea is:

> Do not backpropagate motion rewards through a full diffusion sampling chain. Do not use pixel-space critic gradients. Instead, project clean generated videos onto a geometry-defined driving dynamics manifold, and distill the projection-induced local denoising score into the low-noise region of the diffusion model.

In Chinese:

> 不反传完整扩散链，不用像素级 critic 梯度攻击视频，而是把 clean-level 生成视频投影到驾驶动态流形上，并把这个投影诱导出的局部 score 蒸馏到低噪声 denoiser 中。

This re-framing avoids four major traps:

1. **SDS misnaming**: the method is not Score Distillation Sampling in the DreamFusion sense.
2. **Adversarial artifact collapse**: the method avoids using raw pixel gradients from a motion critic.
3. **Tweedie OOD trap**: the motion auditor never evaluates high-noise \(x_t\) or high-noise Tweedie reconstructions.
4. **VRAM explosion**: the heavy perception and projection pipeline is no-grad/offline/cached, and training only backpropagates through the video denoiser.

Recommended paper title:

> **Motion-Proj: Dynamics Projection Distillation for Vision-Centric Driving World Models**

Recommended technical subtitle:

> **Projected Denoising Score Matching on the Driving Dynamics Manifold**

---

## 1. CVPR-Style Introduction Draft

Vision-centric driving world models aim to generate plausible future multi-view videos conditioned on past observations, ego trajectories, maps, and planning commands. Unlike generic video generation, driving world models must respect strong geometric and temporal constraints: static background should remain stable after ego-motion compensation, dynamic agents should follow physically plausible trajectories, cross-view observations should agree under camera calibration, and newly appearing objects should be supported by occlusion, cross-view evidence, or coherent motion history.

Modern video diffusion models have become attractive backbones for this problem because they provide high-capacity generative priors and can synthesize realistic future observations. However, their perceptual quality does not guarantee dynamic correctness. Generated vehicles may flicker, static structures may drift across frames, and unsupported objects may appear without valid motion or visibility support. These errors are especially harmful in driving, where a visually plausible but dynamically inconsistent future can mislead downstream planning or evaluation.

A direct solution is to fine-tune a diffusion model using motion-consistency rewards. Existing reward-alignment strategies, however, are poorly matched to large-scale driving video world models. Full-chain reward backpropagation requires storing activations across tens of denoising steps, creating prohibitive memory cost for multi-view video diffusion. Policy-gradient approaches avoid differentiability requirements but suffer from high variance and poor sample efficiency, while each rollout is itself expensive. Pixel-space differentiable reward guidance appears cheaper, but the gradient of a neural motion critic with respect to RGB pixels often behaves like a white-box adversarial perturbation rather than a semantic motion correction.

This paper argues that the right alignment object for driving video world models is not a pixel-space reward gradient, but a **geometry-constrained projection**. Given a clean-level generated or corrupted video, we first extract a low-dimensional motion state: ego-compensated background flow, object tracks, depth/scale estimates, cross-view correspondences, visibility masks, and support relations. We then project this state onto a driving dynamics manifold defined by physically interpretable energies. The projected state is rendered or warped back into video space, producing a dynamically repaired target. Finally, we train the denoiser with a standard single-step diffusion objective in the low-noise neighborhood between the original sample and its projection.

We call this framework **Motion-Proj**, short for **Dynamics Projection Distillation**. Motion-Proj differs from reward fine-tuning in three important ways. First, the motion auditor is only applied to clean-level videos, never to noisy intermediate diffusion states. Second, the correction target is not produced by pixel-space gradients, but by a low-dimensional geometric projection, preventing adversarial high-frequency artifacts. Third, the projection pipeline is detached from training and can be cached offline, so the memory cost remains close to ordinary diffusion fine-tuning.

The resulting method can be interpreted as **projected denoising score matching** in a tubular neighborhood of the driving dynamics manifold. The projection defines a local Gaussian target centered at the dynamically repaired video, and the denoiser learns the corresponding local score field. This gives a mathematically clean alternative to mislabeling the method as Score Distillation Sampling: we do not use a frozen diffusion prior score to optimize a separate representation. Instead, we construct a projection-induced denoising target and distill it into the video world model.

### Contributions

1. **Dynamics projection distillation**: a diffusion alignment framework that projects clean generated driving videos onto a geometry-defined dynamics manifold and distills the induced local score into the low-noise denoiser.
2. **Adversarial-safe motion correction**: a correction mechanism constrained to the image of a geometric renderer/warper, avoiding raw RGB-space critic gradients.
3. **Compute-conscious training**: a no-grad/offline motion auditor and projection pipeline that avoids full-chain diffusion backpropagation and heavy critic backward graphs.
4. **Driving-specific dynamic consistency benchmark**: evaluation metrics for ego-compensated static drift, object trajectory continuity, cross-view consistency, unsupported hallucinations, and downstream detector/tracker stability.

---

## 2. Problem Formulation

Let \(c\) denote the conditioning context, including past multi-view frames, camera intrinsics/extrinsics, ego poses, maps, and optional future ego commands. Let \(x_0 \in \mathbb{R}^{K \times V \times H \times W \times 3}\) be a clean future video with \(K\) future frames and \(V\) cameras.

A diffusion model defines a forward noising process:

\[
z_t = \alpha_t x_0 + \sigma_t \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, I),
\]

and trains a denoiser \(\epsilon_\theta(z_t,t,c)\) or a velocity predictor \(v_\theta(z_t,t,c)\).

The ordinary denoising objective is:

\[
\mathcal{L}_{\text{real}}(\theta)
=
\mathbb{E}_{x_0,c,t,\epsilon}
\left[
\left\|
\epsilon_\theta(\alpha_t x_0+\sigma_t \epsilon,t,c)-\epsilon
\right\|_2^2
\right].
\]

This objective teaches visual realism from data but does not explicitly penalize dynamically inconsistent futures produced by the model. We therefore need an alignment objective that improves dynamic consistency without creating an infeasible training graph.

---

## 3. Why Not Motion-SDS

The earlier name **Motion-SDS** should be avoided. In the original DreamFusion-style formulation, Score Distillation Sampling uses the score of a frozen pretrained diffusion prior to optimize another parameterized representation, often a 3D scene:

\[
\nabla_x \log p_\phi(x_t \mid c).
\]

In contrast, a motion critic energy \(R_{\text{motion}}(x)\) is not a diffusion prior score. Its pixel gradient

\[
\nabla_x R_{\text{motion}}(x)
\]

is a reward or energy gradient, not a denoising score. Calling it SDS invites a fundamental criticism: the method confuses reward guidance with denoising score matching.

Motion-Proj therefore does not claim to perform SDS. It defines a projection operator \(P\) onto a driving dynamics manifold and trains the denoiser to match the score of a local Gaussian target centered at \(P(y)\). This is **projected denoising score matching**, not SDS.

---

## 4. Method Overview

Motion-Proj has three components:

1. **Motion Auditor**: a frozen/no-grad perception and geometry pipeline that extracts low-dimensional motion state from clean-level videos.
2. **Dynamics Projector**: an optimization or filtering module that projects the extracted state onto a driving dynamics manifold.
3. **Projection Distillation**: a diffusion training objective that distills the projection-induced local score into the low-noise denoiser.

The high-level pipeline is:

```text
clean-level sample y
        |
        v
Motion Auditor: extract state s_y and appearance cache a_y
        |
        v
Dynamics Projector: solve s^\dagger = argmin dynamics energy
        |
        v
Renderer/Warper: x^\dagger = Gamma(a_y, s^\dagger)
        |
        v
Low-noise diffusion target: train denoiser around y toward x^\dagger
```

The clean-level sample \(y\) can come from:

1. ground-truth video with synthetic geometry corruption;
2. no-grad samples from the base diffusion model;
3. no-grad samples from a shortcut/few-step student;
4. hard replay samples mined from failure cases.

The key rule is:

> The motion auditor and projector never evaluate high-noise \(x_t\), and they never participate in denoiser backpropagation.

---

## 5. Motion State and Dynamics Manifold

For a clean-level video \(y\), the auditor extracts:

\[
s_y =
\{
u_{\text{static}},
\tau_{\text{obj}},
d,
m_{\text{vis}},
r_{\text{support}},
\kappa
\},
\]

where:

- \(u_{\text{static}}\): ego-compensated static background flow;
- \(\tau_{\text{obj}}\): object trajectories, boxes, masks, or keypoints;
- \(d\): depth or scale estimates;
- \(m_{\text{vis}}\): visibility and occlusion masks;
- \(r_{\text{support}}\): support relation for object appearance/reappearance;
- \(\kappa\): camera calibration and ego-motion metadata.

The appearance cache \(a_y\) stores RGB textures, background patches, object crops, inpainted regions, or latent appearance features. The renderer/warper

\[
\Gamma(a_y,s)
\]

maps low-dimensional state \(s\) back to image/video space. It can be implemented with differentiable or non-differentiable components; differentiability is not required during training because projection is detached.

The dynamics manifold is implicitly defined by low-dimensional energies:

\[
E_{\text{dyn}}(s)
=
\lambda_{\text{static}}E_{\text{static}}(s)
+
\lambda_{\text{obj}}E_{\text{obj}}(s)
+
\lambda_{\text{xview}}E_{\text{xview}}(s)
+
\lambda_{\text{sup}}E_{\text{sup}}(s)
+
\lambda_{\text{prior}}E_{\text{prior}}(s).
\]

### Static Drift Energy

Static regions should align with ego-motion induced flow:

\[
E_{\text{static}}(s)
=
\sum_{v,k}
\sum_{p \in M_{\text{static}}^{v,k}}
\rho
\left(
u_{\text{static}}^{v,k}(p)
-
u_{\text{ego}}^{v,k}(p;\kappa,d)
\right),
\]

where \(\rho\) is a robust penalty such as Huber or Charbonnier.

### Object Motion Smoothness

Dynamic agents should not flicker, teleport, or undergo impossible acceleration:

\[
E_{\text{obj}}(s)
=
\sum_i \sum_k
\rho
\left(
b_{i,k+1}-2b_{i,k}+b_{i,k-1}
\right)
+
\rho
\left(
\Delta \log z_{i,k}
\right),
\]

where \(b_{i,k}\) is a 2D/3D box or track state and \(z_{i,k}\) is depth or scale.

### Cross-View Consistency

The same physical agent should project consistently across cameras:

\[
E_{\text{xview}}(s)
=
\sum_i \sum_k \sum_{v \neq v'}
w_{i,k}^{v,v'}
\rho
\left(
\Pi_v(X_{i,k})-\Pi_{v'}(X_{i,k})
\right),
\]

where \(X_{i,k}\) is a shared 3D state and \(\Pi_v\) is camera projection.

### Support Energy

New dynamic objects should be supported by at least one valid source:

\[
E_{\text{sup}}(s)
=
\sum_{i,k}
\mathbf{1}[\text{new}(i,k)]
\cdot
\ell
\left(
r_{\text{prev}}(i,k),
r_{\text{xview}}(i,k),
r_{\text{occ}}(i,k)
\right).
\]

The support variables indicate whether the object was previously visible, transferred from another view, or plausibly emerged from an occlusion boundary. Unsupported hallucinations are not blindly repaired with arbitrary pixels. They are either masked out, downweighted, or added to replay mining for later robust training.

---

## 6. Dynamics Projection

Given a clean-level sample \(y\), extract \(s_y\) and \(a_y\). The projector solves:

\[
s^\dagger
=
\arg\min_s
\frac{1}{2}
\left\|
s-s_y
\right\|_{\Sigma^{-1}}^2
+
\lambda E_{\text{dyn}}(s),
\]

where \(\Sigma\) encodes confidence in the extracted state. High-confidence tracks and calibrated static regions move less; low-confidence or inconsistent regions can move more.

The repaired video is:

\[
x^\dagger = P(y) = \Gamma(a_y, s^\dagger).
\]

This projection can be solved by:

1. robust least squares for static flow and object trajectories;
2. Kalman smoothing / Rauch-Tung-Striebel smoothing for object tracks;
3. bundle-adjustment-style optimization for cross-view trajectory states;
4. rule-based support filtering for unsupported hallucination;
5. cached warping/rendering back to RGB or latent space.

Practical first implementation:

```text
1. Run detector/tracker/depth/flow no-grad on clean-level videos.
2. Estimate ego-induced static flow using calibration and ego pose.
3. Smooth object boxes/tracks with confidence-weighted temporal regularization.
4. Remove or downweight unsupported object appearances.
5. Warp/render corrected frames or corrected latent targets.
6. Save y, x_dagger, mask, metadata to cache.
```

---

## 7. Projection-Induced Score Matching

For a clean-level imperfect sample \(y\) and its projected target \(x^\dagger=P(y)\), sample a low-noise diffusion timestep \(t \in \mathcal{T}_{\text{tube}}\):

\[
z_t = \alpha_t y + \sigma_t \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I).
\]

Instead of training the denoiser to reconstruct \(y\), train it to reconstruct \(x^\dagger\). The target noise is:

\[
\epsilon^\dagger
=
\frac{z_t-\alpha_t x^\dagger}{\sigma_t}
=
\epsilon
+
\frac{\alpha_t}{\sigma_t}
(y-x^\dagger).
\]

The projection distillation loss is:

\[
\mathcal{L}_{\text{proj}}(\theta)
=
\mathbb{E}_{y,x^\dagger,t,\epsilon}
\left[
\left\|
M_y \odot
\left(
\epsilon_\theta(z_t,t,c)
-
\text{sg}(\epsilon^\dagger)
\right)
\right\|_2^2
\right],
\]

where:

- \(M_y\) is a reliability mask;
- \(\text{sg}(\cdot)\) is stop-gradient;
- the projector is not in the training graph.

The total objective is:

\[
\mathcal{L}(\theta)
=
\mathcal{L}_{\text{real}}
+
\lambda_{\text{proj}}\mathcal{L}_{\text{proj}}
+
\beta\mathcal{L}_{\text{anchor}}
+
\gamma\mathcal{L}_{\text{latent/perceptual}}.
\]

The anchor loss keeps the tuned model close to the base model:

\[
\mathcal{L}_{\text{anchor}}
=
\left\|
\epsilon_\theta(z_t,t,c)
-
\epsilon_{\theta_0}(z_t,t,c)
\right\|_2^2.
\]

This reduces quality regression and discourages overfitting to imperfect projection targets.

---

## 8. Mathematical Defense

### 8.1 Motion-Proj Is Not SDS

Motion-Proj does not use a frozen diffusion prior score to optimize an external representation. It defines a projection operator \(P\) and trains the model on a denoising target centered at \(P(y)\).

For fixed \(y\), define a local target distribution:

\[
q_t(z \mid y)
=
\mathcal{N}
\left(
z;
\alpha_t P(y),
\sigma_t^2 I
\right).
\]

Its score is:

\[
\nabla_z \log q_t(z \mid y)
=
-
\frac{z-\alpha_t P(y)}{\sigma_t^2}.
\]

Predicting \(\epsilon^\dagger=(z-\alpha_t P(y))/\sigma_t\) is equivalent to matching this local score up to the standard diffusion scaling:

\[
\nabla_z \log q_t(z \mid y)
=
-
\frac{1}{\sigma_t}\epsilon^\dagger.
\]

Therefore the method is properly described as **projection-induced denoising score matching**, not SDS.

### 8.2 Why It Avoids Pixel-Space Adversarial Artifacts

The dangerous version constructs:

\[
x_0^+ = x_0 + \eta \nabla_x R(x_0).
\]

In high-dimensional RGB space, \(\nabla_x R(x_0)\) can exploit the critic's non-robust directions and produce adversarial high-frequency noise.

Motion-Proj instead constructs:

\[
x^\dagger-y
=
\Gamma(a_y,s^\dagger)-\Gamma(a_y,s_y).
\]

Assuming \(\Gamma\) is locally smooth in \(s\), a first-order expansion gives:

\[
x^\dagger-y
=
J_s\Gamma(a_y,s_y)(s^\dagger-s_y)
+
O(\|s^\dagger-s_y\|^2).
\]

Thus the correction lies approximately in:

\[
\operatorname{Im}(J_s\Gamma),
\]

the image of the geometry renderer's motion tangent space. This space encodes warps, object track adjustments, depth/scale corrections, and cross-view geometric changes. It does not contain arbitrary RGB perturbations. Hence the method restricts supervision to semantically meaningful motion directions instead of exposing the generator to white-box critic artifacts.

### 8.3 Why It Avoids the Tweedie OOD Trap

The motion auditor is never asked to evaluate:

\[
x_t
\quad \text{or} \quad
\hat{x}_0(x_t)
=
\frac{x_t-\sigma_t\epsilon_\theta(x_t,t,c)}{\alpha_t}
\]

at high-noise timesteps. It only sees clean-level samples \(y\), such as ground-truth corrupted videos or no-grad generated videos.

The projection loss is applied only for timesteps:

\[
t \in \mathcal{T}_{\text{tube}},
\]

where the noisy sample remains in a semantic neighborhood of \(y\). A practical criterion is:

\[
\frac{\alpha_t}{\sigma_t}
\|y-P(y)\|
\le B,
\]

so that the target noise correction does not explode. This prevents the method from poisoning early high-noise denoising with unreliable motion targets.

### 8.4 Why It Avoids VRAM Explosion

Full-chain reward backprop stores activations for:

\[
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0
\rightarrow R(x_0),
\]

which scales roughly with the number of denoising steps and model activation cost.

Pixel-space critic gradient training stores:

\[
\text{Video denoiser backward}
+
\text{Motion critic backward}
+
\text{perception pipeline activations}.
\]

Motion-Proj training stores only:

\[
z_t
\rightarrow
\epsilon_\theta(z_t,t,c)
\rightarrow
\mathcal{L}.
\]

The projection target \(x^\dagger\), mask \(M_y\), and metadata are tensors loaded from cache. No detector, tracker, depth model, flow model, or projector activation graph is retained in GPU memory during denoiser training.

Therefore the peak memory is close to ordinary LoRA/adapter diffusion fine-tuning:

\[
\text{VRAM}_{\text{Motion-Proj}}
\approx
\text{VRAM}_{\text{denoiser backward}}
+
\text{VRAM}_{\text{batch tensors}}.
\]

---

## 9. Training Recipe

### Stage 0: Base Model

Use an existing video diffusion world model if possible. Do not train a large video world model from scratch within the first iteration.

Possible starting points:

1. an existing internal driving video diffusion model;
2. an open video diffusion backbone adapted to multi-view driving;
3. a latent video diffusion model with camera/time conditioning;
4. a small research-scale model for initial validation.

### Stage 1: Build Motion Auditor

Run frozen models and geometry code on clean videos:

```text
Input: multi-view future video, calibration, ego pose
Output: tracks, masks, depth/scale, static flow, confidence, support metadata
```

Candidate modules:

- detector: 2D/3D vehicle and pedestrian detector;
- tracker: multi-object tracker;
- optical flow: frozen flow model;
- depth: monocular or multi-view depth estimator;
- ego compensation: calibration + ego pose;
- support checker: previous-frame, cross-view, and occlusion-boundary logic.

The first implementation can be intentionally simple. The important design principle is that outputs are low-dimensional and cached.

### Stage 2: Create Projection Cache

For each sample \(y\):

1. extract state \(s_y\);
2. compute confidence \(\Sigma\);
3. solve projection \(s^\dagger\);
4. render/warp \(x^\dagger\);
5. produce mask \(M_y\);
6. store metadata.

Cache format recommendation:

```text
sample_id/
  y_latent.pt or y_rgb.mp4
  x_dagger_latent.pt or x_dagger_rgb.mp4
  mask.pt
  context.pt
  metadata.json
```

Use latent-space cache if the base model is latent diffusion. It reduces disk and GPU memory cost.

### Stage 3: Projection Distillation Fine-Tuning

Recommended default:

```text
Backbone: frozen or mostly frozen video diffusion model
Trainable params: LoRA / adapter / temporal attention adapter
Precision: bf16 if stable, fp16 otherwise
Memory tricks: gradient checkpointing, FlashAttention, xFormers if supported
Timesteps: low-noise tube only
Batch: small micro-batches + gradient accumulation
```

Loss:

\[
\mathcal{L}
=
\mathcal{L}_{\text{real}}
+
\lambda_{\text{proj}}\mathcal{L}_{\text{proj}}
+
\beta\mathcal{L}_{\text{anchor}}.
\]

Initial hyperparameters:

```text
lambda_proj: 0.05 to 0.2
beta_anchor: 0.1 to 1.0
LoRA rank: 8 to 32
learning rate: 1e-5 to 5e-5
low-noise timesteps: last 20% to 40% of schedule
gradient accumulation: 4 to 16
```

### Stage 4: Replay Mining

Every few training rounds:

1. sample futures from the current model no-grad;
2. run motion auditor offline;
3. find high-error generated samples;
4. project repairable failures;
5. add them to projection cache.

This closes the gap between synthetic corruption and real model failures.

### Stage 5: Optional Shortcut Baseline

Train or use a 1-4 step shortcut student. Then perform short-chain reward or projection fine-tuning as a baseline:

```text
teacher diffusion model
    -> few-step student
    -> clean generated future
    -> motion projection / reward
    -> 1-4 step fine-tuning
```

This is not the main theoretical contribution, but it is a strong engineering comparison.

---

## 10. Experiments

### Datasets

Primary:

- **Argoverse 2 Sensor**: multi-sensor driving scenes with camera, LiDAR, annotations, calibration, and ego pose. Suitable for dynamic consistency evaluation.

Secondary:

- **nuScenes**: six-camera driving dataset with annotations and ego poses. Suitable for cross-dataset generalization.

Initial debug subset:

```text
AV2 20-50 scenes
short future horizon: 4-8 frames
small resolution: 256x448 or 320x576
front camera first, then multi-view
```

Main experiments:

```text
AV2 200-500 scenes
future horizon: 8-16 frames
resolution: 384x704 or model-native latent size
multi-view: 3 views first, then 6/7 views if memory allows
```

### Metrics

Visual quality:

- FVD;
- LPIPS;
- PSNR/SSIM for reconstruction-style settings;
- CLIP/image embedding distance if relevant.

Driving dynamics:

- ego-compensated static drift;
- dynamic object trajectory acceleration/kink score;
- object track survival and IDF1 on generated video;
- detector AP on generated future;
- cross-view reprojection error;
- unsupported hallucination rate;
- reappearance consistency.

Compute:

- peak VRAM;
- training throughput;
- projection cache generation cost;
- samples per GPU hour;
- wall-clock time to reproduce main table.

### Baselines

Required:

1. base RGB diffusion fine-tuning;
2. RGB + optical flow auxiliary loss;
3. RGB + detector/tracker perceptual loss;
4. DDPO/DPOK-style reward fine-tuning on small setting;
5. DRaFT-K/truncated reward backprop on small setting;
6. Shortcut-MotionFT;
7. Motion-Proj.

Optional but useful:

1. pixel-gradient reward target \(x+\eta\nabla_xR(x)\), showing artifact collapse;
2. Motion-Proj without replay mining;
3. Motion-Proj without anchor loss;
4. Motion-Proj with RGB cache vs latent cache;
5. low-noise tube ablation.

### Main Hypotheses

1. Motion-Proj improves dynamic consistency metrics over base diffusion without degrading FVD significantly.
2. Motion-Proj achieves better compute-quality tradeoff than full-chain or policy-gradient reward alignment.
3. Projection targets avoid high-frequency artifacts seen in pixel-gradient reward guidance.
4. Replay-mined projection samples improve robustness compared with synthetic corruption only.

---

## 11. Compute and Server Recommendation

### Key Principle

This project should be run as:

```text
pretrained model + LoRA/adapter fine-tuning + offline projection cache
```

not:

```text
train a full driving video world model from scratch
```

The latter is a compute black hole.

### Recommended AutoDL Rental Strategy

#### Phase A: Code Bring-Up

Use:

```text
GPU: 1 x RTX 4090 24GB
CPU: 8-16 vCPU
RAM: 64-128GB
Disk: 500GB-1TB
Duration: 1-3 days
```

Purpose:

- dataset loading;
- calibration and ego pose parsing;
- small projection cache;
- one-camera toy training;
- loss sanity check.

Expected constraints:

- multi-view video training will be tight;
- batch size likely 1;
- use low resolution and short horizon.

#### Phase B: Projection Cache Generation

Use:

```text
GPU: 1 x RTX 4090 24GB / L20 48GB / A40 48GB / A100 80GB
CPU: 16-32 vCPU
RAM: 128-256GB
Disk: 1-2TB
```

Purpose:

- run detector/tracker/depth/flow no-grad;
- generate state cache;
- generate projected target cache.

If budget allows, 48GB cards are more comfortable than 4090 for heavy perception modules.

#### Phase C: Main Training

Recommended:

```text
GPU: 1 x A100 80GB / A800 80GB / H20 96GB
CPU: 16-32 vCPU
RAM: 128-256GB
Disk: 2TB
Precision: bf16 preferred
Training: LoRA/adapter + gradient checkpointing + FlashAttention
```

This is the best cost-performance tier for the main method.

Why:

- 80GB/96GB lets you run longer horizons and more views;
- single-card debugging is much simpler than distributed training;
- LoRA/adapter fine-tuning avoids full-parameter memory pressure.

#### Phase D: Paper-Grade Ablation

Use:

```text
GPU: 2 x A100/A800 80GB or 2 x H20 96GB
CPU: 32-64 vCPU
RAM: 256GB
Disk: 2-4TB
Training: DDP or FSDP depending on model size
```

Purpose:

- main table;
- ablations;
- multiple seeds;
- final high-resolution runs;
- baseline comparison.

#### Phase E: Only If Absolutely Necessary

Use:

```text
GPU: 4-8 x A100/H800/H20 class cards
```

Only for:

- full-parameter fine-tuning of a very large Video-DiT;
- large-scale multi-view high-resolution training;
- training a base world model from scratch.

This is not recommended for the first 1-2 months.

### What Not To Rent

Avoid using multiple 24GB gaming cards as the main solution. With ordinary DDP, each GPU still holds a full model replica, so 4 x 4090 does not behave like one 96GB GPU. It only helps throughput after the model fits on one card.

Avoid old low-memory cards such as 16GB V100 or similar. Video diffusion plus multi-view tensors will quickly become painful.

### Practical Server Choice

Best first serious server:

```text
1 x H20 96GB, 24 vCPU, 256GB RAM, 2TB local disk
```

or:

```text
1 x A100/A800 80GB, 24 vCPU, 256GB RAM, 2TB local disk
```

Budget-friendly first step:

```text
1 x RTX 4090 24GB, 16 vCPU, 128GB RAM, 1TB local disk
```

### Disk Planning

Minimum:

```text
1TB: code + subset + small cache
2TB: comfortable AV2 subset + projection cache + checkpoints
4TB: AV2 + nuScenes + replay cache + multiple experiment versions
```

Recommended data policy:

```text
Keep raw dataset in object storage or long-term file storage.
Keep only current shards and active caches on local disk.
Delete intermediate RGB videos after latent cache is verified.
Version projection cache by hash of auditor/projector config.
```

### Memory-Saving Settings

Use:

```text
LoRA/adapters
bf16
gradient checkpointing
FlashAttention/xFormers
latent-space projection targets
micro-batch size 1
gradient accumulation 4-16
low-noise projection loss only
frozen VAE if using latent diffusion
```

Avoid:

```text
full-chain reward backprop
critic backward inside training
pixel-space gradient target
large high-noise projection loss
full-resolution RGB cache for every experiment
```

---

## 12. Implementation Milestones

### Week 1: Minimal Data and Model Loop

Deliverables:

- load a small AV2/nuScenes subset;
- parse camera calibration and ego pose;
- run base model inference or toy video diffusion;
- train one diffusion step on tiny data;
- save generated futures.

Success criterion:

```text
One-camera, short-horizon diffusion fine-tuning runs without OOM.
```

### Week 2: Motion Auditor Prototype

Deliverables:

- static mask or simple segmentation;
- optical flow or feature tracking;
- ego-motion compensation;
- simple object detector/tracker;
- dynamic consistency diagnostic metrics.

Success criterion:

```text
The auditor can rank obvious static drift and object flicker failures above clean samples.
```

### Week 3: Projection Cache V1

Deliverables:

- object track smoothing;
- static background warp correction;
- reliability mask;
- cache writer and loader;
- visual inspection script.

Success criterion:

```text
Projected targets look cleaner than corrupted/generated samples in at least 60-70% of inspected cases.
```

### Week 4: Motion-Proj Training V1

Deliverables:

- projection distillation loss;
- anchor loss;
- low-noise timestep sampling;
- LoRA/adapter training;
- first metric table on tiny subset.

Success criterion:

```text
Motion-Proj improves at least one dynamic metric without obvious visual collapse.
```

### Weeks 5-6: Replay Mining and Baselines

Deliverables:

- sample generated futures;
- mine high-error failures;
- project repairable samples;
- compare against flow auxiliary and detector/tracker perceptual loss;
- optional shortcut baseline.

Success criterion:

```text
Replay mining improves generated failure cases more than synthetic corruption only.
```

### Weeks 7-8: Main Experiments

Deliverables:

- main AV2 table;
- ablations;
- compute table;
- qualitative figure panels;
- failure analysis.

Success criterion:

```text
The method has a clear quality/compute advantage over at least two meaningful baselines.
```

---

## 13. Risks and Fallbacks

### Risk 1: Projection Targets Are Too Noisy

Symptoms:

- warped targets contain tearing;
- masks are unreliable;
- model learns visual artifacts.

Fixes:

- move supervision to latent space;
- use stronger reliability masks;
- reduce \(\lambda_{\text{proj}}\);
- increase anchor loss;
- use only object boxes/tracks at first, not full RGB warping;
- train on high-confidence samples only.

### Risk 2: Dynamic Metrics Improve but FVD Gets Worse

Fixes:

- lower projection loss weight;
- apply projection loss only to dynamic/static error masks;
- add base-model anchor;
- mix more real denoising batches;
- use LoRA rank 8-16 rather than high-rank adapters.

### Risk 3: Auditor Does Not Generalize to Generated Samples

Fixes:

- run replay mining;
- calibrate confidence scores;
- reject low-confidence projection cases;
- use generated failures only after visual verification;
- keep synthetic corruption for stable early training.

### Risk 4: Main Model Still Does Not Fit

Fixes:

- reduce views first;
- reduce future horizon;
- use latent cache;
- use smaller LoRA rank;
- train temporal adapters only;
- switch from 4090 to A100/H20.

### Risk 5: Method Looks Like a Heuristic

Fixes:

- emphasize projection-induced score matching;
- include mathematical propositions;
- show memory complexity;
- show pixel-gradient artifact baseline;
- show low-noise tube ablation;
- show geometric tangent-space interpretation.

---

## 14. Minimal First Experiment

If starting tomorrow on a rented server, do this:

```text
Hardware:
  1 x RTX 4090 24GB first, then 1 x A100 80GB/H20 96GB

Data:
  AV2 subset, 20 scenes
  front camera only
  4-8 future frames
  256x448 or latent equivalent

Model:
  pretrained latent video diffusion if available
  LoRA rank 8 or 16
  base model mostly frozen

Projection:
  static ego compensation + object track smoothing
  no unsupported hallucination repair in V1
  high-confidence masks only

Loss:
  L_real + 0.1 L_proj + 0.5 L_anchor
  low-noise timesteps only

Metrics:
  static drift
  object track smoothness
  FVD/LPIPS
  qualitative panels
```

Pass/fail criterion:

```text
Pass if dynamic consistency improves without visible texture collapse.
Fail if projection targets are visually broken or training degrades base quality.
```

---

## 15. Suggested Abstract Draft

Driving video world models must generate futures that are not only visually realistic but also dynamically consistent under ego motion, object motion, and multi-view geometry. Existing reward-based alignment methods are difficult to scale to this setting: full-chain reward backpropagation is memory-intensive, policy-gradient fine-tuning is sample-inefficient, and pixel-space critic gradients can produce adversarial artifacts rather than semantic motion corrections. We propose Motion-Proj, a dynamics projection distillation framework for vision-centric driving world models. Given a clean-level generated video, Motion-Proj extracts low-dimensional motion state using a frozen geometry and perception auditor, projects the state onto a driving dynamics manifold, and renders a dynamically repaired target. The diffusion model is then fine-tuned with a projection-induced denoising score matching objective in the low-noise neighborhood of the original sample. Since the auditor and projector are no-grad and cacheable, Motion-Proj avoids both denoising-chain backpropagation and critic-backward memory overhead. Experiments on driving video benchmarks evaluate static drift, object trajectory continuity, cross-view consistency, hallucination support, and visual quality, demonstrating improved dynamic consistency under a practical compute budget.

---

## 16. One-Sentence Pitch

> Motion-Proj aligns driving video diffusion models by projecting clean generated futures onto a geometry-defined dynamics manifold and distilling the induced local denoising score, avoiding full-chain reward backpropagation, pixel-space critic gradients, and high-noise critic evaluation.

---

## 17. Server Checklist for New Conversation

When opening a new server conversation, paste this checklist:

```text
Goal:
  Implement Motion-Proj V1 for driving video diffusion alignment.

Hardware:
  Start with 1 x RTX 4090 24GB for code bring-up.
  Move to 1 x A100 80GB or H20 96GB for main training.

Immediate tasks:
  1. Prepare AV2/nuScenes mini subset.
  2. Build dataloader for multi-view future video + calibration + ego pose.
  3. Implement no-grad motion auditor.
  4. Generate projection cache: y, x_dagger, mask, metadata.
  5. Add projection distillation loss to diffusion training.
  6. Run one-camera short-horizon sanity experiment.
  7. Track VRAM, throughput, static drift, object continuity, and FVD/LPIPS.

Non-negotiables:
  Do not call the method SDS.
  Do not use pixel-space critic gradients as targets.
  Do not evaluate critic on high-noise x_t.
  Do not backprop through detector/tracker/flow/depth modules during denoiser training.
  Do not full-chain backprop through 30-50 diffusion steps.
```

---

## 18. References to Cite

Use these categories in the paper draft:

1. Diffusion reward fine-tuning:
   - AlignProp
   - DDPO
   - DPOK
   - DRaFT / DRaFT-K
2. Score distillation:
   - DreamFusion / SDS
   - related text-to-3D score distillation work
3. Driving world models:
   - vision-centric world model and video prediction papers
   - multi-view driving generation papers
4. Datasets:
   - Argoverse 2
   - nuScenes
5. Perception modules:
   - optical flow
   - tracking
   - detection
   - depth estimation
6. Geometry and smoothing:
   - bundle adjustment
   - Kalman smoothing
   - robust least squares

The method should be positioned as a bridge between diffusion alignment and classical driving geometry:

```text
Diffusion model supplies visual generative capacity.
Driving geometry supplies low-dimensional correction structure.
Projection distillation connects them without blowing up compute.
```
