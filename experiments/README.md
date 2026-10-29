# Quick experiments on the follow-up questions in `thoughts/papers.md`

Each folder is one experiment and holds its scripts, raw results and an `analysis.txt` or `summary.csv`.
Run everything from the repo root. These are small-scale pilots on a laptop CPU (October 29, 2026). Treat the numbers as first estimates.

| Folder | Paper it follows up | Question |
|---|---|---|
| `h1_learning_order/` | Let's agree to agree | Do networks share a learning order, and does an exact complexity measure predict it? |
| `h2_adaptive_implicit_bias/` | On the implicit bias in deep-learning algorithms | Does Adam keep GD's min-L2 implicit bias? Is its bias set by the landscape or by the optimizer? |
| `h3_leap_adam/` | Leap complexity and saddle-to-saddle dynamics | Does Adam keep the staircase order and the leap-dependent scaling with dimension? |
| `h4_adam_sharpness/` | README hypothesis 1 | Does √v_t normalization act as negative feedback that makes Adam minimize sharpness? |

## H1: learning order on the exact AIXI prior

**Setup.** This uses the same task as `aixi/` experiment 1, where every behaviour's shortest program length K is known exactly.
- Six transformers: d = 32 and 64, three seeds each, 3,000 steps.
- A behaviour's *accessibility* is its prediction accuracy averaged over all checkpoints, which is the score used in the paper.
- The baseline for "intrinsically easy" is the accuracy of the exact Bayes predictor on the same input orders.

**Findings.**
1. **The learning order is shared.** Accessibility correlates at Spearman 0.85–0.92 across seeds, and just as strongly across widths. Subtracting the Bayes-optimal accuracy barely lowers it (0.83–0.90). So the order is not only "what is intrinsically predictable"; it is also a shared network bias.
2. **Exact program length (Kolmogorov complexity) adds nothing beyond frequency.** K correlates with accessibility at ρ = −0.24. Controlling for how often a behaviour is sampled (log2 P) flips it to +0.07.
3. **Boolean sensitivity explains the order.** It has ρ = −0.63 raw and −0.56 after controlling for both frequency and Bayes accuracy, the strongest of any measure. Runtime (−0.20) and LZ complexity (−0.17) are weaker but real. Sensitivity is a *statistical* simplicity measure, not an algorithmic one, which matches the `aixi/` finding.
4. **The weaker model learns roughly a subset of what the stronger one learns.** 86% of the behaviours d=32 learns, d=64 also learns, against a 39% base rate.
5. **The networks are furthest behind Bayes on mid-complexity behaviours (K = 3–9).** Their accessibility trails the Bayes predictor's by 0.12–0.17. On the simplest and the most complex behaviours the shortfall is only 0.05–0.08.

**Answer to the question in `papers.md`:** yes, the shared order survives a different complexity notion. But the measure that predicts it is sensitivity (how many single-bit input flips change the output), not program length.

## H2: does Adam keep GD's implicit bias? (overparametrized linear regression)

**Setup.**
- n = 30 points in d = 200 dimensions, with a sparse or dense teacher, 5 seeds.
- Every optimizer is trained to an exact interpolant.
- **Rotation test:** train again on rotated features X·Q. The loss landscape is the same up to rotation, so a bias set by the landscape alone must give identical test predictions.

**Findings.**
1. **GD and SGD land exactly on the interpolant closest to the initialization** (distance 0.000), as theory says, and they pass the rotation test with zero disagreement.
2. **Every adaptive method lands somewhere else.** Adam, RMSprop, Adagrad and sign-GD all end 0.6–0.75 (in units of ‖w_GD − w0‖) away from that interpolant, and they move 15–25% further from the initialization.
3. **The adaptive bias comes from the optimizer's coordinate system, not the landscape.** On the rotated problem, Adam's test predictions differ by about 70% (relative RMS) from the unrotated ones. Its implicit bias depends on the basis it was trained in.
4. **The adaptive methods cluster near sign-GD.** They sit 0.2–0.34 from the sign-GD solution, against 0.6–0.75 from GD's. Learning rate and β2 move Adam only a little inside that cluster.
5. **It is not a clean norm minimizer.**
   - Adam has a smaller L∞ norm than GD (about 2.1× the minimum, against GD's 2.5–3.1×), but it is far from the min-L∞ interpolant.
   - Its L1 norm is larger than GD's (3.4× the minimum, against 2.5×). It spreads weight over all coordinates, as sign-like updates do.
   - So on this data the README hypothesis that Adam "maximizes the L1 norm" looks directionally right. A better description is that its solutions follow L∞/sign geometry.
6. **Generalization cost.** With a sparse teacher, Adam's test MSE is about 24% worse than GD's. With a dense teacher the difference is small (about 12%). Neither generalizes well at n/d = 0.15.

## H3: leap staircase under Adam vs SGD

**Setup.**
- Targets: staircase x1 + x1x2 + x1x2x3 (leap 1), and the single parity x1x2x3 (leap 3).
- A 2-layer ReLU net, width 256, trained online.
- d = 10–80, 4 seeds, two learning rates per optimizer. T is the step at which the top monomial is learned.

**Findings.**
1. **Adam keeps the staircase order.** In 100% of runs, for both optimizers, x1 is learned before x1x2, which is learned before x1x2x3.
2. **The leap-dependent scaling survives Adam.** T grows as d^0.4 for leap 1 and as d^1.0 for leap 3 under Adam. SGD shows the same gap (d^0.35–0.52 vs d^1.06–1.53). At these sizes the exponents are below the asymptotic predictions of ~d and ~d², but leap still sets the scaling.
3. **No evidence that Adam escapes leap complexity.** In the first coarse run, Adam seemed to scale better on the parity (d^1.0 vs SGD's d^1.5). That disappears at SGD lr = 0.1 (d^1.06), so it was a learning-rate effect, not an optimizer effect.

**Caveats.** Default (not small) initialization, so the plateaus are short; a cleaner saddle-to-saddle test would use small init. Only d ≤ 80.

## H4: does √v_t make Adam minimize sharpness? (README hypothesis 1)

**Setup.**
- MLP 64-128-128-10 on sklearn digits (600 train / 1197 test), full-batch MSE, 5,000 steps, 3 seeds.
- Optimizers: GD, GD with momentum and Adam, three learning rates each.
- Control: **Adam with v_t frozen at step 300**. It keeps the diagonal scaling but removes the feedback.
- Every 100 steps we measure:
  - the top Hessian eigenvalue (raw sharpness) and the Hessian trace;
  - for Adam, the top eigenvalue of the preconditioned Hessian P⁻¹H, with P = √v̂ + ε. Its stability threshold is 38/lr.

**Findings.**
1. **There is a negative feedback loop, but it acts on preconditioned sharpness, not raw sharpness.** In every run, Adam's preconditioned sharpness sits at 0.98–0.99 × its threshold of 38/lr at every checkpoint after step 1,000. This is the "adaptive edge of stability" of Cohen et al. (2022).
2. **Adam does not end flatter. It usually ends sharper.**
   - At the end of training, Adam's raw sharpness is 26–245 and its trace is 1,100–2,500.
   - GD's raw sharpness is 10–40 and its trace is about 600.
   - Comparing at matched training loss gives the same picture.
   - Adam's raw sharpness also isn't monotone in the learning rate: 137 at lr 1e-3, 26 at 1e-2, 245 at 3e-2.
3. **The feedback isn't specific to Adam.**
   - GD sits at its own threshold of 2/lr in 100% of late checkpoints. Its final sharpness is 40, 20 and 10 at lr 0.05, 0.1 and 0.2.
   - Momentum reaches its threshold only at the largest learning rate.
   - So self-stabilization is generic. Adam's √v_t mainly *changes which quantity is held at the edge*, from H to P⁻¹H.
4. **Freezing v_t mostly doesn't remove it.** With a fixed preconditioner, Adam stays at the edge at lr 1e-3 and 1e-2 (0.96 and 0.98 of threshold). It drops off only at lr 3e-2 (0.72). The regulation comes from edge-of-stability dynamics on the preconditioned landscape, not from v_t adapting.
5. **Sharpness barely predicts generalization here.** Across all 36 runs, Spearman of sharpness vs test accuracy is −0.21 (p = 0.22) at matched loss and −0.08 at the end. For the trace it is −0.26 and −0.14. The sharpest Adam runs (lr 3e-2) do generalize worst (94.6% vs 96–97%).

**Verdict.** The feedback-loop part of the hypothesis holds: √v_t does create a stabilizing loop. The "therefore it minimizes a generalization measure" part does not. The loop pins *preconditioned* sharpness at a learning-rate-set value, raw sharpness ends up higher than GD's, and neither predicts test accuracy well.

A sharper follow-up would ask whether P⁻¹H-sharpness, or a reparametrization-invariant measure, predicts generalization better than raw sharpness.

**Caveats.** One small dataset and architecture, full-batch only, 3 seeds. Raw sharpness is not invariant to reparametrization.

## Not tested yet

- README hypothesis 3: a conservation law in function space.
