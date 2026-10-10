# Progress Log

_Last updated: 2026-10-10_

First iteration of the baseline in [`PROPOSAL.md`](./PROPOSAL.md): a recurrent world model trained only on next-state prediction, then probed and intervened on for the hidden parameters φ. Commands are in [`QUICKSTART.md`](./QUICKSTART.md).

## Summary

- **Prediction.** The GRU infers φ in context: at force_prob ≥ 0.1 its late one-step error is 0.2–7% of a memoryless MLP's, and it approaches a φ-given oracle as the data become more informative.
- **Identifiability drives decodability (H6, strongest result).** Mass decodability rises with force_prob in both environments and is zero when mass is not identifiable. The unforced spring encodes the identifiable ratios k/m and c/m (R² ≈ 0.75), and m stays under its identifiability ceiling.
- **Untrained baselines are strong.** A random GRU already decodes forced-motion mass at R² 0.63–0.81 (trained 0.83–0.98), so decodability alone is weak evidence. Spring mass is the clean case (0.91 vs. 0.29).
- **The model uses its φ belief (§3.3).** Swapping a small learned subspace of the memory (DAS) from trajectory B into A makes the model predict A with B's physics. In the spring, one direction gives a paired correlation of 0.37–0.46 and 8 directions give 0.46–0.57. Random, probe-derived and whole-memory swaps give ≤ 0.13, and the same search on an untrained GRU gives 0.05–0.11 (force_prob ≥ 0.3). The probes point to where φ is _readable_, not where the model _reads_ it.
- **Adoption is partial.** The spring model takes up about 10% of the size of B's effect, and A's own evidence overwrites it within a few steps.
- **Generalization.** With forcing, errors grow about 2× on held-out parameter combinations, and 7–20× in the spring when stiffness is extrapolated.

## 1. Setup

| env | dynamics | φ (training range) | step |
| --- | --- | --- | --- |
| `forced` | m·v̇ = F − μ·m·g·sign(v), with stick/slip | m ∈ [1,5], μ ∈ [0.05,0.3] | dt 0.05 |
| `spring` | m·ẍ + c·ẋ + k·x = F | m ∈ [0.5,2], k ∈ [1,10], c ∈ [0.1,1] | dt 0.1 |

Trajectories have T = 100 steps; forces are constant over segments of 5–20 steps, each pushed with probability force_prob. Splits: `id` (8000 train, 1000 val, 2000 test), `comp` (high-high corner of (m, μ) or (m, k)), `extrap` (forced m ∈ [5,10], spring k ∈ [10,25]). Two fixes from §2: forced motion does not feed x to the model, and pushes above |v| = 15 become brakes.

Models: a GRU world model (encoder, 2 GRU layers of 128, MLP decoder), a memoryless MLP, and an oracle MLP given φ. One-step MSE, AdamW, 60 epochs (MLPs 30). A run takes about 25 s on an RTX 5070.

The 30-run sweep is {forced, spring} × force_prob {0, 0.1, 0.3, 0.7, 1} × seeds {0, 1, 2}. Values below are means over seeds.

## 2. Method notes

Problems found along the way; each would have misled the conclusions.

1. **Late-horizon error was covariate shift.** A tail of runaway velocities (|v| ≥ 35) carried ~100× error. Turning fast pushes into brakes fixed it (GRU/oracle at force_prob 0.7: 4.5 → 2.1).
2. **Steering along probe weights is not a causal test.** Probe weights sit on low-variance dimensions, so a shift that moves the probe readout barely moves the state. Swaps that use a partner's real memory replaced it.
3. **Swap partners must match recent history**, or the swap mostly changes the remembered trajectory. Partners are nearest neighbours on the last two inputs.
4. **The first step after a swap shows a sign-flipped "surprise".** Scores are taken over the 10 steps after the swap.
5. **Last-timestep R² penalizes forgetting.** Friction stops mattering once a block stops, and the model drops it. Decodability is measured only on steps where the param affects the next state; params that never matter are reported as N/A.
6. **Identifiability ceilings.** From k/m and c/m alone, the best possible R² is 0.55 (m), 0.83 (k), 0.84 (c).
7. **Partner matching is not the bottleneck.** Partners simulated with exactly A's φ give a noise floor of 7–27% of the true effect (§3.3).
8. **Plain swap correlation rewards erasing A's belief.** The target f(φ_B) − f(φ_A) has a half that depends on A alone. Subspace swaps are scored with a paired difference over two partners, which cancels it (`corr_import`).
9. **The control for an over-flexible subspace search is the untrained GRU.** A shuffled-target control is zero by construction under the paired score.

## 3. Results

### 3.1 Prediction and generalization

| env | force_prob | GRU / MLP | GRU / oracle | comp / id | extrap / id |
| --- | --- | --- | --- | --- | --- |
| forced | 0 | 0.0003 | 0.04 | 1.15 | 1.01 |
| forced | 0.1 | 0.073 | 11.4 | 2.18 | 2.91 |
| forced | 0.3 | 0.036 | 5.8 | 2.20 | 2.78 |
| forced | 0.7 | 0.012 | 2.1 | 2.30 | 2.44 |
| forced | 1 | 0.007 | 1.4 | 2.16 | 2.08 |
| spring | 0 | 0.0001 | 0.62 | 1.05 | n/a* |
| spring | 0.1 | 0.008 | 6.6 | 2.04 | 7.2 |
| spring | 0.3 | 0.006 | 9.9 | 2.06 | 11.0 |
| spring | 0.7 | 0.003 | 5.8 | 1.84 | 16.6 |
| spring | 1 | 0.002 | 4.7 | 1.71 | 19.6 |

\* id error ≈ 7e-6, so the ratio is meaningless. In forced motion the gap to the oracle shrinks as the data become more informative. Without forcing the GRU beats the oracle, because extrapolating from history is easier than computing from φ.

### 3.2 Decodability

R² on steps where the param matters (best trained layer; untrained control in parentheses):

| force_prob | 0 | 0.1 | 0.3 | 0.7 | 1 |
| --- | --- | --- | --- | --- | --- |
| forced mass | N/A | 0.83 (0.63) | 0.91 (0.71) | 0.96 (0.76) | 0.98 (0.81) |
| forced friction | 1.00 (0.99) | 0.63 (0.30) | 0.49 (0.15) | 0.41 (0.09) | 0.46 (0.07) |
| spring mass | 0.42 (0.40) | 0.54 (0.24) | 0.82 (0.26) | 0.91 (0.29) | 0.93 (0.31) |
| spring stiffness | 0.77 (0.71) | 0.79 (0.53) | 0.87 (0.62) | 0.89 (0.67) | 0.91 (0.71) |
| spring damping | 0.72 (0.65) | 0.54 (0.31) | 0.60 (0.42) | 0.65 (0.49) | 0.69 (0.52) |

- Mass decodability rises with force_prob. In the unforced spring, last-step R² is 0.39 for m (ceiling 0.55) and 0.75 / 0.77 for k/m / c/m.
- Untrained GRUs are strong wherever φ is a near-linear function of the last few inputs (forced mass during pushes, friction while coasting, spring stiffness).
- Dissipative params get harder to read under forcing (friction 0.63 → 0.41–0.46; spring c/m 0.77 → 0.53–0.59). Nonlinear probes recover more, so friction looks entangled with the force response (H3).

### 3.3 Causal use

**1-D swaps (baseline).** Swapping the memory's component along one probe-derived direction transfers forced-motion mass (corr 0.46–0.51 at force_prob ≥ 0.3, random directions ≈ 0.1), but barely transfers any spring parameter (0.03–0.16). The whole-memory swap fails in the spring as well, so the spring question was open: is φ spread over more dimensions (Outcome B) or not used (Outcome C)?

**Noise floor.** Swaps with partners simulated with exactly A's φ have zero true effect, so whatever they change is matching noise. That noise is 7–16% of the true effect in the spring (20–27% in forced), so a model that fully adopted B's belief could reach corr ≥ 0.96. The test is sensitive enough. At the first step after a swap, though, noise makes up 56–104% of the whole-memory response, so scores below start from step 1.

**Subspace swaps.** A k-dim subspace per GRU layer (k = 1–32) is swapped once, or clamped to B's value for all 10 steps. Subspaces are random, encode-PCA (principal directions of the φ-explained part of the memory), or learned by distributed alignment search (DAS: trained so the swap moves predictions like the simulator moves when A takes B's φ; trained and scored on separate trajectories). The score is `corr_import` over steps 1–9.

Once mode, mean over 3 seeds (DAS std ≤ 0.05):

| env | force_prob | whole memory | random k=8 | encode-PCA (best k) | DAS k=1 | DAS k=8 | DAS on untrained GRU k=1 / 8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forced | 0 | 0.84 | 0.46 | 0.48 | 0.92 | 0.96 | |
| forced | 0.1 | 0.37 | 0.10 | 0.15 | 0.62 | 0.71 | |
| forced | 0.3 | 0.41 | 0.09 | 0.33 | 0.61 | 0.67 | |
| forced | 0.7 | 0.46 | 0.16 | 0.45 | 0.67 | 0.72 | |
| forced | 1 | 0.47 | 0.19 | 0.45 | 0.70 | 0.76 | 0.10 / 0.12 |
| spring | 0 | 0.13 | 0.10 | 0.08 | 0.37 | 0.55 | 0.30 / 0.32 |
| spring | 0.1 | 0.01 | −0.02 | 0.09 | 0.38 | 0.46 | |
| spring | 0.3 | 0.07 | 0.03 | 0.12 | 0.44 | 0.53 | 0.05 / 0.09 |
| spring | 0.7 | 0.09 | −0.01 | 0.08 | 0.45 | 0.55 | |
| spring | 1 | 0.12 | 0.05 | 0.11 | 0.46 | 0.57 | 0.07 / 0.11 |

- **The spring belief is used, and it sits in a few directions.** Gains level off by k ≈ 4. The untrained-GRU control stays ≤ 0.11 at force_prob ≥ 0.3, so this is not an artifact of the search. Outcome C is ruled out.
- **The learned direction is a natural φ coordinate.** In the spring at force_prob 1, the single DAS direction holds 9% of the memory's variance (12× chance) and reads mass at R² 0.75; random and encode-PCA directions read 0.11–0.15. In forced, one direction reads mass at R² 0.85.
- **Probe-derived directions miss it.** In the spring, encode-PCA overlaps the DAS subspace no more than a random subspace does.
- **Swapping everything fails where one direction works.** The rest of B's memory cancels its belief once A's inputs arrive, in line with the first-step surprise (§2, note 4).
- **Adoption is small and short.** Once-mode slope is 0.06–0.11 in the spring (forced 0.32–0.40). In the spring, transfer is strongest one step after the swap (0.85–0.92) and decays to 0.34–0.68 by step 9.
- **Clamped:** DAS reaches 0.81–0.91 (spring) and 0.81–0.98 (forced), slope 0.36–0.86.

Caveats: in the spring at force_prob 1, the clamped 1-dim DAS direction holds 3% of the variance and carries no linear φ information, so clamp scores may run through a dormant pathway and are an upper bound. In the unforced spring the untrained GRU reaches 0.30, which leaves a thin margin. The target swaps all params at once. Ablation metrics are computed in each `report.md` but not yet analyzed.

## 4. Status

| Hypothesis | Status |
| --- | --- |
| H1 identifiable φ becomes recoverable | Supported |
| H2 kinematics immediate, φ needs time | Supported (v at R² ≈ 1 from t = 0; φ emerges over t) |
| H3 φ not stored as explicit scalars | Mixed: friction and c/m look entangled, but one learned direction carries a usable mass coordinate |
| H4 only part of decodable φ is used | Not supported as tested: the spring φ is used, just not along probe directions |
| H5 causal structure predicts OOD | Not tested |
| H6 identifiability constrains emergence | Strongly supported |
| H7 informative diversity beats sample count | Not tested |

Proposal outcomes: both environments look like A/B (decodable, causally used, low-dimensional, missed by probes). E holds for decodability.

## 5. Limitations

- One architecture (2-layer GRU, hidden 128) and one data scale; 3 seeds and no significance tests.
- The untrained control (spring at force_prob 0, 0.3, 1; forced at 1) and the variance/R² profile (force_prob 0 and 1) cover only part of the conditions.
- No collision environment yet.
- `runs/` is gitignored; results are regenerated with the commands below.

## 6. Next steps

1. Train DAS per parameter (mass, stiffness, damping, ratios) and check whether their subspaces separate (Level 3).
2. Swap only the complement of the DAS subspace to find what cancels the belief in whole-memory swaps.
3. Compare the decay of the swapped belief with an ideal Bayesian filter.
4. Run the controls in every condition; test clamp mode for dormant directions.
5. H5 within force_prob levels (needs more models per level, e.g. from item 6); analyze the ablations.
6. H7 (data size vs. informativeness), capacity sweep, a Transformer context model, the collision environment, bootstrap CIs.

## 7. Reproducing

The sweep and follow-ups are in QUICKSTART. The partial-grid runs of §3.3:

```bash
uv run python -m latent_physics.experiments.subspace_swap runs/{spring,forced}_full_fp{0,1}_s{0,1,2} --ks 1 2 8 --tag _diag
uv run python -m latent_physics.experiments.subspace_swap runs/spring_full_fp{0,0.3,1}_s{0,1,2} runs/forced_full_fp1_s{0,1,2} \
  --ks 1 2 8 --tag _untrained --untrained
```
