# Progress Log

_Last updated: 2026-10-10_

This log covers the first iteration of the baseline described in [`PROPOSAL.md`](./PROPOSAL.md): a state-based recurrent world model, trained only on next-state prediction, probed for hidden physical parameters φ. It records what was built, the methodological problems found and how they were fixed, the results of a 30-run sweep and of the follow-up Level-4 analyses on it, and open questions. For how to run things, see [`QUICKSTART.md`](./QUICKSTART.md).

## TL;DR

- **Prediction (Level 1).** A 2-layer GRU world model infers φ in context. Its late-timestep one-step MSE is 0.2–7% of a memoryless MLP's at force_prob ≥ 0.1, and approaches a φ-conditioned oracle as the data becomes more informative.
- **Identifiability drives emergence (H6, strongest result).** Linear decodability of mass rises monotonically with the probability of applied forcing in both environments, and is exactly zero when mass is structurally non-identifiable. In the unforced spring, the model encodes the identifiable ratios k/m and c/m (R² ≈ 0.75–0.77), while m stays under its Bayes identifiability ceiling (0.39 vs. 0.55). All of this holds across 3 seeds.
- **Strong untrained baselines.** Measured on the steps where a parameter matters, a randomly initialized GRU already linearly decodes forced-motion mass at R² 0.63–0.81, against 0.83–0.98 trained. Decodability alone is therefore weak evidence; the evidence comes from the trained − untrained gap, the identifiability dose-response, and the causal tests. Spring mass is the cleanest Level-2 result (≈0.91 trained vs. ≈0.29 untrained).
- **Causal use (Level 4, exploratory).** In forced motion, interchange interventions on a 1-D mass direction transfer the counterfactual mass belief (corr 0.72 at the first step vs. 0.09 for random natural directions), and this effect tracks identifiability too. In the spring, swaps along 1-D directions barely transfer the belief (corr 0.06–0.16, random 0.02–0.06). Whole-memory swaps produce a consistent sign-flipped "surprise" response at the first post-swap step, which points to a filtering-like memory that mixes the parameter belief with next-state expectation. A same-φ partner control (§3.4) shows that the swap test's noise floor is low (correlation ceiling ≥ 0.96), so the weak spring transfer is not a partner-matching artifact.
- **The spring belief is causally used, in a small subspace the probes miss (§3.5).** A subspace learned by distributed alignment search (DAS) transfers B's φ belief: one direction gives paired corr 0.37–0.46 and 8 dims give 0.46–0.57 in the spring (forced: 0.61–0.92 and 0.67–0.96), against ≤ 0.13 for random, probe-derived and whole-memory swaps. On the untrained GRU, the same search reaches only 0.05–0.11 (spring, force_prob ≥ 0.3). The learned direction is a natural φ coordinate: one dim reads spring mass at R² 0.75. The model adopts only about 10% of the size of B's effect, and A's evidence overwrites it within a few steps.
- **Generalization.** Forced motion degrades about 2–3× on compositional and extrapolation splits. The spring degrades 7–20× when stiffness is extrapolated.

## 1. Setup

### Environments (`latent_physics/envs/`)

Both environments are deterministic 1-D systems with per-trajectory φ, observed state (x, v), and a scalar force action. Forces are piecewise constant over segments of 5–20 steps; each segment is driven with probability `force_prob` (otherwise F = 0). T = 100 steps per trajectory.

| env | dynamics | φ (training range) | integration |
|---|---|---|---|
| `forced` | m·v̇ = F − μ·m·g·sign(v), Coulomb stick/slip with static friction | m ∈ [1,5] (log-uniform), μ ∈ [0.05,0.3] | dt = 0.05, 10 substeps |
| `spring` | m·ẍ + c·ẋ + k·x = F | m ∈ [0.5,2], k ∈ [1,10], c ∈ [0.1,1] (all log-uniform) | dt = 0.1, 20 substeps, semi-implicit Euler |

Splits:
- `id`: the training distribution, with the compositional corner excluded.
- `comp`: the held-out high-high corner, above the 70th percentile of the range in both (m, μ) for forced and both (m, k) for spring, in probe space.
- `extrap`: forced m ∈ [5,10]; spring k ∈ [10,25].

Sizes: 8000 train, 1000 val, 2000 per test split.

Two design changes came out of the diagnostics in §2:
- For `forced`, the model input drops x (`input_dims = (1,)`), because x is dynamically irrelevant and random-walks unboundedly. The model still predicts Δx.
- For `forced`, actions are sampled closed-loop. A push on a block with |v| > 15 is converted into a brake of the same magnitude (`v_soft_max`), which keeps the velocity distribution stationary in t.

### Models (`latent_physics/models/`)

- **GRU world model:** Linear+GELU encoder → two stacked single-layer GRUs (hidden 128) → MLP decoder. It predicts standardized Δs from standardized (s_t, a_t). The layers are separate modules so each layer's state sequence is observable, and `step(..., edit=)` allows interventions on the recurrent carry.
- **Memoryless MLP:** (s_t, a_t) → Δs. It cannot infer φ, so it sets the "φ unknown" error level.
- **Oracle MLP:** (s_t, a_t, φ) → Δs. It sets the "φ known" level. It is *not* a strict lower bound: when the dynamics are trivially extrapolable from history, the GRU can beat it (see §3.1).

Training is teacher-forced one-step MSE with AdamW and a OneCycle schedule (lr 2e-3, batch 128). The GRU trains for 60 epochs and the MLPs for 30, keeping best-validation weights. A full run (training + analysis) takes about 25 s on an RTX 5070, or about 8 min on a 24-thread CPU.

### Analyses (`latent_physics/analysis/`)

| Level | Method |
|---|---|
| 1 | Late-timestep (t ≥ 20) one-step MSE on id/comp/extrap; 25-step open-loop rollouts after 50 steps of context |
| 2 | Ridge probes (per-output α chosen on a trajectory-grouped validation split) on `enc`, `gru1`, `gru2`. Two readouts: **per-timestep emergence curves**, and **R² when relevant**: a pooled probe evaluated only on (trajectory, t) cells where the simulator's sensitivity of the next state to the parameter exceeds 10% of its mean. Controls: raw input, the same architecture at initialization, shuffled labels. Plus a 1-hidden-layer MLP probe as a nonlinear diagnostic |
| 4 | **Interchange intervention** (activation patching). For trajectory A, find a partner B by nearest-neighbour on the last two inputs. Replace the component of every GRU layer's carry along a 1-D parameter direction with B's, and follow A's real inputs for K = 10 steps. Correlate the change in predictions with the simulator's counterfactual (A's states and actions under A's φ with that parameter set to B's value). Directions tested: `decode` (probe weights in raw space) and `encode` (OLS of the hidden state on [φ, two most recent inputs], keeping the φ coefficients). Control: random directions drawn from N(0, Σ_h). Also **mean-ablation** of the same directions in all layers at every step, reported with the fraction of variance removed and the correlation of the extra error with physical sensitivity |
| 4 (follow-ups) | **Noise floor** (§3.4): the same swaps with partners simulated with exactly A's φ. **Subspace swaps** (§3.5): k-dim subspaces per layer (random, encode-PCA, DAS), swapped once or clamped, scored with a paired score that cancels the A-only part of the effect; controls are random subspaces and DAS on the untrained GRU |
| 5/6 | Prediction error and pooled-probe RMSE on comp/extrap |

The verdict thresholds in `experiments/baseline/verdict.py` are heuristics, not significance tests.

## 2. Methodological log

These are problems found during the iteration, in order. Several of them would have produced misleading conclusions if they had gone unnoticed.

1. **Late-horizon error growth was a covariate shift, not physics.** The first baseline's GRU error rose for t > 70. Removing the unbounded x input did not fix it. The actual cause was a heavy right tail in |v|: about 0.7% of samples reached |v| ≥ 35, mostly late in trajectories, with roughly 100× normal error. Closed-loop braking made the |v| distribution stationary (p99 ≈ 22–23 for every t ≥ 25) and cut GRU/oracle at force_prob = 0.7 from 4.5 to 2.1.

2. **Additive steering along probe (decoding) directions is not a valid causal test here.** Shifting h by δ·w/‖w‖² so that the probe readout moves by 0.5σ moved the state by only 0.3–3% of its spread. Probe weights concentrate on low-variance dimensions, the "filter vs. pattern" distinction. Effect slopes were about 0.05, and using encoding directions instead flipped the conclusions (e.g. spring damping corr −0.77). Variance-unmatched random ablation controls made random directions look 50× more important than parameter directions; probe directions carried 0.000–0.002 of the hidden variance against about 0.2–0.3 for random ones. We replaced this with interchange interventions, which use B's actual hidden values, plus random controls with natural variance.

3. **Whole-memory swaps are confounded by recent-state context.** With a random partner, a whole-memory swap gave corr ≈ 0 even on well-trained models, because the carry also encodes the recent trajectory. Partners are now history-matched. Encode directions were likewise confounded with state (e.g. heavy ↔ slow), so they are now estimated with the recent inputs as covariates.

4. **One-step swaps show a "surprise" response.** Even with matched partners, swaps in the spring gave a negative first-step correlation. Following the swap for 10 steps showed a sign flip from k = 0 to k ≥ 1: on seed 0 at fp 0.7, spring whole-memory corr went −0.30 → +0.22, and spring damping −0.10 → +0.44 at k = 2. The effect then decays as A's evidence overwrites the swapped belief. This fits a filtering-like recurrent state that stores a next-state expectation alongside the parameter belief, so a mismatched expectation first triggers a correction. Level 4 metrics are now pooled over K = 10, with k = 0 reported separately.

5. **Last-timestep R² penalizes adaptive forgetting.** In unforced motion, friction is decodable at R² 0.99 from t = 1, then falls to 0.19 by t = 99 as 94% of blocks come to rest and μ stops affecting the future. The model drops parameters that are no longer predictive. "R² when relevant" replaces last-t R² in the verdict, and parameters that never matter are reported as N/A (not identifiable) rather than FAIL.

6. **Identifiability ceilings.** For the unforced spring, only (k/m, c/m) are identifiable. The Bayes ceiling for recovering each parameter from those ratios under the sampling prior (kNN regression, 50k samples) is R² = 0.545 (m), 0.834 (k) and 0.843 (c). Decodability of m above zero in this regime is therefore expected and not a leak.

7. **Partner matching is not the bottleneck of the swap test.** History-matched partners still differ from A in recent history, and in the spring the match is looser than in forced. Swapping with partners that have exactly A's φ measures the resulting noise: it is 7–27% of the true φ effect over the 10-step horizon, but 56–104% of the whole-memory response at k = 0. Details in §3.4.

8. **The plain swap correlation rewards erasing A's belief.** The target f(φ_B) − f(φ_A) has a half that depends on A alone. A DAS subspace trained toward a random other trajectory's params, which B cannot supply, still reached plain corr 0.48–0.62 when clamped. Subspace swaps are therefore scored with a paired difference over two partners of the same swap point (`corr_import`, §3.5), which cancels that half.

9. **A shuffled-target control is empty under the paired score.** Its `corr_import` is zero by construction, because neither partner holds information about the shuffled params. The control for an over-expressive subspace search is the same search on the untrained GRU.

## 3. Results

These are 30 runs: {forced, spring} × force_prob ∈ {0, 0.1, 0.3, 0.7, 1.0} × seeds {0, 1, 2}, full preset. Values are means over seeds; ± is the std over seeds, shown where it is non-negligible. Raw per-run outputs are in `runs/` (gitignored) and the aggregate is in `runs/summary.md`.

### 3.1 Level 1: prediction and generalization

Late one-step MSE ratios:

| env | force_prob | GRU / MLP | GRU / oracle | comp / id | extrap / id |
|---|---|---|---|---|---|
| forced | 0.0 | 0.0003 | 0.04 ± 0.01 | 1.15 | 1.01 |
| forced | 0.1 | 0.073 | 11.4 ± 1.4 | 2.18 | 2.91 |
| forced | 0.3 | 0.036 | 5.8 ± 0.9 | 2.20 | 2.78 |
| forced | 0.7 | 0.012 | 2.1 ± 0.4 | 2.30 | 2.44 |
| forced | 1.0 | 0.007 | 1.4 ± 0.3 | 2.16 | 2.08 |
| spring | 0.0 | 0.0001 | 0.62 ± 0.06 | 1.05 | ~2300* |
| spring | 0.1 | 0.008 | 6.6 ± 2.2 | 2.04 | 7.2 |
| spring | 0.3 | 0.006 | 9.9 ± 0.5 | 2.06 | 11.0 |
| spring | 0.7 | 0.003 | 5.8 ± 0.3 | 1.84 | 16.6 ± 2.0 |
| spring | 1.0 | 0.002 | 4.7 ± 0.8 | 1.71 | 19.6 ± 1.8 |

\* The id error is about 7e-6, so the ratio is uninformative. The oracle degrades comparably on extrap.

- The gap to the oracle shrinks as the data become more informative about φ (forced: 11.4 → 1.4), so prediction quality itself tracks identifiability.
- In the unforced regimes the GRU *beats* the oracle. The dynamics reduce to "repeat the last deceleration" (forced) or a free damped oscillator (spring), which is easier to extrapolate from history than to compute from φ with an MLP.
- Stiffness extrapolation breaks the spring model (7–20×); compositional holdout costs about 2× in both environments.

### 3.2 Level 2: decodability

**R² when relevant** (best trained layer, almost always `gru2`). The untrained control is the best of raw input and untrained layers.

| force_prob | 0.0 | 0.1 | 0.3 | 0.7 | 1.0 |
|---|---|---|---|---|---|
| forced mass | N/A | 0.83 (ctl 0.63) | 0.91 (0.71) | 0.96 (0.76) | 0.98 (0.81) |
| forced friction | 1.00 (ctl 0.99) | 0.63 (0.30) | 0.49 (0.15) | 0.41 (0.09) | 0.46 (0.07) |
| spring mass | 0.42 (0.40) | 0.54 (0.24) | 0.82 (0.26) | 0.91 (0.29) | 0.93 (0.31) |
| spring stiffness | 0.77 (0.71) | 0.79 (0.53) | 0.87 (0.62) | 0.89 (0.67) | 0.91 (0.71) |
| spring damping | 0.72 (0.65) | 0.54 (0.31) | 0.60 (0.42) | 0.65 (0.49) | 0.69 (0.52) |

**Last-timestep R²** (trained), including the identifiable spring ratios:

| force_prob | 0.0 | 0.1 | 0.3 | 0.7 | 1.0 |
|---|---|---|---|---|---|
| forced mass | −0.01 | 0.48 | 0.87 | 0.97 | 0.99 |
| spring mass | 0.39 | 0.55 | 0.84 | 0.93 | 0.94 |
| spring k/m | 0.75 | 0.89 | 0.86 | 0.90 | 0.92 |
| spring c/m | 0.77 | 0.58 | 0.53 | 0.56 | 0.59 |

Observations:
- **Dose-response with identifiability.** Mass decodability is monotonic in force_prob in both environments and exactly zero when mass is non-identifiable. At force_prob = 0.1, forced mass R² grows slowly with t (0.08 → 0.48), consistent with evidence accumulating from rare pushes.
- **Ratio encoding without forcing.** The unforced spring encodes k/m and c/m, and m sits below its identifiability ceiling (0.39 < 0.545).
- **Untrained baselines are strong** wherever the parameter is a near-linear function of the last few inputs: forced mass during pushes (Δv ∝ F/m), friction when coasting (Δv = −μ·g·dt), and spring stiffness (oscillation frequency). The trained − untrained gap is about 0.2 for forced mass and about 0.6 for spring mass.
- **Dissipative parameters get harder to read under forcing:** forced friction 0.63 → 0.41–0.46, and spring c/m 0.77 → 0.53–0.59. In earlier runs, nonlinear probes recovered substantially more (forced friction ~0.60 vs. 0.34 linear), which suggests friction is entangled with the force/mass response rather than stored as a linear coordinate (H3).

### 3.3 Level 4: causal use (exploratory)

Interchange-intervention correlation with the simulator counterfactual. For each run, the better of the decode/encode directions is taken, which is post-hoc and therefore optimistic. Shown as pooled over 10 steps / first step k = 0 / random natural directions:

| force_prob | 0.0 | 0.1 | 0.3 | 0.7 | 1.0 |
|---|---|---|---|---|---|
| forced mass | 0.01 / 0.00 / 0.01 | 0.28 / 0.39 / 0.05 | 0.46 / 0.54 / 0.12 | 0.51 / 0.72 / 0.09 | 0.47 / 0.73 / 0.12 |
| forced friction | 0.19 / 0.23 / 0.09 | 0.19 / 0.03 / 0.03 | 0.22 / 0.10 / 0.03 | 0.31 / 0.27 / 0.02 | 0.37 / 0.38 / 0.03 |
| forced whole memory | 0.73 / 0.69 | 0.19 / −0.01 | 0.37 / 0.23 | 0.40 / 0.27 | 0.39 / 0.21 |
| spring mass | 0.09 / 0.11 / 0.03 | 0.06 / 0.21 / 0.05 | 0.10 / 0.11 / 0.04 | 0.12 / 0.15 / 0.04 | 0.15 / 0.21 / 0.06 |
| spring stiffness | 0.06 / 0.26 / 0.03 | 0.11 / 0.03 / 0.03 | 0.13 / −0.01 / 0.04 | 0.03 / −0.13 / 0.05 | 0.10 / 0.02 / 0.03 |
| spring damping | 0.11 / 0.27 / 0.01 | 0.05 / −0.09 / 0.02 | 0.13 / 0.13 / 0.02 | 0.16 / 0.11 / 0.03 | 0.15 / 0.01 / 0.02 |
| spring whole memory | −0.05 / −0.44 | −0.03 / −0.26 | −0.01 / −0.32 | −0.01 / −0.37 | 0.01 / −0.34 |

- **Forced mass:** the 1-D swap transfers the counterfactual belief well above control, and the effect is monotonic in identifiability. Regression slopes are still far below 1: about 0.02 for decode and about 0.1 for encode at force_prob ≥ 0.7, and even the whole-memory swap only reaches about 0.15. The swapped belief is adopted only partially and is quickly overwritten by A's own evidence, and the 1-D direction carries only part of the functional mass representation.
- **Spring:** the same parameters that decode at R² ≈ 0.9 barely transfer through 1-D swaps. The whole-memory first-step response is consistently negative (−0.26 to −0.44 across all force_prob and seeds), i.e. the surprise effect of §2.4. Whether the spring's φ representation is distributed (Outcome B: needs a higher-dimensional subspace swap) or decodable but not causally used in this form (Outcome C) was left **unresolved** here. §3.4 rules out partner-matching noise as the reason, and §3.5 resolves it: the belief is causally used, in a subspace the probe directions do not find.
- Ablation metrics (Δloss, variance fraction removed, sensitivity correlation) are computed in every `report.md` but have not been analyzed yet.

### 3.4 Level 4: noise floor of the swap test

Partners in the swap test are matched on the last two inputs only, so A's and B's memories differ in recent history as well as in φ, and the model reacts to both. The spring is the harder case for matching: its history has three dims (x, v, F) instead of two, the median match distance reaches 0.10 at force_prob ≥ 0.7 (forced: ≤ 0.02), and a φ change moves one step at dt = 0.1 only a little. So one candidate explanation for the weak spring transfer in §3.3 was that history mismatch drowns out the φ effect.

`latent_physics/experiments/noise_floor.py` measures this directly. For each swap point in A it simulates 599 new trajectories with exactly A's φ (the same pool size as the standard partners), picks the best history match among them, and runs the same swaps. The true effect of these swaps is zero, so the prediction change they cause is the noise floor. The **corr ceiling** is the correlation that a model fully adopting B's belief (slope 1) would reach on top of that noise.

Whole-memory swap, mean over 3 seeds, RMS in standardized Δs units, pooled over K = 10 (the measured corr comes from this script's own sample of swap points, so it differs slightly from §3.3):

| env | force_prob | true effect | model response | noise (same φ) | noise / true | corr ceiling | measured corr |
|---|---|---|---|---|---|---|---|
| forced | 0.0 | 0.035 | 0.013 | 0.002 | 0.07 | 1.00 | 0.83 |
| forced | 0.1 | 0.367 | 0.125 | 0.092 | 0.26 | 0.97 | 0.18 |
| forced | 0.3 | 0.305 | 0.115 | 0.081 | 0.27 | 0.97 | 0.38 |
| forced | 0.7 | 0.279 | 0.111 | 0.059 | 0.21 | 0.98 | 0.44 |
| forced | 1.0 | 0.236 | 0.095 | 0.048 | 0.20 | 0.98 | 0.39 |
| spring | 0.0 | 0.201 | 0.057 | 0.013 | 0.07 | 1.00 | −0.05 |
| spring | 0.1 | 0.486 | 0.110 | 0.073 | 0.16 | 0.99 | −0.07 |
| spring | 0.3 | 0.525 | 0.134 | 0.072 | 0.14 | 0.99 | −0.02 |
| spring | 0.7 | 0.551 | 0.155 | 0.073 | 0.13 | 0.99 | 0.00 |
| spring | 1.0 | 0.547 | 0.145 | 0.068 | 0.13 | 0.99 | 0.02 |

By step after the swap, at force_prob = 0.7:

| env | | k=0 | k=1 | k=2 | k=5 | k=9 |
|---|---|---|---|---|---|---|
| spring | true effect | 0.389 | 0.439 | 0.482 | 0.574 | 0.653 |
| | model response | 0.383 | 0.147 | 0.119 | 0.084 | 0.076 |
| | noise | 0.215 | 0.056 | 0.031 | 0.019 | 0.016 |
| forced | true effect | 0.236 | 0.243 | 0.252 | 0.289 | 0.314 |
| | model response | 0.201 | 0.105 | 0.098 | 0.098 | 0.081 |
| | noise | 0.162 | 0.043 | 0.042 | 0.022 | 0.017 |

- **History mismatch does not explain the weak spring transfer.** The noise is 7–16% of the true effect in the spring (20–27% in forced), and the corr ceiling is at least 0.96 in every condition. Same-φ partners match a little closer than standard ones (median distance 0.07 vs. 0.10 for the spring at force_prob ≥ 0.7). Restricting the noise estimate to same-φ swaps matched no closer than the median standard partner still leaves a ceiling of at least 0.96 in the spring and 0.92 in forced. Building exact partners by inverse dynamics, which was the fallback plan, is not needed.
- **The spring model reacts to B's memory, but mostly not as it would to B's physics.** From k = 1 on, the whole-memory response is 2.7–4.9× the noise floor at every force_prob, yet its correlation with the true effect over steps 1–9 is only 0.02–0.16 (measured in §3.5; an earlier version of this sentence quoted the correlation pooled over all steps including k = 0, −0.07 to +0.02). The model reads something from B's memory other than φ_B, or reads φ_B in a form that does not transfer to A's state. §3.5 finds that a small part of the memory does transfer φ_B.
- **For whole-memory swaps, k = 0 is mostly noise.** At the first step after the swap, same-φ partners already produce 56–74% of the RMS response in the spring and 73–104% in forced (force_prob ≥ 0.1). The magnitude of the "surprise" response in §2.4 is therefore largely history mismatch. Its sign is not explained: noise from same-φ partners cannot correlate with the φ difference, yet the spring k = 0 correlation is consistently negative (−0.27 to −0.39). This control does not cover mismatch that is itself caused by the φ difference (B's history was shaped by φ_B), which remains a candidate.
- **A 1-D swap transfers more than the whole memory.** In the spring at force_prob ≥ 0.7, the encode/mass swap reaches corr 0.15 with noise at 24–25% of its response, while the whole-memory swap stays at about 0. Swapping the whole memory brings in content that overrides B's mass belief, and a swap restricted to a φ subspace avoids part of it. This supports running the subspace swaps in §6.
- Decode directions barely move the prediction (response under 5% of the true effect at force_prob ≥ 0.1), which is consistent with §2.2.

Limitations: the same-φ control only measures mismatch that is independent of φ. The encode and decode directions are refit by this script with its own validation split, so they differ slightly from those in the original runs.

### 3.5 Level 4: subspace swaps

§3.4 showed that the swap test can detect transfer, yet in the spring neither the whole memory nor the 1-D probe directions transfer B's belief. This section asks whether a small subspace does. `latent_physics/experiments/subspace_swap.py` swaps a k-dim subspace in each GRU layer (k = 1–32) and scores the result on steps 1–9 after the swap.

**Subspaces.**
- `random`: k directions drawn with the memory's own covariance (control).
- `encode-PCA`: top-k principal directions of the param-explained part of the memory. The memory is regressed on [z, z², z_i·z_j, z × recent inputs, recent inputs], and the fitted param part is kept. Its rank caps k at 13 (forced) and 27 (spring).
- `DAS` (distributed alignment search): an orthonormal subspace per layer trained by gradient descent so that the swap moves A's predictions like the simulator moves when A takes all of B's params. It is trained on swaps between probe-train trajectories (11,200) and scored on swaps between probe-test trajectories (1,200). Train and test correlations agree within 0.04, so it does not overfit.

**Modes.** `once` swaps at one step and then lets the memory update from A's inputs. `clamp` also holds the subspace at B's value for all 10 steps, so A's evidence cannot overwrite it.

**Score.** The plain correlation compares the response with f(φ_B) − f(φ_A). Its −f(φ_A) half depends on A alone, so a swap that only erases A's own belief scores on it. A DAS run trained toward the effect of a random other trajectory's params (`das_shuffled`), which B's memory cannot hold, reaches plain corr 0.11–0.29 (once) and 0.48–0.62 (clamp). The reported score, `corr_import`, removes that half. Each test swap point gets a second, different partner B2, and the difference of the two responses is correlated with f(φ_B) − f(φ_B2). Since `das_shuffled`'s `corr_import` is zero by construction, the control for an over-expressive search is DAS on the **untrained** GRU (same seed and architecture, never trained). It was run for spring at force_prob 0, 0.3 and 1 and for forced at 1, with k = 1, 2, 8.

`corr_import`, once mode, mean over 3 seeds (DAS std over seeds ≤ 0.05):

| env | force_prob | whole memory | random k=8 | encode-PCA (best k) | DAS k=1 | DAS k=8 | DAS untrained k=1 / k=8 | DAS slope k=8 |
|---|---|---|---|---|---|---|---|---|
| forced | 0.0 | 0.84 | 0.46 | 0.48 | 0.92 | 0.96 | | 0.76 |
| forced | 0.1 | 0.37 | 0.10 | 0.15 | 0.62 | 0.71 | | 0.33 |
| forced | 0.3 | 0.41 | 0.09 | 0.33 | 0.61 | 0.67 | | 0.32 |
| forced | 0.7 | 0.46 | 0.16 | 0.45 | 0.67 | 0.72 | | 0.37 |
| forced | 1.0 | 0.47 | 0.19 | 0.45 | 0.70 | 0.76 | 0.10 / 0.12 | 0.40 |
| spring | 0.0 | 0.13 | 0.10 | 0.08 | 0.37 | 0.55 | 0.30 / 0.32 | 0.11 |
| spring | 0.1 | 0.01 | −0.02 | 0.09 | 0.38 | 0.46 | | 0.06 |
| spring | 0.3 | 0.07 | 0.03 | 0.12 | 0.44 | 0.53 | 0.05 / 0.09 | 0.09 |
| spring | 0.7 | 0.09 | −0.01 | 0.08 | 0.45 | 0.55 | | 0.10 |
| spring | 1.0 | 0.12 | 0.05 | 0.11 | 0.46 | 0.57 | 0.07 / 0.11 | 0.10 |

Clamp mode, k = 8: DAS reaches 0.81–0.98 in forced and 0.81–0.91 in the spring, against 0.09–0.53 and 0.21–0.36 for random subspaces and 0.37–0.68 and 0.33–0.49 for encode-PCA. Untrained DAS reaches −0.03 (forced, 1.0), 0.04 and 0.01 (spring, 0.3 and 1.0), and 0.36 (spring, 0.0). Clamped slopes are 0.53–0.86 (forced) and 0.36–0.51 (spring).

To check whether a DAS subspace is a direction the model uses on its own, or a "dormant" one that only an intervention activates, `subspace_profile` measures the share of the memory's variance that lies in it and the held-out linear R² of φ from the memory's coordinates in it (k = 1, 2, 8; force_prob 0 and 1; 3 seeds).

| k = 1, force_prob = 1 | variance share (chance 0.008) | R² mass | R² k/m |
|---|---|---|---|
| spring, DAS once | 0.093 | 0.75 | 0.44 |
| spring, DAS clamp | 0.030 | 0.00 | 0.01 |
| spring, random | 0.161 | 0.11 | 0.07 |
| spring, encode-PCA | 0.207 | 0.15 | 0.07 |
| forced, DAS once | 0.023 | 0.85 | |
| forced, DAS clamp | 0.013 | 0.94 | |
| forced, random | 0.232 | 0.07 | |

- **The spring model does use its φ belief, and the belief sits in a small subspace.** One learned direction transfers B's belief at corr_import 0.37–0.46, and 8 dims at 0.46–0.57. Random 8-dim subspaces reach 0.10 or less, encode-PCA 0.12 or less at any k, and the whole memory 0.13 or less. Gains level off by k ≈ 4. At force_prob ≥ 0.3 the untrained-GRU control stays at 0.05–0.11, so the method does not produce this on a model that never learned the dynamics. This settles the question left open in §3.3: the spring is not Outcome C.
- **The once-mode DAS direction is a natural φ coordinate.** In the spring at force_prob 1, the single DAS direction holds 9% of the memory's variance (12× chance) and reads mass at R² 0.75 on its own, where random and encode-PCA directions read 0.11–0.15. In forced, one direction reads mass at R² 0.85. A DAS subspace trained toward a shuffled target, which never sees B's params, transfers B's params almost as well (plain corr 0.48–0.58 in the spring, 0.58–0.75 in forced, at k = 8), so the subspace is determined by where A's own belief is read, not by fitting B's labels.
- **The probe-derived directions miss it.** encode-PCA overlaps the DAS subspace no more than a random subspace with the memory's covariance does (spring, k = 8: 0.23–0.24 against 0.21–0.22; chance 0.06). Decode and encode directions (§3.3) and their principal subspace find where φ is *readable*; DAS finds where it is *read*. In this model those differ.
- **The whole memory fails where part of it succeeds.** In the spring, swapping everything transfers almost nothing, while swapping one direction transfers a clear signal. The rest of B's memory carries content that cancels B's belief once A's inputs arrive, in line with the filtering picture of §2.4.
- **Adoption is partial and short-lived.** In once mode the spring model takes up about 10% of the size of B's effect (slope 0.06–0.11; forced 0.32–0.40 at force_prob > 0). Import is strongest at the first step after the swap (spring corr_import 0.85–0.92 at k = 1, step 1) and decays to 0.34–0.68 by step 9 as A's own evidence replaces the swapped belief.
- **Transfer does not track identifiability the way decodability does.** DAS once-mode scores are flat across force_prob > 0 in both environments, while mass decodability (§3.2) rises with force_prob. The target is the effect of all params together, which stays identifiable in every regime (k/m and c/m in the unforced spring, friction in unforced forced motion).

Caveats:
- At force_prob 1, the clamped 1-dim DAS direction in the spring holds little natural variance (3%) and carries no linear φ information (R² ≤ 0.01), yet transfers at 0.69. They may act through a dormant pathway, so clamp-mode scores are an upper bound rather than evidence of the model's own code.
- In the unforced spring, the untrained-GRU control reaches 0.30–0.32 (once) and 0.36–0.42 (clamp), so the trained model's margin there is 0.07–0.23 (once). Without forcing, the dynamics are a linear oscillator, and any memory of the recent motion already reflects k/m and c/m.
- The target swaps all params at once. Whether mass, stiffness and damping occupy separate directions inside the DAS subspace has not been tested.
- The untrained control and the variance/R² profile cover only a subset of conditions (listed above).

## 4. Status against the proposal

| Hypothesis | Status |
|---|---|
| H1: identifiable φ becomes recoverable | Supported (both envs, 3 seeds) |
| H2: kinematics are immediate, φ needs temporal integration | Supported (v decodable at R² ≈ 1 from t = 0; φ emerges over t) |
| H3: φ is not stored as explicit scalars | Partially supported (friction and c/m are nonlinearly or entangled encoded). Against it: one learned direction carries a natural, causally used mass coordinate (§3.5) |
| H4: only part of decodable info is causally operative | Not supported in the form tested: the decodable spring φ is causally used (§3.5). What differs is the direction: probe-derived directions show where φ is readable, not where the model reads it |
| H5: causal structure predicts OOD better than in-distribution loss | Not tested yet |
| H6: identifiability constrains emergence | **Strongly supported** (dose-response, N/A at zero identifiability, ratio encoding, ceiling respected) |
| H7: informative diversity beats sample count | Not tested yet |

In terms of the proposal's outcomes, both environments look like **A/B**: φ is decodable and causally used, through a low-dimensional subspace that the probe directions miss. **C** is ruled out for the spring at force_prob 0.3 and 1, where the untrained control was run, and the same transfer appears at 0.1 and 0.7. **E** (emergence only under identifiable interaction) holds for decodability in both.

## 5. Known limitations

- A single architecture (2-layer GRU, hidden 128) at a single data scale.
- Verdict thresholds are heuristic. There are no significance tests or confidence intervals beyond std over 3 seeds.
- In the baseline, Level 4 uses 1-D directions, and choosing the better of decode/encode per run is post-hoc. The subspace swaps (§3.5) avoid both, but swap all params together, and their untrained control and variance/R² profile cover only a subset of conditions.
- Clamped DAS subspaces may act through dormant directions (§3.5 caveats), so clamp-mode scores are upper bounds.
- The oracle MLP is a reference, not a bound.
- The "relevant" mask uses a fixed 10%-of-mean sensitivity threshold, and the identifiability ceilings were computed only for the unforced spring.
- The collision-dynamics domain from the proposal is not implemented.
- `runs/` (checkpoints, metrics, figures) is gitignored. Results exist only locally and are regenerable with the commands in QUICKSTART (about 15 min on an RTX 5070).

## 6. Next steps

1. **Per-parameter structure of the DAS subspace (Level 3).** Train DAS toward the effect of a single parameter (mass, stiffness, damping, or k/m, c/m) and measure how much the per-parameter subspaces overlap. Separate subspaces would mean the causally used code is factorized.
2. **Why the whole memory fails where one direction succeeds.** Swap only the complement of the DAS subspace and check whether it alone produces the k = 0 "surprise" and cancels the transfer. Then regress its effect on B's older history and predicted next state to identify that content.
3. **Size and decay of adoption.** In once mode the spring model adopts about 10% of B's effect, and import decays over 9 steps. Compare this decay with how fast an ideal Bayesian filter's posterior would move from φ_B to φ_A on the same evidence.
4. **Extend the controls** (untrained DAS, variance/R² profile) to every condition, and add a dormant-direction test for clamp mode (e.g. restrict DAS to the top principal subspace of the memory).
5. **H5 with existing runs:** correlate causal-transfer scores with comp/extrap degradation across the 30 runs, controlling for in-distribution loss. Both quantities track force_prob, so the comparison has to be made within a force_prob level, which needs more variation per level than 3 seeds (see item 8).
6. **Analyze the ablation metrics,** especially sensitivity correlation against variance-matched random directions.
7. **H7:** sweep data size against informativeness, e.g. `--n-train` vs. `--force-prob` at a matched number of informative transitions.
8. **Capacity sweep** (`--hidden 32–256`) and a non-recurrent context model (e.g. a Transformer over the history) to test architecture dependence.
9. **Collision environment** (hidden mass and restitution, identifiable only through contact).
10. Add bootstrap CIs and replace the heuristic verdict with effect sizes.

## 7. Reproducing the results

```bash
uv sync && uv run pytest
for env in forced spring; do for fp in 0.0 0.1 0.3 0.7 1.0; do for s in 0 1 2; do
  uv run python -m latent_physics.experiments.baseline --env $env --preset full --force-prob $fp --seed $s --reuse-models; done; done; done
uv run python -m latent_physics.experiments.summarize   # -> runs/summary.md
uv run python -m latent_physics.experiments.noise_floor # -> runs/*/noise_floor.json (§3.4, about 15 s per run on the GPU)
uv run python -m latent_physics.experiments.subspace_swap runs/*_full_*   # §3.5, about 2.5 min per run
uv run python -m latent_physics.experiments.subspace_swap runs/{spring,forced}_full_fp{0,1}_s{0,1,2} --ks 1 2 8 --tag _diag   # variance/R² profile
uv run python -m latent_physics.experiments.subspace_swap runs/spring_full_fp{0,0.3,1}_s{0,1,2} runs/forced_full_fp1_s{0,1,2} \
  --ks 1 2 8 --tag _untrained --untrained
uv run python -m latent_physics.experiments.subspace_swap --summary   # -> runs/subspace_summary*.md, runs/subspace_*.png
```
