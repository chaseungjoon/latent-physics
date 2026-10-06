# Progress Log

_Last updated: 2026-10-06_

This log covers the first iteration of the baseline described in [`PROPOSAL.md`](./PROPOSAL.md): a state-based recurrent world model, trained only on next-state prediction, probed for hidden physical parameters φ. It records what was built, the methodological problems found and how they were fixed, the results of a 30-run sweep, and open questions. For how to run things, see [`QUICKSTART.md`](./QUICKSTART.md).

## TL;DR

- **Prediction (Level 1).** A 2-layer GRU world model infers φ in context. Its late-timestep one-step MSE is 0.2–7% of a memoryless MLP's at force_prob ≥ 0.1, and approaches a φ-conditioned oracle as the data becomes more informative.
- **Identifiability drives emergence (H6, strongest result).** Linear decodability of mass rises monotonically with the probability of applied forcing in both environments, and is exactly zero when mass is structurally non-identifiable. In the unforced spring, the model encodes the identifiable ratios k/m and c/m (R² ≈ 0.75–0.77), while m stays under its Bayes identifiability ceiling (0.39 vs. 0.55). All of this holds across 3 seeds.
- **Strong untrained baselines.** Measured on the steps where a parameter matters, a randomly initialized GRU already linearly decodes forced-motion mass at R² 0.63–0.81, against 0.83–0.98 trained. Decodability alone is therefore weak evidence; the evidence comes from the trained − untrained gap, the identifiability dose-response, and the causal tests. Spring mass is the cleanest Level-2 result (≈0.91 trained vs. ≈0.29 untrained).
- **Causal use (Level 4, exploratory).** In forced motion, interchange interventions on a 1-D mass direction transfer the counterfactual mass belief (corr 0.72 at the first step vs. 0.09 for random natural directions), and this effect tracks identifiability too. In the spring, swaps along 1-D directions barely transfer the belief (corr 0.06–0.16, random 0.02–0.06). Whole-memory swaps produce a consistent sign-flipped "surprise" response at the first post-swap step, which points to a filtering-like memory that mixes the parameter belief with next-state expectation.
- **Generalization.** Forced motion degrades about 2–3× on compositional and extrapolation splits. The spring degrades 7–20× when stiffness is extrapolated.

## 1. Setup

### Environments (`latent_physics/envs.py`)

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

### Models (`latent_physics/models.py`)

- **GRU world model:** Linear+GELU encoder → two stacked single-layer GRUs (hidden 128) → MLP decoder. It predicts standardized Δs from standardized (s_t, a_t). The layers are separate modules so each layer's state sequence is observable, and `step(..., edit=)` allows interventions on the recurrent carry.
- **Memoryless MLP:** (s_t, a_t) → Δs. It cannot infer φ, so it sets the "φ unknown" error level.
- **Oracle MLP:** (s_t, a_t, φ) → Δs. It sets the "φ known" level. It is *not* a strict lower bound: when the dynamics are trivially extrapolable from history, the GRU can beat it (see §3.1).

Training is teacher-forced one-step MSE with AdamW and a OneCycle schedule (lr 2e-3, batch 128). The GRU trains for 60 epochs and the MLPs for 30, keeping best-validation weights. A full run (training + analysis) takes about 25 s on an RTX 5070, or about 8 min on a 24-thread CPU.

### Analyses (`latent_physics/analysis.py`)

| Level | Method |
|---|---|
| 1 | Late-timestep (t ≥ 20) one-step MSE on id/comp/extrap; 25-step open-loop rollouts after 50 steps of context |
| 2 | Ridge probes (per-output α chosen on a trajectory-grouped validation split) on `enc`, `gru1`, `gru2`. Two readouts: **per-timestep emergence curves**, and **R² when relevant**: a pooled probe evaluated only on (trajectory, t) cells where the simulator's sensitivity of the next state to the parameter exceeds 10% of its mean. Controls: raw input, the same architecture at initialization, shuffled labels. Plus a 1-hidden-layer MLP probe as a nonlinear diagnostic |
| 4 | **Interchange intervention** (activation patching). For trajectory A, find a partner B by nearest-neighbour on the last two inputs. Replace the component of every GRU layer's carry along a 1-D parameter direction with B's, and follow A's real inputs for K = 10 steps. Correlate the change in predictions with the simulator's counterfactual (A's states and actions under A's φ with that parameter set to B's value). Directions tested: `decode` (probe weights in raw space) and `encode` (OLS of the hidden state on [φ, two most recent inputs], keeping the φ coefficients). Control: random directions drawn from N(0, Σ_h). Also **mean-ablation** of the same directions in all layers at every step, reported with the fraction of variance removed and the correlation of the extra error with physical sensitivity |
| 5/6 | Prediction error and pooled-probe RMSE on comp/extrap |

The verdict thresholds in `run.py` are heuristics, not significance tests.

## 2. Methodological log

These are problems found during the iteration, in order. Several of them would have produced misleading conclusions if they had gone unnoticed.

1. **Late-horizon error growth was a covariate shift, not physics.** The first baseline's GRU error rose for t > 70. Removing the unbounded x input did not fix it. The actual cause was a heavy right tail in |v|: about 0.7% of samples reached |v| ≥ 35, mostly late in trajectories, with roughly 100× normal error. Closed-loop braking made the |v| distribution stationary (p99 ≈ 22–23 for every t ≥ 25) and cut GRU/oracle at force_prob = 0.7 from 4.5 to 2.1.

2. **Additive steering along probe (decoding) directions is not a valid causal test here.** Shifting h by δ·w/‖w‖² so that the probe readout moves by 0.5σ moved the state by only 0.3–3% of its spread. Probe weights concentrate on low-variance dimensions, the "filter vs. pattern" distinction. Effect slopes were about 0.05, and using encoding directions instead flipped the conclusions (e.g. spring damping corr −0.77). Variance-unmatched random ablation controls made random directions look 50× more important than parameter directions; probe directions carried 0.000–0.002 of the hidden variance against about 0.2–0.3 for random ones. We replaced this with interchange interventions, which use B's actual hidden values, plus random controls with natural variance.

3. **Whole-memory swaps are confounded by recent-state context.** With a random partner, a whole-memory swap gave corr ≈ 0 even on well-trained models, because the carry also encodes the recent trajectory. Partners are now history-matched. Encode directions were likewise confounded with state (e.g. heavy ↔ slow), so they are now estimated with the recent inputs as covariates.

4. **One-step swaps show a "surprise" response.** Even with matched partners, swaps in the spring gave a negative first-step correlation. Following the swap for 10 steps showed a sign flip from k = 0 to k ≥ 1: on seed 0 at fp 0.7, spring whole-memory corr went −0.30 → +0.22, and spring damping −0.10 → +0.44 at k = 2. The effect then decays as A's evidence overwrites the swapped belief. This fits a filtering-like recurrent state that stores a next-state expectation alongside the parameter belief, so a mismatched expectation first triggers a correction. Level 4 metrics are now pooled over K = 10, with k = 0 reported separately.

5. **Last-timestep R² penalizes adaptive forgetting.** In unforced motion, friction is decodable at R² 0.99 from t = 1, then falls to 0.19 by t = 99 as 94% of blocks come to rest and μ stops affecting the future. The model drops parameters that are no longer predictive. "R² when relevant" replaces last-t R² in the verdict, and parameters that never matter are reported as N/A (not identifiable) rather than FAIL.

6. **Identifiability ceilings.** For the unforced spring, only (k/m, c/m) are identifiable. The Bayes ceiling for recovering each parameter from those ratios under the sampling prior (kNN regression, 50k samples) is R² = 0.545 (m), 0.834 (k) and 0.843 (c). Decodability of m above zero in this regime is therefore expected and not a leak.

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
- **Spring:** the same parameters that decode at R² ≈ 0.9 barely transfer through 1-D swaps. The whole-memory first-step response is consistently negative (−0.26 to −0.44 across all force_prob and seeds), i.e. the surprise effect of §2.4. Whether the spring's φ representation is distributed (Outcome B: needs a higher-dimensional subspace swap) or decodable but not causally used in this form (Outcome C) is **unresolved**.
- Ablation metrics (Δloss, variance fraction removed, sensitivity correlation) are computed in every `report.md` but have not been analyzed yet.

## 4. Status against the proposal

| Hypothesis | Status |
|---|---|
| H1: identifiable φ becomes recoverable | Supported (both envs, 3 seeds) |
| H2: kinematics are immediate, φ needs temporal integration | Supported (v decodable at R² ≈ 1 from t = 0; φ emerges over t) |
| H3: φ is not stored as explicit scalars | Partially supported (friction and c/m are nonlinearly or entangled encoded; 1-D causal slopes ≪ 1) |
| H4: only part of decodable info is causally operative | Suggestive (spring: high R², weak 1-D causal transfer); needs subspace interventions |
| H5: causal structure predicts OOD better than in-distribution loss | Not tested yet |
| H6: identifiability constrains emergence | **Strongly supported** (dose-response, N/A at zero identifiability, ratio encoding, ceiling respected) |
| H7: informative diversity beats sample count | Not tested yet |

In terms of the proposal's outcomes: forced motion looks like **A/B** (decodable and causally used, partly distributed), the spring is between **B and C**, and **E** (emergence only under identifiable interaction) holds in both.

## 5. Known limitations

- A single architecture (2-layer GRU, hidden 128) at a single data scale.
- Verdict thresholds are heuristic. There are no significance tests or confidence intervals beyond std over 3 seeds.
- Level 4 uses 1-D directions only. Choosing the better of decode/encode per run is post-hoc.
- The oracle MLP is a reference, not a bound.
- The "relevant" mask uses a fixed 10%-of-mean sensitivity threshold, and the identifiability ceilings were computed only for the unforced spring.
- The collision-dynamics domain from the proposal is not implemented.
- `runs/` (checkpoints, metrics, figures) is gitignored. Results exist only locally and are regenerable with the commands in QUICKSTART (about 15 min on an RTX 5070).

## 6. Next steps

1. **Resolve spring Level 4:** k-dimensional subspace interchange (k = 2–8), using either iterative probe directions (INLP) or a learned rotation (distributed alignment search). If transfer appears at k > 1, that indicates Outcome B; otherwise C.
2. **H5 with existing runs:** correlate causal-transfer scores with comp/extrap degradation across the 30 runs, controlling for in-distribution loss.
3. **Analyze the ablation metrics,** especially sensitivity correlation against variance-matched random directions.
4. **H7:** sweep data size against informativeness, e.g. `--n-train` vs. `--force-prob` at a matched number of informative transitions.
5. **Capacity sweep** (`--hidden 32–256`) and a non-recurrent context model (e.g. a Transformer over the history) to test architecture dependence.
6. **Collision environment** (hidden mass and restitution, identifiable only through contact).
7. Add bootstrap CIs and replace the heuristic verdict with effect sizes.

## 7. Reproducing the results

```bash
uv sync && uv run pytest
for env in forced spring; do for fp in 0.0 0.1 0.3 0.7 1.0; do for s in 0 1 2; do
  uv run python -m latent_physics.run --env $env --preset full --force-prob $fp --seed $s --reuse-models; done; done; done
uv run python -m latent_physics.summarize   # -> runs/summary.md
```
