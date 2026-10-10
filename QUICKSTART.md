# Quickstart

The baseline answers one question end to end: **does a GRU world model trained only on next-state prediction develop internal representations of hidden physical parameters, and does it use them?**

One command does everything: it simulates data, trains three models, evaluates them, runs the probes and causal tests, and writes a report with a pass/fail verdict.

## 1. Setup (once)

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10.

```bash
uv sync                     # creates .venv with torch, numpy, matplotlib, pytest
uv run pytest               # physics sanity checks for the simulators (< 1 s)
uv run python -c "import torch; print(torch.cuda.is_available())"
```

The default PyPI torch wheel (CUDA 13) supports the RTX 5070 (Blackwell). If `cuda.is_available()` is `False` on another machine, reinstall torch for that CUDA version, e.g. `uv pip install torch --index-url https://download.pytorch.org/whl/cu128`.

## 2. Run

```bash
# smoke test: ~15 s on CPU, only checks that the pipeline runs (its numbers mean nothing)
uv run python -m latent_physics.experiments.baseline --env forced --preset quick --device cpu

# the baseline: ~10 min on a 24-thread CPU, ~0.5 min on the GPU
uv run python -m latent_physics.experiments.baseline --env forced --preset full
uv run python -m latent_physics.experiments.baseline --env spring --preset full
```

`--device auto` (the default) uses CUDA when it's available. Every run writes to `runs/<env>_<preset>_fp<force-prob>_s<seed>/`:

| file | contents |
|---|---|
| `report.md` | verdict table, all metrics, embedded figures (**start here**) |
| `metrics.json` | everything in machine-readable form |
| `loss_vs_t.png` | one-step error vs. history length for GRU / memoryless MLP / oracle |
| `probe_emergence.png` | linear-probe R² for each hidden param vs. timestep, per layer, with controls |
| `interchange.png` | swap test: model's reaction vs. the simulator's reaction to the same param change |
| `swap_horizon.png` | swap test over the 10 steps after the swap (immediate "surprise" vs. later belief) |
| `gru.pt`, `mlp.pt`, `oracle.pt` | trained weights |
| `log.txt` | training log |

Useful flags: `--reuse-models` (load the trained `.pt` files from the run folder and only redo the analysis; data is regenerated identically from the seed), `--seed N`, `--force-prob P`, `--epochs`, `--hidden`, `--n-train`, `--n-test`, `--lr`, `--batch-size`, `--out DIR`, `--threads N` (CPU).

## 3. What gets run

**Environments** (`latent_physics/envs/`). The model observes the state `(x, v)` and the action (force `F`). Each trajectory has its own hidden parameters φ, which the model never sees.

| env | hidden φ | how φ shows up in the data |
|---|---|---|
| `forced` | mass `m ∈ [1,5]`, Coulomb friction `μ ∈ [0.05,0.3]` | `m` only through the response to `F`; `μ` through coasting deceleration `μg` and stick/slip |
| `spring` | `m ∈ [0.5,2]`, stiffness `k ∈ [1,10]`, damping `c ∈ [0.1,1]` | without forcing, only `k/m` and `c/m` are identifiable (also probed) |

Forces are piecewise constant. Each segment is pushed with probability `--force-prob`. In `forced`, a push on a block already faster than 15 becomes a brake, so speeds stay in a steady range instead of a few trajectories running away. The model is not fed `forced`'s position `x`, which is irrelevant to its dynamics and drifts without bound; it still predicts Δx.

**Splits.** `id` is the training distribution. `comp` is the held-out high-high corner of (param 0, param 1), used for compositional OOD. `extrap` puts one param beyond its training range (`forced`: mass ∈ [5,10]; `spring`: k ∈ [10,25]).

**Models** (`latent_physics/models/`):
- `gru`: an encoder, 2 stacked GRUs and an MLP decoder. Its recurrent state is the only place where φ can accumulate.
- `mlp`: memoryless. It can't infer φ, so it gives the error floor for not knowing the physics.
- `oracle`: an MLP that is given the true φ. It gives the error ceiling for knowing the physics.

**Analyses** (`latent_physics/analysis/`) map onto the proposal's hierarchy:

| level | test | what to look for |
|---|---|---|
| 1 prediction | late-timestep one-step MSE, 25-step rollouts, on id/comp/extrap | GRU ≪ MLP, and GRU → oracle |
| 2 decodability | ridge probe for each (layer, t), split by trajectory; controls: raw input, untrained model, shuffled labels. **R² when it matters** = probe measured only on steps where the param physically affects the next step; this is the number the verdict uses (N/A if the param never matters) | R² high, well above controls, rising with t |
| 4 causal use (exploratory) | **swap test**: copy one param's component of the GRU memory from a trajectory with matching recent history, then, over the next 10 steps of A's real observations, compare the model's change in prediction with the simulator's change when that param really is swapped. The first step (k=0) can be a "surprise" reaction in the opposite direction; the belief shows up from k=1 on. Both `decode` (probe readout) and `encode` (how the memory moves with the param, state held fixed) directions are tested. **Ablation** of the same directions, with variance-matched random controls and a check of whether the damage lands where the param physically matters | corr ≫ random |
| 5/6 generalization | probe RMSE and prediction error on comp / extrap | how much worse than id |

The **Verdict** table at the top of `report.md` turns these into PASS/FAIL checks. The thresholds are rules of thumb (see `latent_physics/experiments/baseline/verdict.py`), not significance tests. Level 4 is still exploratory: a GRU memory mixes "belief about φ" with "expectation of the next state", which is why the swap is followed over several steps.

To compare many runs (mean ± std over seeds, grouped by env × force_prob):

```bash
uv run python -m latent_physics.experiments.summarize        # prints a table and writes runs/summary.md
```

## 4. Experiments

Each full run takes about 0.5 min on the RTX 5070, and a `--reuse-models` re-analysis takes about 0.3 min. Results of the sweep below are written up in [`PROGRESS.md`](./PROGRESS.md).

1. **Seeds × identifiability sweep (H6).** With no applied forces, `forced` mass never affects the trajectory (reported as N/A), while friction stays readable. In `spring`, `m` alone should fall to its identifiability ceiling (≈0.55 R² from `k/m`, `c/m` alone), while `k/m` and `c/m` remain. `--reuse-models` re-analyzes runs that already exist and trains the missing ones:
   ```bash
   for env in forced spring; do for fp in 0.0 0.1 0.3 0.7 1.0; do for s in 0 1 2; do
     uv run python -m latent_physics.experiments.baseline --env $env --preset full --force-prob $fp --seed $s --reuse-models; done; done; done
   ```
2. `uv run python -m latent_physics.experiments.summarize`
3. **Swap-test noise floor.** Repeats the whole-memory and 1-D swaps with partners that share A's exact φ, so their true effect is zero. About 15 s per run; writes `runs/<run>/noise_floor.json`:
   ```bash
   uv run python -m latent_physics.experiments.noise_floor           # every run under runs/
   ```
4. **Subspace swaps (Level 4).** Swaps a k-dim subspace per GRU layer (k = 1–32) found by four methods: random, encode-PCA, alignment search (DAS) and DAS toward a shuffled target as a control. Each is run once or clamped, and scored on steps k ≥ 1 after the swap. About 2.5 min per run; writes `runs/<run>/subspace_swap.json`, then `runs/subspace_summary.md`, `runs/subspace_swap.png` and `runs/subspace_vs_force_prob.png`:
   ```bash
   uv run python -m latent_physics.experiments.subspace_swap runs/spring_full_*   # or no args: every run
   uv run python -m latent_physics.experiments.subspace_swap --summary            # re-collate only
   ```
5. **Capacity / data sweeps (H7):** `--hidden 32|64|256`, `--n-train 1000|2000|16000`.

## 5. Code map

```
latent_physics/
  envs/          simulators: base.py (Param, Env), forced.py, spring.py; ENVS / make_env
  data/          datasets.py (seeded splits), normalize.py (Normalizer, oracle inputs)
  models/        world_models.py (GRUWorldModel with step/edit hooks, MLPWorldModel), training.py
  analysis/
    stats.py         r2, corr
    prediction.py    per-t error, open-loop rollouts                       (Level 1)
    probes.py        ridge / MLP probes, emergence, R² when it matters     (Level 2)
    sensitivity.py   how much each param matters for the next step
    directions.py    decode / encode directions, random controls
    partners.py      history-matched swap partners (standard, same-φ)
    swaps.py         swap rollouts along a subspace, one-shot or clamped    (Level 4)
    interchange.py   the baseline's 1-D swap test                          (Level 4)
    ablation.py      mean-ablation of directions                           (Level 4)
    subspaces.py     k-dim subspaces: random, encode-PCA, alignment search (Level 4)
  experiments/   runnable with `uv run python -m latent_physics.experiments.<name>`
    baseline/        pipeline.py (CLI and stages), verdict.py, figures.py, report.py
    summarize.py     mean ± std over seeds -> runs/summary.md
    noise_floor.py   swap-test noise floor -> runs/*/noise_floor.json
    subspace_swap.py k-dim subspace swaps -> runs/*/subspace_swap.json
    loading.py       reload a finished run (data, split, GRU, hidden states)
tests/test_envs.py   physics sanity checks
```

To add an environment, subclass `Env` in a new module under `latent_physics/envs/`: set `params`, `dt`, `substeps`, `f_max`, `sample_init` and a vectorized `step` (optionally `input_dims`, `v_soft_max`, `derived_targets`). Then register it in `ENVS` and add a test.
