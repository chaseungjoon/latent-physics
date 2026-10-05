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
uv run python -m latent_physics.run --env forced --preset quick --device cpu

# the baseline: ~10 min on a 24-thread CPU, a few minutes on the GPU
uv run python -m latent_physics.run --env forced --preset full
uv run python -m latent_physics.run --env spring --preset full
```

`--device auto` (the default) uses CUDA when it's available. Every run writes to `runs/<env>_<preset>_fp<force-prob>_s<seed>/`:

| file | contents |
|---|---|
| `report.md` | verdict table, all metrics, embedded figures (**start here**) |
| `metrics.json` | everything in machine-readable form |
| `loss_vs_t.png` | one-step error vs. history length for GRU / memoryless MLP / oracle |
| `probe_emergence.png` | linear-probe R² for each hidden param vs. timestep, per layer, with controls |
| `interventions.png` | model's response to a latent shift vs. simulator's response to the true param change |
| `gru.pt`, `mlp.pt`, `oracle.pt` | trained weights |
| `log.txt` | training log |

Useful flags: `--seed N`, `--force-prob P`, `--epochs`, `--hidden`, `--n-train`, `--n-test`, `--lr`, `--batch-size`, `--out DIR`, `--threads N` (CPU).

## 3. What gets run

**Environments** (`latent_physics/envs.py`). The model observes the state `(x, v)` and the action (force `F`). Each trajectory has its own hidden parameters φ, which the model never sees.

| env | hidden φ | how φ shows up in the data |
|---|---|---|
| `forced` | mass `m ∈ [1,5]`, Coulomb friction `μ ∈ [0.05,0.3]` | `m` only through the response to `F`; `μ` through coasting deceleration `μg` and stick/slip |
| `spring` | `m ∈ [0.5,2]`, stiffness `k ∈ [1,10]`, damping `c ∈ [0.1,1]` | without forcing, only `k/m` and `c/m` are identifiable |

**Splits.** `id` is the training distribution. `comp` is the held-out high-high corner of (param 0, param 1), used for compositional OOD. `extrap` puts one param beyond its training range (`forced`: mass ∈ [5,10]; `spring`: k ∈ [10,25]).

**Models** (`latent_physics/models.py`):
- `gru`: an encoder, 2 stacked GRUs and an MLP decoder. Its recurrent state is the only place where φ can accumulate.
- `mlp`: memoryless. It can't infer φ, so it gives the error floor for not knowing the physics.
- `oracle`: an MLP that is given the true φ. It gives the error ceiling for knowing the physics.

**Analyses** (`latent_physics/analysis.py`) map onto the proposal's hierarchy:

| level | test | what to look for |
|---|---|---|
| 1 prediction | late-timestep one-step MSE, 25-step rollouts, on id/comp/extrap | GRU ≪ MLP, and GRU → oracle |
| 2 decodability | ridge probe for each (layer, t), split by trajectory; controls: raw input, untrained model, shuffled labels | R² high, well above controls, rising with t |
| 4 causal use | shift the GRU state along the probe direction by ±0.5σ and correlate the predicted Δs with the simulator's Δs when φ truly moves; 1-D ablation | corr ≫ random-direction corr |
| 5/6 generalization | probe RMSE and prediction error on comp / extrap | how much worse than id |

The **Verdict** table at the top of `report.md` turns these into PASS/FAIL checks. The thresholds are rules of thumb (see `verdict()` in `run.py`), not significance tests.

## 4. First experiments once the GPU is free

1. **Baseline on both envs, 3 seeds.** Check that the verdict is stable across seeds:
   ```bash
   for env in forced spring; do for s in 0 1 2; do
     uv run python -m latent_physics.run --env $env --preset full --seed $s; done; done
   ```
2. **Identifiability (H6).** With no applied forces, mass has no effect on `forced` trajectories, so mass decodability should collapse while friction survives:
   ```bash
   for fp in 0.0 0.1 0.3 0.7 1.0; do
     uv run python -m latent_physics.run --env forced --preset full --force-prob $fp; done
   ```
   Do the same on `spring`: at `--force-prob 0`, `m` alone is unidentifiable, but `k/m` and `c/m` remain identifiable.
3. **Capacity / data sweeps (H7):** `--hidden 32|64|256`, `--n-train 1000|2000|16000`.

## 5. Code map

```
latent_physics/
  envs.py      simulators, parameter ranges, OOD splits, action sampling
  data.py      dataset generation, normalization, oracle inputs
  models.py    GRUWorldModel (step/forward_stepwise with edit hooks), MLPWorldModel
  train.py     teacher-forced one-step training, best-val checkpointing
  analysis.py  per-t error, rollouts, ridge probes, interventions, ablations
  report.py    figures + report.md
  run.py       CLI / pipeline / verdict
tests/test_envs.py   physics sanity checks
```

To add an environment, subclass `Env` in `envs.py`: set `params`, `dt`, `substeps`, `f_max`, `sample_init` and a vectorized `step`. Then register it in `ENVS` and add a test.
