# Quickstart

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.10.

```bash
uv sync          # .venv with torch, numpy, matplotlib, pytest
uv run pytest    # simulator physics checks and swap-code checks (~1 s)
```

The default torch wheel (CUDA 13) runs on the RTX 5070. On other GPUs, install the matching build, e.g. `uv pip install torch --index-url https://download.pytorch.org/whl/cu128`.

## Commands

Every experiment is `uv run python -m latent_physics.experiments.<name>`.

```bash
# one baseline run: simulate, train, evaluate, probe, intervene -> runs/<env>_<preset>_fp<p>_s<seed>/report.md
uv run python -m latent_physics.experiments.baseline --env spring --preset full --force-prob 0.7 --seed 0
#   --preset quick --device cpu   smoke test (~15 s; numbers are meaningless)
#   --reuse-models                reload the trained .pt files and redo only the analysis

# the 30-run sweep used in PROGRESS.md (~15 min on the GPU)
for env in forced spring; do for fp in 0.0 0.1 0.3 0.7 1.0; do for s in 0 1 2; do
  uv run python -m latent_physics.experiments.baseline --env $env --force-prob $fp --seed $s --reuse-models; done; done; done

uv run python -m latent_physics.experiments.summarize       # mean ± std over seeds -> runs/summary.md
uv run python -m latent_physics.experiments.noise_floor     # swap-test noise floor -> runs/*/noise_floor.json (~15 s/run)
uv run python -m latent_physics.experiments.subspace_swap   # subspace swaps -> runs/*/subspace_swap.json (~2.5 min/run)
uv run python -m latent_physics.experiments.subspace_swap --summary   # -> runs/subspace_summary.md, runs/subspace_*.png
```

Without run arguments, the follow-up experiments process every `runs/*/` that has a `gru.pt`. `runs/` is gitignored.

## What a baseline run does

**Environments.** The model sees the state (x, v) and a force F. Each trajectory has its own hidden φ. Forces are piecewise constant, and `--force-prob` sets how often a segment is pushed. This is the identifiability knob: at 0, forced-motion mass never affects anything.

| env | hidden φ | how φ shows |
| --- | --- | --- |
| `forced` | mass m, Coulomb friction μ | m only through the response to F; μ through coasting and stick/slip |
| `spring` | m, stiffness k, damping c | without forcing only k/m and c/m are identifiable |

Test splits: `id`, `comp` (held-out high-high corner of two params) and `extrap` (one param beyond its training range).

**Models.** `gru`: encoder, 2 stacked GRUs, MLP decoder; its memory is the only place φ can accumulate. `mlp`: no memory, so it shows the error of not knowing φ. `oracle`: an MLP given the true φ.

**Analyses**, by level of the proposal:

| level | test |
| --- | --- |
| 1 prediction | one-step error over history length, open-loop rollouts, on all splits |
| 2 decodability | ridge probes per layer and timestep; "R² when it matters" counts only steps where the param affects the next state; controls: raw input, untrained GRU, shuffled labels |
| 4 causal use | swap test: copy part of the memory from a trajectory B with matching recent history into A, and compare the change in A's predictions with the simulator's change when A takes B's φ; plus ablations |
| 5/6 generalization | error and probe RMSE on `comp` / `extrap` |

`report.md` opens with a PASS/FAIL table. Its thresholds are rules of thumb (`experiments/baseline/verdict.py`), not significance tests.

## Code map

```text
latent_physics/
  envs/          simulators (base.py, forced.py, spring.py)
  data/          seeded datasets, normalization
  models/        GRU / MLP world models, training loop
  analysis/      prediction, probes, sensitivity, directions, partners, swaps,
                 interchange, ablation, subspaces (random, encode-PCA, DAS)
  experiments/   baseline/ (pipeline, verdict, figures, report), summarize,
                 noise_floor, subspace_swap, loading (reload a finished run)
tests/           simulator physics, swap machinery
```

New environment: subclass `Env` under `envs/` (params, `dt`, `substeps`, `f_max`, `sample_init`, vectorized `step`), register it in `ENVS`, and add a test.
