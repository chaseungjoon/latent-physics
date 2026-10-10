"""Runnable experiments (each is `uv run python -m latent_physics.experiments.<name>`).

baseline       train + analyze one (env, force_prob, seed) run -> runs/<name>/report.md
summarize      mean ± std over seeds of all baseline runs -> runs/summary.md
noise_floor    swap-test noise floor from same-phi partners -> runs/<name>/noise_floor.json
subspace_swap  k-dim subspace swaps -> runs/<name>/subspace_swap.json, runs/subspace_summary.md
loading        (library) reload a finished run for a follow-up analysis
"""
