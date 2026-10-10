"""Analyses of a trained world model, by level of the proposal's hierarchy.

stats         r2, corr
prediction    Level 1: per-timestep error, open-loop rollouts
probes        Level 2: ridge / MLP probes, emergence over t, R^2 when the param matters
sensitivity   how much each param physically matters for the next step
directions    1-D parameter directions in the memory (decode / encode) and random controls
partners      history-matched swap partners (standard and same-phi)
swaps         swap rollouts along a subspace (one-shot or clamped) and their statistics
interchange   Level 4: the 1-D swap test of the baseline
ablation      Level 4: mean-ablation of parameter directions
subspaces     Level 4: k-dim subspaces (random, encode-PCA, learned by alignment search)
"""
