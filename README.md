# latent-physics

Investigating whether state-based neural world models spontaneously learn meaningful physical structure.

The project studies whether hidden physical properties such as mass, friction, restitution, stiffness, and damping emerge inside predictive dynamics models trained only to predict future states.

## Research Questions

- Which physical quantities become decodable from hidden representations?
- Where in the model do those representations emerge?
- Are physical factors represented independently or in distributed subspaces?
- Are decodable physical features actually used causally for prediction?
- Do physically meaningful representations improve compositional and out-of-distribution generalization?
- How does physical identifiability depend on the type and amount of observed interaction?

## Core Idea

A system evolves according to hidden physical parameters:

```text
(state, action, hidden physics) -> next state
```

The world model only observes:

```text
(state, action) -> next state
```

The hidden physical parameters are never provided during training. After training, the model's internal representations are analyzed through probing, intervention, ablation, and generalization experiments.

## Scope

Initial work focuses on controlled state-based dynamical systems rather than video, allowing physical reasoning to be studied independently from perception.

Primary systems include:

- forced motion with mass and friction
- mass-spring-damper dynamics
- collision dynamics with hidden mass and restitution

## Goal

The central question is:

> Does accurate prediction imply that a world model has learned physical structure?

This project treats physical understanding as a hierarchy:

```text
prediction
    -> decodability
    -> representation structure
    -> causal use
    -> generalization
    -> identifiability
```

See [`PROPOSAL.md`](./PROPOSAL.md) for the full research proposal.
