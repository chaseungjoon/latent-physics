# latent-physics

Does a neural world model trained only to predict the next state learn the hidden physics behind it?

Each trajectory follows `(state, action, hidden φ) -> next state`, where φ is mass, friction, stiffness, damping or restitution. The model only sees `(state, action) -> next state` and never gets φ. After training, we probe, intervene on and ablate its memory, and test how it generalizes.

The systems are state-based (no video), so dynamics can be studied without perception: forced motion with friction, a mass-spring-damper, and later collisions.

Physical understanding is treated as a ladder, each rung harder than the last:

```text
prediction -> decodability -> structure -> causal use -> generalization -> identifiability
```

Questions: which parameters become decodable and where; whether they are stored separately or spread out; whether the model actually uses them; whether that helps out of distribution; and how all of this depends on how informative the data are.

## Documents

- [`PROPOSAL.md`](./PROPOSAL.md): the research proposal
- [`QUICKSTART.md`](./QUICKSTART.md): setup, commands, code map
- [`PROGRESS.md`](./PROGRESS.md): results so far and next steps
