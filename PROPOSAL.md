# Physics Inside Predictive World Models

**Do state-based neural dynamics models spontaneously internalize physical structure?**

## Abstract

A world model can predict a dynamical system well without learning variables that correspond to its physics. When it forecasts accurately, has it learned the physics, or only the input-output mapping on its training distribution?

We study this in a controlled **state-based** setting. The model sees system states and actions but never the trajectory's physical parameters (mass, friction, restitution, stiffness, damping), and is trained only to predict future states. Its internal representations are then analyzed for those hidden parameters.

We separate increasingly strong notions of physical understanding: accurate prediction, decodability, structured representation, causal use in prediction, compositional and out-of-distribution generalization, and identifiability from informative data. The goal is to determine **what physical structure a world model contains, whether that structure is causally used, and when predictive learning is enough to make it emerge**.

## 1. Motivation

A world model learns \(\hat{s}_{t+1} = f_\theta(s_t, a_t)\). A model that minimizes prediction error may encode mass, friction or contact explicitly, spread them across many dimensions, or avoid stable representations altogether and interpolate between observed trajectories. Prediction error alone cannot tell these apart.

So the question is not only _what physical structure emerges_ in a model trained to predict, but _when that structure counts as evidence of physical understanding_ rather than a correlate of good prediction. We treat understanding as a hierarchy of properties, each tested separately.

## 2. Why state-based

Video models must solve appearance, object identity, localization, occlusion, lighting and camera geometry along with dynamics. If one fails to encode mass, it is unclear whether physics or perception failed. Here the model observes states directly:

\[
s_{t+1} = F(s_t, a_t; \phi), \qquad \phi = [\phi_1, \ldots, \phi_n],
\]

where φ holds trajectory-specific properties that the model never receives as input or target. If knowing φ helps prediction, does predictive learning make a representation of φ emerge without supervision? State-based experiments isolate this dynamics question before perception is added; they do not replace visual world models.

## 3. Levels of physical understanding

| Level | Property | What it shows |
| --- | --- | --- |
| 0 | Trajectory memorization | Works only on familiar trajectories; no claim about physics |
| 1 | Accurate prediction, \(\hat{s}_{t+1} \approx s_{t+1}\) | Predictive competence, not physical representation |
| 2 | Decodability, \(z_t \rightarrow \hat{\phi}\) | The information is present, not that the model relies on it |
| 3 | Organization | How φ is represented: separate directions, subspaces, or a distributed code |
| 4 | Causal use | Intervening on the representation changes predictions in the physically implied direction (e.g. more "mass" → less acceleration at fixed force) |
| 5 | Compositional generalization | Works on unseen combinations of familiar factors (heavy + high friction) |
| 6 | Extrapolation | Works outside the training range; failure exposes reliance on interpolation |
| 7 | Identifiability | φ emerges only when the data contain enough information to identify it |

## 4. Problem formulation

φ is fixed within a trajectory and varies across trajectories. The model \(\hat{s}_{t+1}=f_\theta(s_t,a_t,h_t)\) is trained without φ; its state \(h_t\) and layer activations \(z_t^{(l)}\) are analyzed afterwards. The learning objective (prediction) is kept separate from the analysis target (φ): any physical structure that appears is a consequence of prediction, not supervision.

## 5. Research questions

**RQ1. Which physical variables emerge?** Observable quantities (position, velocity, acceleration, contact) can be inferred from short local history. Latent properties (mass, friction, restitution, stiffness, damping) must be inferred from how the system responds over time. If velocity becomes decodable early and mass only after evidence accumulates, representation formation follows the information requirements of each quantity.

**RQ2. How are they represented?** Factorized (separate directions), subspace-based, distributed, or entangled with motion state. _Information present_ must be distinguished from _information in a simple, factorized coordinate system_. Video world models show physical variables can be accessible yet high-dimensional and distributed.

**RQ3. Are they causally used?** A probe reveals correlation; the model may compute through a correlated feature instead. Intervening along a mass direction (\(z' = z + \alpha v_m\)) should reduce predicted acceleration at fixed force (\(a = F/m\)) if the direction is used. Likewise for friction (more deceleration), restitution (more rebound), stiffness (more restoring force) and damping (less oscillation). Decodability and causal use are separate claims.

**RQ4. Does physical structure predict generalization?** Three test regimes: in-distribution; compositional (e.g. train on \((m_{\text{high}},\mu_{\text{low}})\) and \((m_{\text{low}},\mu_{\text{high}})\), test on \((m_{\text{high}},\mu_{\text{high}})\)); and extrapolation (\(m_{\text{train}}\in[1,5]\), \(m_{\text{test}}>5\)). The question is not whether interpretable models always generalize better, but whether causally meaningful, separable structure goes with robustness under distribution shift.

**RQ5. When is φ identifiable, and how much informative data is needed?** In free fall \(a = g\), so mass is invisible; under a known force \(a = F/m\), it is not. More data is different from more informative data. For each parameter: which trajectories carry information about it, when it is non-identifiable, how many informative transitions it takes to decode, whether causal use appears at the same point, and whether prediction saturates before φ becomes identifiable. A model's failure to represent a quantity may reflect the data, not the learner.

## 6. Hypotheses

- **H1.** Prediction produces recoverable φ when φ is necessary and identifiable from the trajectory.
- **H2.** Kinematic quantities emerge from short context; trajectory-level parameters need temporal integration or interaction.
- **H3.** φ is not necessarily stored as explicit scalars; high probe scores do not imply a compact "physics engine".
- **H4.** Only part of the decodable information is causally used.
- **H5.** Causal physical structure predicts compositional generalization better than one-step accuracy does.
- **H6.** Identifiability constrains emergence: indistinguishable values are not recovered; informative interactions make recovery possible.
- **H7.** Informative diversity matters more than sample count.

## 7. Domains

- **Forced translational motion** (mass, friction): separates kinematics, applied force, inertia and dissipation; the same object can be observed with informative or uninformative trajectories.
- **Mass-spring-damper**, \(m\ddot{x}+c\dot{x}+kx = F(t)\): several parameters affect trajectories in distinct but entangled ways.
- **Collisions** (masses, restitution): some parameters become identifiable only through interaction.

## 8. Methods

**Layerwise probing.** Linear probes \(\hat{\phi}=Wz^{(l)}+b\), scored by out-of-sample R², are the primary diagnostic, since an expressive probe could solve the inference itself. Questions: where does each variable first become decodable, where is it strongest, does it persist, and do observable and latent quantities emerge differently?

**Controls.** Shuffled labels; models with similar error but different seeds; probes on raw states; held-out parameter combinations; checks that φ is not trivially inferable from a single state.

**Subspace geometry.** Estimate each factor's dimensionality; whether mass and friction subspaces are independent; whether nearby φ values map to nearby representations; how geometry changes during training.

**Causal tests.** (a) Directional intervention along a direction or subspace, over several strengths α, checking monotonicity, smoothness, sign consistency and which layers matter. (b) Subspace ablation: removing a mass subspace should hurt predictions under applied force more than where mass is irrelevant.

**Identifiability as a manipulated variable.** Build non-informative, weakly informative and strongly informative trajectory sets for each φ and test whether decodability and causal strength rise with identifiability. The question shifts from how many trajectories are needed to what evidence they contain.

**Generalization.** Matched in-distribution error (compare models at similar competence), factor recombination, range extrapolation, and long-horizon rollouts (stable dynamics vs. accurate one-step corrections).

**Linking structure to generalization.** Across models, seeds, checkpoints and architectures, relate probe R², subspace dimensionality, intervention effect and sign consistency, ablation effect, and compositional, extrapolation and rollout error. The aim is not "the model has a mass probe" but whether having something that behaves like mass explains generalization. Two models with equal one-step accuracy but different causal physics and different OOD behavior would show that accuracy alone does not characterize what a world model learned.

## 9. Related work

1. **Joseph et al. (2026), _Interpreting Physics in Video World Models_.** Layerwise probing, geometry and ablations in video encoders; a middle-layer "Physics Emergence Zone"; some quantities held in distributed population codes. We remove perception and use exact ground truth, which makes identifiability and intervention experiments possible.
2. **Baumgartner et al. (2026), _Disentangling Dynamical Systems_.** Designs models so system parameters are identifiable and disentangled. We ask instead whether a generic predictive model finds such structure on its own.
3. **Yao, Muller & Locatello (2024), _Marrying Causal Representation Learning with Dynamical Systems for Science_.** The \(s_{t+1}=F(s_t,a_t;\phi)\) formulation and structural identifiability. For us, identifying φ is an analysis target, not the learning objective.
4. **Alam (2026), _Causal Physics Steering in Video World Models via Concept Activation Vectors_.** Shows physics directions can be steered as well as read. In the state-based setting, an intervention can be compared with a known physical counterfactual.
5. **Xu et al. (2026), _Dynamics Are Learned, Not Told_.** A smooth latent geometry of dynamics supports adaptation without parameter labels. Success should therefore not be defined only as recovering human-named parameters; functionally useful latent physics counts too.

**Gap.** Physical variables can be decoded from video models and steered; causal representation learning can recover parameters by design; latent dynamics can be useful without supervision. Less is known about the controlled question: when does a model trained only to predict develop representations of hidden physics, does it use them, and do they explain generalization? This project places prediction, probing, intervention, generalization and identifiability in one framework.

## 10. Expected contributions

1. A graded operational definition of physical understanding: prediction → decodability → structure → causal use → generalization.
2. Controlled evidence on whether hidden parameters emerge without labels or inputs.
3. A causal test of latent physical concepts by intervention and ablation.
4. A test of whether physical representation quality explains OOD robustness beyond prediction loss.
5. An identifiability-centered view of data efficiency.

## 11. Possible outcomes

- **A. Factorized physics emerges:** simple decoding, correct counterfactuals, OOD support.
- **B. Distributed but causal:** no clean coordinates, but subspace interventions work and geometry predicts generalization.
- **C. Decodable but not used:** probes succeed, interventions and ablations have no specific effect.
- **D. Prediction without physics:** accurate on familiar data, fails probing, intervention and OOD.
- **E. Emergence only under identifiable interaction:** φ appears once the data contain discriminative interventions.

Each outcome is informative, including the negative ones.

## 12. Limitations and scope

- State access is privileged: results concern learned dynamics, not visual reasoning.
- Human parameters may not be the model's coordinates; failing to decode mass linearly is not absence of physics.
- Large interventions can leave the representation manifold; effects must be interpreted locally.
- Identifiability depends on the observation and intervention regime; claims are conditional on it.
- Extrapolation is not understanding: a physically structured model may still fail far outside its training range, and success does not prove it recovered a physical law.

In scope: state-based predictive models, hidden trajectory-level parameters, probing, geometry, interventions and ablations, compositional and OOD generalization, identifiability. Out of scope for now: raw video, large generative video models, symbolic equation discovery, language-model physics, full robotic control, learned active experiment selection, and physics-engine architectures as the main subject.

## 13. Thesis

> Accurate physical prediction is not sufficient evidence that a world model has learned physical structure. A stronger notion requires that hidden physical properties become accessible when they are identifiable, participate causally in prediction, and support generalization to new physical conditions.

## References

1. **Joseph, S., Garrido, Q., Balestriero, R., Kowal, M., Fel, T., Bakhtiari, S., Richards, B. A., & Rabbat, M.** (2026). _Interpreting Physics in Video World Models_. ICML, PMLR 306, 54687–54724. <https://proceedings.mlr.press/v306/joseph26a.html>
2. **Baumgartner, M. W., Lei, A., Watson, J., & Posner, I.** (2026). _Disentangling Dynamical Systems: Causal Representation Learning Meets Local Sparse Attention_. CLeaR, PMLR 323, 119–165. <https://proceedings.mlr.press/v323/baumgartner26a.html>
3. **Yao, D., Muller, C. J., & Locatello, F.** (2024). _Marrying Causal Representation Learning with Dynamical Systems for Science_. NeurIPS 37. DOI: 10.52202/079017-2290. <https://proceedings.neurips.cc/paper_files/paper/2024/hash/83eb339ed42297658fa24b5cec939285-Abstract-Conference.html>
4. **Alam, N.** (2026). _Causal Physics Steering in Video World Models via Concept Activation Vectors_. CVPR Workshops, 5890–5896. <https://openaccess.thecvf.com/content/CVPR2026W/VideoWorldModel/html/Alam_Causal_Physics_Steering_in_Video_World_Models_via_Concept_Activation_CVPRW_2026_paper.html>
5. **Xu, Z., Zhou, W., Pan, X., Deng, N., Liu, C., Chen, Q., & Yao, C.** (2026). _Dynamics Are Learned, Not Told: Semi-Supervised Discovery of Latent Dynamics Geometries for Zero-Shot Policy Adaptation_. ICML, PMLR 306, 142962–142988. <https://proceedings.mlr.press/v306/xu26cq.html>
