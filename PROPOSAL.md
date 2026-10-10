# Physics Inside Predictive World Models
## A Research Proposal on the Emergence, Causal Use, Identifiability, and Generalization of Physical Structure in State-Based Neural Dynamics Models

### Working Title

**Physics Inside Predictive World Models: Do State-Based Neural Dynamics Models Spontaneously Internalize Physical Structure?**

---

## Abstract

Neural world models can achieve low prediction error on dynamical systems without necessarily learning representations that correspond to physically meaningful variables or laws. This creates a fundamental ambiguity: when a predictive model accurately forecasts future states, has it learned the underlying physics of the system, or has it merely approximated the input-output mapping over the distribution on which it was trained?

This proposal studies that question in a deliberately controlled **state-based** setting. Rather than beginning with video, where physical reasoning is confounded with perception, appearance, and representation learning from pixels, the proposed study assumes direct access to observable system states while withholding trajectory-specific physical parameters such as mass, friction, restitution, stiffness, or damping from the predictive model. A neural world model is trained only to predict future states from present states and actions. Its internal representations are then analyzed to determine whether hidden physical properties emerge spontaneously as a consequence of predictive learning.

The central objective is not simply to ask whether physical parameters can be decoded from hidden activations. The study distinguishes several increasingly strong notions of physical understanding: accurate prediction, linear decodability of hidden physical variables, structured or distributed representation of those variables, causal use of the representations in prediction, compositional and out-of-distribution generalization, and identifiability from informative trajectories. This hierarchy allows the project to separate superficial correlations from representations that are genuinely implicated in the model's learned dynamics.

The proposed research focuses on five questions. First, which physical quantities become recoverable from hidden representations, and at what stages of the model? Second, are physical factors represented independently, jointly, or in distributed subspaces? Third, does intervening on a representation associated with a physical quantity cause predicted dynamics to change in the physically expected direction? Fourth, does the emergence of physically meaningful structure predict improved generalization to unseen combinations or ranges of physical parameters? Fifth, under what observational and interaction conditions is a hidden physical quantity identifiable at all, and how much informative experience is required before it emerges in the representation?

The work is positioned between recent studies of physics representations in video world models, causal representation learning for dynamical systems, and latent dynamics adaptation. Its intended contribution is a controlled framework for determining not merely whether a world model predicts physics, but **what physical structure it contains, whether that structure is causally operative, and when predictive learning is sufficient to make it emerge**.

---

# 1. Motivation

World models are increasingly used as internal predictive models for control, reinforcement learning, robotics, and embodied intelligence. At a high level, a world model attempts to represent how a system evolves:

\[
\hat{s}_{t+1} = f_\theta(s_t, a_t),
\]

or, for longer horizons,

\[
\hat{s}_{t+k} = f_\theta^{(k)}(s_t, a_t,\ldots,a_{t+k-1}),
\]

where \(s_t\) denotes the current state, \(a_t\) an action or external intervention, and \(\hat{s}_{t+k}\) a predicted future state.

A model that minimizes prediction error may become highly accurate without constructing an internal representation that resembles human physical concepts. It may encode mass, friction, velocity, momentum, or contact state explicitly; it may represent them only as distributed directions in a high-dimensional space; or it may avoid representing them in any stable form and instead learn a complicated interpolation rule over observed trajectories.

These possibilities are behaviorally difficult to distinguish if evaluation is based only on prediction error.

This motivates a more fundamental question:

> **When a neural world model is trained only to predict the evolution of a physical system, what physical structure emerges inside the model?**

A related question is even more important:

> **When should such structure be regarded as evidence of physical understanding rather than a correlate of successful prediction?**

This proposal treats "understanding physics" not as a binary claim but as a hierarchy of increasingly demanding properties. A model may predict correctly without exposing interpretable physical variables. It may contain information about a variable without using that information causally. It may use a physical representation on familiar trajectories but fail under new combinations of parameters. A physically meaningful representation should therefore be evaluated through multiple complementary tests rather than by prediction accuracy alone.

---

# 2. Why a State-Based Study?

Most recent discussion of physical reasoning in large world models concerns video. Video is attractive because it resembles the observational input available to biological or embodied agents, but it introduces a major confound.

A video model must simultaneously solve problems involving:

- appearance,
- object identity,
- spatial localization,
- motion extraction,
- occlusion,
- lighting,
- texture,
- camera geometry,
- and physical dynamics.

If a video world model fails to encode mass or friction, it can be difficult to determine whether it failed to learn the relevant physics or whether the physical information was obscured by the perceptual problem.

A state-based setting removes that ambiguity.

The proposed model receives only observable dynamical state variables and actions, while important physical parameters remain hidden. For a generic dynamical system,

\[
s_{t+1} = F(s_t, a_t; \phi),
\]

the vector

\[
\phi = [\phi_1,\phi_2,\ldots,\phi_n]
\]

contains trajectory- or system-specific physical properties such as mass, friction, restitution, stiffness, or damping. The predictive model observes \(s_t\), \(a_t\), and subsequent states, but it never receives \(\phi\) as an input or training target.

This produces a clean experimental question:

> **If knowledge of \(\phi\) is useful for predicting the trajectory, will predictive learning cause a representation of \(\phi\) to emerge without explicit supervision?**

State-based experiments are therefore not intended as a replacement for visual world models. They are intended as a controlled scientific instrument for isolating the dynamics question before perception is introduced.

---

# 3. Conceptual Definition of "Physical Understanding"

The proposal avoids treating physical understanding as a single measurable property. Instead, it defines a hierarchy.

## Level 0 — Trajectory Memorization

The model performs well only on familiar trajectory distributions or near-duplicate states. No claim about physical structure is justified.

## Level 1 — Accurate Prediction

The model predicts one-step or multi-step state transitions accurately:

\[
\hat{s}_{t+1} \approx s_{t+1}.
\]

This demonstrates predictive competence but not necessarily physical representation.

## Level 2 — Physical Information Is Decodable

A simple diagnostic model can recover a hidden physical quantity from an internal representation:

\[
z_t \rightarrow \hat{\phi}.
\]

For example, mass or friction may be linearly predictable from \(z_t\).

This establishes that the information is present in the representation, but not that the world model actually relies on it.

## Level 3 — Physical Structure Is Organized

Physical information may exhibit meaningful structure in representation space. Different factors may correspond to distinct directions or subspaces, or they may be encoded in a distributed geometry.

This level concerns **how** physical information is represented, not merely whether it is present.

## Level 4 — The Representation Is Causally Used

An intervention on the representation associated with a physical factor changes future predictions in the direction implied by physics.

For example, increasing a latent "mass" direction while holding the observable state and applied force fixed should decrease the predicted acceleration if that direction is causally involved in representing mass.

This level distinguishes information that is merely readable from information that actually participates in the computation.

## Level 5 — The Representation Supports Compositional Generalization

The model generalizes to combinations of physical factors that were individually observed during training but never observed together.

For example, if the model has seen:

- heavy objects with low friction,
- light objects with high friction,

but has never seen:

- heavy objects with high friction,

successful prediction in the held-out combination suggests that the model has learned factors that can be recombined rather than memorized entire trajectory regimes.

## Level 6 — The Model Generalizes Outside the Training Distribution

A stronger criterion is extrapolation to physical parameters outside the training range.

This cannot by itself prove that a model has learned a human-interpretable law, but failure under extrapolation can expose representations that depend heavily on interpolation within the training distribution.

## Level 7 — Physical Properties Are Efficiently Identified From Informative Experience

A hidden physical property should emerge only when the observed trajectory contains enough information to identify it.

A strong model should therefore exhibit a relationship between physical identifiability and representation emergence, rather than appearing to recover hidden parameters from fundamentally uninformative observations.

This final criterion connects representation learning to the structure of the physical evidence itself.

---

# 4. Problem Formulation

Consider a family of dynamical systems indexed by hidden physical parameters \(\phi\):

\[
s_{t+1}=F(s_t,a_t;\phi).
\]

The physical parameter vector is fixed over a trajectory or episode but varies across trajectories.

A predictive neural model is trained without access to \(\phi\):

\[
\hat{s}_{t+1}=f_\theta(s_t,a_t,h_t),
\]

where \(h_t\) or intermediate activations \(z_t^{(l)}\) represent the model's internal state or layerwise hidden representations.

The model's primary training objective is predictive. The hidden parameters are used only after training for scientific analysis.

The core problem is therefore:

> Given a model optimized only for predictive dynamics, characterize the emergence, geometry, causal role, generalization value, and identifiability of hidden physical factors in its internal representations.

This formulation intentionally separates **learning objective** from **analysis target**. The model is not rewarded for producing an interpretable mass or friction variable. If such structure appears, it is a consequence of predictive learning rather than direct supervision.

---

# 5. Research Questions

## RQ1. Which physical variables spontaneously emerge in predictive representations?

The first question asks whether trajectory-specific physical quantities can be decoded from internal activations after training.

Candidate quantities fall into two broad classes.

### Directly observable or near-observable quantities

Examples include:

- position,
- velocity,
- acceleration,
- contact state.

These can often be inferred from a short local history.

### Latent system properties

Examples include:

- mass,
- friction coefficient,
- restitution,
- stiffness,
- damping.

These are not generally observable from a single instantaneous state. They must be inferred from how the system responds over time or under interventions.

If velocity becomes decodable very early while mass emerges only after temporal evidence accumulates, this would indicate that representation formation reflects the information requirements of the underlying physical quantities.

### Main question

> Does prediction pressure alone make hidden physical parameters linearly or otherwise simply accessible in internal representations?

---

## RQ2. How are physical variables represented?

Linear decodability does not imply that a model contains one clean neuron or coordinate corresponding to a physical parameter.

Representations may be:

1. **factorized**, where different physical factors correspond approximately to separate directions;
2. **subspace-based**, where a small multidimensional subspace contains most information about a factor;
3. **distributed**, where a quantity is encoded across many dimensions;
4. **entangled**, where mass, friction, motion state, and other properties cannot be cleanly separated.

This question asks whether the latent geometry resembles an explicit parameterization of physics or whether physically predictive information is encoded in a more distributed form.

The study should therefore distinguish:

\[
\text{information present}
\]

from

\[
\text{information represented in a simple, factorized coordinate system}.
\]

This distinction is directly motivated by recent work on video world models, which found that physical variables can be accessible while remaining high-dimensional and distributed rather than resembling the compact state variables of a classical simulator.

---

## RQ3. Is a decodable physical representation causally used by the model?

This is the most important step beyond probing.

A probe can reveal correlations. If mass can be predicted from a hidden representation, the model may nevertheless perform its actual future-state computation through another correlated feature.

The proposed study therefore treats causal intervention as a necessary stronger test.

Suppose a direction \(v_m\) in representation space is associated with mass. Construct an intervention

\[
z' = z + \alpha v_m.
\]

If this representation is genuinely used as a mass-like variable, increasing \(\alpha\) should produce systematic changes in predicted dynamics consistent with increased effective mass.

For a fixed applied force,

\[
a = \frac{F}{m},
\]

so increasing the mass representation should reduce predicted acceleration.

Equivalent tests can be constructed for other parameters:

- increasing friction should increase predicted deceleration during sliding;
- increasing restitution should increase post-impact rebound;
- increasing stiffness should increase restoring force for a fixed displacement;
- increasing damping should suppress oscillation.

This leads to a central distinction:

> **Decodability asks whether physical information exists in the representation. Intervention asks whether the model uses that representation as part of its dynamics computation.**

The proposal treats these as separate empirical claims.

---

## RQ4. Does physically meaningful representation predict better generalization?

If a model truly decomposes dynamics into reusable physical factors, this should be useful when the test distribution recombines those factors in unfamiliar ways.

Three forms of generalization should be distinguished.

### In-distribution prediction

Physical parameter combinations resemble those seen during training.

### Compositional out-of-distribution generalization

Individual factors are familiar but their combination is held out.

For instance:

\[
(m_{\text{high}},\mu_{\text{low}})
\]

and

\[
(m_{\text{low}},\mu_{\text{high}})
\]

may appear in training, while

\[
(m_{\text{high}},\mu_{\text{high}})
\]

is reserved for evaluation.

This tests whether the model can recombine separately learned factors.

### Parameter extrapolation

Physical parameters fall outside the training range.

For example,

\[
m_{\text{train}}\in[1,5], \qquad m_{\text{test}}>5.
\]

Extrapolation is a demanding test and should not be assumed to follow automatically from an interpretable representation. However, comparing representational structure with extrapolation performance can reveal whether some forms of internal physical organization are associated with more robust dynamics modeling.

The key hypothesis is not simply that interpretable models will always generalize better. The more precise question is:

> **Are models whose latent spaces contain causally meaningful and more separable physical structure systematically more robust under distribution shift?**

---

## RQ5. When is a hidden physical parameter identifiable, and how much informative experience is required?

A model cannot infer information that is absent from the observations.

This is especially important for physical parameters.

Consider mass under ideal free fall:

\[
a=g.
\]

The trajectory does not reveal mass. A one-kilogram object and a one-hundred-kilogram object have the same ideal gravitational acceleration.

By contrast, under a known external force,

\[
a=\frac{F}{m},
\]

mass affects the observed acceleration and may become identifiable.

This motivates a critical distinction between:

- **more data**, and
- **more informative data**.

The project should therefore investigate whether representation emergence reflects structural identifiability.

For each hidden parameter, the study asks:

1. What types of trajectories contain information about the parameter?
2. Under what conditions is the parameter theoretically or empirically non-identifiable?
3. How many informative transitions or trajectories are required before the parameter becomes reliably decodable?
4. Does causal usefulness emerge at the same point as decodability?
5. Does predictive accuracy saturate before physical structure becomes identifiable?

This research question is central because it prevents misleading conclusions. A model's failure to represent a quantity may be evidence of inadequate learning, but it may equally be a consequence of the experimental data not uniquely determining that quantity.

---

# 6. Hypotheses

## H1 — Predictive training will produce recoverable representations of hidden physical parameters when those parameters are necessary and identifiable from the trajectory.

Parameters such as mass or friction should become decodable when variations in those quantities systematically affect future transitions and the observed trajectories contain sufficient information to distinguish them.

## H2 — Observable kinematic quantities and latent physical parameters will emerge differently.

Quantities such as velocity or acceleration should become accessible from shorter context and potentially earlier representations, while trajectory-level physical parameters should require temporal integration or interaction evidence.

## H3 — Physical information will not necessarily be encoded as explicit scalar variables.

The model may use distributed subspaces rather than a classical factorized representation. High probe performance alone should therefore not be interpreted as evidence of a compact "physics engine" inside the model.

## H4 — Only a subset of decodable physical information will be causally operative.

Some properties may be readable from hidden representations because they correlate with trajectory regimes, while intervention reveals that the predictive computation does not directly depend on the corresponding direction.

## H5 — Causally meaningful physical representations will be more strongly associated with compositional generalization than raw one-step prediction accuracy.

Models with similar in-distribution prediction errors may differ substantially in latent physical structure and OOD performance.

## H6 — Identifiability will constrain representation emergence.

When two values of a hidden parameter produce observationally indistinguishable trajectories under a given observation/intervention regime, the model should not reliably recover that parameter. Introducing informative interactions should make recovery possible.

## H7 — Informative diversity will matter more than raw sample count.

A smaller set of trajectories that distinguishes alternative physical hypotheses may produce stronger parameter representations than a much larger set of redundant trajectories.

---

# 7. Proposed Study Domains

The proposal should use multiple simple dynamical families rather than relying on a single system. The goal is not realism but scientific control.

The domains should collectively expose different forms of hidden physical structure.

## 7.1 Forced Translational Motion

Hidden factors can include mass and friction.

This domain provides a direct setting for separating:

- kinematic state,
- applied force,
- inertia,
- dissipative effects.

It is particularly useful for identifiability studies because the same object can be observed under trajectories that are informative or uninformative about its mass.

## 7.2 Mass-Spring-Damper Dynamics

A standard form is

\[
m\ddot{x}+c\dot{x}+kx = F(t),
\]

where \(m\) is mass, \(c\) is damping, and \(k\) is stiffness.

This system is useful because multiple hidden parameters jointly influence trajectories in distinct but potentially entangled ways. It enables analysis of whether a predictive model separates physical factors or encodes a trajectory-specific aggregate representation.

## 7.3 Collision Dynamics

Collision systems introduce hidden quantities such as object masses and restitution.

They are useful because some physical parameters become identifiable primarily through interaction. The same object may reveal little about its properties during free motion but become highly informative after collision.

Together, these domains provide complementary tests of observability, interaction, temporal context, and hidden parameter recovery.

---

# 8. Representation Analysis

## 8.1 Layerwise Probing

For each hidden representation \(z^{(l)}\), a lightweight probe predicts physical quantities:

\[
\hat{\phi}=Wz^{(l)}+b.
\]

Linear probes are preferred as the primary diagnostic because a highly expressive nonlinear probe could itself learn substantial parts of the physical inference problem.

The principal metric for continuous quantities is out-of-sample \(R^2\), supplemented by error measures appropriate to each variable.

Layerwise analysis asks:

- when does a variable first become decodable?
- where is it represented most strongly?
- does information persist or disappear in later layers?
- do directly observable quantities and inferred physical parameters exhibit different emergence profiles?

This is conceptually analogous to identifying a "physics emergence" region, but in a state-based predictive model where physical ground truth is exactly known.

## 8.2 Probe Selectivity and Controls

Probe results can be misleading if the probe succeeds through trivial correlations.

Controls should therefore include:

- shuffled physical labels,
- matched models with similar prediction error but different random initialization,
- probes applied to raw observable state,
- probes evaluated across held-out parameter combinations,
- simple baselines that test whether the target parameter is already trivially inferable from a single state.

The purpose is to ensure that successful decoding reflects information integrated by the world model rather than accidental dataset structure.

## 8.3 Subspace Geometry

Physical variables should not be assumed to correspond to single neurons.

For each factor, analysis should estimate the dimensionality and geometry of the informative subspace.

Questions include:

- Is mass represented primarily along one direction or many?
- Are mass and friction subspaces approximately independent?
- Does the geometry change as the model becomes more predictive?
- Do nearby values of a physical parameter occupy nearby regions of representation space?
- Does representation distance reflect physical similarity?

This analysis distinguishes factorized representations from distributed population codes.

---

# 9. Causal Representation Tests

Probing alone is insufficient.

The proposal therefore includes two complementary causal tests.

## 9.1 Directional Intervention

A physical direction or subspace identified through probes is modified while holding other inputs fixed:

\[
z' = z + \alpha v_\phi.
\]

The resulting future prediction is compared with the physically expected effect of changing \(\phi\).

The intervention should be evaluated across multiple strengths \(\alpha\), allowing analysis of whether the response is:

- monotonic,
- approximately local and smooth,
- physically sign-consistent,
- localized to specific layers or representations.

## 9.2 Subspace Ablation

If a subspace contains information about a physical parameter, that information can be removed or projected out.

If prediction specifically degrades on transitions whose outcome depends on that physical factor, this provides stronger evidence that the representation is functionally important.

For example, removing a mass-related subspace should affect predictions under applied force more than predictions in conditions where mass is observationally irrelevant.

Together, intervention and ablation distinguish **physical information that is present** from **physical information that participates in computation**.

---

# 10. Identifiability as an Experimental Variable

Identifiability should not be treated merely as a theoretical assumption. It should become a manipulated variable in the experiments.

For a hidden parameter \(\phi\), construct sets of trajectories that differ in how strongly they constrain \(\phi\).

### Non-informative condition

Observed trajectories are compatible with many possible values of \(\phi\).

### Weakly informative condition

The parameter affects the trajectory, but different values remain difficult to distinguish.

### Strongly informative condition

Interactions or forces produce clearly distinguishable outcomes for different parameter values.

The key prediction is:

\[
\text{identifiability} \uparrow
\quad\Rightarrow\quad
\text{physical decodability} \uparrow
\]

and, potentially,

\[
\text{identifiability} \uparrow
\quad\Rightarrow\quad
\text{causal representation strength} \uparrow.
\]

This provides a principled way to study sample efficiency. Rather than asking only how many trajectories are required, the study asks **what evidence those trajectories contain**.

---

# 11. Generalization Analysis

The proposal distinguishes prediction quality from physical generalization.

## 11.1 Matched In-Distribution Evaluation

Establish standard predictive performance.

This acts as a control: models with different representational properties should be compared at approximately matched predictive competence when possible.

## 11.2 Factor-Recombination Evaluation

Hold out combinations of latent physical parameters during training.

This directly tests whether representations are compositional.

## 11.3 Parameter-Range Extrapolation

Evaluate beyond the training range.

The aim is not to demand perfect extrapolation, but to measure how representational structure relates to degradation outside the training distribution.

## 11.4 Long-Horizon Rollout

Small local errors can accumulate dramatically.

Multi-step rollout evaluation can reveal whether a model has learned stable dynamics or merely accurate one-step corrections.

---

# 12. Connecting Representation to Generalization

A central analysis should explicitly relate internal physics to external performance.

Across models, training checkpoints, architectures, or random seeds, collect measures such as:

- in-distribution prediction error,
- physical probe \(R^2\),
- physical subspace dimensionality,
- intervention effect size,
- intervention sign consistency,
- subspace-ablation effect,
- compositional OOD error,
- extrapolation error,
- long-horizon rollout error.

Then test whether models with stronger internal physical structure are systematically better at generalization.

This is critical because the primary scientific claim should not be:

> "The model has a mass probe."

It should instead ask:

> **Does possessing a representation that behaves like mass explain anything about the model's ability to reason about unfamiliar physical situations?**

A particularly informative outcome would be the existence of two models with nearly identical one-step prediction accuracy but substantially different causal physics scores and substantially different OOD behavior.

Such a result would demonstrate that ordinary predictive accuracy is insufficient to characterize what a world model has learned.

---

# 13. Relevant Prior Work

The most directly relevant works are summarized below in approximate order of conceptual proximity.

## 13.1 Joseph et al. (2026), *Interpreting Physics in Video World Models*

Joseph et al. provide a direct precedent for asking where and how physical information is represented inside a predictive representation model. They analyze large video encoders using layerwise probing, subspace geometry, patch-level decoding, and targeted ablations. They identify a middle-layer **Physics Emergence Zone** where physical variables become linearly accessible and show that some quantities are represented in high-dimensional distributed population codes rather than as compact scalar variables.

This work motivates several elements of the present proposal:

- layerwise physical probing,
- explicit analysis of representation geometry,
- skepticism toward the assumption that models reproduce classical factorized state variables,
- and the distinction between representation accessibility and representation organization.

The proposed project differs in that it removes the visual perception problem and studies **state-based predictive dynamics with exact hidden physical ground truth**. This enables controlled variation of mass, friction, restitution, and related factors, as well as direct identifiability and intervention experiments that are difficult to isolate in video.

## 13.2 Baumgartner et al. (2026), *Disentangling Dynamical Systems: Causal Representation Learning Meets Local Sparse Attention*

Baumgartner et al. study whether system parameters can be disentangled from raw trajectories and derive identifiability conditions using causal representation learning. Their work is particularly relevant to the present proposal's treatment of physical parameter recovery as an identifiability problem rather than merely a regression task.

Their goal, however, is explicitly to construct identifiable and disentangled representations of system parameters. The present proposal instead asks what happens when the world model is **not instructed to learn such a representation**. The focus is on whether predictive learning spontaneously gives rise to physically meaningful structure, and whether that structure is causally used.

Thus, the two projects address complementary questions:

- **Baumgartner et al.:** How can a model be designed so that hidden dynamical parameters are identifiable and disentangled?
- **This proposal:** To what extent does a generic predictive dynamics model discover such structure on its own, and under what data conditions?

## 13.3 Yao, Muller, and Locatello (2024), *Marrying Causal Representation Learning with Dynamical Systems for Science*

Yao et al. connect causal representation learning with dynamical systems parameter estimation. They formulate trajectory-specific, time-invariant dynamical parameters and show how identifiable causal representation methods can isolate such parameters for downstream use.

This work strongly motivates the present proposal's formulation

\[
s_{t+1}=F(s_t,a_t;\phi),
\]

where \(\phi\) denotes a hidden trajectory-specific physical parameter vector.

It also motivates the emphasis on **structural identifiability**: a hidden parameter cannot be expected to emerge if distinct parameter values generate indistinguishable observations.

The proposed work departs from Yao et al. by making parameter identification an **analysis target rather than the learning objective**.

## 13.4 Alam (2026), *Causal Physics Steering in Video World Models via Concept Activation Vectors*

Alam extends probing-based physics interpretability by intervening directly on representation directions associated with physical concepts. The study shows that such a representation can be steered as well as read, which is evidence that intervention is a stronger test than decodability alone.

This directly motivates the proposed causal intervention framework.

The state-based setting allows the intervention criterion to be made substantially more precise. Instead of shifting a generic judgment of physical plausibility, an intervention can be compared against a known physical counterfactual, such as the expected change in acceleration caused by increasing mass or the expected change in rebound caused by increasing restitution.

## 13.5 Xu et al. (2026), *Dynamics Are Learned, Not Told: Semi-Supervised Discovery of Latent Dynamics Geometries for Zero-Shot Policy Adaptation*

Xu et al. challenge the assumption that robust dynamics representations must explicitly recover predefined physical parameters. They show that a smooth latent geometry of trajectory dynamics can support adaptation even without privileged parameter labels.

This is an important conceptual counterpoint for the present proposal.

A model may contain a useful physical representation even if mass and friction do not appear as clean independent coordinates. Consequently, the proposed research should not define success solely as recovering human-named parameters. Distributed or nonlinearly organized representations may still capture the relevant physical structure.

This motivates the distinction between:

- **human-interpretable factorization**, and
- **functionally useful latent physics**.

The intervention and generalization tests are therefore essential complements to probing.

---

# 14. Research Gap

The reviewed literature leaves a specific gap.

Recent work has shown that:

1. physical variables can be decoded from video model representations;
2. physics-related representation directions can be intervened upon;
3. causal representation learning can deliberately recover identifiable dynamical parameters;
4. latent dynamics representations can be useful without explicit physical parameter supervision.

What remains less directly characterized is the following controlled question:

> **If a state-based world model is trained only for predictive accuracy, with no physical-parameter supervision or explicit disentanglement objective, under what conditions does it spontaneously develop representations of hidden physical structure, are those representations causally used in its predictions, and do they explain its ability to generalize beyond the training distribution?**

The proposed project addresses this gap by placing **prediction, probing, intervention, generalization, and identifiability within a single controlled framework**.

The intended novelty is therefore not the invention of another dynamics architecture. It is the empirical decomposition of what it means for a learned dynamics model to "know physics."

---

# 15. Expected Contributions

The project is expected to make five main contributions.

## Contribution 1 — A graded operational definition of physical understanding in world models

Rather than equating prediction accuracy with understanding, the study distinguishes:

\[
\text{prediction}
\rightarrow
\text{decodability}
\rightarrow
\text{representational structure}
\rightarrow
\text{causal use}
\rightarrow
\text{generalization}.
\]

## Contribution 2 — Controlled evidence about spontaneous physical representation emergence

The project tests whether hidden physical parameters emerge without being supplied as labels or model inputs.

## Contribution 3 — A causal test of latent physical concepts

Representation interventions and subspace ablations assess whether decodable physical information actually participates in future-state computation.

## Contribution 4 — A connection between latent physical structure and OOD generalization

The project evaluates whether physical representation quality explains robustness to unseen factor combinations or parameter ranges beyond what ordinary prediction loss explains.

## Contribution 5 — An identifiability-centered view of data efficiency

The study distinguishes the amount of experience from the informativeness of experience and tests whether representation emergence tracks the theoretical or empirical identifiability of hidden physical quantities.

---

# 16. Possible Outcomes and Their Interpretation

The proposal is valuable even if the strongest hypotheses fail.

## Outcome A — Strong factorized physics emerges

Mass, friction, or related variables become simply decodable, interventions produce correct counterfactual dynamics, and the representation supports OOD generalization.

This would provide strong evidence that predictive learning alone can induce internally meaningful physical variables under controlled conditions.

## Outcome B — Physics is distributed but causally meaningful

Physical variables are difficult to isolate as individual coordinates, but subspace interventions produce coherent physical effects and the geometry predicts generalization.

This would support a view similar to recent video-model findings: useful physical reasoning may rely on distributed representations rather than a classical simulator-like state vector.

## Outcome C — Physics is decodable but not causally used

Probe performance is high, but interventions and ablations have little specific effect.

This would be an important negative result demonstrating that physical information can be present as a correlate without functioning as a causal computational variable.

## Outcome D — Prediction is strong while physics representation is weak

The model predicts familiar trajectories accurately but fails probing, intervention, and OOD tests.

This would support the hypothesis that prediction objectives can be satisfied through trajectory interpolation without learning reusable physical abstractions.

## Outcome E — Physical variables emerge only under identifiable interactions

The same model fails to represent mass or friction under uninformative trajectories but develops strong representations once the data contain discriminative interventions.

This would directly connect representation learning with structural identifiability and provide a principled explanation for sample efficiency.

---

# 17. Limitations

Several limitations should be acknowledged from the outset.

### State access is privileged

The study deliberately bypasses perception. Results will therefore characterize learned dynamics rather than end-to-end visual physical reasoning.

### Human physical parameters may not be the model's preferred coordinates

Failure to linearly decode mass does not imply absence of useful physics. This is why representation geometry, causal intervention, and OOD performance are included.

### Interventions on neural representations can leave the learned representation manifold

A sufficiently large perturbation may create hidden states the model never encounters naturally. Intervention strength must therefore be interpreted cautiously, with emphasis on local and controlled effects.

### Identifiability depends on modeling assumptions

A parameter that is identifiable under one family of trajectories may be non-identifiable under another. Claims must always be conditioned on the observational and intervention regime.

### Extrapolation is not synonymous with understanding

Even a physically structured neural model may fail far outside its training range, and successful extrapolation alone does not prove that the model has recovered a human physical law.

---

# 18. Scope

The initial project should remain deliberately focused.

### Core scope

- state-based predictive world models;
- hidden trajectory-specific physical parameters;
- physical representation probing;
- representation geometry;
- causal interventions and ablations;
- compositional and out-of-distribution generalization;
- identifiability and informative-data requirements.

### Explicitly outside the initial scope

- raw video perception;
- large generative video world models;
- symbolic equation discovery;
- language-model physical reasoning;
- full robotic control;
- active experiment selection as a learned policy;
- explicit physics-engine architectures as the primary research subject.

These topics may become follow-up work, but including them initially would weaken the central scientific question.

---

# 19. Proposed Central Thesis

The project can ultimately be summarized by the following thesis:

> **Accurate physical prediction is not sufficient evidence that a neural world model has learned physical structure. A stronger notion of learned physics requires that hidden physical properties become representationally accessible when they are identifiable, participate causally in prediction, and support generalization across novel physical conditions.**

A concise research question is:

> **Do predictive state-space world models spontaneously form causally useful representations of latent physical parameters?**

A more complete formulation is:

> **When do hidden physical variables emerge inside predictive neural dynamics models, how are they represented, are they causally used to generate future predictions, and does their emergence explain compositional and out-of-distribution generalization?**

---

# References

1. **Joseph, S., Garrido, Q., Balestriero, R., Kowal, M., Fel, T., Bakhtiari, S., Richards, B. A., & Rabbat, M.** (2026). *Interpreting Physics in Video World Models*. Proceedings of the 43rd International Conference on Machine Learning (ICML), PMLR 306, 54687–54724.  
   https://proceedings.mlr.press/v306/joseph26a.html

2. **Baumgartner, M. W., Lei, A., Watson, J., & Posner, I.** (2026). *Disentangling Dynamical Systems: Causal Representation Learning Meets Local Sparse Attention*. Proceedings of the Fifth Conference on Causal Learning and Reasoning (CLeaR), PMLR 323, 119–165.  
   https://proceedings.mlr.press/v323/baumgartner26a.html

3. **Yao, D., Muller, C. J., & Locatello, F.** (2024). *Marrying Causal Representation Learning with Dynamical Systems for Science*. Advances in Neural Information Processing Systems 37 (NeurIPS 2024).  
   https://proceedings.neurips.cc/paper_files/paper/2024/hash/83eb339ed42297658fa24b5cec939285-Abstract-Conference.html  
   DOI: 10.52202/079017-2290

4. **Alam, N.** (2026). *Causal Physics Steering in Video World Models via Concept Activation Vectors*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 5890–5896.  
   https://openaccess.thecvf.com/content/CVPR2026W/VideoWorldModel/html/Alam_Causal_Physics_Steering_in_Video_World_Models_via_Concept_Activation_CVPRW_2026_paper.html

5. **Xu, Z., Zhou, W., Pan, X., Deng, N., Liu, C., Chen, Q., & Yao, C.** (2026). *Dynamics Are Learned, Not Told: Semi-Supervised Discovery of Latent Dynamics Geometries for Zero-Shot Policy Adaptation*. Proceedings of the 43rd International Conference on Machine Learning (ICML), PMLR 306, 142962–142988.  
   https://proceedings.mlr.press/v306/xu26cq.html

---

## One-Sentence Summary

**Train a state-based world model only to predict dynamics, hide the true physical parameters from it, and then determine whether those parameters emerge internally, whether the model actually uses them causally, and whether such internal physics explains generalization beyond familiar trajectories.**
