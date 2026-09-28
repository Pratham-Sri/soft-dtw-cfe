# DTW-Guided Constrained Deformation for Efficient Time-Series Counterfactual Explanations

## 1. Executive Summary

### Problem

We are working on **counterfactual explanations (CFE) for time-series classifiers**.

Given a time series \(X\) that is classified as class \(y\), the goal is to generate a minimally modified time series \(X'\) such that:

\[
f(X') = y_{\text{target}}
\]

while keeping \(X'\) close to \(X\), plausible, and structurally meaningful.

The existing approach we investigated uses **Soft-DTW directly inside the counterfactual optimization loop**. This provides a useful temporal similarity measure, but creates a major computational bottleneck because Soft-DTW is approximately quadratic in sequence length.

The central research question therefore became:

> **Can DTW alignment information be extracted once and converted into a low-dimensional deformation space, so that counterfactual optimization no longer needs to repeatedly calculate Soft-DTW?**

Our proposed architecture is:

\[
\boxed{
X
\rightarrow
\text{DTW alignment}
\rightarrow
\text{DTW-informed deformation space}
\rightarrow
\text{CMA-ES}
\rightarrow
X'
}
\]

The key conceptual change is:

> **Use DTW as a one-time geometric prior rather than as an optimization-time distance function.**

The architecture is a research hypothesis, not yet a proven method. Therefore, we will first run a **Representation Probe** to determine whether the proposed deformation space is expressive and useful before introducing CMA-ES.

---

# 2. Original Problem

For an input time series

\[
X = [x_1,x_2,\ldots,x_T]
\]

and a classifier

\[
f(X),
\]

we want to find a counterfactual

\[
X'
\]

such that:

\[
f(X') = y_t
\]

for a desired target class \(y_t\).

At the same time, we want:

\[
D(X,X')
\]

to remain small.

A generic formulation is:

\[
\min_{X'} D(X,X')
\]

subject to:

\[
f(X') = y_t.
\]

The difficulty is that \(X'\) is a \(T\)-dimensional object, potentially with multiple channels:

\[
X\in\mathbb{R}^{T\times d}.
\]

Directly optimizing every individual time point gives a very high-dimensional search space and can easily produce unrealistic signals.

Therefore, the existing method uses temporal similarity information, particularly **Soft-DTW**, to encourage plausible temporal modifications.

---

# 3. Existing Soft-DTW Approach

## 3.1 Why DTW is useful

Ordinary Euclidean distance assumes that two sequences should be compared point-by-point:

\[
D_{\text{Euc}}(X,P)
=
\sum_t
\|X_t-P_t\|^2.
\]

This is problematic for time-series data because two signals can have essentially the same shape while being shifted or stretched in time.

DTW solves this by allowing temporal alignment.

Given:

\[
X=[x_1,\ldots,x_T]
\]

and

\[
P=[p_1,\ldots,p_T],
\]

DTW searches over valid alignment paths:

\[
\pi =
\{(i_1,j_1),(i_2,j_2),\ldots\}.
\]

The classical DTW objective is:

\[
DTW(X,P)
=
\min_{\pi}
\sum_{(i,j)\in\pi}
c(x_i,p_j)
\]

where \(c\) is typically a local squared distance.

This makes DTW particularly useful for time-series counterfactuals because it captures **temporal morphology**, rather than only pointwise similarity.

---

# 4. Main Bottleneck Observed

The major problem is not DTW itself.

The problem is **repeated DTW/Soft-DTW evaluation during optimization**.

Suppose the optimization has:

- \(I\) optimization iterations,
- \(K\) target prototypes/neighbours,
- sequence length \(T\).

Soft-DTW/DTW is approximately:

\[
O(T^2).
\]

Therefore, repeatedly comparing candidates against \(K\) references across \(I\) optimization steps gives approximately:

\[
\boxed{
O(IKT^2)
}
\]

temporal-alignment computation.

This becomes extremely expensive for long time series.

---

# 5. Evidence From the Existing Repository

The repository we investigated uses Soft-DTW in the optimization process.

The existing results used approximately:

- 500 optimization iterations
- \(K=10\) target neighbours by default

and showed the following approximate runtimes for five test samples:

| Dataset | Length / Channels | Runtime for 5 samples |
|---|---:|---:|
| ItalyPowerDemand | \(T=24\) | 18.6 s |
| CBF | \(T=128\) | 37.8 s |
| TwoLeadECG | \(T=82\) | 22.0 s |
| GunPoint | \(T=150\) | 38.0 s |
| Coffee | \(T=286\) | 108.5 s |
| Earthquakes | \(T=512\) | 331.5 s |
| Epilepsy | \(T=206,\ d=3\) | 64.1 s |
| Cricket | \(T=1197,\ d=6\) | 1619.7 s |

The Cricket result is particularly important:

\[
1619.7\text{ s}\approx27\text{ minutes}
\]

for only five test samples.

That is roughly:

\[
5.4\text{ minutes/sample}.
\]

This demonstrates that the quadratic temporal computation becomes increasingly problematic as \(T\) grows.

---

# 6. Other Problems Observed

The existing results also revealed that computational cost is not the only issue.

## 6.1 Validity problems

Some datasets showed poor target-class validity:

| Dataset | Existing validity |
|---|---:|
| Earthquakes | 0.20 |
| Coffee | 0.00 |
| Epilepsy | 0.00 |

Therefore, simply making Soft-DTW faster would not necessarily solve the overall problem.

We need to determine whether the **counterfactual representation itself** is capable of reaching the target decision region.

---

## 6.2 Classifier quality matters

Some datasets had relatively weak classifier accuracy.

Examples included approximately:

- CBF: 64.1%
- Coffee: 53.6%
- Earthquakes: 74.8%

This creates an important experimental issue.

If the classifier itself has poor discrimination, a counterfactual validity result may not mean much.

Therefore, all experiments must report the classifier's performance and use the classifier's **actual decision boundary** rather than an arbitrary probability threshold.

---

## 6.3 Very small evaluation sample

The existing results were based on only five test samples per dataset.

Therefore, they are useful for identifying computational behaviour and failure cases, but they are not sufficient for strong statistical conclusions.

Our new representation probe should initially focus on understanding the mechanism, followed later by a proper evaluation.

---

# 7. Research Hypothesis

The central hypothesis is:

> **The useful information contained in DTW does not necessarily need to be recomputed during every optimization step. It may be possible to extract the alignment once and use it as a prior over a low-dimensional deformation space.**

Instead of:

\[
X
\rightarrow
\text{candidate}
\rightarrow
DTW
\rightarrow
\text{fitness}
\]

repeated hundreds of times, we want:

\[
X,P
\rightarrow
DTW
\rightarrow
\text{alignment prior}
\]

once.

Then:

\[
\text{alignment prior}
\rightarrow
\text{cheap deformation generator}
\rightarrow
X'
\]

during optimization.

The approximate complexity becomes:

\[
\boxed{
O(KT^2)+O(IKT)
}
\]

instead of:

\[
\boxed{
O(IKT^2)
}
\]

The important qualification is:

> We have not eliminated quadratic computation.

We have **removed it from the inner optimization loop**.

That distinction is important.

---

# 8. First Proposed Architecture: DTW-Morphing Genetic Algorithm

Our initial idea was a low-dimensional genome such as:

\[
\theta=[\alpha,\beta]
\]

where:

- \(\alpha\) controls amplitude interpolation
- \(\beta\) controls temporal deformation

The idea was roughly:

\[
X'
=
\text{Morph}(X,P;\alpha,\beta)
\]

and the fitness function would primarily depend on classifier output.

This would avoid DTW during fitness evaluation.

However, after examining the mathematics more carefully, several problems emerged.

---

# 9. Why the Original DTW-Morphing Idea Was Rejected

## 9.1 DTW path is not a normal time axis

A DTW path contains:

- horizontal steps
- vertical steps
- diagonal steps

Therefore it represents an **alignment relation**, not a simple time transformation.

A path such as:

\[
(i,j)
\]

does not directly mean:

> "move time \(t\) by this amount."

It tells us which points correspond.

Therefore, the path must first be converted into a continuous monotonic mapping.

---

## 9.2 One global temporal parameter is insufficient

A single parameter \(\beta\) cannot represent complex local temporal deformation.

For example:

```text
Original:

----/\---------/\----

Target:

------/\---/\--------
```

The first event may need to move right while the second moves left.

A single global time-shift parameter cannot express this.

Therefore, we need **multiple local temporal degrees of freedom**.

---

## 9.3 Linear interpolation is not a manifold guarantee

A naive approach such as:

\[
X_\lambda
=
(1-\lambda)X+\lambda P
\]

does not guarantee that intermediate signals are realistic.

Linear interpolation can create:

- destructive interference
- unnatural amplitudes
- flattened morphology
- unrealistic intermediate shapes

Therefore, we should not claim:

> "Interpolation between real time series stays on the data manifold."

It does not.

Our representation instead creates a **constrained deformation space** designed to be smoother and more plausible.

We should use language such as:

> DTW-informed, smooth, bounded, constrained deformation space

rather than:

> guaranteed manifold.

---

## 9.4 Alpha/beta are not actual signal-space distances

The parameter vector:

\[
\theta=(\alpha,\beta)
\]

is a coordinate system.

It is not itself a meaningful measure of how far the generated signal moved from \(X\).

Two parameter vectors with similar norms can produce very different signals.

Therefore, actual evaluation must use:

\[
D(X,X')
\]

in signal space.

---

## 9.5 Hard invalidity penalties are problematic

A fitness such as:

\[
L=
\begin{cases}
D(X,X') & f(X')=y_t\\
C & \text{otherwise}
\end{cases}
\]

creates a very sparse optimization signal.

Many candidates can receive essentially the same penalty.

Instead, we want a continuous measure of progress toward the decision boundary.

---

# 10. Final Architecture

The refined architecture is:

# DTW-Guided Constrained Deformation via CMA-ES

The complete conceptual pipeline is:

\[
\boxed{
X
\rightarrow
P
\rightarrow
DTW
\rightarrow
\tau_{\text{DTW}}
\rightarrow
\text{low-dimensional deformation}
\rightarrow
\text{CMA-ES}
\rightarrow
X'
}
\]

where:

- \(X\) = original time series
- \(P\) = correctly classified target-class prototype
- \(\tau_{\text{DTW}}\) = continuous DTW-derived temporal prior
- deformation parameters = low-dimensional representation
- CMA-ES = derivative-free optimizer
- \(X'\) = generated counterfactual

---

# 11. Prototype Selection

For a target class \(y_t\), we select target-class examples from the **training set only**.

The prototype must satisfy:

\[
f(P)=y_t.
\]

Equivalently, using a target-class margin:

\[
m(P)>0.
\]

We deliberately do **not** require an arbitrarily large margin.

A prototype with:

\[
m(P)=0.01
\]

is still a valid target example.

However, low target margin does not automatically mean the prototype is close to \(X\).

Therefore, we record both:

\[
m(P)
\]

and:

\[
D(X,P).
\]

We can use multiple prototypes:

\[
P_1,\ldots,P_K
\]

rather than relying on a single target example.

A reasonable initial \(K\) is:

\[
K=3\text{--}5.
\]

The exact prototype-selection rule must be fixed before examining results to avoid selection bias.

---

# 12. Target Margin

Instead of relying only on class probability, we use classifier logits.

For a multiclass classifier with logits:

\[
z_1(X),\ldots,z_C(X),
\]

the target-class margin is:

\[
\boxed{
m(X)
=
z_{y_t}(X)
-
\max_{c\neq y_t}z_c(X)
}
\]

Then:

\[
m(X)>0
\]

means the target class wins.

This gives us a continuous measure of how far the candidate is toward or inside the target decision region.

For binary classifiers, the equivalent signed decision logit can be used.

---

# 13. DTW Computation

For each prototype \(P_k\):

\[
\pi_k=DTW(X,P_k).
\]

The DTW computation happens **once per prototype**.

The path contains pairs:

\[
(i_X,j_P).
\]

These mean:

> \(X(i_X)\) and \(P(j_P)\) are aligned.

We do not directly treat this staircase path as a differentiable function.

Instead, we convert it into a continuous temporal mapping.

---

# 14. Critical DTW Directionality

This was the most important implementation issue discovered during the design review.

Our temporal deformation generator is defined as:

\[
X_{\text{warp}}(t)
=
X(\tau(t)).
\]

Therefore:

\[
\tau(t)
\]

must answer:

> At output time \(t\), which time coordinate from the **original \(X\)** should be sampled?

The DTW path initially gives:

\[
i_X\rightarrow j_P.
\]

But this is the opposite direction.

For a target-aligned output coordinate \(j_P\), we need:

\[
\boxed{
j_P\rightarrow i_X
}
\]

so that:

\[
\tau(j_P)=i_X.
\]

Therefore, the implementation must explicitly construct the inverse-direction mapping.

This must be tested with a synthetic unit test before any real experiment.

A silent reversal here would produce code that runs successfully but performs the wrong temporal transformation.

---

# 15. DTW Path → Continuous Warp

The raw DTW path is discrete and staircase-like.

We need a smooth function:

\[
\tau:[0,1]\rightarrow[0,1].
\]

The general process is:

\[
\text{DTW path}
\rightarrow
\text{monotonic anchors}
\rightarrow
\text{continuous interpolation}
\rightarrow
\tau_{\text{DTW}}.
\]

For a path represented by:

\[
(i,j),
\]

we can aggregate multiple matches using a robust statistic such as a median.

For example:

\[
j_i=
\operatorname{median}
\{j:(i,j)\in\pi\}.
\]

This gives a monotonic anchor map.

---

# 16. PCHIP

We use **PCHIP** rather than ordinary cubic spline interpolation.

PCHIP is useful because it is shape-preserving and reduces the tendency to introduce artificial overshoots around sharp local features.

Important clarification:

The earlier concern that median aggregation necessarily creates duplicate PCHIP **x-coordinates** was incorrect.

If:

\[
x=[0,1,2,3,4]
\]

and:

\[
y=[1,1,2,3,3],
\]

then \(x\) is still strictly increasing.

Therefore, PCHIP itself does not fail because \(y\) contains plateaus.

The real issue is mathematical:

\[
y=[1,1,2,\ldots]
\]

contains regions where:

\[
\frac{d\tau}{dt}=0.
\]

That conflicts with our later requirement that the temporal velocity be strictly positive.

Therefore, after constructing the smooth DTW warp, we estimate its derivative and impose a small positive floor.

---

# 17. Positive Temporal Velocity Representation

We want a temporal warp that is:

1. monotonic
2. endpoint preserving
3. smooth
4. strictly increasing

Instead of directly optimizing \(\tau(t)\), we parameterize its velocity.

Define:

\[
g(t)
=
g_{\text{DTW}}(t)
+
\sum_{m=1}^{M_t}
\beta_m R_m(t)
\]

where:

- \(R_m(t)\) are RBF basis functions
- \(\beta_m\) are temporal deformation parameters
- \(g_{\text{DTW}}\) encodes the DTW prior

Then define:

\[
\boxed{
v(t)=\operatorname{softplus}(g(t))
}
\]

Since:

\[
\operatorname{softplus}(x)>0,
\]

we obtain:

\[
v(t)>0.
\]

The temporal warp is then:

\[
\boxed{
\tau(t)
=
T
\frac{
\int_0^t v(s)\,ds
}{
\int_0^T v(s)\,ds
}
}
\]

This guarantees:

\[
\tau(0)=0
\]

and:

\[
\tau(T)=T.
\]

Because \(v(t)>0\):

\[
\frac{d\tau}{dt}>0.
\]

Thus the warp is strictly monotonic.

---

# 18. Centering the Temporal Search Around DTW

A subtle issue arises if we simply define:

\[
g(t)=\sum_m\beta_mR_m(t).
\]

At:

\[
\beta=0,
\]

we get approximately the identity warp.

But we want the DTW solution to be the **initial prior**.

Therefore, we construct:

\[
v_{\text{DTW}}(t)
\propto
\frac{d\tau_{\text{DTW}}}{dt}.
\]

After ensuring:

\[
v_{\text{DTW}}(t)>\epsilon,
\]

we compute the inverse softplus:

\[
\boxed{
g_{\text{DTW}}(t)
=
\operatorname{softplus}^{-1}
(v_{\text{DTW}}(t))
}
\]

so that:

\[
\operatorname{softplus}(g_{\text{DTW}}(t))
\approx
v_{\text{DTW}}(t).
\]

Then:

\[
\boxed{
g(t)
=
g_{\text{DTW}}(t)
+
\sum_m\beta_mR_m(t)
}
\]

and therefore:

\[
\beta=0
\]

corresponds to the DTW-centered warp.

This is important because DTW is being used as a **prior**, not discarded.

---

# 19. DTW Is a Prior, Not a Cage

The final counterfactual is not required to remain exactly on the DTW path.

DTW finds a temporal alignment that minimizes alignment cost.

It does **not** know where the classifier decision boundary is.

Therefore, the optimal counterfactual may require a temporal deformation different from the original DTW alignment.

Our intended behaviour is:

\[
\text{DTW}
\rightarrow
\text{good initialization/prior}
\rightarrow
\text{search around it}.
\]

Not:

\[
\text{DTW}
\rightarrow
\text{hard constraint}.
\]

This distinction is central to the research hypothesis.

---

# 20. RBF Temporal Basis

We represent temporal deformation using a small number of smooth basis functions:

\[
R_m(t)
=
\exp
\left(
-\frac{(t-c_m)^2}{2\sigma_m^2}
\right).
\]

The centers \(c_m\) cover normalized time:

\[
t\in[0,1].
\]

Then:

\[
g(t)
=
g_{\text{DTW}}(t)
+
\sum_m\beta_mR_m(t).
\]

The number of basis functions \(M_t\) controls temporal expressiveness.

Initial experiments can test values such as:

\[
M_t\in\{2,4,6,8\}.
\]

---

# 21. Amplitude Deformation

Temporal deformation alone is insufficient.

We also need to modify the signal morphology/amplitude.

Define:

\[
h(t)
=
\sum_{m=1}^{M_a}
\alpha_mR_m(t).
\]

Convert this into a bounded interpolation coefficient:

\[
\boxed{
a(t)=\sigma(h(t))
}
\]

where:

\[
\sigma(x)=\frac{1}{1+e^{-x}}.
\]

Therefore:

\[
0<a(t)<1.
\]

The warped original signal is:

\[
X_{\text{warp}}(t)
=
X(\tau(t)).
\]

We construct an aligned target morphology:

\[
P_{\text{aligned}}(t)
\]

using the DTW alignment.

Then:

\[
\boxed{
X'(t)
=
(1-a(t))X_{\text{warp}}(t)
+
a(t)P_{\text{aligned}}(t)
}
\]

This produces a smooth, bounded amplitude deformation toward the target morphology.

---

# 22. Amplitude and Temporal Parameters

The full parameter vector is:

\[
\boxed{
\theta=
[
\alpha_1,\ldots,\alpha_{M_a},
\beta_1,\ldots,\beta_{M_t}
]
}
\]

with total dimensionality:

\[
M_a+M_t.
\]

Typical configurations:

\[
(2,2),(4,2),(2,4),(4,4),
(6,4),(4,6),(8,4),(4,8).
\]

The important point is that we are optimizing perhaps 4–16 parameters rather than \(T\) independent time points.

---

# 23. Immutability Constraints

We originally considered enforcing both amplitude and temporal immutable regions.

However, temporal immutability contains a subtle mathematical problem.

A naive construction such as:

\[
v(t)
=
1+
(\operatorname{softplus}(g(t))-1)M_{\text{time}}(t)
\]

followed by:

\[
\tau(t)
=
T
\frac{\int_0^t v(s)ds}
{\int_0^T v(s)ds}
\]

does **not** guarantee:

\[
\tau(t)=t
\]

inside an immutable region.

Why?

Because the final normalization multiplies the entire temporal mapping by a global factor.

Even if:

\[
v(t)=1
\]

inside the immutable region, global normalization can still change its effective mapping.

Therefore:

### V1

We enforce **amplitude immutability only**.

Temporal immutability is deferred.

### Future temporal immutability

If required, it must be structurally enforced.

For an immutable interval:

\[
[a,b],
\]

we would require:

\[
\tau(a)=a,
\]

\[
\tau(b)=b,
\]

and:

\[
\tau(t)=t
\qquad
t\in[a,b].
\]

That is a structural/piecewise constraint rather than a simple velocity mask.

---

# 24. Objective Function

We want:

1. target-class validity
2. small perturbation
3. plausible deformation

A generic objective is:

\[
\boxed{
L(\theta)
=
D(X,X_\theta)
+
\lambda_{\text{target}}
L_{\text{target}}(X_\theta)
+
\lambda_{\text{reg}}L_{\text{reg}}(\theta)
}
\]

where:

\[
X_\theta = G(X,P;\theta).
\]

For the target term, a hinge loss based on the target margin can be used:

\[
L_{\text{target}}
=
\max(0,m_0-m(X_\theta)).
\]

Here \(m_0\) is a desired positive margin.

Alternatively, the optimization can primarily use the continuous target margin and evaluate actual boundary crossing separately.

The critical point is that we do **not** rely solely on:

\[
f(X')\in\{0,1\}.
\]

We preserve information about how close a candidate is to the target decision region.

---

# 25. Why CMA-ES?

The final optimization problem is:

- continuous
- low-dimensional
- nonlinear
- potentially non-convex
- black-box from the optimizer's perspective
- not necessarily differentiable through the classifier/deformation pipeline

Therefore, **CMA-ES** is a better fit than the originally proposed genetic algorithm.

CMA-ES maintains a distribution over candidate parameter vectors and learns:

- promising regions
- parameter correlations
- covariance structure
- search scale

This is particularly useful because amplitude and temporal parameters may interact.

For example:

> shifting an event in time may only become useful after increasing its amplitude.

CMA-ES can learn such correlations.

---

# 26. Why Not a Genetic Algorithm?

A conventional GA is possible, but it is not our preferred optimizer because:

- the search space is continuous
- dimensionality is relatively low
- parameter correlations matter
- crossover may be less natural for smooth continuous deformation parameters
- CMA-ES is specifically designed for continuous black-box optimization

Therefore:

\[
\boxed{\text{CMA-ES > conventional GA for our final search}}
\]

given this representation.

However, this is **H3**, not the central scientific contribution.

We should first prove that the representation works.

---

# 27. Representation Probe

Before implementing CMA-ES, we perform a **Representation Probe**.

The purpose is to answer:

> Can our deformation family actually reach the target decision region?

This separates **representation failure** from **optimizer failure**.

The primary reachability metric is:

\[
\boxed{
R(X)
=
\max_\theta
m(X_\theta)
}
\]

where:

\[
m(X_\theta)
=
z_t(X_\theta)
-
\max_{c\neq t}z_c(X_\theta).
\]

If:

\[
R(X)>0,
\]

then at least one sampled deformation crossed the classifier's target decision boundary.

If:

\[
R(X)<0,
\]

that does **not mathematically prove** that the representation cannot reach the target.

It means our current search has not demonstrated reachability.

This distinction is important.

---

# 28. Why LHS Instead of Naive Random Sampling?

The parameter space may contain 8–16 dimensions.

Pure Gaussian random sampling can leave large regions poorly explored.

We therefore use **Latin Hypercube Sampling (LHS)**.

LHS divides each parameter dimension into intervals and ensures that the sample set covers each dimension more systematically.

This provides better parameter-space reconnaissance.

However:

> LHS does not provide uniform coverage of signal space.

Different parameter vectors can generate very similar signals.

Therefore, every sampled candidate must also be evaluated in signal space.

---

# 29. Metrics for Every Candidate

For each generated candidate \(X_\theta\), record:

### Classification

\[
m(X_\theta)
\]

Target probability can also be recorded.

### Signal-space deformation

\[
D(X,X_\theta).
\]

### Prototype proximity

\[
D(P,X_\theta).
\]

### Post-hoc DTW

Calculate DTW only for selected/best candidates after generation.

This allows us to evaluate temporal plausibility without placing DTW back into the optimization loop.

### Runtime

Record:

- DTW preprocessing time
- candidate generation time
- classifier inference time
- total probe time

### Stability

Repeat experiments across multiple seeds.

---

# 30. Actual Signal Distance vs Parameter Distance

This is important.

We must not conclude that:

\[
\|\theta\|
\]

is a good measure of perturbation.

Instead, evaluate:

\[
D(X,X_\theta).
\]

For example, two parameter vectors:

\[
\theta_1,\theta_2
\]

may have similar Euclidean norms but generate radically different signals.

Therefore:

> **Parameter space is the search space; signal space is the evaluation space.**

---

# 31. Deterministic Deformation Tests

Before LHS, we should perform deterministic paths.

## 31.1 Amplitude-only

Keep temporal deformation fixed.

Vary amplitude deformation from:

\[
\lambda=0
\]

to:

\[
\lambda=1.
\]

Evaluate:

\[
X_\lambda^{\text{amp}}.
\]

---

## 31.2 Temporal-only

Keep amplitude fixed.

Interpolate from identity temporal deformation to DTW temporal deformation.

For example, in the velocity/log-velocity representation:

\[
g_\lambda
=
(1-\lambda)g_{\text{identity}}
+
\lambda g_{\text{DTW}}.
\]

Then construct:

\[
\tau_\lambda.
\]

This preserves the monotonic warp construction.

---

## 31.3 Combined

Allow both amplitude and temporal deformation.

The purpose is not necessarily to obtain a monotonic increase in target probability.

A deep classifier decision surface can be non-monotonic along any arbitrary path.

Therefore, we should record:

\[
\max_\lambda m(X_\lambda)
\]

rather than demanding:

\[
m(X_\lambda)
\]

increase monotonically.

---

# 32. DTW Prior Ablation

To determine whether DTW actually contributes useful information, we compare three temporal priors.

## A. Identity prior

Start around:

\[
\tau(t)=t.
\]

## B. DTW prior

Start around:

\[
\tau_{\text{DTW}}(t).
\]

## C. Random monotonic prior

Generate a monotonic temporal deformation without using DTW.

The deformation capacity should otherwise remain comparable.

This directly tests:

\[
\boxed{
\text{Does DTW provide useful structure?}
}
\]

If DTW performs no better than generic monotonic flexibility, then the claimed value of DTW as a prior is weakened.

---

# 33. Hypothesis 1 — DTW Prior Is Useful

### Hypothesis

DTW alignment contains useful temporal information for CFE generation.

### Experiment

Compare:

\[
\text{Identity}
\quad vs \quad
\text{DTW}
\quad vs \quad
\text{Random monotonic}.
\]

### Metrics

- maximum target margin
- target validity
- signal-space distance
- prototype proximity
- runtime

### Interpretation

If DTW consistently provides better reachable target margins or lower-deformation valid candidates, that supports H1.

If generic monotonic deformation performs equally well, the usefulness of DTW as a prior is questionable.

---

# 34. Hypothesis 2 — Low-Dimensional Deformation Is Expressive

### Hypothesis

A small number of amplitude and temporal basis functions can represent sufficiently useful counterfactual transformations.

### Experiment

Vary:

\[
M_a,M_t.
\]

For example:

\[
(2,2),(4,2),(2,4),(4,4),
(6,4),(4,6),(8,4),(4,8).
\]

Measure:

\[
R(X)=\max_\theta m(X_\theta)
\]

and:

\[
\min_\theta D(P,X_\theta).
\]

### Interpretation

If increasing dimensionality dramatically improves reachability, the lower-dimensional representation is too restrictive.

If moderate dimensions already achieve good reachability, this supports the low-dimensional representation hypothesis.

---

# 35. Prototype Approachability Test

We initially considered trying to reproduce the prototype exactly.

That is too strong.

The deformation family intentionally imposes constraints.

Therefore, the correct test is:

\[
\boxed{
A(P)
=
\min_\theta D(P,X_\theta)
}
\]

where \(A(P)\) measures prototype approachability.

We also record:

\[
D(X,P)
\]

and:

\[
m(P).
\]

This gives:

| Quantity | Meaning |
|---|---|
| \(m(P)\) | Is prototype target-class? |
| \(D(X,P)\) | How far is prototype from original? |
| \(A(P)\) | Can deformation family approach prototype? |
| \(R(X)\) | Can deformation family reach target decision region? |

This cleanly separates different failure modes.

---

# 36. Important Distinction: Prototype Reachability vs Counterfactual Reachability

A representation does not necessarily need to reproduce the selected prototype exactly.

The classifier decision boundary may be crossed before the prototype is reached.

Therefore:

\[
\min_\theta D(P,X_\theta)
\]

and:

\[
\min_\theta D(X,X_\theta)
\quad
\text{s.t. }m(X_\theta)>0
\]

are different problems.

This is important because a representation could fail to reproduce \(P\) exactly while still generating an excellent counterfactual.

Therefore, prototype approachability is a **diagnostic**, not the final success criterion.

---

# 37. Visual Diagnostics

For representative examples, visualize:

1. Original \(X\)
2. Target prototype \(P\)
3. Raw DTW path
4. Continuous \(\tau_{\text{DTW}}\)
5. Warped \(X\)
6. Aligned prototype
7. Best LHS candidate
8. Best final CMA-ES candidate

For multichannel data, inspect individual channels as well as aggregate behaviour.

The purpose is to detect:

- jaggedness
- unnatural amplitude changes
- overshooting
- temporal compression
- temporal stretching
- pathological warps
- unrealistic morphology

Numerical validity alone is insufficient.

---

# 38. Datasets for the Representation Probe

The primary probe datasets are:

### Coffee

Important because the existing method had:

\[
\text{validity}=0.
\]

### Earthquakes

Important because:

\[
\text{validity}=0.20.
\]

It also has:

\[
T=512,
\]

making computational effects more visible.

### Epilepsy

Important because:

\[
\text{validity}=0.
\]

It is also multichannel:

\[
d=3.
\]

### Cricket

Although not the primary initial probe dataset, it is particularly useful for demonstrating computational scaling because:

\[
T=1197,\quad d=6
\]

and the existing method took approximately:

\[
1619.7\text{ s}
\]

for five samples.

---

# 39. What Would Constitute Representation Failure?

We should not use a single arbitrary probability threshold.

For example, we should not define:

> "If target probability never reaches 0.4, representation failed."

The correct decision boundary comes from the classifier.

Instead, use:

\[
m(X_\theta)>0.
\]

However, failure should be interpreted carefully.

If LHS does not find a valid candidate, possible explanations include:

1. representation is insufficient
2. sampled parameter bounds are too restrictive
3. the successful region is very small
4. the prototype is inappropriate
5. LHS sample count is insufficient
6. classifier decision surface is difficult

Therefore:

> **LHS failure is evidence, not proof of non-reachability.**

---

# 40. Why We Do Not Start With CMA-ES

Suppose CMA-ES fails to find a valid counterfactual.

We then have two possibilities:

### Possibility A

The deformation family cannot reach the target.

### Possibility B

The deformation family can reach the target, but CMA-ES failed to find it.

Without a representation probe, we cannot distinguish A and B.

Therefore:

\[
\boxed{
\text{Representation first}
\rightarrow
\text{Optimizer second}
}
\]

This is one of the most important methodological decisions in the project.

---

# 41. Hypothesis 3 — CMA-ES Is Efficient

Only after H1/H2 have survived do we introduce CMA-ES.

### Hypothesis

Given a useful deformation representation, CMA-ES can efficiently locate low-distance target-valid counterfactuals.

We compare:

- LHS/random search
- CMA-ES

using the **same representation**.

Metrics:

- target validity
- minimum \(D(X,X')\)
- target margin
- number of evaluations
- wall-clock runtime
- stability across seeds

This isolates the optimizer.

---

# 42. Final Optimization Objective

The final CFE can be expressed as:

\[
\boxed{
\min_{\theta}
D(X,X_\theta)
}
\]

subject to:

\[
m(X_\theta)>0
\]

and:

\[
X_\theta
=
G(X,P;\theta)
\]

where \(G\) is our constrained DTW-guided deformation generator.

A soft optimization formulation is:

\[
\boxed{
L(\theta)
=
D(X,X_\theta)
+
\lambda
\max(0,m_0-m(X_\theta))
+
\lambda_rL_{\text{reg}}(\theta)
}
\]

where \(L_{\text{reg}}\) can encode deformation regularity if needed.

---

# 43. What We Explicitly Rejected

## 43.1 Repeated Soft-DTW

Rejected as the primary optimization-time similarity mechanism because of:

\[
O(IKT^2)
\]

computational cost.

DTW is retained, but moved to preprocessing.

---

## 43.2 Naive linear interpolation

Rejected:

\[
X_\lambda=(1-\lambda)X+\lambda P.
\]

Reason:

- no explicit temporal deformation
- unrealistic intermediate morphology possible
- no manifold guarantee
- not sufficiently expressive

---

## 43.3 One global temporal parameter

Rejected because local temporal changes cannot be represented.

---

## 43.4 Direct unconstrained time-index optimization

Rejected because it can create:

- non-monotonic warps
- discontinuities
- pathological sampling
- unrealistic temporal ordering.

---

## 43.5 Direct displacement parameterization

Instead of:

\[
\tau(t)=t+\Delta(t),
\]

we use a positive velocity formulation.

Reason:

Direct displacement does not naturally guarantee:

\[
\tau'(t)>0.
\]

---

## 43.6 Standard cubic spline as the default interpolation

Not because cubic splines are inherently invalid, but because they can introduce unwanted local overshoot around sharp features.

PCHIP is preferred as a shape-preserving interpolation mechanism.

---

## 43.7 Claiming PCHIP guarantees realism

Rejected.

PCHIP controls interpolation behaviour.

It does not guarantee:

- physical realism
- classifier plausibility
- data-manifold membership.

---

## 43.8 Hard probability/validity penalty

Rejected because it gives little information about progress toward the target boundary.

Continuous logit/margin information is preferred.

---

## 43.9 Arbitrary strong prototype margin

Rejected.

We require:

\[
m(P)>0.
\]

We record the actual margin rather than filtering based on an arbitrary threshold.

---

## 43.10 Exact prototype replication requirement

Rejected.

We measure:

\[
\min_\theta D(P,X_\theta)
\]

as an expressiveness diagnostic rather than demanding:

\[
X_\theta=P.
\]

---

## 43.11 Parameter norm as perturbation size

Rejected.

We use:

\[
D(X,X_\theta)
\]

in signal space.

---

## 43.12 CMA-ES before representation validation

Rejected because optimizer failure and representation failure would become impossible to distinguish.

---

# 44. Complexity Comparison

## Existing approach

Repeated Soft-DTW:

\[
\boxed{
O(IKT^2)
}
\]

where:

- \(I\) = optimization iterations
- \(K\) = target prototypes
- \(T\) = sequence length

---

## Proposed approach

One-time DTW:

\[
O(KT^2)
\]

Then candidate generation:

\[
O(IKT)
\]

approximately, assuming interpolation and deformation are linear in sequence length.

Therefore:

\[
\boxed{
O(KT^2)+O(IKT)
}
\]

The quadratic component is no longer repeated inside the optimization loop.

This should be one of the central computational claims we test empirically.

---

# 45. Experimental Structure

The entire experimental program is therefore:

```text
                     ORIGINAL X
                         │
                         ▼
              Select target prototypes
                         │
                         ▼
                Check m(P) > 0
                         │
                         ▼
                  Compute DTW
                         │
                         ▼
             Correct DTW direction
                         │
                         ▼
             Build continuous warp
                         │
                         ▼
              Validate τ and v(t)
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Identity prior          DTW prior
              │                     │
              └──────────┬──────────┘
                         ▼
              Deformation family
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Deterministic tests          LHS
             │                       │
             └───────────┬───────────┘
                         ▼
             Reachability analysis
                         │
                 ┌───────┴───────┐
                 │               │
               FAIL            PASS
                 │               │
                 ▼               ▼
          Modify representation  CMA-ES
                                 │
                                 ▼
                           Final CFE
```

---

# 46. Exact Implementation Order

We now stop theoretical design and implement.

## Module 1 — `dtw.py`

Responsibilities:

- compute DTW
- return path
- return DTW distance
- support multichannel signals

No classifier logic.

---

## Module 2 — `warp.py`

Responsibilities:

- convert DTW path to the correct coordinate mapping
- construct:

\[
j_P\rightarrow i_X
\]

- normalize coordinates
- build PCHIP warp
- estimate derivative
- apply positive derivative floor
- construct DTW-centered velocity
- produce \(\tau_{\text{DTW}}\)

---

## Module 3 — `basis.py`

Responsibilities:

- RBF centers
- widths
- amplitude basis
- temporal basis

---

## Module 4 — `deformation.py`

Responsibilities:

\[
\theta
\rightarrow
\tau_\theta
\rightarrow
X_{\text{warp}}
\rightarrow
X_\theta.
\]

No optimization logic.

---

## Module 5 — `classifier.py`

Responsibilities:

- model loading
- preprocessing
- logits
- target margin
- probability
- predicted class

---

## Module 6 — `prototypes.py`

Responsibilities:

- target-class selection
- correct-classification filtering
- deterministic prototype selection
- prototype metadata

---

## Module 7 — `diagnostics.py`

Responsibilities:

- signal distances
- DTW distance
- target margin
- runtime
- prototype approachability
- visualization

---

## Module 8 — `run_probe.py`

Responsibilities:

- execute the entire Representation Probe
- run deterministic paths
- run LHS
- compare temporal priors
- save JSON/CSV results
- save plots

CMA-ES should be added only after the probe passes.

---

# 47. First Unit Test: DTW Directionality

This is the first test we write.

We construct a synthetic example where the alignment is known.

Suppose:

```text
P = A B C D E
X = A A B C D E
```

The DTW path should approximately capture:

```text
X → P

0 → 0
1 → 0
2 → 1
3 → 2
4 → 3
5 → 4
```

But our generator requires:

```text
P → X

0 → approximately 0–1
1 → 2
2 → 3
3 → 4
4 → 5
```

The test must verify that:

\[
X(\tau(t))
\]

produces the expected temporal alignment.

This test is more important than immediately testing on real datasets because an incorrect direction can silently produce valid-looking but fundamentally wrong results.

---

# 48. Second Set of Unit Tests

We then test:

### Endpoint preservation

\[
\tau(0)=0
\]

\[
\tau(1)=1.
\]

### Monotonicity

For all sampled \(t\):

\[
\tau(t+\Delta t)>\tau(t).
\]

### Positive velocity

\[
v(t)\ge\epsilon.
\]

### Identity

With zero temporal deformation around the identity prior:

\[
\tau(t)\approx t.
\]

### DTW initialization

With:

\[
\beta=0,
\]

the generated warp should approximately reproduce:

\[
\tau_{\text{DTW}}.
\]

---

# 49. Experimental Outputs

Each probe should produce structured results containing at least:

```text
dataset
sample_id
prototype_id
seed

T
channels

prototype_margin
prototype_distance

dtw_distance
dtw_preprocessing_time

num_amplitude_basis
num_temporal_basis

prior_type

candidate_id

target_margin
target_probability
predicted_class

distance_X_to_candidate
distance_P_to_candidate
posthoc_dtw_distance

generation_time
```

This allows later statistical analysis without rerunning the experiment.

---

# 50. What Success Looks Like

We are not looking for one impressive example.

We want evidence that:

### Representation

The deformation family can produce target-valid candidates.

### DTW prior

DTW provides useful structure compared with identity/random monotonic priors.

### Expressiveness

Moderate \(M_a,M_t\) values are sufficient.

### Plausibility

Generated candidates do not exhibit obvious pathological distortions.

### Efficiency

The expensive DTW computation is paid once rather than repeatedly.

### Optimization

CMA-ES can exploit the representation efficiently.

---

# 51. What Failure Would Tell Us

Failure is useful if it is properly isolated.

### If DTW ≈ identity

The DTW prior may not add meaningful information.

### If random monotonic ≈ DTW

The contribution of DTW becomes questionable.

### If prototype approachability is poor

The deformation family may be too restrictive.

### If target margin remains negative

The representation may not reach the target region.

### If LHS finds valid regions but CMA-ES cannot

The representation works but optimization may be inadequate.

### If CMA-ES finds valid CFEs but they require very large \(D(X,X')\)

The representation may technically work but produce poor/minimally useful counterfactuals.

### If runtime improves dramatically but validity does not

We solved the computational bottleneck but not the CFE quality problem.

### If both runtime and CFE quality improve

Then we have strong evidence supporting the overall approach.

---

# 52. Important Scientific Discipline

We must avoid changing the representation after seeing every dataset's results.

Before running the main experiments, we should freeze:

- RBF basis construction
- parameter bounds
- LHS sample count
- prototype-selection rule
- distance functions
- classifier preprocessing
- random seeds
- prior definitions
- evaluation metrics

If changes become necessary, record them explicitly as representation revisions.

Otherwise, we risk tuning the architecture directly against the evaluation datasets.

---

# 53. Final Research Question

The project is ultimately testing:

> **Can temporal alignment information from DTW be extracted once and transformed into a low-dimensional, smooth, constrained deformation space that enables efficient generation of valid time-series counterfactual explanations without repeatedly computing Soft-DTW?**

This contains three experimentally separable claims:

### H1 — Alignment prior

\[
\boxed{
\text{DTW provides useful temporal prior information.}
}
\]

### H2 — Representation

\[
\boxed{
\text{A low-dimensional constrained deformation family is expressive enough.}
}
\]

### H3 — Optimization

\[
\boxed{
\text{CMA-ES can efficiently search this deformation space.}
}
\]

Only H1 + H2 justify the representation.

H3 is evaluated afterward.

---

# 54. Final Architecture

The final design we are proceeding with is therefore:

\[
\boxed{
\begin{aligned}
&\text{Input }X\\
&\downarrow\\
&\text{Select correctly classified target prototype(s) }P\\
&\downarrow\\
&\text{Compute DTW once}\\
&\downarrow\\
&\text{Convert DTW path to }P\rightarrow X\text{ temporal mapping}\\
&\downarrow\\
&\text{Construct smooth }\tau_{\text{DTW}}\\
&\downarrow\\
&\text{Represent temporal deformation using positive velocity field}\\
&\downarrow\\
&\text{Represent amplitude deformation using bounded RBF basis}\\
&\downarrow\\
&\text{Generate }X_\theta\\
&\downarrow\\
&\text{Representation Probe / LHS}\\
&\downarrow\\
&\text{Validate reachability and expressiveness}\\
&\downarrow\\
&\text{CMA-ES optimization}\\
&\downarrow\\
&\boxed{\text{Final Counterfactual }X'}
\end{aligned}
}
\]

The key complexity change is:

\[
\boxed{
O(IKT^2)
\quad\longrightarrow\quad
O(KT^2)+O(IKT)
}
\]

and the key conceptual change is:

\[
\boxed{
\text{DTW as optimization-time distance}
\quad\longrightarrow\quad
\text{DTW as one-time deformation prior}
}
\]

---

# 55. Current Status

The theoretical architecture is now considered **frozen for experimental testing**.

However, it is important to distinguish:

> **Architecture frozen for testing**

from:

> **Architecture proven to work.**

We have not yet demonstrated:

- that the deformation family can cross the classifier boundary,
- that DTW is better than generic monotonic temporal flexibility,
- that the chosen basis dimensionality is sufficient,
- that the resulting counterfactuals are meaningfully plausible,
- or that CMA-ES will outperform simpler search.

Those are precisely what the Representation Probe is designed to establish.

Therefore, the immediate implementation target is:

\[
\boxed{
\texttt{dtw.py}
+
\texttt{warp.py}
+
\text{directionality unit test}
}
\]

followed by the Representation Probe on:

\[
\boxed{
\text{Coffee, Earthquakes, Epilepsy}
}
\]

before introducing CMA-ES.

That is the point at which the architecture stops being a theoretical proposal and becomes an experimentally falsifiable research system.