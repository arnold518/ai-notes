# Chapter 1: <br> Optimization and Stochastic Gradient Descent 

Mathematical Foundations of Deep Neural Networks
Spring 2024
Department of Mathematical Sciences
Ernest K. Ryu
Seoul National University

## Optimization problem

In an optimization problem, we minimize or maximize a function value, possibly subject to constraints.

$$
\begin{array}{ll}
\underset{\theta \in \mathbb{R}^{p}}{\operatorname{minimize}} & f(\theta) \\
\text { subject to } & h_{1}(\theta)=0
\end{array}
$$

Decision variable: $\theta$
Objective function: $f$
Equality constraint: $h_{i}(\theta)=0$ for $i=1, \ldots, m$
Inequality constraint: $g_{j}(\theta) \leq 0$ for $j=1, \ldots, n$

## Minimization vs. maximization

In machine learning (ML), we often minimize a "loss", but sometimes we maximize the "likelihood".

In any case, minimization and maximization are equivalent since

$$
\text { maximize } f(\theta) \quad \Leftrightarrow \quad \text { minimize }-f(\theta)
$$

## Feasible point and constraints

$\theta \in \mathbb{R}^{p}$ is a feasible point if it satisfies all constraints:

$$
\begin{array}{cc}
h_{1}(\theta)=0 & g_{1}(\theta) \leq 0 \\
\vdots & \vdots \\
h_{m}(\theta)=0 & g_{n}(\theta) \leq 0
\end{array}
$$

Optimization problem is infeasible if there is no feasible point.

An optimization problem with no constraint is called an unconstrained optimization problem. Optimization problems with constraints is called a constrained optimization problem.

## Optimal value and solution

Optimal value of an optimization problem is

$$
p^{\star}=\inf \left\{f(\theta) \mid \theta \in \mathbb{R}^{n}, \theta \text { feasible }\right\}
$$

- $p^{\star}=\infty$ if problem is infeasible
- $p^{\star}=-\infty$ is possible
- In ML, it is often a priori clear that $0 \leq p^{\star}<\infty$.

If $f\left(\theta^{\star}\right)=p^{\star}$, we say $\theta^{\star}$ is a solution or $\theta^{\star}$ is optimal.

- A solution may or may not exist.
- A solution may or may not be unique.


## Example: Curve fitting

Consider setup with data $X_{1}, \ldots, X_{N}$ and corresponding labels $Y_{1}, \ldots, Y_{N}$ satisfying the relationship

$$
Y_{i}=f_{\star}\left(X_{i}\right)+\text { error }
$$

for $i=1, \ldots, N$. Hopefully, "error" is small. True function $f_{\star}$ is unknown.

Goal is to find a function (curve) $f$ such that $f \approx f_{\star}$.

## Example: Least-squares

In least-squares minimization, we solve

$$
\operatorname{minimize}_{\theta \in \mathbb{R} P} \quad \frac{1}{2}\|X \theta-Y\|^{2}
$$

where $X \in \mathbb{R}^{N \times p}$ and $Y \in \mathbb{R}^{N}$. Equivalent to

$$
\operatorname{minimize}_{\theta \in \mathbb{R}^{p}} \frac{1}{2} \sum_{i=1}^{N}\left(X_{i}^{\top} \theta-Y_{i}\right)^{2}
$$

where $X=\left[\begin{array}{c}X_{1}^{\top} \\ \vdots \\ X_{N}^{\top}\end{array}\right]$ and $Y=\left[\begin{array}{c}Y_{1} \\ \vdots \\ Y_{N}\end{array}\right]$.

## Example: Least-squares

To solve

$$
\underset{\theta \in \mathbb{R}^{p}}{\operatorname{minimize}} \frac{1}{2}\|X \theta-Y\|^{2}
$$

take grad and set it to 0 :

$$
\begin{gathered}
X^{\top}\left(X \theta^{\star}-Y\right)=0 \\
\theta^{\star}=\left(X^{\top} X\right)^{-1} X^{\top} Y
\end{gathered}
$$

Here, we assume $X^{\top} X$ is invertible.

Make sure you understand why

$$
\nabla_{\theta} \frac{1}{2}\|X \theta-Y\|^{2}=X^{\top}(X \theta-Y)
$$

## LS is an instance of curve fitting

How is LS curve fitting? Define $f_{\theta}(x)=x^{\top} \theta$. Then LS becomes

$$
\underset{\theta \in \mathbb{R}^{p}}{\operatorname{minimize}} \frac{1}{2} \sum_{i=1}^{N}\left(f_{\theta}\left(X_{i}\right)-Y_{i}\right)^{2}
$$

and the solution hopefully satisfies

$$
Y_{i}=f_{\theta}\left(X_{i}\right)+\text { small. }
$$

Since $X_{i}$ and $Y_{i}$ is assumed to satisfy

$$
Y_{i}=f_{\star}\left(X_{i}\right)+\text { error }
$$

we are searching over linear functions (linear curves) $f_{\theta}$ that best fit (approximate) $f_{\star}$.

## Local vs. global minima

$\theta^{\star}$ is a local minimum if $f(\theta) \geq f\left(\theta^{\star}\right)$ for all feasible $\theta$ within a small neighborhood.
$\theta^{\star}$ is a global minimum if $f(\theta) \geq f\left(\theta^{\star}\right)$ for all feasible $\theta$.

In the worst case, finding the global minimum of an optimization problem is difficult*.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-010.jpg?height=851&width=1103&top_left_y=506&top_left_x=2170)

However, in deep learning, optimization problems are often "solved" without any guarantee of global optimality.

## Gradient descent

Consider the unconstrained optimization problem

$$
\underset{\theta \in \mathbb{R}^{P}}{\operatorname{minimize}} f(\theta)
$$

where $f$ is differentiable.

Gradient Descent (GD) algorithm:

$$
\theta^{k+1}=\theta^{k}-\alpha_{k} \nabla f\left(\theta^{k}\right) \quad \text { for } k=0,1, \ldots,
$$

where $\theta^{0} \in \mathbb{R}^{p}$ is the initial point and $\alpha_{k}>0$ is the learning rate or the stepsize.

The terminology learning rate is common the machine learning literature while stepsize is more common in the optimization literature.

## Definition of "differentiability"

In math, a function is "differentiable" if its derivative exists everywhere.

In deep learning (DL), a function is often said to be differentiable if its derivative exists almost everywhere and the function is nice*. ReLU activation functions are said to be differentiable.

We won't be too concerned with this distinction.

Differentiable in
DL \& Math

Differentiable in
DL but not in Math
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-012.jpg?height=201&width=465&top_left_y=954&top_left_x=2591)

Not differentiable in DL or Math
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-012.jpg?height=74&width=252&top_left_y=1494&top_left_x=2740)
$\longrightarrow$

## Why does GD converge?

$$
\theta^{k+1}=\theta^{k}-\alpha_{k} \nabla f\left(\theta^{k}\right)
$$

Taylor expansion of $f$ about $\theta^{k}$ :

$$
f(\theta)=f\left(\theta^{k}\right)+\nabla f\left(\theta^{k}\right)^{\top}\left(\theta-\theta^{k}\right)+\mathcal{O}\left(\left\|\theta-\theta^{k}\right\|^{2}\right)
$$

Plug in $\theta^{k+1}$ :

$$
f\left(\theta^{k+1}\right)=f\left(\theta^{k}\right)-\alpha_{k}\left\|\nabla f\left(\theta^{k}\right)\right\|^{2}+\mathcal{O}\left(\alpha_{k}^{2}\right)
$$

$-\nabla f\left(\theta^{k}\right)$ is steepest descent direction. For small (cautious) $\alpha_{k}$, GD step reduces function value.

## Is GD a "descent method"?

$$
\theta^{k+1}=\theta^{k}-\alpha_{k} \nabla f\left(\theta^{k}\right)
$$

Without further assumptions, $-\nabla f\left(\theta^{k}\right)$ only gives you directional information. How far should you go? How large should $\alpha_{k}$ be?

A step of GD need not result in descent, i.e., $f\left(\theta^{k+1}\right)>f\left(\theta^{k}\right)$ is possible.

We need an assumption that ensures the first-order Taylor expansion is a good approximation within a sufficiently large neighborhood.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-014.jpg?height=1327&width=1004&top_left_y=306&top_left_x=2326)

## What can we prove?

Without further assumptions, there is no hope of finding the global minimum.

We cannot prove the function value converges to global optimum. We instead prove $\nabla f\left(\theta^{k}\right) \rightarrow 0$. Roughly speaking, this is similar, but weaker than proving that $\theta^{k}$ converges to a local minimum.*
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-015.jpg?height=771&width=1460&top_left_y=941&top_left_x=1252)
${ }^{*}$ Without further assumptions, we cannot show that $\theta^{k}$ converges to a limit, and even $\theta^{k}$ does converge to a limit, we cannot guarantee that that limit is not a saddle point or even a local maximum. Nevertheless, people commonly use the argument that $\theta^{k}$ usually converges and that it is unlikely that the

## Convergence of GD

Theorem) Assume $f: \mathbb{R}^{p} \rightarrow \mathbb{R}$ is differentiable, $\nabla f$ is $L$-Lipschitz continuous, and $\inf _{\theta \in \mathbb{R}^{p}} f(\theta)>-\infty$. Then

$$
\theta^{k+1}=\theta^{k}-\alpha \nabla f\left(\theta^{k}\right)
$$

with $\alpha \in\left(0, \frac{2}{L}\right)$ satisfies $\nabla f\left(\theta^{k}\right) \rightarrow 0$.

## Lipschitz gradient lemma

We say $\nabla f: \mathbb{R}^{p} \rightarrow \mathbb{R}^{p}$ is $L$-Lipschitz if

$$
\|\nabla f(x)-\nabla f(y)\| \leq L\|x-y\| \quad \forall x, y \in \mathbb{R}^{p} .
$$

Roughly, this means $\nabla f$ does not change rapidly. As a consequence, we can trust the first-order Taylor expansion on a non-infinitesimal neighborhood.

Lemma) Let $f: \mathbb{R}^{p} \rightarrow \mathbb{R}$ be differentiable and $\nabla f: \mathbb{R}^{p} \rightarrow \mathbb{R}^{p}$ be L-Lipschitz. Then

$$
f(\theta+\delta) \leq f(\theta)+\nabla f(\theta)^{\top} \delta+\frac{L}{2}\|\delta\|^{2} \quad \forall \theta, \delta \in \mathbb{R}^{p}
$$

$f(\theta)+\nabla f(\theta)^{\top} \delta-\frac{L}{2}\|\delta\|^{2} \leq f(\theta+\delta)$ is also true, but we do not need this other direction. Together the inequalities imply

$$
\left|f(\theta+\delta)-\left(f(\theta)+\nabla f(\theta)^{\top} \delta\right)\right| \leq \frac{L}{2}\|\delta\|^{2} \quad \forall \theta, \delta \in \mathbb{R}^{p}
$$

(I don't think this proof is important enough to cover in class, but I provide it here for completeness.)

Proof) Define $g: \mathbb{R} \rightarrow \mathbb{R}$ as $g(t)=f(\theta+t \delta)$. Then $g$ is differentiable and

$$
g^{\prime}(t)=\nabla f(\theta+t \delta)^{\top} \delta
$$

Note $g^{\prime}$ is $\left(L\|\delta\|^{2}\right)$-Lipschitz continuous since

$$
\begin{gathered}
\left|g^{\prime}\left(t_{1}\right)-g^{\prime}\left(t_{0}\right)\right|=\left|\left(\nabla f\left(\theta+t_{1} \delta\right)-\nabla f\left(\theta+t_{0} \delta\right)\right)^{\top} \delta\right| \\
\leq\left\|\nabla f\left(\theta+t_{1} \delta\right)-\nabla f\left(\theta+t_{0} \delta\right)\right\|\| \| \delta \| \\
\leq L\left\|t_{1} \delta-t_{0} \delta\right\|\|\delta\| \\
=L\|\delta\|^{2}\left|t_{1}-t_{0}\right|
\end{gathered}
$$

Finally, we conclude with

$$
\begin{gathered}
f(\theta+\delta)=g(1)=g(0)+\int_{0}^{1} g^{\prime}(t) \mathrm{d} t \\
\leq f(\theta)+\int_{0}^{1}\left(g^{\prime}(0)+L\|\delta\|^{2} t\right) \mathrm{d} t \\
=f(\theta)+\nabla f(\theta)^{\top} \delta+\frac{L}{2}\|\delta\|^{2}
\end{gathered}
$$

## Summability Lemma

Lemma) Let $V^{0}, V^{1}, \ldots \in \mathbb{R}$ and $S^{0}, S^{1}, \ldots \in \mathbb{R}$ be nonnegative sequences satisfying

$$
V^{k+1} \leq V^{k}-S^{k}
$$

for $k=0,1,2, \ldots$ Then $S^{k} \rightarrow 0$.
Key idea. $S^{k}$ measures progress (decrease) made in iteration $k$. Since $V^{k} \geq 0, V^{k}$ cannot decrease forever, so the progress (magnitude of $S^{k}$ ) must diminish to 0.
Proof) Sum the inequality from $i=0$ to $k$

$$
V^{k+1}+\sum_{i=0}^{k} S^{i} \leq V^{0}
$$

Let $k \rightarrow \infty$

$$
\sum_{i=0}^{\infty} S^{i} \leq V^{0}-\lim _{k \rightarrow \infty} V^{k} \leq V^{0}
$$

Since $\sum_{i=0}^{\infty} S^{i}<\infty, S^{i} \rightarrow 0$

## Convergence of GD: Proof

Theorem) Under the assumptions, if $\theta^{k+1}=\theta^{k}-\alpha \nabla f\left(\theta^{k}\right)$ and $\alpha \in\left(0, \frac{2}{L}\right)$, then $\nabla f\left(\theta^{k}\right) \rightarrow 0$.
Proof) Use Lipschitz gradient lemma with $\theta=\theta^{k}$ and $\delta=-\alpha \nabla f\left(\theta^{k}\right)$ to get

$$
f\left(\theta^{k+1}\right) \leq f\left(\theta^{k}\right)-\alpha\left(1-\frac{\alpha L}{2}\right)\left\|\nabla f\left(\theta^{k}\right)\right\|^{2}
$$

and

$$
\begin{array}{r}
\left(f\left(\theta^{k+1}\right)-\inf _{\theta} f(\theta)\right) \leq\left(f\left(\theta^{k}\right)-\inf _{\theta} f(\theta)\right)-\alpha\left(1-\frac{\alpha L}{2}\right)\left\|\nabla f\left(\theta^{k}\right)\right\|^{2} \\
\geq 0
\end{array}
$$

By the summability lemma, $\left\|\nabla f\left(\theta^{k}\right)\right\|^{2} \rightarrow 0$ and thus $\nabla f\left(\theta^{k}\right) \rightarrow 0$.

## Purpose of GD convergence analysis

In deep learning, the condition that $\nabla f$ is $L$-Lipschitz is usually not true*.

Rather, the purpose of these mathematical analyses is to obtain qualitative insights; this convergence proof and the exercises of hw1 are meant to provide you with intuition on the training dynamics of GD and SGD.

Because analyzing deep learning systems as is rigorously is usually difficult, people usually

- analyze modified (simplified) setups rigorously or
- analyze the full setup heuristically.

In both cases, the goal is to obtain qualitative insights, rather than theoretical guarantees.

## Finite-sum optimization problems

A finite-sum optimization problem has the structure

$$
\underset{\theta \in \mathbb{R}^{p}}{\operatorname{minimize}} \frac{1}{N} \sum_{i=1}^{N} f_{i}(\theta):=F(\theta)
$$

Finite-sum is ubiquitous in ML. $N$ usually corresponds to the number of data points.

Using GD

$$
\theta^{k+1}=\theta^{k}-\frac{\alpha_{k}}{N} \sum_{i=1}^{N} \nabla f_{i}\left(\theta^{k}\right)
$$

is impractical when $N$ is large since $\frac{1}{N} \sum_{i=1}^{N} \cdot$ takes too long to compute.

## Finite-sum $\cong$ Expectation

Although the finite-sum optimization problem has no inherent randomness, we can reformulate this problem with randomness:

$$
\operatorname{minimize}_{\theta \in \mathbb{R}^{p}} \quad \mathbb{E}_{I}\left[f_{I}(\theta)\right]
$$

where $I \sim$ Uniform $\{1, \ldots, N\}$. To see the equivalence,

$$
\mathbb{E}_{I}\left[f_{I}(\theta)\right]=\sum_{i=1}^{N} f_{i}(\theta) \mathbb{P}(I=i)=\frac{1}{N} \sum_{i=1}^{N} f_{i}(\theta)=F(\theta)
$$

## Stochastic gradient descent (SGD)

Stochastic gradient descent (SGD)

$$
\begin{gathered}
i(k) \sim \operatorname{Uniform}\{1, \ldots, N\} \\
\theta^{k+1}=\theta^{k}-\alpha_{k} \nabla f_{i(k)}\left(\theta^{k}\right)
\end{gathered}
$$

for $k=0,1, \ldots$, where $\theta^{0} \in \mathbb{R}^{p}$ is the initial point and $\alpha_{k}>0$ is the learning rate.
$\nabla f_{i(k)}\left(\theta^{k}\right)$ is a stochastic gradient of $F$ at $\theta^{k}$, i.e.,

$$
\mathbb{E}\left[\nabla f_{i(k)}\left(\theta^{k}\right)\right]=\nabla \mathbb{E}\left[f_{i(k)}\left(\theta^{k}\right)\right]=\nabla F\left(\theta^{k}\right)
$$

## GD vs. SGD

GD uses all indices $i=1, \ldots, N$ every iteration

$$
\theta^{k+1}=\theta^{k}-\frac{\alpha_{k}}{N} \sum_{i=1}^{N} \nabla f_{i}\left(\theta^{k}\right)
$$

SGD uses only a single random index $i(k)$ every iteration

$$
\begin{gathered}
i(k) \sim \text { Uniform }\{1, \ldots, N\} \\
\theta^{k+1}=\theta^{k}-\alpha_{k} \nabla f_{i(k)}\left(\theta^{k}\right)
\end{gathered}
$$

When size of the data $N$ is large, SGD is often more effective than GD.

## Digression: Randomized algorithms

A randomized algorithm utilizes artificial randomness to solve an otherwise deterministic problem.

There are problems* for which a randomized algorithm is faster than the best known deterministic algorithm.

The most famous example of this is SGD in deep learning.

## Why does SGD converge?

Plug $\theta^{k+1}$ into Taylor expansion of $F$ about $\theta^{k}$ :

$$
F\left(\theta^{k+1}\right)=F\left(\theta^{k}\right)-\alpha_{k} \nabla F\left(\theta^{k}\right)^{\top} \nabla f_{i(k)}\left(\theta^{k}\right)+\mathcal{O}\left(\alpha_{k}^{2}\right)
$$

Take expectation on both sides:

$$
\mathbb{E}_{k}\left[F\left(\theta^{k+1}\right)\right]=F\left(\theta^{k}\right)-\alpha_{k}\left\|\nabla F\left(\theta^{k}\right)\right\|^{2}+\mathcal{O}\left(\alpha_{k}^{2}\right)
$$

( $\mathbb{E}_{k}$ is expectation conditioned on $\theta^{k}$ )
$-\nabla f_{i(k)}\left(\theta^{k}\right)$ is descent direction in expectation. For small (cautious) $\alpha_{k}$, SGD step reduces function value in expectation.

## Variants of SGD for finite-sum problems

Consider

$$
\operatorname{minimize}_{\theta \in \mathbb{R}^{p}} \frac{1}{N} \sum_{i=1}^{N} f_{i}(\theta)
$$

SGD can be generalized to

$$
\theta^{k+1}=\theta^{k}-\alpha_{k} g^{k}
$$

where $g^{k}$ is a stochastic gradient. The choice $g^{k}=\nabla f_{i(k)}\left(\theta^{k}\right)$ is just one option.

## Sampling with replacement lemma

Lemma) Let $X_{1}, \ldots, X_{N} \in \mathbb{R}^{p}$ be given (non-random) vectors. Let $\frac{1}{N} \sum_{i=1}^{N} X_{i}=\mu$. Let $i(1), \ldots, i(B) \subseteq\{1, \ldots, N\}$ be random indices. Then

$$
\mathbb{E} \frac{1}{B} \sum_{b=1}^{B} X_{i(b)}=\mu
$$

Proof)

$$
\mathbb{E} \frac{1}{B} \sum_{b=1}^{B} X_{i(b)}=\frac{1}{B} \sum_{b=1}^{B} \mathbb{E} X_{i(b)}=\frac{1}{B} \sum_{b=1}^{B} \mu=\mu
$$

## Minibatch SGD with replacement

Minibatch SGD with replacement

$$
\begin{gathered}
i(k, 1), \ldots, i(k, B) \sim \text { Uniform }\{1, \ldots, N\} \\
\theta^{k+1}=\theta^{k}-\frac{\alpha_{k}}{B} \sum_{b=1}^{B} \nabla f_{i(k, b)}\left(\theta^{k}\right)
\end{gathered}
$$

To clarify, we sample $B$ out of $N$ indices with replacement, i.e., the same index can be sampled multiple times.

By previous lemma, $\frac{1}{B} \sum_{b=1}^{B} \nabla f_{i(k, b)}\left(\theta^{k}\right)$ is a stochastic gradient of $F$ at $\theta^{k}$

## Random permutations

A permutation $\sigma$ is a list of length $N$ containing integers $1, \ldots, N$ all exactly once. We write $S_{n}$ for the set of permutations of length $N$.

There are $N$ ! possible permutations of length $N$.

A random permutation is a permutation chosen randomly with uniform probability; each of the $N!$ permutations are chosen with probability $\frac{1}{N}$ :

# Digression: 0-based indexing and random permutations in Python 

In Python, generate random permutations with

```
np.random.permutation(np.arange(N))
```

In Python, array indices start at 0, although in math and in human language, counting starts at 1 . We use permutations containing $0,1, \ldots, N-1$ in our Python code.

## Sampling without replacement lemma

Lemma) Let $X_{1}, \ldots, X_{N} \in \mathbb{R}^{p}$ be given (non-random) vectors. Let $\frac{1}{N} \sum_{i=1}^{N} X_{i}=\mu$. Let $\sigma$ be a random permutation. Then

$$
\mathbb{E} \frac{1}{B} \sum_{b=1}^{B} X_{\sigma(b)}=\mu
$$

Proof)

$$
\mathbb{E} \frac{1}{B} \sum_{b=1}^{B} X_{\sigma(b)}=\frac{1}{B} \sum_{b=1}^{B} \mathbb{E} X_{\sigma(b)}=\frac{1}{B} \sum_{b=1}^{B} \mu=\mu
$$

## Minibatch SGD without replacement

Minibatch SGD without replacement

$$
\begin{gathered}
\sigma^{k} \sim \operatorname{permutation}(N) \\
\theta^{k+1}=\theta^{k}-\frac{\alpha_{k}}{B} \sum_{b=1}^{B} \nabla f_{\sigma^{k}(b)}\left(\theta^{k}\right)
\end{gathered}
$$

We assume $B \leq N$. To clarify, we sample $B$ out of $N$ indices without replacement, i.e., the same index cannot be sampled multiple times.

By previous lemma, $\frac{1}{B} \sum_{b=1}^{B} \nabla f_{\sigma^{k}(b)}\left(\theta^{k}\right)$ is a stochastic gradient of $F$ at $\theta^{k}$.

## How to choose batch size $B$ ?

Note $B=1$ minibatch SGD becomes SGD.

Mathematically (measuring performance per iteration)

- Use large batch is when noise/randomness is large.
- Use small batch is when noise/randomness is small.

Practically (measuring performance per unit time)

- Large batch allows more efficient computation on GPUs.
- Often best to increase batch size up to the GPU memory limit.


## GD and SGD without differentiability

In DL, SGD is applied to nice continuous but non-differentiable* functions that are differentiable almost everywhere.

In this case, if we choose $\theta^{0} \in \mathbb{R}^{n}$ randomly and run

$$
\theta^{k+1}=\theta^{k}-\alpha_{k} \nabla f\left(\theta^{k}\right)
$$

the algorithm is usually well-defined, i.e., $\theta^{k}$ never hits a point of non-differentiability.

With a proof or not, GD and SGD are applied to non-differentiable minimization in ML. The absence of differentiability ${ }^{*}$ does not seem to cause serious problems.

## Cyclic SGD

Consider the sequence of indices

$$
\{\bmod (k, N)+1\}_{k=0,1, \ldots}=1,2, \ldots, N, 1,2, \ldots, N, \ldots
$$

Here, $\bmod (k, N)$ is the remainder of $k$ when divided by $N$. In Python, this is written with $k \% N$.

Cyclic SGD:

$$
\theta^{k+1}=\theta^{k}-\alpha_{\mathbf{k}} \nabla f_{\bmod (k, N)+1}\left(\theta^{k}\right)
$$

To clarify, this samples the indices in a (deterministic) cyclic order.

## Cyclic (mini-batch) SGD

Strictly speaking, cyclic SGD is not an instance of SGD as unbiased estimation property lost.

Advantage:

- Uses all indices (data) every $N$ iterations.

Disadvantage:

- Worse than SGD in some cases, theoretically and empirically.
- In DL, neural networks can learn to anticipate cyclic order.


## Shuffled Cyclic SGD

Shuffled Cyclic SGD:

$$
\theta^{k+1}=\theta^{k}-\left.\alpha_{k} \nabla f\right|_{\left.\left.\right|^{\frac{k}{N}}\right|_{(\bmod (k, N)+1)}}\left(\theta^{k}\right)
$$

where $\sigma^{0}, \sigma^{1}, \ldots$ is a sequence of random permutations, i.e., we shuffle the order every cycle. Again, strictly speaking, shuffled cyclic SGD is not an instance of SGD as unbiased estimation property lost.

Advantages :

- Uses all indices (data) every $N$ iterations.
- Neural network cannot learn to anticipate data order.
- Empirically best performance.

Disadvantages:

- Theory not as strong as regular SGD.


## Which variant of SGD to use?

Theoretical comparison of SGD variants:

- Not that easy.
- Result does not strongly correlate with practical performance in DL.

In DL, the most common choice is

- shuffled cyclic minibatch SGD (without replacement) and
- batchsize $B$ is as large as possible within the GPU memory limit.

One can generally consider this to be the default option.

## Epoch in finite-sum optimization and machine learning training

An epoch is loosely defined as the unit of optimization or training progress of processing all indices or data once.

- 1 iteration of GD constitutes an epoch.
- $N$ iterations of SGD, cyclic SGD, or shuffled cyclic SGD constitute an epoch.
- $N / B$ iterations of minibatch SGD constitute an epoch.

Epoch is often a convenient unit for counting iterations compared to directly counting the iteration number.

## SGD with general expectation

Consider an optimization problem with its objective defined with a general expectation

$$
\operatorname{minimize}_{\theta \in \mathbb{R}^{p}} \quad \mathbb{E}_{\omega}\left[f_{\omega}(\theta)\right]:=F(\theta)
$$

Here, $\omega$ is a random variable. We will encounter these expectations (non-finite sum) when we talk about generative models.

For this setup, the SGD algorithm is

$$
\theta^{k+1}=\theta^{k}-\alpha_{k} \nabla f_{\omega^{k}}\left(\theta^{k}\right)
$$

where $\omega^{0}, \omega^{1}, \ldots$ are IID random samples of $\omega$. If $\nabla_{\theta} \mathbb{E}_{\omega}\left[f_{\omega}(\theta)\right]=\mathbb{E}_{\omega}\left[\nabla_{\theta} f_{\omega}(\theta)\right]$, then $\nabla f_{\omega^{k}}\left(\theta^{k}\right)$ is a stochastic gradient of $F(\theta)$ at $\theta^{k}$. (Make sure you understand why the previous SGD for the finite-sum setup is a special case of this.)

GD for this setup is

$$
\theta^{k+1}=\theta^{k}-\alpha_{k} \mathbb{E}_{\omega}\left[\nabla_{\theta} f_{\omega}\left(\theta^{k}\right)\right]
$$

However, if the expectation is difficult to compute GD is impractical and SGD is preferred.

# Chapter 2: <br> Shallow Neural Networks to Multilayer Perceptrons 

Mathematical Foundations of Deep Neural Networks

Spring 2024
Department of Mathematical Sciences
Ernest K. Ryu
Seoul National University

## Supervised learning setup

We have data $X_{1}, \ldots, X_{N} \in X$ and corresponding labels $Y_{1}, \ldots, Y_{N} \in \mathcal{Y}$.

Example) $X_{i}$ is the $i$ th email and $Y_{i} \in\{-1,+1\}$ denotes whether $X_{i}$ is a spam email.
Example) $X_{i}$ is the $i$ th image and $Y_{i} \in\{0, \ldots, 9\}$ denotes handwritten digit.

Assume there is a true unknown function

$$
f_{\star}: x \rightarrow y
$$

mapping data to its label. In particular, $Y_{i}=f_{\star}\left(X_{i}\right)$ for $i=1, \ldots, N$.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-044.jpg?height=568&width=862&top_left_y=846&top_left_x=2468)

The goal of supervised learning is to use $X_{1}, \ldots, X_{N}$ and $Y_{1}, \ldots, Y_{N}$ to find $f \approx f_{\star}$.

## Formulating the right objective

The goal of "finding $f \approx f_{\star}$ " must be further quantified.

Assume a loss function such that $\ell\left(y_{1}, y_{2}\right)=0$ if $y_{1}=y_{2}$ and $\ell\left(y_{1}, y_{2}\right)>0$ if $y_{1} \neq y_{2}$.

Attempt 1)

$$
\underset{f}{\operatorname{minimize}} \sup _{x \in \mathcal{X}} \ell\left(f(x), f_{\star}(x)\right)
$$

Problems:

- There is a trivial solution $f=f_{\star}$.
- Minimization over all functions $f$ is in general algorithmically intractable ${ }^{1}$. How would one represent a $f$ on a computer?


## Formulating the right objective

Attempt 2) Restrict search to a class of parametrized functions $f_{\theta}(x)$ where $\theta \in \Theta \subseteq \mathbb{R}^{p}$, i.e., only consider $f \in\left\{f_{\theta} \mid \theta \in \Theta\right\}$ where $\Theta \subseteq \mathbb{R}^{p}$. Then solve

$$
\operatorname{minimize}_{f \in\left\{f_{\theta} \mid \theta \in \Theta\right\}} \sup _{x \in \mathcal{X}} \ell\left(f(x), f_{\star}(x)\right)
$$

which is equivalent to

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \sup _{x \in \mathcal{X}} \ell\left(f_{\theta}(x), f_{\star}(x)\right)
$$

Problems:

- The supremum $\sup _{x \in \mathcal{X}}$ is computationally inconvenient to deal with.
- Objective is too pessimistic. We do not need to do well all the time, we just need to do well on average.


## Formulating the right objective

Attempt 3) Take a finite sample ${ }^{*} X_{1}, \ldots, X_{N} \in \mathcal{X}$ and corresponding labels $Y_{1}, \ldots, Y_{N} \in \mathcal{Y}$. Then solve

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \frac{1}{N} \sum_{i=1}^{N} \ell\left(f_{\theta}\left(X_{i}\right), f_{\star}\left(X_{i}\right)\right)
$$

which is equivalent to

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \frac{1}{N} \sum_{i=1}^{N} \ell\left(f_{\theta}\left(X_{i}\right), Y_{i}\right)
$$

This is the standard form of the optimization problem (except regularizers) we consider in the supervised learning. We will talk about regularizers later.

## Aside: Minimum vs. infimum

We clarify terminology.

- "Minimize": Used to specify an optimization problem.
- "Minimizer": A solution to a minimization problem.
- "Minimum": Used to specify the smallest objective value and asserts a minimizer exists.
- "Infimum": Used to specify the limiting smallest objective value, but a minimizer may not exist.

Analogous definitions with "maximize", "maximizer", "maximum", and "supremum"

## Training is optimization

In machine learning, the anthropomorphized word "training" refers to solving an optimization problem such as

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \frac{1}{N} \sum_{i=1}^{N} \ell\left(f_{\theta}\left(X_{i}\right), Y_{i}\right)
$$

In most cases, SGD or variants of SGD are used.

We call $f_{\theta}$ the machine learning model or the neural network.

## Least-squares regression

$\ln \mathrm{LS}, X=\mathbb{R}^{p}, \mathcal{Y}=\mathbb{R}, \Theta=\mathbb{R}^{p}, f_{\theta}(x)=x^{\top} \theta$, and $\ell\left(y_{1}, y_{2}\right)=\frac{1}{2}\left(y_{1}-y_{2}\right)^{2}$.
So we solve

$$
\operatorname{minimize}_{\theta \in \mathbb{R}^{p}} \frac{1}{N} \sum_{i=1}^{N} \frac{1}{2}\left(f_{\theta}\left(X_{i}\right)-Y_{i}\right)^{2}=\frac{1}{N} \sum_{i=1}^{N} \frac{1}{2}\left(X_{i}^{\top} \theta-Y_{i}\right)^{2}=\frac{1}{2 N}\|X \theta-Y\|^{2}
$$

where $X=\left[\begin{array}{c}X_{1}^{\top} \\ \vdots \\ X_{N}^{\top}\end{array}\right]$ and $Y=\left[\begin{array}{c}Y_{1} \\ \vdots \\ Y_{N}\end{array}\right]$.

The model $f_{\theta}(x)=x^{\top} \theta$ is a shallow neural network. (The terminology will makes sense when contrasted with deep neural networks.)

## Binary classification and linear separability

In binary classification, we have $X=\mathbb{R}^{p}$ and $\mathcal{Y}=\{-1,+1\}$.

The data is linearly separable if there is a hyperplane defined by ( $a_{\text {true }}, b_{\text {true }}$ ) such that

$$
y=\left\{\begin{array}{cl}
1 & \text { if } a_{\text {true }}^{\top} x+b_{\text {true }}>0 \\
-1 & \text { otherwis. }
\end{array}\right.
$$

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-051.jpg?height=807&width=2712&top_left_y=1041&top_left_x=274)

## Linear classification

Consider linear (affine) models

$$
f_{a, b}(x)= \begin{cases}+1 & \text { if } a^{\top} x+b>0 \\ -1 & \text { otherwise }\end{cases}
$$

Consider the loss function

$$
\ell\left(y_{1}, y_{2}\right)=\frac{1}{2}\left|1-y_{1} y_{2}\right|= \begin{cases}0 & \text { if } y_{1}=y_{2} \\ 1 & \text { if } y_{1} \neq y_{2}\end{cases}
$$

The optimization problem

$$
\operatorname{minimize}_{a \in \mathbb{R}^{p}, b \in \mathbb{R}} \frac{1}{N} \sum_{i=1}^{N} \ell\left(f_{a, b}\left(X_{i}\right), Y_{i}\right)
$$

has a solution with optimal value 0 when the data is linearly separable.
Problem: Optimization problem is discontinuous and thus cannot be solved with SGD.

## Relaxing into continuous formulation

Even if the underlying function or phenomenon to approximate is discontinuous, the model needs to be continuous* in its parameters. The loss function also needs to be continuous. (The prediction need not be continuous.)

We consider a relaxation, is a continuous proxy of the discontinuous thing. Specifically, consider

$$
f_{a, b}(x)=a^{\top} x+b
$$

Once trained, $f_{a, b}(x)>0$ means the neural network is predicting $y=+1$ to be "more likely", and $f_{a, b}(x)<0$ means the neural network is predicting $y=-1$ to be "more likely".

Therefore, we train the model to satisfy

$$
Y_{i} f_{a, b}\left(X_{i}\right)>0 \text { for } i=1, \ldots, N .
$$

## Relaxing into continuous formulation

Problem with strict inequality $Y_{i} f_{a, b}\left(X_{i}\right)>0$ :

- Strict inequality has numerical problems with round-off error.
- The magnitude $\left|f_{a, b}(x)\right|$ can be viewed as the confidence* of the prediction, but having a small positive value for $Y_{i} f_{a, b}\left(X_{i}\right)$ indicates small confidence of the neural network.

We modify our model's desired goal to be $Y_{i} f_{a, b}\left(X_{i}\right) \geq 1$.

## Support vector machine (SVM)

To train the neural network to satisfy

$$
0 \geq 1-Y_{i} f_{a, b}\left(X_{i}\right) \text { for } i=1, \ldots, N .
$$

we minimize the excess positive component of the RHS

$$
\operatorname{minimize}_{a \in \mathbb{R}^{p}, b \in \mathbb{R}} \frac{1}{N} \sum_{i=1}^{N} \max \left\{0,1-Y_{i} f_{a, b}\left(X_{i}\right)\right\}
$$

which is equivalent to

$$
\operatorname{minimize}_{a \in \mathbb{R}^{P}, b \in \mathbb{R}} \frac{1}{N} \sum_{i=1}^{N} \max \left\{0,1-Y_{i}\left(a^{\top} X_{i}+b\right)\right\}
$$

If the optimal value is 0 , then the data is linearly separable.

## Support vector machine (SVM)

This formulation is called the support vector machine (SVM)*

$$
\operatorname{minimize}_{a \in \mathbb{R}^{p}, b \in \mathbb{R}} \frac{1}{N} \sum_{i=1}^{N} \max \left\{0,1-Y_{i}\left(a^{\top} X_{i}+b\right)\right\}
$$

It is also common to add a regularizer

$$
\operatorname{minimize}_{a \in \mathbb{R}^{p}, b \in \mathbb{R}} \frac{1}{N} \sum_{i=1}^{N} \max \left\{0,1-Y_{i}\left(a^{\top} X_{i}+b\right)\right\}+\frac{\lambda}{2}\|a\|^{2}
$$

We will talk about regularizers later.

## Prediction with SVM

Once the SVM is trained, make predictions with

$$
\operatorname{sign}\left(f_{a, b}(x)\right)=\operatorname{sign}\left(a^{\top} x+b\right)
$$

when $f_{a, b}(x)=0$, we assign $\operatorname{sign}\left(f_{a, b}(x)\right)$ arbitrarily.

Note that the prediction is discontinuous, but predictions are in $\{-1,+1\}$ so it must be discontinuous.

If $\sum_{i=1}^{N} \max \left\{0,1-Y_{i} f_{a, b}\left(X_{i}\right)\right\}=0$, then $\operatorname{sign}\left(f_{a, b}\left(X_{i}\right)\right)=Y_{i}$ for $i=1, \ldots, N$, i.e., the neural network predicts the known labels perfectly. (Make sure you understand this.) Of course, it is a priori not clear how accurate the prediction will be for new unseen data.

## SVM is a relaxation

Directly minimizing the prediction error on the data is

$$
\operatorname{minimize}_{a \in \mathbb{R}^{P}, b \in \mathbb{R}} \frac{1}{N} \sum_{i=1}^{N} \frac{1}{2}\left|1-Y_{i} \operatorname{sign}\left(f_{a, b}\left(X_{i}\right)\right)\right|
$$

The optimization we instead solve is

$$
\operatorname{minimize}_{a \in \mathbb{R}^{p}, b \in \mathbb{R}} \frac{1}{N} \sum_{i=1}^{N} \max \left\{0,1-Y_{i} f_{a, b}\left(X_{i}\right)\right\}
$$

Let the optimal values be $p_{1}^{\star}$ and $p_{2}^{\star}$. Again, SVM is of as a relaxation of the first. The two are not equivalent. (An equivalent formulation is not referred to as a relaxation.)

- It is possible to show ${ }^{\star}$ that $p_{1}^{\star}=0$ if and only if $p_{2}^{\star}=0$.
- If $p_{1}^{\star}>0$ and $p_{2}^{\star}>0$, a solution to the first problem need not correspond to a solution to the second problem, i.e., there solutions may be completely different.


## Relaxed supervised learning setup

We relax the supervised learning setup to predict probabilities, rather than make point predictions*. So, labels are generated based on data, perhaps randomly. Consider data $X_{1}, \ldots, X_{N} \in \mathcal{X}$ and labels $Y_{1}, \ldots, Y_{N} \in \mathcal{Y}$. Assume there exists a function

$$
f_{\star}: \mathcal{X} \rightarrow \mathcal{P}(\mathcal{Y})
$$

where $\mathcal{P}(\mathcal{Y})$ denotes the set of probability distributions on $\mathcal{Y}$.
Assume the generation of $Y_{i}$ given $X_{i}$ is independent of $Y_{j}$ and $X_{j}$ for $j \neq i$.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-059.jpg?height=545&width=567&top_left_y=650&top_left_x=2693)

Example) $f(X)=\left[\begin{array}{l}0.8 \\ 0.2\end{array}\right]$ in dog vs. cat classifier.
Example) An email saying "Buy this thing at our store!" may be spam to some people, but it may not be spam to others.

The relaxed SL setup is more general and further realistic.

## KL-divergence

Let $p, q \in \mathbb{R}^{n}$ represent probability masses, i.e., $p_{i} \geq 0$ for $i=1, \ldots, n$ and $\sum_{i=1}^{n} p_{i}=1$ and the same for $q$. The Kullback-Leibler-divergence (KL-divergence) from $q$ to $p$ is

$$
D_{\mathrm{KL}}(p \| q)=\sum_{i=1}^{n} p_{i} \log \left(\frac{p_{i}}{q_{i}}\right)=-\sum_{i=1}^{n} p_{i} \log \left(q_{i}\right)+\sum_{i=1}^{n} p_{i} \log \left(p_{i}\right)
$$

Properties:

$$
\begin{array}{ll}
\quad=H(p, q) & =-H(p) \\
\text { cross entropy of } q & =-H \\
\text { relative to } p & \text { entropy of } p
\end{array}
$$

- Not symmetric, i.e., $D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)$.
- $D_{\mathrm{KL}}(p \| q)>0$ if $p \neq q$ and $D_{\mathrm{KL}}(p \| q)=0$ if $p=q$.
- $D_{\mathrm{KL}}(p \| q)=\infty$ is possible. (Further detail on the next slide.)

Often used as a "distance" between $p$ and $q$ despite not being a metric.

## KL-divergence

$$
D_{\mathrm{KL}}(p \| q)=\sum_{i=1}^{n} p_{i} \log \left(\frac{p_{i}}{q_{i}}\right)
$$

Clarification: Use the convention

- $0 \log \left(\frac{0}{0}\right)=0\left(\right.$ when $\left.p_{i}=q_{i}=0\right)$
- $0 \log \left(\frac{0}{q_{i}}\right)=0$ if $q_{i}>0$
- $p_{i} \log \left(\frac{p_{i}}{0}\right)=\infty$ if $p_{i}>0$

Probabilistic interpretation:

$$
D_{\mathrm{KL}}(p \| q)=\mathbb{E}_{I}\left[\log \left(\frac{p_{I}}{q_{I}}\right)\right]
$$

with the random variable $I$ such that $\mathbb{P}(I=i)=p_{i}$.

## Empirical distribution for binary classification

In basic binary classification, define the empirical distribution

$$
\mathcal{P}(y)= \begin{cases}{\left[\begin{array}{l}
1 \\
0
\end{array}\right]} & \text { if } y=-1 \\
{\left[\begin{array}{l}
0 \\
1
\end{array}\right]} & \text { if } y=+1\end{cases}
$$

More generally, the empirical distribution describes the data we have seen. In this context, we have only seen one label per datapoint, so our empirical distributions are one-hot vectors.
(If there are multiple annotations per data point $x$ and they don't agree, then the empirical distribution may not be one-hot vectors. For example, given the same email, some users may flag it as spam while others consider it useful information.)

## Logistic regression

Logistic regression (LR), is another model for binary classification:

1. Use the model

$$
f_{a, b}(x)=\left[\begin{array}{c}
\frac{1}{1+e^{a^{\top} x+b}} \\
\frac{e^{a^{\top} x+b}}{1+e^{a^{\top} x+b}}
\end{array}\right]=\left[\begin{array}{c}
\frac{1}{1+e^{a^{\top} x+b}} \\
\frac{1}{1+e^{-\left(a^{\top} x+b\right)}}
\end{array}\right]=\mathbb{P}(y=-1)
$$

2. Minimize KL-Divergence (or cross entropy) from the model $f_{a, b}\left(X_{i}\right)$ output probabilities to the empirical distribution $\mathcal{P}\left(Y_{i}\right)$.

$$
\operatorname{minimize}_{a \in \mathbb{R}^{p}, b \in \mathbb{R}} \sum_{i=1}^{N} D_{\mathrm{KL}}\left(\mathcal{P}\left(Y_{i}\right) \| f_{a, b}\left(X_{i}\right)\right)
$$

## Logistic regression

Note:

$$
\begin{gathered}
\operatorname{minimize}_{a \in \mathbb{R}^{p}, b \in \mathbb{R}} \sum_{i=1}^{N} D_{\mathrm{KL}}\left(\mathcal{P}\left(Y_{i}\right) \| f_{a, b}\left(X_{i}\right)\right) \\
\mathbb{\Downarrow} \\
\operatorname{minimize}_{a \in \mathbb{R}^{p}, b \in \mathbb{R}} \sum_{i=1}^{N} H\left(\mathcal{P}\left(Y_{i}\right), f_{a, b}\left(X_{i}\right)\right)+(\text { terms independent of } a, b) \\
\operatorname{minimize}_{a \in \mathbb{R}^{p}, b \in \mathbb{R}} \sum_{i=1}^{N} \log \left(1+\exp \left(-Y_{i}\left(a^{\top} X_{i}+b\right)\right)\right) \\
\operatorname{minimize}_{a \in \mathbb{R}^{p}, b \in \mathbb{R}} \frac{1}{N} \sum_{i=1}^{N} \ell\left(Y_{i}\left(a^{\top} X_{i}+b\right)\right)
\end{gathered}
$$

where $\ell(z)=\log \left(1+e^{-z}\right)$.

## Point prediction with logistic regression

When performing point prediction with $\mathrm{LR}, a^{\top} x+b>0$ means $\mathbb{P}(y=+1)>0.5$ and vice versa.

Once the LR is trained, make predictions with

$$
\operatorname{sign}\left(a^{\top} x+b\right)
$$

when $a^{\top} x+b=0$, we assign $\operatorname{sign}\left(a^{\top} x+b\right)$ arbitrarily. This is the same as SVM.

Again, it is a priori not clear how accurate the prediction will be for new unseen data.

## SVM vs. LR

Both support vector machine and logistic regression can be written as

$$
\operatorname{minimize}_{a \in \mathbb{R}^{p}, b \in \mathbb{R}} \frac{1}{N} \sum_{i=1}^{N} \ell\left(Y_{i}\left(a^{\top} X_{i}+b\right)\right)
$$

- SVM uses $\ell(z)=\max \{0,1-z\}$. Obtained from relaxing the discontinuous prediction loss.
- LR uses $\ell(z)=\log \left(1+e^{-z}\right)$. Obtained from relaxing the supervised learning setup from predicting the label to predicting the label probabilities.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-066.jpg?height=609&width=1013&top_left_y=1243&top_left_x=2007)


## SVM vs. LR

SVM and LR are both "linear" classifiers:

- Decision boundary $a^{\top} x+b=0$ is linear.
- Model completely ignores information perpendicular to $a$.

LR naturally generalizes to multi-class classification via softmax regression. Generalizing SVM to multi-class classification is trickier and less common.

## Estimation vs. Prediction

Finding $f \approx f_{\star}$ for unknown

$$
f_{\star}: \mathcal{X} \rightarrow \mathcal{P}(\mathcal{Y})
$$

is called estimation*. When we consider a parameterized model $f_{\theta}$, finding $\theta$ is the estimation. However, estimation is usually not the end goal.

The end goal is prediction. It is to use $f_{\theta} \approx f_{\star}$ on new data $X_{1}^{\prime}, \ldots, X_{M}^{\prime} \in X$ to find labels $Y_{1}^{\prime}, \ldots, Y_{M}^{\prime} \in \mathcal{Y}$.

## Is prediction possible?

In the worst hypotheticals, prediction is impossible.

- Even though smoking is harmful for every other human being, how can we be $100 \%$ sure that this one person is not a mutant who benefits from the chemicals of a cigarette?
- Water freezes at $0^{\circ}$, but will the same be true tomorrow? How can we be $100 \%$ sure that the laws of physics will not suddenly change tomorrow?

Of course, prediction is possible in practice.

Theoretically, prediction requires assumptions on the distribution of $X$ and the model of $f_{\star}$ is needed. This is in the realm of statistics of statistical learning theory.

For now, we will take the view that if we predict known labels of the training data, we can reasonably hope to do well on the new data. (We will discuss the issue of generalization and overfitting later.)

## Training data vs. test data

When testing a machine learning model, it is essential that one separates the training data with the test data.

In other classical disciplines using data, one performs a statistical hypothesis test to obtain a $p$-value. In ML, people do not do that.

The only sure way to ensure that the model is doing well is to assess its performance on new data.

Usually, training data and test data is collected together. This ensures that they have the same statistical properties. The assumption is that this test data will be representative of the actual data one intends to use machine learning on.

## Aside: Maximum likelihood estimation $\cong$ minimizing KL divergence

Consider the setup where you have IID discrete random variables $X_{1}, \ldots, X_{N}$ that can take values $1, \ldots, k$. We model the probability masses with $\mathbb{P}_{\theta}(X=1), \ldots, \mathbb{P}_{\theta}(X=k)$. The maximum likelihood estimation (MLE) is obtained by solving

$$
\underset{\theta}{\operatorname{maximize}} \frac{1}{N} \sum_{i=1}^{N} \log \left(\mathbb{P}_{\theta}\left(X_{i}\right)\right)
$$

Next, define

$$
f_{\theta}=\left[\begin{array}{c}
\mathbb{P}_{\theta}(X=1) \\
\vdots \\
\mathbb{P}_{\theta}(X=k)
\end{array}\right], \quad \mathcal{P}\left(X_{1}, \ldots, X_{N}\right)=\frac{1}{N}\left[\begin{array}{c}
\#\left(X_{i}=1\right) \\
\vdots \\
\#\left(X_{i}=k\right)
\end{array}\right] .
$$

Then MLE is equivalent to minimizing the KL divergence from the model to the empirical distribution.

|  | MLE |
| :---: | :---: |
| $\underset{\theta}{\operatorname{minimize}}$ | $\mathbb{\\|}$ |
| $\left.\underset{\theta}{\operatorname{minimize}}\left(X_{1}, \ldots, X_{N}\right), f_{\theta}\right)$ |  |
|  | $D_{\mathrm{KL}}\left(\mathcal{P}\left(X_{1}, \ldots, X_{N}\right) \\| f_{\theta}\right)$ |

## Aside: Maximum likelihood estimation $\cong$ minimizing KL divergence

One can also derive LR equivalently as the MLE.

Generally, one can view the MLE as minimizing the KL divergence between the model and the empirical distribution. (For continuous random variables like the Gaussian, this requires extra work, since we haven't defined the KL divergence for continuous random variables.)

In deep learning, the distance measure need not be KL divergence.

Dataset: MNIST

Images of hand-written digits with $28 \times 28=784$ pixels and integervalued intensity between 0 and 255 . Every digit has a label in $\{0,1, \ldots, 8,9\}$.

70,000 images (60,000 for training 10,000 testing) of 10 almost balanced classes.

One of the simplest data set used in machine learning.

| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 |

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-074.jpg?height=1405&width=1204&top_left_y=301&top_left_x=1099)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-075.jpg?height=1874&width=2484&top_left_y=0&top_left_x=731)

## Dataset: MNIST

The USA government needed a standardized test to assess handwriting recognition software being sold to the government. So the NIST (National Institute of Standards and Technology) created the dataset in the 1990s. In 1990, NIST Special Database 1 distributed on CD-ROMs by mail. NIST SD 3 (1992) and SD 19 (1995) were improvements.

Humans were instructed to fill out handwriting sample forms.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-076.jpg?height=1520&width=1230&top_left_y=99&top_left_x=2081)

## Dataset: MNIST

However, humans cannot be trusted to follow instructions, so a lab technician performed "human ground truth adjudication".

In 1998, Man LeCun, Corinna Cortes, Christopher J. C. Barges took the NIST dataset and modified it to create the MNIST dataset.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-077.jpg?height=1721&width=1719&top_left_y=0&top_left_x=1611)

## Role of Datasets in ML Research

An often underappreciated contribution.

Good datasets play a crucial role in driving progress in ML research.

Thinking about the dataset is the essential first step of understanding the feasibility of a ML task.

Accounting for the cost of producing datasets and leveraging freely available data as much as possible (semi-supervised learning) is a recent trend in machine learning.

## Dataset: CIFAR10

$60,00032 \times 32$ color images in 10 (perfectly) balanced classes.
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-079.jpg?height=1460&width=1451&top_left_y=299&top_left_x=1877)

## Dataset: CIFAR10

In 2008, a MIT and NYU team created the 80 million tiny images data set by searching on Google, Flickr, and Altavista for every non-abstract English noun and downscaled the images to $32 \times 32$. The search term provided an unreliable label for the image. This dataset was not very easy to use since the classes were too numerous.

In 2009, Alex Krizhevsky published the CIFAR10, by distilling just a few classes and cleaning up the labels. Students were paid to verify the labels.

The dataset was named CIFAR-10 after the funding agency Canadian Institute For Advanced Research.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-080.jpg?height=1281&width=1465&top_left_y=295&top_left_x=1869) There is also a CIFAR-100 with 100 classes.

## Shallow learning with PyTorch

PyTorch demo

We follow the following steps

1. Load data
2. Define model
3. Miscellaneous setup

- Instantiate model
- Choose loss function
- Choose optimizer

4. Train with SGD

- Clear previously computed gradients
- Compute forward pass
- Compute gradient via backprop
- SGD update

5. Evaluate trained model
6. Visualize results of trained model

## LR as a 1-layer neural network

In LR, we solve

$$
\operatorname{minimize}_{a \in \mathbb{R}^{p}, b \in \mathbb{R}} \frac{1}{N} \sum_{i=1}^{N} \ell\left(f_{\theta}\left(X_{i}\right), Y_{i}\right)
$$

where $\ell\left(y_{1}, y_{2}\right)=\log \left(1+e^{-y_{1} y_{2}}\right)$ and $f_{\theta}$ is linear.

We can view $f_{\theta}(x)=0=a^{\top} x+b$ as a 1-layer (shallow) neural network.

Output
layer
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-082.jpg?height=736&width=958&top_left_y=610&top_left_x=2187)

## Linear deep networks make no sense

What happens if we stack multiple linear layers?
Problem: This is pointless because composition of linear functions is linear.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-083.jpg?height=915&width=2702&top_left_y=746&top_left_x=316)

## Deep neural networks with nonlinearities

Solution: use a nonlinear activation function $\sigma$ to inject nonlinearities.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-084.jpg?height=924&width=2965&top_left_y=746&top_left_x=304)

## Common activation functions

Rectified Linear Unit (ReLU)
$\operatorname{ReLU}(z)=\max (z, 0)$

Sigmoid
$\operatorname{Sigmoid}(z)=\frac{1}{1+e^{-z}}$

Hyperbolic tangent
$\tanh (z)=\frac{1-e^{-2 z}}{1+e^{-2 z}}$

## Multilayer perceptron (MLP)

The multilayer perceptron, also called fully connected neural network, has the form

$$
\begin{aligned}
y_{L}= & W_{L} y_{L-1}+b_{L} \\
y_{L-1}= & \sigma\left(W_{L-1} y_{L-2}+b_{L-1}\right) \\
& \vdots \\
y_{2}= & \sigma\left(W_{2} y_{1}+b_{2}\right) \\
y_{1}= & \sigma\left(W_{1} x+b_{1}\right),
\end{aligned}
$$

where $x \in \mathbb{R}^{n_{0}}, W_{\ell} \in \mathbb{R}^{n_{\ell} \times n_{\ell-1}}, b_{\ell} \in \mathbb{R}^{n_{\ell}}$, and $n_{L}=1$. To clarify, $\sigma$ is applied element-wise.

## MLP for CIFAR10 binary classification

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-087.jpg?height=866&width=2096&top_left_y=469&top_left_x=296)
activation function $\sigma=\operatorname{ReLU}$

PyTorch demo

## Linear layer: Formal definition

Input tensor: $X \in \mathbb{R}^{B \times n}, B$ batch size, $n$ number of indices.
Output tensor: $Y \in \mathbb{R}^{B \times m}, B$ batch size, $m$ number of indices.

With weight $A \in \mathbb{R}^{m \times n}$, bias $b \in \mathbb{R}^{m}, k=1, \ldots B$, and $i=1, \ldots, m$ :

$$
Y_{k, i}=\sum_{j=1}^{n} A_{i, j} X_{k, j}+b_{i}
$$

Operation is independent across elements of the batch.

If bias=False, then $b=0$.

## Weight initialization

Remember, SGD is

$$
\theta^{k+1}=\theta^{k}-\alpha g^{k}
$$

where $\theta^{0} \in \mathbb{R}^{p}$ is an initial point.

In nice (convex) optimization problems, the initial point $\theta^{0}$ is not important; you converge to the global solution no matter how you initialize.

In deep learning, it is very important to initialize $\theta^{0}$ well. In fact, $\theta^{0}=0$ is a terrible idea.

Example) With an MLP with ReLU activations functions, if all weights and biases are initialized to be zero, then only the output layer's bias is trained and all other parameters do not move. So the training is stuck at a trivial network setting with $f_{\theta}(x)=$ constant.

## Weight initialization

PyTorch layers have default initialization schemes. (The default is not to initialize everything to 0 .) Sometimes this default initialization scheme is sufficient (eg. Chapter 2 code.ipynb) sometimes it is not sufficient (eg. Hw3 problem 1).

How to initialize weights is tricky. More on this later.

## Gradient computation via backprop

PyTorch and other deep learning libraries allows users to specify how to evaluate a function then compute derivatives (gradients) automatically.

No need to work out gradient computation by hand (even though I make you do it in homework assignments).
This feature is called, automatic differentiation, back propagation, or just the chain rule. This is implemented in the torch. autograd module.

More on this later.

## Multi-class classification problem

Consider supervised learning with data $X_{1}, \ldots, X_{N} \in \mathbb{R}^{n}$ and labels $Y_{1}, \ldots, Y_{N} \in\{1, \ldots, k\}$. (A $k$ class classification problem.) Assume there exists a function $f_{\star}: \mathbb{R}^{n} \rightarrow \Delta^{k}$ mapping from data to label probabilities. Here, $\Delta^{k} \subset \mathbb{R}^{k}$ denotes the set of probability mass functions on $\{1, \ldots, k\}$.

Define the empirical distribution $\mathcal{P}(y) \in \mathbb{R}^{k}$ as the one-hot vector:

$$
(\mathcal{P}(y))_{i}=\left\{\begin{array}{cc}
1 & \text { if } y=i \\
0 & \text { otherwise }
\end{array}\right.
$$

for $i=1, \ldots, k$.

## Softmax function

## Examples:

Softmax function $\mu: \mathbb{R}^{k} \rightarrow \Delta^{k}$ is defined by

$$
\mu_{i}(z)=(\mu(z))_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{k} e^{z_{j}}}
$$

for $i=1, \ldots, k$, where $z=\left(z_{1}, \ldots, z_{k}\right) \in \mathbb{R}^{k}$. Since

$$
\sum_{i=1}^{k} \mu_{i}(z)=1, \quad \mu>0
$$

$$
\begin{aligned}
& \mu\left(\left[\begin{array}{l}
1 \\
2 \\
3
\end{array}\right]\right)=\left[\begin{array}{l}
0.09 \\
0.24 \\
0.6
\end{array}\right] \\
& \mu\left(\left[\begin{array}{c}
999 \\
0 \\
-2
\end{array}\right]\right) \approx\left[\begin{array}{l}
1 \\
0 \\
0
\end{array}\right] \\
& \mu\left(\left[\begin{array}{c}
-2 \\
-2 \\
-99
\end{array}\right]\right) \approx\left[\begin{array}{c}
0.5 \\
0.5 \\
0
\end{array}\right]
\end{aligned}
$$

Name "softmax" is a misnomer. "Softargmax" would be more accurate

- $\mu(z) \not \approx \max (z)$
- $\mu(z) \approx \operatorname{argmax}(z)$


## Softmax regression

In softmax regression (SR):

1. Choose the model

$$
\mu\left(f_{A, b}(x)\right)=\frac{1}{\sum_{i=1}^{k} e^{a_{i}^{\top} x+b_{i}}}\left[\begin{array}{c}
e^{a_{1}^{\top} x+b_{1}} \\
e^{a_{2}^{\top} x+b_{2}} \\
\vdots \\
e^{a_{k}^{\top} x+b_{k}}
\end{array}\right], \quad f_{A, b}(x)=A x+b, A=\left[\begin{array}{c}
a_{1}^{\top} \\
a_{2}^{\top} \\
\vdots \\
a_{k}^{\top}
\end{array}\right] \in \mathbb{R}^{k \times n}, \quad b=\left[\begin{array}{c}
b_{1} \\
b_{2} \\
\vdots \\
b_{k}
\end{array}\right] \in \mathbb{R}^{k} .
$$

2. Minimize KL-Divergence (or cross entropy) from the model $\mu\left(f_{A, b}\left(X_{i}\right)\right)$ output probabilities to the empirical distribution $\mathcal{P}\left(Y_{i}\right)$.
$\operatorname{minimize}_{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}} \sum_{i=1}^{N} D_{\mathrm{KL}}\left(\mathcal{P}\left(Y_{i}\right) \| \mu\left(f_{A, b}\left(X_{i}\right)\right)\right) \Leftrightarrow \operatorname{minimize}_{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}} \sum_{i=1}^{N} H\left(\mathcal{P}\left(Y_{i}\right), \mu\left(f_{A, b}\left(X_{i}\right)\right)\right)$

## Softmax regression

$$
\begin{aligned}
& \operatorname{minimize}_{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}} \sum_{i=1}^{N} H\left(\mathcal{P}\left(Y_{i}\right), \mu\left(f_{A, b}\left(X_{i}\right)\right)\right) \\
& \operatorname{minimize}_{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}} \frac{1}{N} \sum_{i=1}^{N}-\log \left(\mu_{Y_{i}}\left(f_{A, b}\left(X_{i}\right)\right)\right) \\
& \operatorname{minimize}_{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}} \frac{1}{N} \sum_{i=1}^{N}-\log \left(\frac{\exp \left(a_{Y_{i}}^{\top} X_{i}+b_{Y_{i}}\right)}{\sum_{j=1}^{k} \exp \left(a_{j}^{\top} X_{i}+b_{j}\right)}\right) \\
& \operatorname{minimize}_{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}} \frac{1}{N} \sum_{i=1}^{N}\left(-\left(a_{Y_{i}}^{\top} X_{i}+b_{Y_{i}}\right)+\log \left(\sum_{j=1}^{k} \exp \left(a_{j}^{\top} X_{i}+b_{j}\right)\right)\right)
\end{aligned}
$$

## Cross entropy loss

So

$$
\begin{array}{cc}
\underset{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}}{\operatorname{minimize}} & \sum_{i=1}^{N} H\left(\mathcal{P}\left(Y_{i}\right), \mu\left(f_{A, b}\left(X_{i}\right)\right)\right) \\
\operatorname{minimize}_{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}} & \frac{1}{N} \sum_{i=1}^{N} \ell^{\mathrm{CE}}\left(f_{A, b}\left(X_{i}\right), Y_{i}\right)
\end{array}
$$

where

$$
\ell^{\mathrm{CE}}(f, y)=-\log \left(\frac{\exp \left(f_{y}\right)}{\sum_{j=1}^{k} \exp \left(f_{j}\right)}\right)
$$

is the cross entropy loss.

## Classification with deep networks

$\mathrm{SR}=$ linear model $f_{A, b}$ with cross entropy loss:

$$
\operatorname{minimize}_{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}} \frac{1}{N} \sum_{i=1}^{N} \ell^{\mathrm{CE}}\left(f_{A, b}\left(X_{i}\right), Y_{i}\right) \Leftrightarrow \operatorname{minimize}_{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}} \sum_{i=1}^{N} D_{\mathrm{KL}}\left(\mathcal{P}\left(Y_{i}\right) \| \mu\left(f_{A, b}\left(X_{i}\right)\right)\right)
$$

(Note $e^{\mathrm{CE}}(f, y)>0$. More on homework 3.)

The natural extension of $S R$ is to consider

$$
\underset{\theta \in \mathbb{R}^{P}}{\operatorname{minimize}} \frac{1}{N} \sum_{i=1}^{N} \ell^{\mathrm{CE}}\left(f_{\theta}\left(X_{i}\right), Y_{i}\right) \Leftrightarrow \underset{\theta \in \mathbb{R}^{\boldsymbol{P}}}{\operatorname{minimize}} \quad \sum_{i=1}^{N} D_{\mathrm{KL}}\left(\mathcal{P}\left(Y_{i}\right) \| \mu\left(f_{\theta}\left(X_{i}\right)\right)\right)
$$

where $f_{\theta}$ is a deep neural network.

## History of GPU Computing

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-098.jpg?height=624&width=3216&top_left_y=376&top_left_x=46)

Rendering graphics involves computing many small tasks in parallel. Graphics cards provide many small processors to render graphics.
In 1999, Nvidia released GeForce 256 and introduced programmability in the form of vertex and pixel shaders. Marketed as the first 'Graphical Processing Unit (GPU)'.
Researchers quickly learned how to implement linear algebra by mapping matrix data into textures and applying shaders.

## General Purpose GPUs (GPGPU)

In 2007, Nvidia released 'Compute Unified Device Architecture (CUDA)', which enabled general purpose computing on a CUDA-enabled GPUs.

Unlike CPUs which provide fast serial processing, GPUs provide massive parallel computing with its numerous slower processors.
The 2008 financial crisis hit Nvidia very hard as GPUs were luxury items used for games. This encouraged Nvidia to invest further in GPGPUs and create a more stable consumer base.

## CPU computing model

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-100.jpg?height=1230&width=1800&top_left_y=435&top_left_x=797)

## GPU computing model

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-101.jpg?height=1060&width=3279&top_left_y=448&top_left_x=19)

## GPUs for machine learning

Raina et al.'s 2009* paper demonstrated that GPUs can be used to train large neural networks. (This was not the first to use of GPUs in machine learning, but it was one of the most influential.)
Modern deep learning is driven by big data and big compute, respectively provided by the internet and GPUs.

Krizhevsky et al.'s 2012* landmark paper introduced AlexNet trained on GPUs and kickstarted the modern deep learning boom.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-102.jpg?height=544&width=1662&top_left_y=1157&top_left_x=1624)

## Example: Power iteration with GPUs

Computing $x^{100}=A^{100} x^{0}$ with a GPU:

```
send A from host (CPU) to device (GPU)
send x=x0 from host (CPU) to device (GPU)
for _ in range(100):
    tell GPU to compute x=A*x
send x from device (GPU) to host (CPU)
```

In this example and deep learning, GPU accelerates computation since:
Amount of computation > data communication.
Large information resides in the GPU, and CPU issues commands to perform computation on the data. ( $A$ in this example, neural network architecture in deep learning.)

PyTorch demo

## Deep learning on GPUs

Steps for training neural network on GPU:

1. Create the neural network on CPU and send it to GPU. Neural network parameters stay on GPU.

- Sometimes you load parameters from CPU to GPU.

2. Select data batch (image, label) and send it to GPU every iteration

- Data for real-world setups is large, so keeping all data on GPU is infeasible.

3. On GPU, compute network output (forward pass)
4. On GPU, compute gradients (backward pass)
5. On GPU, perform gradient update
6. Once trained, perform prediction on GPU.

- Send test data to GPU.
- Compute network output.
- Retrieve output on CPU.
- Alternatively, neural network can be loaded on CPU and prediction can be done on CPU.

PyTorch demo

# Chapter 3: Convolutional Neural Networks 

Mathematical Foundations of Deep Neural Networks
Spring 2024
Department of Mathematical Sciences
Ernest K. Ryu
Seoul National University

## Fully connected layers

Advantages of fully connected layers:

- Simple.
- Very general, in theory. (Sufficiently large MLPs can learn any function, in theory.)

Disadvantage of fully connected layers:

- Too many trainable parameters.
- Does not encode shift equivariance/invariance and therefore has poor inductive bias. (More on this later.)


## Shift equivariance/invariance in vision

Many tasks in vision are equivariant/invariant with respect shifts/translations.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-107.jpg?height=333&width=770&top_left_y=684&top_left_x=644)

Cat
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-107.jpg?height=337&width=779&top_left_y=682&top_left_x=1592)

Still a Cat

Roughly speaking, equivariance/invariance means shifting the object does not change the meaning (it only changes the position).

## Shift equivariance/invariance in vision

Logistic regression (with a single fully connected layer) does not encode shift invariance.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-108.jpg?height=696&width=3056&top_left_y=634&top_left_x=173)

Since convolution is equivariant with respect to translations, constructing neural network layers with them is a natural choice.

## Convolution

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-109.jpg?height=1409&width=3207&top_left_y=418&top_left_x=68)

## Multiple filters

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-110.jpg?height=1498&width=3156&top_left_y=374&top_left_x=38)

## 2D convolutional layer: Formal definition

Input tensor: $X \in \mathbb{R}^{B \times C_{\text {in }} \times m \times n}, B$ batch size, $C_{\text {in }} \#$ of input channels, $m, n \#$ of vertical and horizontal indices.
Output tensor: $Y \in \mathbb{R}^{B \times C_{\text {out }} \times\left(m-f_{1}+1\right) \times\left(n-f_{2}+1\right)}, B$ batch size, $C_{\text {out }} \#$ of output channels.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-111.jpg?height=108&width=2799&top_left_y=907&top_left_x=238) and $j=1, \ldots, n-f_{2}+1$ :

$$
Y_{k, \ell, i, j}=\sum_{\gamma=1}^{c_{\text {in }}} \sum_{\alpha=1}^{f_{1}} \sum_{\beta=1}^{f_{2}} w_{\ell, \gamma, \alpha, \beta} X_{k, \gamma, i+\alpha-1, j+\beta-1}+b_{\ell}
$$

Operation is independent across elements of the batch. The vertical and horizontal indices are referred to as spatial dimensions. If bias=False, then $b=0$.

## Notes on convolution

Mind the indexing. In math, indices start at 1. In Python, indices start at 0.

1D conv is commonly used with 1D data, such as audio.

3D conv is commonly used with 3D data, such as video.

1D and 3D conv are defined analogously.

## Zero padding

$(C \times 7 \times 7$ image $) \circledast(C \times 5 \times 5$ filter $)=(1 \times 3 \times 3$ feature map $)$.
Spatial dimension 7 reduced to 3 .
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-113.jpg?height=869&width=2359&top_left_y=824&top_left_x=412)

## Zero padding

$(C \times 7 \times 7$ image with zero padding $=2) \circledast(C \times 5 \times 5$ filter $)=(1 \times 7 \times 7$ feature map $)$.
Spatial dimension is preserved.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-114.jpg?height=1085&width=3156&top_left_y=640&top_left_x=51)

## Stride

$(7 x 7$ image $) \circledast(3 x 3$ filter with stride 2$)=($ output $3 x 3)$.
(With stride 1 , output is $5 \times 5$.)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-115.jpg?height=1018&width=2239&top_left_y=603&top_left_x=0)

|  |  |  |  |  |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |

If stride 3, dimensions don't fit.
$7 \times 7$ image with zero padding of 1 becomes $9 \times 9$ image.
$(7 \times 7$ image, padding of 1$) \circledast(3 x 3$ filter $)$ with stride 3 does fit.

## Convolution summary

Input $C_{\text {in }} \times W_{\text {in }} \times H_{\text {in }}$
Conv layer parameters

- $C_{\text {out }}$ filters
- $F$ spatial extent ( $C_{\mathrm{in}} \times F \times F$ filters)
- $S$ stride
- $\quad P$ padding

Output $C_{\text {out }} \times W_{\text {out }} \times H_{\text {out }}$

$$
\begin{aligned}
& W_{\mathrm{out}}=\left\lfloor\frac{W_{\mathrm{in}}-F+2 P}{S}+1\right\rfloor \\
& H_{\mathrm{out}}=\left\lfloor\frac{H_{\mathrm{in}}-F+2 P}{S}+1\right\rfloor
\end{aligned}
$$

$\lfloor\cdot\rfloor$ denotes the floor (rounding down) operation. To avoid the complication of this floor operation, it is best to ensure the formula inside evaluates to an integer.

Number of trainable parameters:
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-116.jpg?height=354&width=588&top_left_y=1014&top_left_x=2164)

Make sure you are able to derive these formulae yourself.

## Aside: Geometric deep learning

More generally, given a group $\mathcal{G}$ encoding a symmetry or invariance, one can define operations "equivariant" with respect $\mathcal{G}$ and construct equivariant neural networks.

This is the subject of geometric deep learning, and its formulation utilizes graph theory and group theory.

Geometric deep learning is particularly useful for non-Euclidean data. Examples include as protein molecule data and social network service connections.

## Pooling

Primarily used to reduce spatial dimension. Similar to conv.
Operates over each channel independently.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-118.jpg?height=878&width=1539&top_left_y=811&top_left_x=902)

## Pooling

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-119.jpg?height=1370&width=3292&top_left_y=489&top_left_x=38)

## LeNet5

## Modern instances of LeNet5 use

- $\sigma=$ ReLu
- MaxPool instead of AvgPool
- No $\sigma$ after S2, S4 (Why?)
- Full connection instead of Gaussian connections
$1 \times 28 \times 28$ MNIST image
- Complete C3 connections
with $p=2 \Rightarrow 1 \times 32 \times 32$
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-120.jpg?height=886&width=2970&top_left_y=650&top_left_x=284)

$$
\begin{aligned}
& \text { Something like } \\
& \text { average pool } \\
& f=2, s=2
\end{aligned}
$$

## LeNet5

PyTorch demo

## Architectural contribution: LeNet

One of the earliest demonstration of using a deep CNN to learn a nontrivial task.

Laid the foundation of the modern CNN architecture.

## Weight sharing

In neural networks, weight sharing is a way to reduce the number of parameters by reusing the same parameter in multiple operations. Convolutional layers are the primary example.

$$
A_{w}=\left[\begin{array}{cccccccc}
w_{1} & \cdots & w_{r} & 0 & \cdots & & & 0 \\
0 & w_{1} & \cdots & w_{r} & 0 & \cdots & & 0 \\
0 & 0 & w_{1} & \cdots & w_{r} & 0 & \cdots & 0 \\
\vdots & & & \ddots & & \ddots & & \vdots \\
0 & & \cdots & 0 & w_{1} & \cdots & w_{r} & 0 \\
0 & & \cdots & 0 & 0 & w_{1} & \cdots & w_{r}
\end{array}\right]
$$

If we consider convolution with filter $w$ as a linear operator, the components of $w$ appear may times in the matrix representation. This is because the same $w$ is reused for every patch in the convolution. Weight sharing in convolution may now seem obvious, but it was a key contribution back when the LeNet architecture was presented.
Some models (not studied in this course) use weight sharing more explicitly in other ways.

## Data augmentation

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-124.jpg?height=567&width=1589&top_left_y=412&top_left_x=253)

Invariances

- Translation
- Horizontal flip
- Vertical flip
- Color change (?)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-124.jpg?height=450&width=1551&top_left_y=1041&top_left_x=265)

Invariances

- Translation
- Horizontal flip
- Vertical flip
- Color change

Translation invariance encoded in convolution, but other invariances are harder to encode (unless one uses geometric deep learning). Therefore encode invariances in data and have neural networks learn the invariance.

## Data augmentation

Data augmentation (DA) applies transforms to the data while preserving meaning and label.

Option 1: Enlarge dataset itself.

- Usually cumbersome and unnecessary.

Option 2: Use randomly transformed data in training loop.

- In PyTorch, we use Torchvision.transforms.

PyTorch demo

## Spurious correlation

Hypothetical: A photographer prefers to take pictures with cats looking to the left and dogs looking to the right. Neural network learns to distinguish cats from dogs by which direction it is facing. This learned correlation will not be useful for pictures taken by another photographer.

This is a spurious correlation, a correlation between the data and labels that does not capture the "true" meaning. Spurious correlations are not robust in the sense that the spurious correlation will not be a useful predictor when the data changes slightly.

Removing spurious correlations is another purpose of DA.

## Data augmentation

## We use DA to:

- Inject our prior knowledge of the structure of the data and force the neural network to learn it.
- Remove spurious correlations.
- Increase the effective data size. In particular, we ensure neural network never encounters the exact same data again and thereby prevent the neural network from performing exact memorization. (Neural network can memorize quite well.)


## Effects of DA:

- DA usually worsens the training error (but we don't care about training error).
- DA often, but not always, improves the test error.
- If DA removes a spurious correlation, then the test error can be worsened.
- DA usually improves robustness.


## Data augmentation on test data?

DA is usually applied only on training data.

DA is usually not applied on test data, because we want to ensure test scores are comparable. (There are many different DAs, and applying different DAs on test data will make the metric different.)

However, one can perform "test-time data augmentation" to improve predictions without changing the test. More on this later.

## ImageNet dataset

ImageNet contains more 14 million hand-annotated images in more than 20,000 categories.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-129.jpg?height=707&width=1421&top_left_y=659&top_left_x=863)

Many classes, higher resolution, non-uniform image size, multiple objects per image.

## History

- Fei-Fei Li started the ImageNet project in 2006 with the goal of expanding and improving the data available for training Al algorithms.
- Images were annotated with Amazon Mechanical Turk.
- The ImageNet team first presented their dataset in the 2009 Conference on Computer Vision and Pattern Recognition (CVPR).
- From 2010 to 2017, the ImageNet project ran the ImageNet Large Scale Visual Recognition Challenge (ILSVRC).
- In the 2012 ILSVRC challenge, 150,000 images of 1000 classes were used.
- In 2017, 29 teams achieved above 95\% accuracy. The organizers deemed task complete and ended the ILSVRC competition.


## ImageNet-1k

Commonly referred to as "the ImageNet dataset". Also called ImageNet2012

However, ImageNet-1k is really a subset of full ImageNet dataset.

ImageNet-1k has 150,000 images of 1000 roughly balanced classes.

## List of categories:

https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-131.jpg?height=1404&width=2067&top_left_y=229&top_left_x=1263)

## ImageNet-1k

Data has been removed from the ImageNet website. Downloading peer-to-peer via torrent is now the most convenient way to access the data.

Privacy concerns: Although dataset is about recognizing objects, rather than humans, many human faces are in the images. Troublingly, identifying personal information is possible.

NSFW concerns: Sexual and non-consensual content.

Creating datasets while protecting privacy and other social values is an important challenge going forward.

## Top-1 vs. top-5 accuracy

Classifiers on ImageNet-1k are often assessed by their top-5 accuracy, which requires the 5 categories with the highest confidence to contain the label.

In contrast, the top-1 accuracy simply measures whether the network's single prediction is the label.

For example, AlexNet had a top-5 accuracy of $84.6 \%$ and a top- 1 accuracy of $63.3 \%$.

Nowadays, accuracies of classifiers has improved, so the top 1 accuracy is becoming the more common metric.

## Classical statistics: Over vs. underfitting

Given separate train and test data

- When (training loss) << (testing loss) you are overfitting. What you have learned from the training data does not carry over to test data.
- When (training loss) $\approx$ (testing loss) you are underfitting. You have the potential to learn more from the training data.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-134.jpg?height=1205&width=1829&top_left_y=401&top_left_x=1501)


## Classical statistics: Over vs. underfitting

The goal of ML is to learn patterns that generalize to data you have not seen. From each datapoint, you want to learn enough (don't underfit) but if you learn too much you overcompensate for an observation specific to the single experience.

In classical statistics, underfitting vs. overfitting (bias vs. variance tradeoff) is characterized rigorously.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-135.jpg?height=911&width=1511&top_left_y=799&top_left_x=767)

## Modern deep learning: Double descent

In modern deep learning, you can overfit, but the state-of-the art neural networks do not overfit (or "benignly overfit") despite having more model parameters than training data.

We do not yet have clarity with this new phenomenon.

When overfitting happens and when it does not is unclear.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-136.jpg?height=886&width=2459&top_left_y=807&top_left_x=786)

## Double descent on 2-layer neural network on MNIST <br> Belkin et al. experimentally demonstrates the double descent phenomenon with an MLP trained on the MNIST dataset. <br> ![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-137.jpg?height=524&width=1123&top_left_y=299&top_left_x=1718) <br> ![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-137.jpg?height=604&width=1120&top_left_y=842&top_left_x=1728)

Fig. 3. Double-descent risk curve for a fully connected neural network on MNIST. Shown are training and test risks of a network with a single layer of $H$ hidden units, learned on a subset of MNIST ( $n=4 \cdot 10^{3}, d=784$, $K=10$ classes). The number of parameters is $(d+1) \cdot H+(H+1) \cdot K$. The interpolation threshold (black dashed line) is observed at $n \cdot K$.

## Double descent example: 2-layer ReLU NN with fixed hidden layer weights

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-138.jpg?height=1234&width=2880&top_left_y=506&top_left_x=227)

## How to avoid overfitting

Regularization is loosely defined as mechanisms to prevent overfitting.

When you are overfitting, regularize with:

- Smaller NN (fewer parameters) or larger NN (more parameters).
- Improve data by:
- using data augmentation
- acquiring better, more diverse, data
- acquiring more of the same data
- Weight decay
- Dropout
- Early stopping on SGD or late stopping on SGD


## How to avoid underfitting

When you are underfitting, use:

- Larger NN (if computationally feasible)
- Less weight decay
- Less dropout
- Run SGD longer (if computationally feasible)


## Weight decay $\cong \ell^{2}$-regularization

$\ell^{2}$-regularization augments the loss function with

$$
\operatorname{minimize}_{\theta \in \mathbb{R}^{p}} \frac{1}{N} \sum_{i=1}^{N} \ell\left(f_{\theta}\left(x_{i}\right), y_{i}\right)+\frac{\lambda}{2}\|\theta\|^{2}
$$

SGD on the augmented loss is usually implemented by changing SGD update rather than explicitly changing the loss since

$$
\begin{gathered}
\theta^{k+1}=\theta^{k}-\alpha\left(g^{k}+\lambda \theta^{k}\right) \\
=(1-\alpha \lambda) \theta^{k}-\alpha g^{k}
\end{gathered}
$$

Where $g^{k}$ is stochastic gradient of original (unaugmented) loss.
In classical statistics, this is called ridge regression or maximum a posteriori (MAP) estimation with Gaussian prior.

## Weight decay $\cong \ell^{2}$-regularization

In Pytorch, you can use SGD + weight decay by:
augmenting the loss function

```
for param in model.parameters():
    loss += (lamda/2)*param.pow(2.0).sum()
torch.optim.SGD(model.parameters(), lr=... , weight_decay=0)
```

or by using weight_decay in the optimizer
torch.optim.SGD(model.parameters(), lr=... , weight_decay=lamda)

For plain SGD, weight decay and $\ell^{2}$-regularization are equivalent. For other optimizers, the two are similar but not the same. More on this later.

## Dropout

Dropout is a regularization technique that randomly disables neurons.

Standard layer,

$$
h_{2}=\sigma\left(W_{1} h_{1}+b_{1}\right)
$$

Dropout with drop probability $p$ defines

$$
h_{2}=\sigma\left(W_{1} h_{1}^{\prime}+b_{1}\right)
$$

with

$$
\left(h_{1}^{\prime}\right)_{j}= \begin{cases}0 & \text { with probability } p \\ \frac{\left(h_{1}\right)_{j}}{1-p} & \text { otherwise }\end{cases}
$$

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-143.jpg?height=639&width=571&top_left_y=790&top_left_x=1760)
(a) Standard Neural Net
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-143.jpg?height=631&width=566&top_left_y=790&top_left_x=2477)
(b) After applying dropout.

Figure 1: Dropout Neural Net Model. Left: A standard neural net with 2 hidden layers. Right: An example of a thinned net produced by applying dropout to the network on the left. Crossed units have been dropped.

## Why is dropout helpful?

"A motivation for dropout comes from a theory of the role of sex in evolution (Livnat et al., 2010)."

Sexual reproduction, compared to asexual reproduction, creates the criterion for natural selection mix-ability of genes rather than individual fitness, since genes are mixed in a more haphazard manner.
"Since a gene cannot rely on a large set of partners to be present at all times, it must learn to do something useful on its own or in collaboration with a small number of other genes. ... Similarly, each hidden unit in a neural network trained with dropout must learn to work with a randomly chosen sample of other units. This should make each hidden unit more robust and drive it towards creating useful features on its own without relying on other hidden units to correct its mistakes.

## Why is dropout helpful?

The analogy to evolution is very interesting, but it is ultimately a heuristic argument. It also shifts the burden to the question: "why is sexual evolution more powerful than asexual evolution?"

However, dropout can be shown to be loosely equivalent to $\ell^{2}$-regularization. However, we do not yet have a complete understanding of the mathematical reason behind dropout's performance.

## Dropout in PyTorch

Dropout simply multiplies the neurons with a random $0-\frac{1}{1-p_{\text {drop }}}$ mask.

A direct implementation in PyTorch:

```
def dropout_layer(X, p_drop):
    mask = (torch.rand(X.shape) > p_drop).float()
    return mask * X / (1.0 - p_drop)
```

PyTorch provides an implementation of dropout through torch.nn. Dropout.

## Dropout in training vs. test

Typically, dropout is used during training and turned off during prediction/testing.
(Dropout should be viewed as an additional onus imposed during training to make training more difficult and thereby effective, but it is something that should be turned off later.)

In PyTorch, activate the training mode with
model.train()
and activate evaluation mode with
model.eval()
dropout (and batchnorm) will behave differently in these two modes.

## When to use dropout

Dropout is usually used on linear layers but not on convolutional layers.

- Linear layers have many weights and each weight is used only once per forward pass. (If $y=\operatorname{Linear}_{A, b}(x)$, then $A_{i j}$ only affect $y_{i}$.) So regularization seems more necessary.
- A convolutional filter has fewer weights and each weight is used multiple times in each forward pass. (If $y=\operatorname{Conv} 2 \mathrm{D}_{w, b}(x)$, then $w_{i j k t}$ affects $\left.y_{i, .,:}.\right)$ So regularization seems less necessary.

Dropout seems to be going out of fashion:

- Dropout's effect is somehow subsumed by batchnorm. (This is poorly understood.)
- Linear layers are less common due to their large number of trainable parameters.

There is no consensus on whether dropout should be applied before or after the activation function. However, Dropout- $\sigma$ and $\sigma$-Dropout are equivalent when $\sigma$ is $\operatorname{ReLU}$ or leaky $\operatorname{ReLU}$, or, more generally, when $\sigma$ is nonnegative homogeneous.

## SGD early stopping

Early stopping of SGD refers to stopping the training early even if you have time for more iterations.

The rationale is that SGD fits data, so too many iterations lead to overfitting.

A similar phenomenon (too many iterations hurt) is observed in classical algorithms for inverse problems.

Typical training and testing loss Us. iterations
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-149.jpg?height=797&width=1612&top_left_y=669&top_left_x=1722)

## Epochwise double descent

Recently, however, an epochwise double descent has been observed.

So perhaps one should stop SGD early or very late.

We do not yet have clarity with this new phenomenon.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-150.jpg?height=983&width=1672&top_left_y=593&top_left_x=1656)

## More data (by data auagmentation)

With all else fixed, using more data usually* leads to less overfitting.

However, collecting more data is often expansive.

Think of data augmentation (DA) as a mechanism to create more data for free. You can view DA as a form of regularization.

## Summary of over vs. underfitting

In modern deep learning, the double descent phenomenon has brought a conceptual and theoretical crisis regarding over and underfitting. Much of the machine learning practice is informed by classical statistics and learning theory, which do not take the double descent phenomenon into account.

Double descent will bring fundamental changes to statistics, and researchers need more time to figure things out. Most researchers, practitioners and theoreticians, agree that not all classical wisdom is invalid, but what part do we keep, and what part do we replace?

In the meantime, we will have to keep in mind the two contradictory viewpoints and move forward in the absence of clarity.

## AlexNet

Won the 2012 ImageNet challenge by a large margin: top-5 error rate $15.3 \%$ vs. $26.2 \%$ second place.

Started the era of deep neural networks and their training via GPU computing.
AlexNet was split into 2 as GPU memory was limited. (A single modern GPU can easily hold AlexNet.)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-153.jpg?height=707&width=2200&top_left_y=982&top_left_x=958)

## AlexNet for ImageNet

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-154.jpg?height=1179&width=3101&top_left_y=201&top_left_x=142)

[^0]
## AlexNet CIFAR10

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-155.jpg?height=682&width=3254&top_left_y=433&top_left_x=23)

Conv.-ReLU
Max pool $f=3, s=2$ (overlapping max pool)
Network not split into 2
No local response normalization

## Architectural contribution: AlexNet

A scaled-up version of LeNet.

Demonstrated that deep CNNs can learn significantly complex tasks. (Some thought CNNs could only learn simple, toy tasks like MNIST.)

Demonstrated GPU computing to be an essential component of deep learning.

Demonstrated effectiveness of ReLU over sigmoid or tanh in deep CNNs for classification.

## SGD-type optimizers

In modern NN training, SGD and variants of SGD are usually used. There are many variants of SGD.

The variants are compared mostly on an experimental basis. There is some limited theoretical basis in their comparisons. (Cf. Adam story.)

So far, all efforts to completely replace SGD have failed.

## SGD with momentum

SGD:

$$
\theta^{k+1}=\theta^{k}-\alpha g^{k}
$$

SGD with momentum:

$$
\begin{gathered}
v^{k+1}=g^{k}+\beta v^{k} \\
\theta^{k+1}=\theta^{k}-\alpha v^{k+1}
\end{gathered}
$$

$\beta=0.9$ is a common choice.

When different coordinates (parameters) have very different scalings (i.e., when the problem is ill-conditioned, momentum can help find a good direction of progress.

## RMSProp

RMSProp:

$$
\begin{gathered}
m_{2}^{k+1}=\beta_{2} m_{2}^{k}+\left(1-\beta_{2}\right)\left(g^{k} \circledast g^{k}\right) \\
\theta^{k+1}=\theta^{k}-\alpha g^{k} \oslash \sqrt{m_{2}^{k+1}+\epsilon}
\end{gathered}
$$

$\beta_{2}=0.99$ and $\epsilon=10^{-8}$ are common values. $\circledast$ and $\oslash$ are elementwise mult. and div.
$m_{2}^{k}$ is a running estimate of the $2^{\text {nd }}$ moment of the stochastic gradients, i.e., $\left(m_{2}^{k}\right)_{i} \approx \mathbb{E}\left(g^{k}\right)_{i}^{2}$.
$\alpha \oslash \sqrt{m_{2}^{k+1}+\epsilon}$ is the learning rate scaled elementwise. Progress along steep and noisy directions are dampened while progress along flat and non-noisy directions are accelerated.

## Adam (Adaptive moment estimation)

Adam:

$$
\begin{gathered}
m_{1}^{k+1}=\beta_{1} m_{1}^{k}+\left(1-\beta_{1}\right) g^{k}, m_{2}^{k+1}=\beta_{2} m_{2}^{k}+\left(1-\beta_{2}\right)\left(g^{k} \circledast g^{k}\right) \\
\tilde{m}_{1}^{k+1}=\frac{m_{1}^{k+1}}{1-\beta_{1}^{k+1}}, \quad \widetilde{m}_{2}^{k+1}=\frac{m_{2}^{k+1}}{1-\beta_{2}^{k+1}} \\
\theta^{k+1}=\theta^{k}-\alpha \widetilde{m}_{1}^{k+1} \oslash \sqrt{\widetilde{m}_{2}^{k+1}+\epsilon}
\end{gathered}
$$

- $\quad \beta_{1}^{k+1}$ means $\beta_{1}$ to the $(k+1)$ th power.
- $\beta_{1}=0.9, \beta_{2}=0.999$, and $\epsilon=10^{-8}$ are common values. Initialize with $m_{1}^{0}=m_{2}^{0}=0$.
- $m_{1}^{k}$ and $m_{2}^{k}$ are running estimates of the $1^{\text {st }}$ and $2^{\text {nd }}$ moments of $g^{k}$.
- $\tilde{m}_{1}^{k}$ and $\tilde{m}_{2}^{k}$ are bias-corrected estimates of $m_{1}^{k}$ and $m_{2}^{k}$.
- Using $\widetilde{m}_{1}^{k}$ instead of $g^{k}$ adds the effect of momentum.


## Bias correction of Adam

To understand the bias correction, consider the hypothetical $g^{k}=g$ for $k=0,1, \ldots$. Then

$$
m_{1}^{k}=\left(1-\beta_{1}^{k}\right) g
$$

and

$$
m_{2}^{k}=\left(1-\beta_{2}^{k}\right)(g \circledast g)
$$

while $m_{1}^{k} \rightarrow g$ and $m_{2}^{k} \rightarrow(g \circledast g)$ as $k \rightarrow \infty$, the estimators are not exact despite there being no variation in $g^{k}$.

On the other hand, there is bias-corrected estimators are exact:

$$
\widetilde{m}_{1}^{k}=g
$$

and

$$
\widetilde{m}_{2}^{k}=(g \circledast g)
$$

## The cautionary tale of Adam

Adam's original 2015 paper justified the effectiveness of the algorithm through experiments training deep neural networks with Adam. After all, this non-convex optimization is what Adam was proposed to do.

However, the paper also provided a convergence proof under the assumption of convexity. This was perhaps unnecessary in an applied paper focusing on non-convex optimization.

The proof was later shown to be incorrect! Adam does not always converge in the convex setup, i.e., the algorithm, rather than the proof, is wrong.

Reddi and Kale presented the AMSGrad optimizer, which does come with a correct convergence proof, but AMSGrad tends to perform worse than Adam, empirically.

## How to choose the optimizer

Extensive research has gone into finding the "best" optimizer. Schmidt et al." reports that, roughly speaking, that Adam works well most of the time.

So, Adam is a good default choice. Currently, it seems to be the best default choice.

However, Adam does not always work. For example, it seems to be that the widely used EfficientNet model can only be trained ${ }^{\dagger}$ with RMSProp.

However, there are some setups where the LR of SGD is harder to tune, but SGD outperforms Adam when properly tuned.\#

[^1]
## How to tune parameters

Everything should be chosen by trial and error. The weight parameters and $\beta, \beta_{1}, \beta_{2}$ and the weight decay parameter $\lambda$, and the optimizers should be chosen based on trial and error.

The LR (the stepsize $\alpha$ ) of different optimizers are not really comparable between the different optimizers. When you change the optimizer, the LR should be tuned again.

Roughly, large stepsize, large momentum, small weight decay is faster but less stable, while small stepsize, small momentum, and large weight decay is slower but more stable.

## Using different optimizers in PyTorch

In PyTorch, the torch. optim module implements the commonly used optimizers.

Using SGD:
torch.optim.SGD(model.parameters(), lr=X)
Using SGD with momentum:
torch.optim.SGD(model.parameters(), momentum=0.9, lr=X)
Using RMSprop:
torch.optim.RMSprop(model.parameters(), lr=X)
Using Adam:

```
torch.optim.Adam(model.parameters(), lr=X)
```

Exercise: Try Homework 3 problem 1 with Adam but without the custom weight initialization.

## Learning rate scheduler

Sometimes, it is helpful to change (usually reduce) the learning rate as the training progresses. PyTorch provides learning rate schedulers to do this.

```
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.9) # lr = 0.9*lr
for _ in range(...):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step() # .step() call updates (changes) the learning rate
```


## Diminishing learning rate

One common choice is to specify a diminishing learning rate via a function (a lambda expression). Choices like C/epoch or $\mathrm{C} /$ sqrt(iteration), where C is an appropriately chosen constant, are common.

```
# lr_lambda allows us to set lr with a function
scheduler = LambdaLR(optimizer, lr_lambda = lambda ep: 1e-2/ep)
for epoch in range(...):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step() # lr=0.01/epoch
```


## Cosine learning rate

The cosine learning rate scheduler, which sets the learning rate with the cosine function, is also commonly used.

The $2^{\text {nd }}$ case in the specification means $k$ and its purpose is to prevent the learning rate from becoming 0 .
It is also common to use only a half-period of the cosine rather than having the learning rate oscillate.

## COSINEANNEALINGLR

$$
\begin{aligned}
& \text { CLASS torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, } \\
& \text { last_epoch=- 1, verbose=False) [SOURCE] } \\
& \text { Set the learning rate of each parameter group using a cosine annealing schedule, where } \eta_{\max } \text { is set to the } \\
& \text { initial Ir and } T_{\text {cur }} \text { is the number of epochs since the last restart in SGDR: } \\
& \eta_{t}=\eta_{\min }+\frac{1}{2}\left(\eta_{\max }-\eta_{\min }\right)\left(1+\cos \left(\frac{T_{\text {cur }}}{T_{\max }} \pi\right)\right), \quad T_{\text {cur }} \neq(2 k+1) T_{\max } ; \\
& \eta_{t+1}=\eta_{t}+\frac{1}{2}\left(\eta_{\max }-\eta_{\min }\right)\left(1-\cos \left(\frac{1}{T_{\max }} \pi\right)\right), \quad T_{\text {cur }}=(2 k+1) T_{\max } . \\
& \text { When last_epoch=-1, sets initial Ir as Ir. Notice that because the schedule is defined recursively, the learning } \\
& \text { rate can be simultaneously modified outside this scheduler by other operators. If the learning rate is set solely } \\
& \text { by this scheduler, the learning rate at each step becomes: } \\
& \eta_{t}=\eta_{\text {min }}+\frac{1}{2}\left(\eta_{\text {max }}-\eta_{\text {min }}\right)\left(1+\cos \left(\frac{T_{\text {cur }}}{T_{\text {max }}} \pi\right)\right) \\
& \text { CLASS torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, } \\
& \text { last_epoch=- 1, verbose=False) [SOURCE] } \\
& \text { by this scheduler, the learning rate at each step becomes: } \\
& \text { It has been proposed in SGDR: Stochastic Gradient Descent with Warm Restarts. Note that this only } \\
& \text { implements the cosine annealing part of SGDR, and not the restarts. } \\
& \text { nen }
\end{aligned}
$$

a

## Wide vs. sharp minima

As alluded to in hw1:

- Large step makes large and rough progress towards regions with small loss.
- Small steps refines the model by finding sharper minima.

Also small steps better suppress the effect of noise. Mathematically, one can show that SGD with small steps becomes very similar to GD with small steps.\#

However, using small steps to converge to sharp minima may not always be optimal. There is some empirical evidence that wide minima have better test error than sharp minima.*

## Weight initialization

Remember, SGD is

$$
\theta^{k+1}=\theta^{k}-\alpha g^{k}
$$

where $\theta^{0} \in \mathbb{R}^{p}$ is an initial point. Using a good initial point is important in NN training.
Prescription by LeCun et al.: "Weights should be chosen randomly but in such a way that the [tanh] is primarily activated in its linear region. If weights are all very large then the [tanh] will saturate resulting in small gradients that make learning slow. If weights are very small then gradients will also be very small." (Cf. Vanishing gradient homework problem.)
"Intermediate weights that range over the [tanh's] linear region have the advantage that (1) the gradients are large enough that learning can proceed and (2) the network will learn the linear part of the mapping before the more difficult nonlinear part."

## Quick math review

Using the $1^{\text {st }}$ order Taylor approximation,

$$
\tanh (z) \approx z
$$

Write $X \sim \mathcal{N}\left(\mu, \sigma^{2}\right)$ to denote that $X$ is a Gaussian (normal) random variable with mean $\mu$ and standard deviation $\sigma$.

If $X$ and $Y$ are independent mean-zero random variables, then

$$
\begin{gathered}
\mathbb{E}[X Y]=0 \\
\operatorname{Var}(X Y)=\operatorname{Var}(X) \operatorname{Var}(Y)
\end{gathered}
$$

If $X$ and $Y$ are uncorrelated, i.e., if $\mathbb{E}\left[\left(X-\mu_{X}\right)\left(Y-\mu_{Y}\right)\right]=0$, then $\operatorname{Var}(X+Y)=\operatorname{Var}(X)+$ $\operatorname{Var}(Y)$. (Uncorrelated R.V. need not be independent.)

## Weight initialization

Consider
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-172.jpg?height=750&width=877&top_left_y=535&top_left_x=1224)

If $w_{i} \sim \mathcal{N}\left(0, \sigma^{2}\right)$ (zero-mean variance $\sigma^{2}$ Gaussian) then $\operatorname{Var}(\mathrm{z})=3 \sigma^{2}$.
If $\sigma=\frac{1}{\sqrt{3}}$, then $\operatorname{Var}(\mathrm{z})=1$.

## LeCun initialization

Consider the layer

$$
\begin{gathered}
y=\tanh (\tilde{y}) \\
\tilde{y}=A x+b
\end{gathered}
$$

where $x \in \mathbb{R}^{n_{\text {in }}}$ and $y, \tilde{y} \in \mathbb{R}^{n_{\text {out }}}$. Assume $x_{j}$ have mean $=0$ variance $=1$ and are uncorrelated. If we initialize $A_{i j} \sim \mathcal{N}\left(0, \sigma_{A}^{2}\right)$ and $b_{i} \sim \mathcal{N}\left(0, \sigma_{b}^{2}\right)$, IID, then

$$
\begin{aligned}
& \tilde{y}_{i}=\sum_{j=1}^{n_{\mathrm{in}}} A_{i j} x_{j}+b_{i} \quad \text { has mean }=0 \text { variance }=n_{\mathrm{in}} \sigma_{A}^{2}+\sigma_{b}^{2} \\
& y_{i}=\tanh \left(\tilde{y}_{i}\right) \approx \tilde{y}_{i} \quad \text { has mean } \approx 0 \text { variance } \approx n_{\mathrm{in}} \sigma_{A}^{2}+\sigma_{b}^{2}
\end{aligned}
$$

If we choose

$$
\sigma_{A}^{2}=\frac{1}{n_{\text {in }}}, \quad \sigma_{b}^{2}=0,
$$

(so $b=0$ ) then we have $y_{i}$ mean $\approx 0$ variance $\approx 1$ and are uncorrelated.

## LeCun initialization

By induction, with an $L$-layer MLP,

- if the input to has mean $=0$ variance $=1$ and uncorrelated elements,
- the weights and biases are initialized with $A_{i j} \sim \mathcal{N}\left(0, \frac{1}{n_{\text {in }}}\right)$ and $b_{i}=0$, and
- the linear approximations $\tanh (z) \approx z$ are valid,
then we can expect the output layer to have mean $\approx 0$ variance $\approx 1$.


## Xavier initialization

Consider the layer

$$
\begin{gathered}
y=\tanh (\tilde{y}) \\
\tilde{y}=A x+b
\end{gathered}
$$

where $x \in \mathbb{R}^{n_{\text {in }}}$ and $y, \tilde{y} \in \mathbb{R}^{n_{\text {out }}}$. Consider the gradient with respect to some loss $\ell(y)$. Assume $\left(\frac{\partial \ell}{\partial y}\right)_{i}$ have mean $=0$ variance $=1$ and are uncorrelated. Then

$$
\frac{\partial y}{\partial x}=\operatorname{diag}\left(\tanh ^{\prime}(A x+b)\right) A \approx A
$$

if $\tanh (\tilde{y}) \approx \tilde{y}$ and

$$
\frac{\partial \ell}{\partial x}=\frac{\partial \ell}{\partial y} A
$$

If we initialize $A_{i j} \sim \mathcal{N}\left(0, \sigma_{A}^{2}\right)$ and $b_{i} \sim \mathcal{N}\left(0, \sigma_{b}^{2}\right)$, IID, and assume that $\frac{\partial \ell}{\partial y}$ and $A$ are independent ${ }^{*}$, then

$$
\left(\frac{\partial \ell}{\partial x}\right)_{j}=\sum_{i=1}^{n_{\text {out }}}\left(\frac{\partial \ell}{\partial y}\right)_{i} A_{i j} \text { has mean } \approx 0 \text { and variance } \approx n_{\text {out }} \sigma_{A}^{2}
$$

If we choose

$$
\sigma_{A}^{2}=\frac{1}{n_{\mathrm{out}}}
$$

then $\left(\frac{\partial \ell}{\partial x}\right)_{j}$ have mean $\approx 0$ variance $\approx 1$ and are uncorrelated.

## Xavier initialization

$\frac{\partial \ell}{\partial y}$ and $A$ are not independent; $\frac{\partial \ell}{\partial y}$ depends on the forward evaluation, which in turn depends on $A$. Nevertheless, the calculation is an informative exercise and its result seems to be representative of common behavior.

If $y=\tanh (A x+b)$ is an early layer (close to input) in a deep neural network, then the randomness of $A$ is diluted through the forward and backward propagation and $\frac{\partial \ell}{\partial y}$ and $A$ will be nearly independent.

If $y=\tanh (A x+b)$ is an later layer (close to output) in a deep neural network, then $\frac{\partial \ell}{\partial y}$ and $A$ will have strong dependency.

## Xavier initialization

Consideration of forward and backward passes result in different prescriptions.
The Xavier initialization uses the harmonic mean of the two:

$$
\sigma_{A}^{2}=\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}, \quad \sigma_{b}^{2}=0
$$

In the literature, the alternate notation fan $_{\text {in }}$ and fan out $_{\text {out }}$ are often used instead of $n_{\text {in }}$ and $n_{\text {out }}$. The fan-in and fan-out terminology originally refers to the number of electric connections entering and exiting a circuit or an electronic device.

## (Kaiming) He initialization

Consider the layer

$$
y=\operatorname{ReLU}(A x+b)
$$

We cannot use the Taylor expansion with ReLU.

However, a similar line of reasoning with the forward pass gives rise to

$$
\sigma_{A}^{2}=\frac{2}{n_{\mathrm{in}}}
$$

And a similar consideration with backprop gives rise to

$$
\sigma_{A}^{2}=\frac{2}{n_{\text {out }}}
$$

In PyTorch, use mode='fan_in' and mode='fan_out' to toggle between the two modes.

## Discussions on initializations

In the original description of the Xavier and He initializations, the biases are all initialized to 0 . However, the default initialization of Linear* and Conv2d ${ }^{\#}$ layers in PyTorch uses initialize the biases randomly. A documented reasoning behind this choice (in the form of papers or GitHub discussions) do not seem to exist.

Initializing weights with the proper scaling is sometimes necessary to get the network to train, as you will see with the VGG network. However, so long as the network gets trained, the choice of initialization does not seem to affect the final performance.

Since initializations rely on the assumption that the input to each layer has roughly unit variance, it is important that the data is scaled properly. This is why PyTorch dataloader scales pixel intensity values to be in $[0,1]$, rather than $[0,255]$.

## Initialization for conv

Consider the layer

$$
\begin{aligned}
& y=\tanh (\tilde{y}) \\
& \tilde{y}=\operatorname{Conv} 2 \mathrm{D}_{w, b}(x)
\end{aligned}
$$

where $w \in \mathbb{R}^{C_{\text {out }} \times C_{\text {in }} \times f_{1} \times f_{2}}$ and $b \in \mathbb{R}^{C_{\text {out }}}$. Assume $x_{j}$ have mean $=0$ variance $=1$ and are uncorrelated*. If we initialize $w_{i j k \ell} \sim \mathcal{N}\left(0, \sigma_{w}^{2}\right)$ and $b_{i} \sim \mathcal{N}\left(0, \sigma_{b}^{2}\right)$, IID, then

$$
\begin{aligned}
& \tilde{y}_{i} \quad \text { has mean }=0 \text { variance }=\left(C_{\text {in }} f_{1} f_{2}\right) \sigma_{w}^{2}+\sigma_{b}^{2} \\
& y_{i} \approx \tilde{y}_{i} \text { has mean } \approx 0 \text { variance } \approx\left(C_{\text {in }} f_{1} f_{2}\right) \sigma_{w}^{2}+\sigma_{b}^{2}
\end{aligned}
$$

If we choose

$$
\sigma_{w}^{2}=\frac{1}{c_{\text {in }} f_{1} f_{2}}, \quad \sigma_{b}^{2}=0
$$

(so $b=0$ ) then we have $y_{i}$ mean $\approx 0$ variance $\approx 1$ and are correlated.

## Initialization for conv

Outputs from a convolutional layer are correlated. The uncorrelated assumption is false. Nevertheless, the calculation is an informative exercise and its result seems to be representative of common behavior.

Xavier and He initialization is usually used with

$$
n_{\mathrm{in}}=C_{\mathrm{in}} f_{1} f_{2}
$$

and

$$
n_{\text {out }}=C_{\text {out }} f_{1} f_{2}
$$

Justification of $n_{\text {out }}=C_{\text {out }} f_{1} f_{2}$ requires working through the complex indexing or considering the "transpose convolution". We will return to it later.

## ImageNet after AlexNet

AlexNet won the 2012 ImageNet challenge with 8 layers.
ZFNet won the 2013 ImageNet challenge also with 8 layers but with better parameter tuning.

Research since AlexNet indicated that depth is more important than width.
VGGNet ranked 2nd in the 2014 ImagNet challenge with 19 layers.
GoogLeNet ranked 1st in the 2014 ImageNet challenge with 22 layers.

VGG16

- 16 layers with trainable parameters


## VGGNet

- $3 \times 3$ conv. $p=1$ (spatial dimension preserved)
- No local response normalization
- Weight decay $5 \times 10^{-4}$


## By the Oxford Visual Geometry Group

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-183.jpg?height=1121&width=1399&top_left_y=643&top_left_x=31)

- Dropout(0.5) used
- Max pool $f=2, s=2$
- ReLU activation function (except after pool and FC1000)


## VGGNet

VGG19

- 19 layers with trainable parameters
- Slightly better than VGG16
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-184.jpg?height=1025&width=1940&top_left_y=742&top_left_x=29)


## VGGNet-CIFAR10

## 13-layer modification of VGGNet for CIFAR10

- $3 \times 3$ conv. $p=1$
- Max pool $f=2, s=2$
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-185.jpg?height=1196&width=3067&top_left_y=661&top_left_x=31)


## VGGNet training

Training VGGNet was tricky. A shallower version was first trained and then additional layers were gradually added.

Our VGGNet-CIFAR10 is much easier to train since there are fewer layers and the task is simpler. However, good weight initialization is still necessary

Batchnorm (not available when VGGNet was published) makes training VGGNet much easier. With Batchnorm, the complicated initialization scheme of training a smaller version first becomes unnecessary.

PyTorch demo

## Architectural contribution: VGGNet

Demonstrated simple deep CNNs can significantly improve upon AlexNet.

In a sense, VGGNet represents the upper limit of the simple CNN architecture. (It is the best simple model.) Future architectures make gains through more complex constructions.

Demonstrated effectiveness of stacked $3 \times 3$ convolutions over larger $5 \times 5$ or 11x11 convolutions. Large convolutions (larger than $5 \times 5$ ) are now uncommon.

Due to its simplicity, VGGNet is one of the most common test subjects for testing something on deep CNNs.

## Backprop $\subseteq$ autodiff

Autodiff (automatic differentiation) is an algorithm that automates gradient computation. In deep learning libraries, you only need to specify how to evaluate the function. Backprop (back propagation) is an instance of autodiff.

Gradient computation costs roughly $5 \times$ the computation cost $^{*}$ of forward evaluation.

To clarify, backprop and autodiff are not

- finite difference or
- symbolic differentiation.

Autodiff $\approx$ chain rule of vector calculus

## Autodiff example

This complicated gradient computation is simplified by autodiff.

PyTorch demo
$\operatorname{In}[\cdot]:=\mathrm{fn}=\frac{\operatorname{Sin}\left[\operatorname{Cosh}\left[y^{2}+\frac{x}{z}\right]+\operatorname{Tanh}[x y z]\right]}{\log [1+\operatorname{Exp}[x]]}$;
$D[f n, x]$
$\% / .\{x \rightarrow 3.3, y \rightarrow 1.1, z \rightarrow 2.3\} / / N$
D[fn, $y$ ]
$\% / .\{x \rightarrow 3.3, y \rightarrow 1.1, z \rightarrow 2.3\} / / N$
D[fn, z]
$\% / .\{x \rightarrow 3.3, y \rightarrow 1.1, z \rightarrow 2.3\} / / N$

$$
\text { Out }[\cdot]=-\frac{e^{x} \operatorname{Sin}\left[\operatorname{Cosh}\left[y^{2}+\frac{x}{z}\right]+\operatorname{Tanh}[x y z]\right]}{\left(1+e^{x}\right) \log \left[1+\mathbb{e}^{x}\right]^{2}}+\frac{\operatorname{Cos}\left[\operatorname{Cosh}\left[y^{2}+\frac{x}{z}\right]+\operatorname{Tanh}[x y z]\right]\left(y z \operatorname{Sech}[x y z]^{2}+\frac{\sinh \left[y^{2}+\frac{x}{z}\right]}{z}\right)}{\log \left[1+\mathbb{e}^{x}\right]}
$$

Out $[0]=-0.285274$
$\frac{\operatorname{Cos}\left[\operatorname{Cosh}\left[y^{2}+\frac{x}{z}\right]+\operatorname{Tanh}[x y z]\right]\left(x z \operatorname{Sech}[x y z]^{2}+2 y \operatorname{Sinh}\left[y^{2}+\frac{x}{z}\right]\right)}{\log \left[1+\mathbb{e}^{x}\right]}$
Out[ $[0]=-1.01578$
$\frac{\operatorname{Cos}\left[\operatorname{Cosh}\left[y^{2}+\frac{x}{z}\right]+\operatorname{Tanh}[x y z]\right]\left(x y \operatorname{Sech}[x y z]^{2}-\frac{x \operatorname{Sinh}\left[y^{2}+\frac{x}{z}\right]}{z^{2}}\right)}{\log \left[1+\mathbb{e}^{x}\right]}$
Out[ $[=0.288027$

## The power of autodiff

Autodiff is an essential yet often an underappreciated feature of the deep learning libraries. It allows deep learning researchers to use complicated neural networks, while avoiding the burden of performing derivative calculations by hand.

Most deep learning libraries support $2^{\text {nd }}$ and higher order derivative computation, but we will only use $1^{\text {st }}$ order derivatives (gradients) in this class.

Autodiff includes forward-mode, reverse-mode (backprop), and other orders. In deep learning, reverse-mode is most commonly used.

## Autodiff by Jacobian multiplication

Consider $g=f_{L} \circ f_{L-1} \circ \cdots \circ f_{2} \circ f_{1}$, where $f_{\ell}: \mathbb{R}^{n_{\ell-1}} \rightarrow \mathbb{R}^{n_{\ell}}$ for $\ell=1, \cdots, L$.

Chain rule: $D g=D f_{L} \quad D f_{L-1} \quad \cdots \quad D f_{2} \quad D f_{1}$

$$
n_{L} \times n_{L-1} \quad n_{L-1} \times n_{L-2} \quad n_{2} \times n_{1} \quad n_{1} \times n_{0}
$$

Forward-mode: $D f_{L}\left(D f_{L-1}\left(\cdots\left(D f_{2} D f_{1}\right) \cdots\right)\right)$

Reverse-mode: $\left(\left(\left(D f_{L} D f_{L-1}\right) D f_{L-2}\right) \cdots\right) D f_{1}$

Reverse mode is optimal ${ }^{*}$ when $n_{L} \leq n_{L-1} \leq \cdots \leq n_{1} \leq n_{0}$. The number of neurons in each layer tends to decrease in deep neural networks for classification. So reverse-mode is often close to the most efficient mode of autodiff in deep learning.

## General backprop

## Backprop in PyTorch:

1. When the loss function is evaluated, a computation graph is constructed.
2. The computation graph is a directed acyclic graph (DAG) that encodes dependencies of the individual computational components.
3. A topological sort is performed on the DAG and the backprop is performed on the reversed order of this topological sort. (The topological sort ensures that nodes ahead in the DAG are processed first.)

The general form combines a graph theoretic formulation with the principles of backprop that you have seen in the homework assignments.

## Computation graph example

Consider $f(x, y)=y \log x+\sqrt{y \log x}$. Evaluate $f$ with the computation graph:
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-193.jpg?height=392&width=1923&top_left_y=693&top_left_x=552)

The chain rule: $\frac{\partial f}{\partial x}=\frac{\partial f}{\partial c} \frac{\partial c}{\partial b}\left(\frac{\partial b}{\partial a} \frac{\partial a}{\partial x} \frac{\partial x}{\partial x}+\frac{\partial b}{\partial y} \frac{\partial y}{\partial x}\right)+\frac{\partial f}{\partial b}\left(\frac{\partial b}{\partial a} \frac{\partial a}{\partial x} \frac{\partial x}{\partial x}+\frac{\partial b}{\partial y} \frac{\partial y}{\partial x}\right)$

$$
\frac{\partial f}{\partial y}=\frac{\partial f}{\partial c} \frac{\partial c}{\partial b}\left(\frac{\partial b}{\partial a} \frac{\partial a}{\partial x} \frac{\partial x}{\partial y}+\frac{\partial b}{\partial y} \frac{\partial y}{\partial y}\right)+\frac{\partial f}{\partial b}\left(\frac{\partial b}{\partial a} \frac{\partial a}{\partial x} \frac{\partial x}{\partial y}+\frac{\partial b}{\partial y} \frac{\partial y}{\partial y}\right)
$$

But in what order do you evaluate the chain rule expression?

## Computation graph

Let $y_{1}, \ldots, y_{L}$ be the output values (neurons) of the computational nodes. Assume $y_{1}, \ldots, y_{L}$ follow a linear topological ordering, i.e., the computation of $y_{\ell}$ depends on $y_{1}, \ldots, y_{\ell-1}$ and does not depend on $y_{\ell+1}, \ldots, y_{L}$.

Define the graph $G=(V, E)$, where $V=\{1, \ldots, L\}$ and $(i, \ell) \in E$, i.e., $i \rightarrow \ell$, if the computation of $y_{\ell}$ directly depends on $y_{i}$. Write the computation of $y_{1}, \ldots, y_{L}$ as

$$
y_{\ell}=f_{\ell}\left(\left[y_{i}: \text { for } i \rightarrow \ell\right]\right)
$$

## Forward pass on computation graph

In the forward pass, sequentially compute $y_{1}, \ldots, y_{L}$ via

$$
y_{\ell}=f_{\ell}\left(\left[y_{i}: \text { for } i \rightarrow \ell\right]\right)
$$

```
# Use 1-based indexing
# y[1] given
for l = 2,...,L
    inputs = [y[i] for j such that (i->l)]
    y[l] = f[l].eval(inputs)
end
```


## Forward-mode autodiff

$\begin{array}{lllll}\text { Step } 0 & \text { Step } 1 & \text { Step } 2 & \text { Step } 3 & \text { Step } 4\end{array}$
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-196.jpg?height=349&width=2050&top_left_y=506&top_left_x=187)
0. $x=3, y=2, \frac{\partial x}{\partial x}=1, \frac{\partial x}{\partial y}=0, \frac{\partial y}{\partial x}=0, \frac{\partial y}{\partial y}=1$

1. $a=\log x=\log 3, \frac{\partial a}{\partial x}=\frac{1}{x} \cdot \frac{\partial x}{\partial x}=\frac{1}{3}, \frac{\partial a}{\partial y}=0$
2. $b=y a=2 \log 3, \frac{\partial b}{\partial x}=\frac{\partial y}{\partial x} a+y \frac{\partial a}{\partial x}=\frac{2}{3}, \frac{\partial b}{\partial y}=\frac{\partial y}{\partial y} a+y \frac{\partial a}{\partial y}=a=\log 3$
3. $c=\sqrt{b}=\sqrt{2 \log 3}, \frac{\partial c}{\partial x}=\frac{1}{2 \sqrt{b}} \frac{\partial b}{\partial x}=\frac{1}{3 \sqrt{2 \log 3}}, \frac{\partial c}{\partial y}=\frac{1}{\sqrt{b}} \frac{\partial b}{\partial y}=\frac{1}{2} \sqrt{\frac{\log 3}{2}} \longleftarrow \quad$ Computation only depends on node b
4. $f=c+b=\sqrt{2 \log 3}+2 \log 3, \frac{\partial f}{\partial x}=\frac{\partial c}{\partial x}+\frac{\partial b}{\partial x}=\frac{1}{3}\left(2+\frac{1}{3 \sqrt{2 \log 3}}\right), \frac{\partial f}{\partial y}=\frac{\partial c}{\partial y}+\frac{\partial b}{\partial y}=\frac{1}{2} \sqrt{\frac{\log 3}{2}}+\log 3$

## Backprop on computation graph

```
# Use 1-based indexing
# y[1],...,y[L] already computed
g[:] = 0 // .zero_grad()
g[L] = 1 // dy[L]/dy[L]=1
for l = L,...,2
    for i such that (i->l)
        g[i] += g[l]*f[l].grad(i)
    end
end
```

To perform backprop\#, use

$$
\frac{\partial y_{L}}{\partial y_{i}}=\sum_{\ell: i \rightarrow \ell} \frac{\partial y_{L}}{\partial y_{\ell}} \frac{\partial f_{\ell}}{\partial y_{i}}
$$

to sequentially compute $\frac{\partial y_{L}}{\partial y_{L}}, \frac{\partial y_{L}}{\partial y_{L-1}}, \ldots, \frac{\partial y_{L}}{\partial y_{1}}$.

## Reverse-mode autodiff (backprop)

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-198.jpg?height=665&width=2591&top_left_y=386&top_left_x=176)
0. $x=3, y=2$

1. $a=\log 3$
2. $b=2 \log 3$
3. $c=\sqrt{2 \log 3}$
4. $f=\sqrt{2 \log 3}+2 \log 3$
$0^{\prime} \cdot \frac{\partial f}{\partial f}=1$
1'. $\frac{\partial f}{\partial c}=\frac{\partial f}{\partial f} \frac{\partial f}{\partial c}=\frac{\partial f}{\partial f} 1=1$
2'. $\frac{\partial f}{\partial b}=\frac{\partial f}{\partial c} \frac{\partial c}{\partial b}+\frac{\partial f}{\partial f} \frac{\partial f}{\partial c}=\frac{1}{2 \sqrt{b}} 1+1=\frac{1}{2 \sqrt{2 \log 3}}+1$
3'. $\frac{\partial f}{\partial a}=\frac{\partial f}{\partial b} \frac{\partial b}{\partial a}=\frac{\partial f}{\partial b} y=2+\frac{1}{\sqrt{2 \log 3}}$
4'. $\frac{\partial f}{\partial x}=\frac{\partial f}{\partial a} \frac{\partial a}{\partial x}=\frac{\partial f}{\partial a} \frac{1}{x}=\frac{1}{3}\left(2+\frac{1}{\sqrt{2 \log 3}}\right)$
$\frac{\partial f}{\partial y}=\frac{\partial f}{\partial b} \frac{\partial b}{\partial y}=\frac{\partial f}{\partial b} a=\frac{1}{2} \sqrt{\frac{\log 3}{2}}+\log 3$

## Backprop in PyTorch

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-199.jpg?height=630&width=2221&top_left_y=327&top_left_x=501)

In NN training, parameters and fixed inputs are distinguished. In PyTorch, you (1) clear the existing gradient with .zero_grad() (2) forward-evaluate the loss function by providing the input and label and (3) perform backprop with . backward().

The forward pass stores the intermediate neuron values so that they can later be used in backprop. In the test loop, however, we don't compute gradients so the intermediate neuron values are unnecessary. The torch. no_grad() context manager allows intermediate node values to discarded or not be stored. This saves memory and can accelerate the test loop.

## Linear layers have too may parameters

AlexNet: Conv layer params: 2,469,696 (4\%)
Linear layer params: 58,631,144 (96\%)
Total params: 61,100,840
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-200.jpg?height=1039&width=2733&top_left_y=833&top_left_x=598)

## Linear layers have too may parameters

VGG19: Conv layer params: 20,024,384 (14\%)
Linear layer params: 123,642,856 (86\%)
Total params: 143,667,240
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-201.jpg?height=1005&width=1949&top_left_y=850&top_left_x=25)
fc8

## Network in Network (NiN) Network

NiN for CIFAR10.

- Remove linear layers to reduce parameters. Use global average pool instead.
- Weight decay $1 \times 10^{-5}$.
- Dropout(0.5). (dropout after pool is not consistent with modern practice.)
- Maxpool $f=3, s=2$. Use ceil_mode=True so that $\frac{32-3}{2}+1=15.5$ is rounded up to 16 . Default behavior of PyTorch is to round down.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-202.jpg?height=614&width=3187&top_left_y=1198&top_left_x=0)


## 1x1 convolution

A $1 \times 1$ convolution is like a fully connected layer acting independently and identically on each spatial location.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-203.jpg?height=473&width=1753&top_left_y=691&top_left_x=212)

## 192x32x32 <br> 96x32x32

- 96 filters act on 192 channels separately for each pixel
- $96 \times 192+96$ parameters for weights and biases


## Regular conv. layer

Input: $X \in \mathbb{R}^{C_{0} \times m \times n}$

- Select an $f \times f$ patch $\tilde{X}=X[:, i: i+f, j: j+f]$.
- Inner product $\tilde{X}$ and $w_{1}, \ldots, w_{C_{1}} \in \mathbb{R}^{C_{0} \times f \times f}$ and add bias $b_{1} \in \mathbb{R}^{C_{1}}$.
- Apply $\sigma$. (Output in $\mathbb{R}^{C_{1}}$.)

Repeat this for all patches. Output in $X \in \mathbb{R}^{C_{1} \times(m-f+1) \times(n-f+1)}$.
Repeat this for all batch elements.

## "Network in Network"

Input: $X \in \mathbb{R}^{c_{0} \times m \times n}$

- Select an $f \times f$ patch $\tilde{X}=X[i: i+f, j: j+f]$.
- Inner product $\tilde{X}$ and $w_{1}, \ldots, w_{C_{1}} \in \mathbb{R}^{C_{0} \times f \times f}$ and add bias $b_{1} \in \mathbb{R}^{C_{1}}$.
- Apply $\sigma$. (Output in $\mathbb{R}^{C_{1}}$.)
- Apply Linear $A_{A_{2}, b_{2}}(x)$ where $A_{2} \in \mathbb{R}^{C_{2} \times C_{1}}$ and $b_{2} \in \mathbb{R}^{C_{2}}$.
- Apply $\sigma$. (Output in $\mathbb{R}^{C_{2}}$.)
- Apply Linear $A_{A_{3}, b_{3}}(x)$ where $A_{3} \in \mathbb{R}^{C_{3} \times C_{2}}$ and $b_{3} \in \mathbb{R}^{C_{3}}$.
- Apply $\sigma$. (Output in $\mathbb{R}^{C_{3}}$.)

Repeat this for all patches. Output in $X \in \mathbb{R}^{C_{3} \times(m-f+1) \times(n-f+1)}$. Repeat this for all batch elements.
Why is this equivalent to ( $3 \times 3$ conv)-( $1 \times 1$ conv)-( $1 \times 1$ conv)?

## Global average pool

When using CNNs for classification, position of object is not important.

The global average pool has no trainable parameters (linear layers have many) and it is translation invariant. Global average pool removes the spatial dependency.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-206.jpg?height=554&width=3184&top_left_y=1216&top_left_x=0)

## Architectural contribution: NiN Network

Used $1 \times 1$ convolutions to increase the representation power of the convolutional modules.

Replaced linear layer with average pool to reduce number of trainable parameters.

First step in the trend of architectures becoming more abstract. Modern CNNs are built with smaller building blocks.

## GoogLeNet (Inception v1)

Utilizes the inception module. Structure inspired by NiN and name inspired by 2010 Inception movie meme.

Used $1 \times 1$ convolutions.

- Increased depth adds representation power (improves ability to represent nonlinear functions).
- Reduce the number of channels before the expensive $3 \times 3$ and $5 \times 5$ convolutions, and thereby reduce number of trainable weights and computation time. (Cf. hw5)

The name GoogLeNet is a reference to the authors' Google affiliation and is an homage to LeNet.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-208.jpg?height=513&width=915&top_left_y=0&top_left_x=2417)

Inception module
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-208.jpg?height=239&width=809&top_left_y=608&top_left_x=2453)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-208.jpg?height=970&width=669&top_left_y=880&top_left_x=2642)

## GoogLeNet

## Cl! ! ! ! !

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-209.jpg?height=1293&width=3330&top_left_y=153&top_left_x=2)

Two auxiliary classifiers used to slightly improve training. No longer necessary with batch norm.

## GoogLeNet for CIFAR10

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-210.jpg?height=782&width=2248&top_left_y=467&top_left_x=2)

## 1024x7x7 1024x1x1

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-210.jpg?height=239&width=516&top_left_y=748&top_left_x=2736)

| $2 \mathbf{x}$ |  |
| :---: | :---: |
| $k=256480$ |  |
| $k=64$ | $k=128$ |
| 96128 | 128256 |
| 1632 | 2464 |
| 32 | 64 |


| 512 |  |  |  |  | 512 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $k=192$ | 512 | 528 | 832 |  |  |
| $96=160$ | $k=128$ | $k=112$ | $k=256$ |  |  |
| 96 | 208112 | 224128 | 256144 | 288160 | 320 |
| 1648 | 2464 | 2464 | 3264 | 32 | 128 |
| 64 | 64 | 64 | 64 | 128 |  |


| $\mathbf{2 x}$ |  |
| :---: | :---: |
| 832 | 1024 |
| $k=256$ | $k=384$ |
| 160320 | 192384 |
| 32128 | 48128 |
| 128 | 128 |

## Architectural contribution: GoogLeNet

Demonstrated that more complex modular neural network designs can outperform VGGNet's straightforward design.

Together with VGGNet, demonstrated the importance of depth.

Kickstarted the research into deep neural network architecture design.

## Batch normalization

The first step of many data processing algorithms is often to normalize data to have zero mean and unit variance.

- Step 1. Compute $\hat{\mu}=\frac{1}{N} \sum_{i=1}^{N} X_{i}, \widehat{\sigma^{2}}=\frac{1}{N} \sum_{i=1}^{N}\left(X_{i}-\hat{\mu}\right)^{2}$

$$
\hat{X}_{i}=\frac{X_{i}-\widehat{\mu}}{\sqrt{\sigma^{2}}+\varepsilon}
$$

- Step 2. Run method with data $\hat{X}_{1}, \ldots, \hat{X}_{N}$

Batch normalization (BN) (sort of) enforces this normalization layer-by-layer. BN is an indispensable tool for training very deep neural networks. Theoretical justification is weak.

## BN for linear layers

Underlying assumption: Each element of the batch is an IID sample.
Input: $X$ (batch size) $\times(\#$ entries)
output: $\mathrm{BN}_{\beta, \gamma}(X)$. shape $\left(\mathrm{BN}_{\beta, \gamma}(X)\right)=\operatorname{shape}(X)$
$\mathrm{BN}_{\beta, \gamma}$ for linear layers acts independently over neurons.

$$
\begin{gathered}
\hat{\mu}[:]=\frac{1}{B} \sum_{b=1}^{B} X[b,:] \quad \text { FC } \\
\hat{\sigma}^{2}[:]=\frac{1}{B} \sum_{b=1}^{B}(X[b,:]-\hat{\mu}[:])^{2} \\
\mathrm{BN}_{\gamma, \beta}(X)[b,:]=\gamma[:] \frac{X[b,:]-\hat{\mu}[:]}{\sqrt{\hat{\sigma}^{2}[:]+\varepsilon}}+\beta[:] \quad b=1, \ldots, B
\end{gathered}
$$

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-213.jpg?height=341&width=1145&top_left_y=578&top_left_x=2149)

Batch Norm
$\beta, \gamma$
where operations are elementwise. BN normalizes each output neuron. The mean and variance are explicitly controlled through learned parameters $\beta$ and $\gamma$. In Pytorch, nn.BatchNorm1d.

## BN for convolutional layers

Underlying assumption: Each element of the batch, horizontal pixel, and vertical pixel is an IID sample.* Input: $X$ (batch size $) \times($ channels $) \times($ vertical dim $) \times($ horizontal dim $)$ output: $\mathrm{BN}_{\beta, \gamma}(X)$. shape $\left(\mathrm{BN}_{\beta, \gamma}(X)\right)=\operatorname{shape}(X)$ $\mathrm{BN}_{\beta, \gamma}$ for conv. layers acts independently over channels.

$$
\begin{gathered}
\hat{\mu}[:]=\frac{1}{B P Q} \sum_{b=1}^{B} \sum_{i=1}^{P} \sum_{j=1}^{Q} X[b,:, i, j] \\
\hat{\sigma}^{2}[:]=\frac{1}{B P Q} \sum_{b=1}^{B} \sum_{i=1}^{P} \sum_{j=1}^{Q}(X[b,:, i, j]-\hat{\mu}[:])^{2} \\
\operatorname{BN}_{\gamma, \beta}(X)[b,:, i, j]=\gamma[:] \frac{X[b,:, i, j]-\hat{\mu}[:]}{\sqrt{\hat{\sigma}^{2}[:]+\varepsilon}}+\beta[:] \quad \begin{array}{l}
b=1, \ldots, B \\
i=1, \ldots, P \\
j=1, \ldots, Q
\end{array}
\end{gathered}
$$

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-214.jpg?height=384&width=1161&top_left_y=565&top_left_x=2154)
Batch Norm

$$
\beta, \gamma
$$

BN normalizes over each convolutional filter. The mean and variance are explicitly controlled through learned parameters $\beta$ and $\gamma$. In Pytorch, nn. BatchNorm2d.

## BN during testing

$\hat{\mu}$ and $\hat{\sigma}$ are estimated from batches during training. During testing, we don't update the NN, and we may only have a single input (so no batch).
There are 2 strategies for computing final values of $\hat{\mu}$ and $\hat{\sigma}$ :

1. After training, fix all parameters and evaluate NN on full training set to compute $\hat{\mu}$ and $\hat{\sigma}$ layer-by-layer. Store this computed value. (Computation of $\hat{\mu}$ and $\hat{\sigma}$ must be done sequentially layer-by-layer. Why?)
2. During training, compute running average of $\hat{\mu}$ and $\hat{\sigma}$. This is the default behavior of PyTorch.
In PyTorch, use model.train() and model.eval() to switch BN behavior between training and testing.

## Discussion of BN

BN does not change the representation power of NN ; since $\beta$ and $\gamma$ are trained, the output of each layer can have any mean and variance. However, controlling the mean and variance as explicit trainable parameters makes training easier.

With BN, the choice of batch size becomes a more important hyperparameter to tune.

BN is indispensable in practice. Training of VGGNet and GoogLeNet becomes much easier with BN. Training of ResNet requires BN.

## BN and internal covariate shift

BN has insufficient theoretical justification.
The original paper by loffe and Szegedy hypothesized that BN mitigates internal covariate shift (ICS), the shift in the mean and variance of the intermediate layer neurons throughout the training, and that this mitigation leads to improved training.

$$
\mathrm{BN} \Rightarrow(\text { reduced ICS }) \Rightarrow \text { (improved training })
$$

However, Santukar et al. demonstrated that when experimentally measured, BN does not mitigate ICS, but nevertheless improves the training.

$$
\mathrm{BN} \nRightarrow \text { (reduced ICS) }
$$

Nevertheless

$$
\mathrm{BN} \Rightarrow \text { (improved training performance) }
$$

## BN and internal covariate shift

Santukar et al. argues that

$$
\mathrm{BN} \Rightarrow \text { (smoother loss landscape) } \Rightarrow \text { (improved training performance) }
$$

While this claim is more evidence-based than that of loffe and Szegedy, it is still not conclusive. It is also unclear why BN makes the loss landscape smoother, and it is not clear whether the smoother loss landscape fully explains the improved training performance.

This story is a cautionary tale: we should carefully distinguish between speculative hypotheses and evidence-based claims, even in a primarily empirical subject.

## BN has trainable parameters

BN is usually not considered a trainable layer, much like pooling or dropout, and they are usually excluded when counting the "depth" of a NN. However, BN does have trainable parameters. Interestingly, if one randomly initializes a CNN, freezes all other parameters, and only train BN parameters, the performance is surprisingly good.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-219.jpg?height=724&width=1702&top_left_y=901&top_left_x=731)

Figure 2: Accuracy of ResNets for CIFAR-10 (top left, deep; top right, wide) and ImageNet (bottom left, top-1 accuracy; bottom right, top-5 accuracy) with different sets of parameters trainable.

## Discussion of BN

BN seems to also act as a regularizer, and for some reason subsumes effect Dropout. (Using dropout together with BN seems to worsen performance.) Since BN has been popularized, Dropout is used less often.*

After training, functionality of BN can be absorbed into the previous layer when the previous layer is a linear layer or a conv layer. (Cf. homework 6.)

The use of batch norm makes the scaling of weight initialization less important irrelevant.

Use bias=false on layers preceding BN , since $\beta$ subsumes the bias.

## Residual Network (ResNet)

Winner of 2015 ImageNet Challenge
Observation: Excluding the issue of computation cost, more layers it not always better
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-221.jpg?height=634&width=1430&top_left_y=593&top_left_x=297)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-221.jpg?height=456&width=724&top_left_y=635&top_left_x=1828)
aeneric function classes
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-221.jpg?height=447&width=647&top_left_y=644&top_left_x=2683)
nested function classes

Hypothesis 1: Deeper networks are harder to train.
Is there a way to train a shallow network and embed it in a deeper network?
Hypothesis 2: The deeper networks may be worse approximations of the true unknown function. Find an architecture representing a strictly increasing function class as a function of depth.

## Residual blocks

Use a residual connection so that [all weights=0] correspond to [block=identity] ${ }^{*}$
regular residual block
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-222.jpg?height=733&width=1452&top_left_y=684&top_left_x=133)
downsampling residual block
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-222.jpg?height=711&width=1315&top_left_y=716&top_left_x=1877)

Regular block must preserve spatial dimension and number of channels.
Downsampling block halves the spatial dimension and changes the number of channels.

## ResNet18

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-223.jpg?height=677&width=3041&top_left_y=614&top_left_x=223)

Layer count excludes BN even though BN has trainable parameters.

## ResNet34

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-224.jpg?height=1116&width=1902&top_left_y=216&top_left_x=1430)

A trained ResNet18 architecture can be exactly fitted into a ResNet34: copy over the parameters and set parameters of the additional blocks to be 0 . The additional blocks with only serve to apply an additional ReLU, but this makes no difference as ReLU is idempotent.

## ResNet blocks for deeper ResNets

ResNet in fact goes deeper. For the deeper variants, computation cost becomes more significant. To remedy this cost, use $1 \times 1$ conv to reduce number of channels, perform costly $3 \times 3$ convolution, and use $1 \times 1$ conv to restore the number of channels. This bottleneck" structure is adapted from GoogLeNet.
regular residual block with bottleneck
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-225.jpg?height=635&width=1588&top_left_y=1026&top_left_x=99)
downsampling residual block with bottleneck
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-225.jpg?height=496&width=1432&top_left_y=1164&top_left_x=1874)

## ResNet50, 101, 152

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-226.jpg?height=886&width=2894&top_left_y=607&top_left_x=254)

## ResNet18 for cifar10

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-227.jpg?height=954&width=3058&top_left_y=403&top_left_x=240)

ResNet\{34,50,101,152\} for CIFAR10. The intermediate layers are the same as before.

## ResNet v1.5

In the bottleneck blocks performing downsampling, the use of $1 \times 1$ conv with stride 2 is suboptimal as the operation simply ignores $75 \%$ of the neurons. ResNet v 1.5 replaces them with $3 \times 3$ conv with stride 2 .
downsampling residual block v1
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-228.jpg?height=601&width=1008&top_left_y=1154&top_left_x=440)
downsampling residual block v1.5
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-228.jpg?height=827&width=1493&top_left_y=939&top_left_x=1486)

## ResNet v1.5

The fix is more important for the deeper downsampling residual blocks.
downsampling residual block with bottleneck v1
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-229.jpg?height=532&width=1621&top_left_y=474&top_left_x=1639)
downsampling residual block with bottleneck v1.5
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-229.jpg?height=516&width=1581&top_left_y=1218&top_left_x=1676)

## ResNet v2

Permutations of the ordering of conv, BN, and ReLU were tested. BN-ReLUconv had the best performance.

Perform all operations before the residual connection so that the identity mapping can be learned.
modifies residual block
BN-ReLU-conv
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-230.jpg?height=622&width=1477&top_left_y=1122&top_left_x=1677)

## Architectural contribution: ResNet

Introduced residual connections as a key architectural component.

Demonstrated that extremely deep neural networks can be trained with residual connections and BN. ResNet152 concluded the progression of depth. ImageNet challenge winners:

- 2012. AlexNet with 8 layers.
- 2013. ZFNet with 8 layers.
- 2014. GoogLeNet with 22 layers.
- 2015. ResNet152 with 152 layers.
- 2016. Shao et al.. with 152 layers.
- 2017. SENet with 152 layers.

Residual connections and BN are very common throughout all of deep learning.

## ResNext

2016 ImageNet challenge $2^{\text {nd }}$ place. Introduced cardinality as another network parameter, in addition to width (number of channels) and depth. Cardinality is the number of independent paths in the split-transform-merge structure.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-232.jpg?height=966&width=3186&top_left_y=784&top_left_x=40)
S. Xie, R. Girshick, P. Dollár, Z. Tu, and K. He, Aggregated residual transformations for deep neural networks, CVPR, 2017.

## ResNext

## equivalent

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-233.jpg?height=622&width=1021&top_left_y=63&top_left_x=1365)
(a)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-233.jpg?height=614&width=911&top_left_y=63&top_left_x=2381)
(b)

Blocks (a) and (b) almost equivalent due to the by the following observation.

Difference: Block (a) has 32 bias terms which are added to serve the role of the single bias term of block (b).
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-233.jpg?height=852&width=1553&top_left_y=854&top_left_x=1605)

## Ensemble learning

Let $(X, Y)$ be a data-label pair. Let $m_{1}, \ldots, m_{K}$ be models estimating the $Y$ given $X$.

An ensemble is a model

$$
M=\theta_{1} m_{1}+\cdots+\theta_{K} m_{K}
$$

where $\theta_{1}, \ldots, \theta_{K} \in \mathbb{R}$. Often $\theta_{1}+\cdots+\theta_{K}=1$ and $\theta_{i} \geq 0$ for $i=1, \ldots, K$. (So $M$ is often a nonnegative weighted average $m_{1}, \ldots, m_{K}$.)

If $\theta_{1}, \ldots, \theta_{K}$ is chosen well, then

$$
\mathbb{E}_{(X, Y)}\left[\|M(X)-Y\|^{2}\right] \leq \min _{i=1, \ldots, K} \mathbb{E}_{(X, Y)}\left[\left\|m_{i}(X)-Y\right\|^{2}\right]
$$

(The ensemble can be worse if $\theta_{1}, \ldots, \theta_{K}$ is chosen poorly.)

## 2016 ImageNet Challenge ensemble

Trimps-Soushen* won the 2016 ImageNet Challenge with an ensemble of

- Inception-v3[1]
- Inception-v4 ${ }^{[2]}$
- Inception-Resnet-v2 ${ }^{[2]}$
- ResNet-200[3]
- WRN-68-3 ${ }^{[4]}$
*J. Shao, X. Zhang, Z. Ding, Y. Zhao, Y. Chen, J. Zhou, W. Wang, L. Mei, and C. Hu, Trimps-Soushen, 2016.
${ }^{[1]}$ C. Szegedy, V. Vanhoucke, S. loffe, J. Shlens, and Z. Wojna, Rethinking the inception architecture for computer vision, CVPR, 2016.
${ }^{[2]} \mathrm{C}$. Szegedy, S. loffe, V. Vanhoucke, and A. Alemi, Inception-v4, Inception-ResNet and the impact of residual connections on learning, AAAI, 2017.
${ }^{[3]} \mathrm{K} . \mathrm{He}, \mathrm{X}$. Zhang, S. Ren, and J. Sun, Identity mappings in deep residual networks, ECCV, 2016.
${ }^{[4]}$ S. Zagoruyko and N. Komodakis, Wide residual networks, BMVC, 2016.


## Dropout ensemble interpretation

Let $m$ be a model with dropout applied to $K$ neurons. The there are $2^{K}$ possible configurations, which we label $m_{1}, \ldots, m_{2^{K}}$. These models share weights.

Dropout can be viewed as randomly selecting one of these models and updating it with an iteration of SGD.

Turning off dropout at test time can be interpreted and making predictions with an ensemble of these $2^{K}$, since each neuron is scaled so that the neuron value has the same expectation as when dropout is applied.

However, this is not a very precise connection, and I am unsure as to how much to trust it.

## Test-time data augmentation

Test-time data augmentation is an ensemble technique to improve the prediction. (This is not a regularization or data augmentation technique)

Given a single model $M$ and input $X$, make predictions with

$$
\frac{1}{K} \sum_{i=1}^{K} M\left(T_{i}(X)\right)
$$

where $T_{1}, \ldots, T_{K}$ are random data augmentations.

The original AlexNet paper uses test-time data augmentation with random crops and horizontal reflections: "At test time, the network makes a prediction by extracting five ... patches ... as well as their horizontal reflections ..., and averaging the predictions made by the network's softmax layer on the ten patches." Most ImageNet classifiers use similar tricks.

## SENet

2017 ImageNet challenge $1^{\text {st }}$ place. Introduced the squeeze-and-excitation mechanism, which is referred to attention in more modern papers.

Attention multiplicatively reweighs channels.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-238.jpg?height=703&width=3332&top_left_y=1031&top_left_x=1)

## Squeeze-and-excitation

Squeeze is a global average pool. Excitation is a bottleneck structure with $1 \times 1$ convolutions and outputs weights in $(0,1)$ by passing through sigmoid. Finally, scale each channel.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-239.jpg?height=1077&width=2450&top_left_y=712&top_left_x=799)

## Conclusion

We followed the ImageNet challenge from 2012 to 2017 and learned the foundations of the design and training of deep neural networks.

With the advent of deep learning, research in computer vision shifted from "feature engineering" to "network engineering". Loosely speaking, the transition was from what to learn to learn to how to learn.

A natural progression may be to continue studying the more recent neural network architectures, beyond the 2017 SENet. However, we will stop here to move on to learning about other machine learning tasks.

# Chapter 4: <br> CNNs for Other Supervised Learning Tasks 

Mathematical Foundations of Deep Neural Networks
Spring 2024
Department of Mathematical Sciences
Ernest K. Ryu
Seoul National University

## Inverse problem model

In inverse problems, we wish to recover a signal $X_{\text {true }}$ given measurements $Y$. The unknown and the measurements are related through

$$
\mathcal{A}\left[X_{\text {true }}\right]+\varepsilon=Y,
$$

where $\mathcal{A}$ is often, but not always, linear, and $\varepsilon$ represents small error.

The forward model $\mathcal{A}$ may or may not be known. In other words, the goal of an inverse problem is to find an approximation of $\mathcal{A}^{-1}$.

In many cases, $\mathcal{A}$ is not even be invertible. In such cases, we can still hope to find an mapping that serves as an approximate inverse in practice.

## Gaussian denoising

Given $X_{\text {true }} \in \mathbb{R}^{w \times h}$, we measure

$$
Y=X_{\text {true }}+\varepsilon
$$

where $\varepsilon_{i j} \sim \mathcal{N}\left(0, \sigma^{2}\right)$ is IID Gaussian noise. For the sake of simplicity, assume we know $\sigma$. Goal is to recover $X_{\text {true }}$ from $Y$.

Guassian denoising is the simplest setup in which the goal is to remove noise from the image. In more realistic setups, the noise model will be more complicated and the noise level $\sigma$ will be unknown.

## DnCNN

In 2017, Zhang et al. presented the denoising convolutional neural networks (DnCNNs). They trained a 17-layer CNN $f_{\theta}$ to learn the noise with the loss

$$
\mathcal{L}(\theta)=\sum_{i=1}^{N}\left\|f_{\theta}\left(Y_{i}\right)-\left(Y_{i}-X_{i}\right)\right\|^{2}
$$

so that the clean recovery can be obtained with $Y_{i}-f_{\theta}\left(Y_{i}\right)$. (This is equivalent to using a residual connection from beginning to end.)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-244.jpg?height=626&width=1502&top_left_y=1141&top_left_x=1520)

## DnCNN

Image denoising is was an area with a large body of prior work. DnCNN dominated all prior approaches that were not based on deep learning.

Nowadays, all state-of-the-art denoising algorithms are based on deep learning.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-245.jpg?height=1690&width=1663&top_left_y=65&top_left_x=1669)

## Inverse problems via deep learning

In deep learning, we use a neural network to approximate the inverse mapping

$$
f_{\theta} \approx \mathcal{A}^{-1}
$$

i.e., we want $f_{\theta}(Y) \approx X_{\text {true }}$ for the measurements $X$ that we care about.

If we have $X_{1}, \ldots, X_{N}$ and $Y_{1}, \ldots, Y_{N}$ (but no direct knowledge of $\mathcal{A}$ ), we can solve

$$
\underset{\theta \in \mathbb{R}^{\boldsymbol{P}}}{\operatorname{minimize}} \sum_{i=1}^{N}\left\|f_{\theta}\left(Y_{i}\right)-X_{i}\right\|
$$

If we have $X_{1}, \ldots, X_{N}$ and knowledge of $\mathcal{A}$, we can solve

$$
\operatorname{minimize}_{\theta \in \mathbb{R}^{\mathfrak{P}}} \sum_{i=1}^{N}\left\|f_{\theta}\left[\mathcal{A}\left(X_{i}\right)\right]-X_{i}\right\|
$$

If we have $Y_{1}, \ldots, Y_{N}$ and knowledge of $\mathcal{A}$, we can solve

$$
\underset{\theta \in \mathbb{R}^{\mathcal{P}}}{\operatorname{minimize}} \sum_{i=1}^{N}\left\|\mathcal{A}\left[f_{\theta}\left(Y_{i}\right)\right]-Y_{i}\right\|
$$

## Image super-resolution

Given $X_{\text {true }} \in \mathbb{R}^{w \times h}$, we measure

$$
Y=\mathcal{A}\left(X_{\text {true }}\right)
$$

where $\mathcal{A}$ is a "downsampling" operator. So $Y \in \mathbb{R}^{w_{2} \times h_{2}}$ with $w_{2}<w$ and $h_{2}<h$. Goal is to recover $X_{\text {true }}$ from $Y$.

In the simplest setup, $\mathcal{A}$ is an average pool operator with $r \times r$ kernel and a stride $r$.

## SRCNN

In 2015, Dong et al. presented super-resolution convolutional neural network (SRCNN). They trained a 3-layer $\operatorname{CNN} f_{\theta}$ to learn the high-resolution reconstruction with the loss

$$
\mathcal{L}(\theta)=\sum_{i=1}^{N}\left\|f_{\theta}\left(\tilde{Y}_{i}\right)-X_{i}\right\|^{2}
$$

where $\tilde{Y}_{i} \in \mathbb{R}^{w \times h}$ is an upsampled version of $Y_{i} \in \mathbb{R}^{(w / r) \times(h / r)}$, i.e., $\tilde{Y}_{i}$ has the same number of pixels as $X_{i}$, but the image is pixelated or blurry. The goal is to have $f_{\theta}\left(\tilde{Y}_{i}\right)$ be a sharp reconstruction.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-248.jpg?height=630&width=1697&top_left_y=1131&top_left_x=980)

## SRCNN

SRCNN showed that simple learning based approaches can match the state-of theart performances of superresolution task.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-249.jpg?height=1205&width=1201&top_left_y=435&top_left_x=2036)

## VDSR

In 2016, Kim et al. presented VDSR. They trained a 20-layer CNN with a residual connection $f_{\theta}$ to learn the high-resolution reconstruction with the loss

$$
\mathcal{L}(\theta)=\sum_{i=1}^{N}\left\|f_{\theta}\left(\tilde{Y}_{i}\right)-X_{i}\right\|^{2}
$$

The residual connection was the key insight that enabled the training of much deeper CNNs.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-250.jpg?height=734&width=1420&top_left_y=1015&top_left_x=1511)

## VDSR

VDSR dominated all prior approaches not based on deep learning.
showed that simple learning based approaches can batch the state-of theart performances of super-resolution task.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-251.jpg?height=1630&width=1655&top_left_y=27&top_left_x=1673)

## Other inverse problem tasks and results

There are many other inverse problems. Almost all of them now require deep neural networks to achieve state-of-the-art results.

We won't spend more time on inverse problems in this course, but let's have fun and see a few other tasks and results. (These results are based on much more complex architectures and loss functions.)

## SRGAN

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-253.jpg?height=1366&width=945&top_left_y=363&top_left_x=225)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-253.jpg?height=1320&width=907&top_left_y=369&top_left_x=1171)

SRGAN
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-253.jpg?height=1362&width=920&top_left_y=369&top_left_x=2113)
C. Ledig, L. Theis, F. Huszar, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, and W. Shi, Photo-realistic single image super-resolution using a generative adversarial network, CVPR, 2017.

## SRGAN

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-254.jpg?height=903&width=1110&top_left_y=637&top_left_x=0)
bicubic interpolation
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-254.jpg?height=894&width=1116&top_left_y=642&top_left_x=1109)

SRGAN
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-254.jpg?height=890&width=1094&top_left_y=644&top_left_x=2234)
ground truth

## SRGAN

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-255.jpg?height=996&width=3190&top_left_y=561&top_left_x=59)

## Image colorization

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-256.jpg?height=1409&width=2055&top_left_y=333&top_left_x=588)
R. Zhang, P. Isola, and A. A. Efros, Colorful image colorization, ECCV, 2016.

## Image inpainting

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-257.jpg?height=1141&width=3332&top_left_y=497&top_left_x=0)

## Image inpainting

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-258.jpg?height=1354&width=2025&top_left_y=416&top_left_x=399)

## Linear operator $\cong$ matrix

Core tenet of linear algebra: matrices are linear operators and linear operators are matrices.

Let $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$ be linear, i.e.,

$$
f(x+y)=f(x)+f(y) \text { and } f(\alpha x)=\alpha f(x)
$$

for all $x, y \in \mathbb{R}^{n}$ and $\alpha \in \mathbb{R}$.

There exists a matrix $A \in \mathbb{R}^{m \times n}$ that represents $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$, i.e.,

$$
f(x)=A x
$$

for all $x \in \mathbb{R}^{n}$.

## Linear operator $\cong$ matrix

Let $e_{i}$ be the $i$-th unit vector, i.e., $e_{i}$ has all zeros elements except entry 1 in the $i$-th coordinate.

Given a linear $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}$, we can find the matrix

$$
A=\left[\begin{array}{llll}
A_{;, 1} & A_{;, 2} & \cdots & A_{;, n}
\end{array}\right] \in \mathbb{R}^{m \times n}
$$

representing $f$ with

$$
f\left(e_{j}\right)=A e_{j}=A_{;, j}
$$

for all $j=1, \ldots, n$, or with

$$
e_{i}^{\top} f\left(e_{j}\right)=e_{i}^{\top} A e_{j}=A_{i, j}
$$

for all $i=1, \ldots, m$ and $j=1, \ldots, n$.

## Linear operator $\not \neq$ matrix

In applied mathematics and machine learning, there are many setups where explicitly forming the matrix representation $A \in \mathbb{R}^{m \times n}$ is costly, even though the matrix-vector products $A x$ and $A^{\top} y$ are efficient to evaluate.

In machine learning, convolutions are the primary example. Other areas, linear operators based on FFTs are the primary example.

In such setups, the matrix representation is still a useful conceptual tool, even if we never intend to form the matrix.

## Transpose (adjoint) of a linear operator

Given a matrix $A$, the transpose $A^{\top}$ is obtained by flipping the row and column dimensions, i.e., $\left(A^{\top}\right)_{i j}=(A)_{j i}$. However, using this definition is not always the most effective when understanding the action of $A^{\top}$.

Another approach is to use the adjoint view. Since

$$
y^{\top}(A x)=\left(A^{\top} y\right)^{\top} x
$$

for any $x \in \mathbb{R}^{n}$ and $y \in \mathbb{R}^{m}$, understand the action of $A^{\top}$ by finding an expression of the form

$$
y^{\top} A x=\sum_{j=1}^{n}(\text { something })_{j} x_{j}=\left(A^{\top} y\right)^{\top} x
$$

## Example: 1D transpose convolution

Consider the 1D convolution represented by $A \in \mathbb{R}^{(n-f+1) \times n}$ defined with a given $w \in \mathbb{R}^{f}$ and

$$
A=\left[\begin{array}{cccccccc}
w_{1} & \cdots & w_{f} & 0 & \cdots & & & 0 \\
0 & w_{1} & \cdots & w_{f} & 0 & \cdots & & 0 \\
0 & 0 & w_{1} & \cdots & w_{f} & 0 & \cdots & 0 \\
\vdots & & & \ddots & & \ddots & & \vdots \\
0 & & \cdots & 0 & w_{1} & \cdots & w_{f} & 0 \\
0 & & \cdots & 0 & 0 & w_{1} & \cdots & w_{f}
\end{array}\right]
$$

Then we have

$$
(A x)_{j}=\sum_{i=1}^{f} w_{i} x_{j+i-1}
$$

## Example: 1D transpose convolution

and we have the following formula which coincides with transposing the matrix $A$.

$$
\begin{aligned}
y^{\top} A x & =\sum_{j=1}^{n-f+1} y_{j} \sum_{i=1}^{f} w_{i} x_{j+i-1} \\
& =\sum_{j=1}^{n-f+1} \sum_{i=1}^{f} y_{j} w_{i} x_{j+i-1} \sum_{k=1}^{n} \mathbf{1}_{\{k=j+i-1\}}
\end{aligned}
$$

$$
=\sum_{k=1}^{n} \sum_{j=1}^{n-f+1} \sum_{i=1}^{f} y_{j} w_{i} x_{k} \mathbf{1}_{\{k-j+1=i\}}
$$

$$
=\sum_{k=1}^{n} x_{k} \sum_{j=1}^{n-f+1} \sum_{i=1}^{f} w_{k-j+1} y_{j} \mathbf{1}_{\{k-j+1=i\}}
$$

$$
=\sum_{k=1}^{n} x_{k} \sum_{j=1}^{n-f+1} w_{k-j+1} y_{j} \sum_{i=1}^{f} \mathbf{1}_{\{k-j+1=i\}}
$$

$$
=\sum_{k=1}^{n} x_{k} \sum_{j=1}^{n-f+1} w_{k-j+1} y_{j} \mathbf{1}_{\{1 \leq k-j+1 \leq f\}}
$$

$$
=\sum_{k=1}^{n} x_{k} \sum_{j=1}^{n-f+1} w_{k-j+1} y_{j} \mathbf{1}_{\{j \leq k\}} \mathbf{1}_{\{k-f+1 \leq j\}}
$$

$$
=\sum_{k=1}^{n} x_{k} \sum_{j=\max (k-f+1,1)}^{\min (n-f+1, k)} w_{k-j+1} y_{j}=\left(A^{\top} y\right)^{\top} x
$$

## Operations increasing spatial dimensions

In image classification tasks, the spatial dimensions of neural networks often decrease as the depth progresses.

This is because we are trying to forget location information. (In classification, we care about what is in the image, but we do not where it is in the image.)

However, there are many networks for which we want to increase the spatial dimension:

- Linear layers
- Upsampling
- Transposed convolution


## Upsampling: Nearest neighbor

torch.nn.Upsample with mode='nearest'
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-266.jpg?height=681&width=1863&top_left_y=710&top_left_x=1029)

## Upsampling: Bilinear interpolation

Torch.nn.Upsample with mode='bilinear'
(We won't pay attention to the interpolation formula.)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-267.jpg?height=443&width=1028&top_left_y=833&top_left_x=1024)
'linear' interpolation is available for 1D data 'trilinear' interpolation is available for 3D data

| 6.0000 | 6.5000 | 7.5000 | 8.0000 |
| :--- | :--- | :--- | :--- |
| 5.2500 | 5.6875 | 6.5625 | 7.0000 |
| 3.7500 | 4.0625 | 4.6875 | 5.0000 |
| 3.0000 | 3.2500 | 3.7500 | 4.0000 |

## Transposed convolution

In transposed convolution, input neurons additively distribute values to the output via the kernel.

Before people noticed that this is the transpose

Input
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-268.jpg?height=239&width=252&top_left_y=918&top_left_x=1728)

Kernel
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-268.jpg?height=252&width=252&top_left_y=912&top_left_x=2460)

Output
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-268.jpg?height=379&width=1804&top_left_y=1320&top_left_x=1099)

| 0 | 0 | 1 |
| :--- | :---: | :---: |
| 0 | 4 | 6 |
| 4 | 12 | 9 |

## Transposed convolution

Input
Kernel
For each input neuron, multiply the kernel and add (accumulate) the value in the output.

Can accommodate strides, padding, and multiple channels.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-269.jpg?height=413&width=430&top_left_y=882&top_left_x=1214)
$+$
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-269.jpg?height=421&width=433&top_left_y=878&top_left_x=1757)
$+$
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-269.jpg?height=409&width=448&top_left_y=884&top_left_x=2302)
$+$
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-269.jpg?height=405&width=430&top_left_y=886&top_left_x=2864)
$=$

| 0 | 0 | 0 | 1 |
| :--- | :--- | :--- | :--- |
| 0 | 0 | 2 | 3 |
| 0 | 2 | 0 | 3 |
| 4 | 6 | 6 | 9 |

## Convolution visualized

## Transpose convolution visualized

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-271.jpg?height=1085&width=1039&top_left_y=495&top_left_x=1071)

## 2D trans. Conv. Iayer: Formal definition

Input tensor: $Y \in \mathbb{R}^{B \times C_{\mathrm{in}} \times m \times n}, B$ batch size, $C_{\mathrm{in}} \#$ of input channels.
Output tensor: $X \in \mathbb{R}^{B \times C_{\text {out }} \times\left(m+f_{1}-1\right) \times\left(n+f_{2}-1\right)}, B$ batch size, $C_{\text {out }} \#$ of output channels, $m, n \#$ of vertical and horizontal indices.
Filter $w \in \mathbb{R}^{C_{\text {in }} \times C_{\text {out }} \times f_{1} \times f_{2}}$, bias $b \in \mathbb{R}^{C_{\text {out }}}$. (If bias=False, then $b=0$.)

```
def trans_conv(Y, w, b):
    c_in, c_out, f1, f2 = w.shape
    batch, c_in, m, n = Y.shape
    X = torch.zeros(batch, c_out, m + f1 - 1, n + f2 - 1)
    for k in range(c_in):
        for i in range(Y.shape[2]):
            for j in range(Y.shape[3]):
                X[:, :, i:i+f1, j:j+f2] += Y[:, k, i, j].view(-1,1,1,1)*w[k, :, :, :].unsqueeze(0)
    return X + b.view(1,-1,1,1)
```


## Dependency by sparsity pattern

In a matrix representation $A$ of convolution. The dependencies of the inputs and outputs are represented by the non-zeros of $A$, i.e., the sparsity pattern of $A$.
If $A_{i j}=0$, then input neuron $j$ does not affect the output neuron $i$. If $A_{i j} \neq 0$, then $\left(A^{\top}\right)_{j i} \neq 0$. So if input neuron $j$ affects output neuron $i$ in convolution, then input neuron $i$ affects output neuron $j$ in transposed convolution.

We can combine this reasoning with our visual understanding of convolution. The diagram simultaneously illustrates the dependencies for both convolution and transposed convolution.

Input for conv
Output for trans.conv
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-273.jpg?height=775&width=1174&top_left_y=646&top_left_x=2156)

Output for conv Input for trans.conv.

## Semantic segmentation

In semantic
segmentation, the goal is to segment the image into semantically meaningful regions by classifying each pixel.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-274.jpg?height=1219&width=1944&top_left_y=372&top_left_x=1146)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-274.jpg?height=231&width=2055&top_left_y=1628&top_left_x=1158)

## Other related tasks

Object localization localizes a single object usually via a bounding box.

Object detection detects many objects, with the same class often repeated, usually via bounding boxes.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-275.jpg?height=520&width=626&top_left_y=369&top_left_x=1762)

CAT
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-275.jpg?height=532&width=613&top_left_y=372&top_left_x=2381)

CAT
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-275.jpg?height=813&width=1251&top_left_y=1056&top_left_x=1760)

## Other related tasks

Instance segmentation distinguishes multiple instances of the same object type.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-276.jpg?height=481&width=1230&top_left_y=678&top_left_x=406)

Image Recognition
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-276.jpg?height=478&width=1243&top_left_y=1292&top_left_x=395)

Object Detection
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-276.jpg?height=486&width=1235&top_left_y=671&top_left_x=1696)

Semantic Segmentation
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-276.jpg?height=486&width=1239&top_left_y=1288&top_left_x=1694)

Instance Segmentation

## Pascal VOC

We will use PASCAL Visual Object Classes (VOC) dataset for semantic segmentation.
(Dataset also contains labels for object detection.)

There are 21 classes: 20 main classes and 1 "unlabeled" class.

Data $X_{1}, \ldots, X_{N} \in \mathbb{R}^{3 \times m \times n}$ and labels $Y_{1}, \ldots, Y_{N} \in\{0,1, \ldots, 20\}^{m \times n}$, i.e., $Y_{i}$ provides a class label for every pixel of $X_{i}$.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-277.jpg?height=1667&width=864&top_left_y=0&top_left_x=2468)
image

## Loss for semantic segmentation

Consider the neural network

$$
f_{\theta}: \mathbb{R}^{3 \times m \times n} \rightarrow \mathbb{R}^{k \times m \times n}
$$

such that $\mu\left(f_{\theta}(X)\right)_{i j} \in \Delta^{k}$ is the probabilities for the $k$ classes for pixel $(i, j)$.

We minimize the sum of pixel-wise cross-entropy losses

$$
\mathcal{L}(\theta)=\sum_{l=1}^{N} \sum_{i=1}^{m} \sum_{j=1}^{n} \ell^{\mathrm{CE}}\left(f_{\theta}\left(X_{l}\right)_{i j},\left(Y_{l}\right)_{i j}\right)
$$

where $\ell^{C E}$ is the cross entropy loss.

## U-Net

The U-Net architecture:

- Reduce the spatial dimension to obtain high-level (coarse scale) features
- Upsample or transpose convolution to restore spatial dimension.
- Use residual connections across each dimension reduction stage.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-279.jpg?height=1727&width=1999&top_left_y=0&top_left_x=1331)


## Magnetic resonance imaging

Magnetic resonance imaging (MRI) is an inverse problem in which we partially* measure the Fourier transform of the patient and the goal is to reconstruct the patient's image.

So $X_{\text {true }} \in \mathbb{R}^{n}$ is the true original image (reshaped into a vector) with $n$ pixels or voxels and $\mathcal{A}\left[X_{\text {true }}\right] \in \mathbb{C}^{k}$ with $k \ll n$. (If $k=n$, MRI scan can take hours.)

Classical reconstruction algorithms rely on Fourier analysis, total variation regularization, compressed sensing, and optimization.

Recent state-of-the-art use deep neural networks.

## fastMRI dataset

A team of researchers from Facebook AI Research and NYU released a large MRI dataset to stimulate datadriven deep learning research for MRI reconstruction.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-281.jpg?height=1124&width=1702&top_left_y=514&top_left_x=1318)

## U-Net for inverse problems

Although U-Net was originally proposed as an architecture for semantic segmentation, it is also being used widely as one of the default architectures in inverse problems, including MRI reconstruction.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-282.jpg?height=966&width=2233&top_left_y=699&top_left_x=1095)

## Computational tomography

Computational tomography (CT) is an inverse problem in which we partially* measure the Radon transform of the patient and the goal is to reconstruct the patient's image.

So $X_{\text {true }} \in \mathbb{R}^{n}$ is the true original image (reshaped into a vector) with $n$ pixels or voxels and $\mathcal{A}\left[X_{\text {true }}\right] \in \mathbb{R}^{k}$ with $k \ll n$. (If $k=n$, the X -ray exposure to perform the CT scan can be harmful.)

Recent state-of-the-art use deep neural networks.

## U-Net for CT reconstruction

U-Net is also used as one of the default architectures in CT reconstruction
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-284.jpg?height=1115&width=3271&top_left_y=612&top_left_x=57)

# Chapter 5: Unsupervised Learning 

Mathematical Foundations of Deep Neural Networks
Fall 2022
Department of Mathematical Sciences
Ernest K. Ryu
Seoul National University

## Unsupervised learning

Unsupervised learning utilizes data $X_{1}, \ldots, X_{N}$ to learn the "structure" of the data. No labels are utilized.

There are a wide range of unsupervised learning tasks. In this class, we discuss just a few.

Generally, unsupervised learning tasks tend to have more mathematical complexity.

## Low-dimensional latent representation

Many high-dimensional data has some underlying low-dimensional structure.*
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-287.jpg?height=652&width=729&top_left_y=703&top_left_x=1311)

If you randomly generate the pixels of a color image $X \in \mathbb{R}^{3 \times m \times n}$, it will likely make no sense. Only a very small subset of pixel values correspond to meaningful images.

## Finding latent representations

In machine learning, especially in unsupervised learning, finding a "meaningful" lowdimensional latent representation is of interest.

A good lower-dimensional representation of the data implies you have a good understanding of the data.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-289.jpg?height=567&width=1128&top_left_y=91&top_left_x=2064)

An autoencoder (AE) has encoder $E_{\theta}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{r}$ and decoder $D_{\varphi}: \mathbb{R}^{r} \rightarrow \mathbb{R}^{n}$ networks, where $r \ll n$. (If $r \geq n$, AE learns identity mapping, so pointless.) The two networks are trained through the loss

$$
\mathcal{L}(\theta, \varphi)=\sum_{i=1}^{N}\left\|X_{i}-D_{\varphi}\left(E_{\theta}\left(X_{i}\right)\right)\right\|^{2}
$$

The low-dimensional output $E_{\theta}(X)$ is the latent vector. The encoder performs dimensionality reduction.

The autoencoder can be thought of as a deep non-linear generalization of the principle component analysis (PCA).

## Autoencoder with MNIST

PyTorch demo

## Applications of AE: Denoising

Autoencoders can be used to denoise or reconstruct corrupted images.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-291.jpg?height=1231&width=2818&top_left_y=439&top_left_x=240)
P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, and P.-A. Manzagol, Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion, $J M L R, 2010$.

## Applications of AE: Compression

Once an AE has been trained, storing the latent variable representation, rather than the original image can be used as a compression mechanism.

More generally, latent variable representations can be used for video compression. https://youtu.be/NqmMnjJ6GEg

## Applications of AE: Clustering

Train an AE and then perform clustering on the latent variables. For the clustering algorithm, one can use things like k-means, which groups together
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-293.jpg?height=1022&width=1855&top_left_y=714&top_left_x=744)

## Applications of AE: Clustering

Clustering is also referred to as unsupervised classification. Without labels, we want the group "similar" data.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-294.jpg?height=1107&width=2135&top_left_y=535&top_left_x=1199)

## Anomaly/outlier detection

Problem: detecting data that is significantly different from the data seen during training.

Insight: AE should not be able to faithfully reconstruct novel data.

Solution: Train an AE and define the score function to be the reconstruction loss:

$$
s(X)=\left\|X-D_{\varphi}\left(E_{\theta}(X)\right)\right\|^{2}
$$

If score is high, determine the datapoint to be an outliner. (Cf. hw7.)

## Probabilistic generative models

A probabilistic generative model learns a distribution $p_{\theta}$ from $X_{1}, \ldots, X_{N} \sim p_{\text {true }}$ such that $p_{\theta} \approx p_{\text {true }}$ and such that we can generate new samples $X \sim p_{\theta}$.

The ability to generate new synthetic data is interesting, but by itself not very useful.*

The structure of the data learned through the unsupervised learning is of higher value. However, we won't talk about the downstream applications in this course.

In this class, we will talk about flow models, VAEs, and GANs.

## Flow model: Change of variable formula combined with deep neural networks

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-297.jpg?height=1281&width=3330&top_left_y=457&top_left_x=2)

## Flow models

Fit a probability density function $p_{\theta}(x)$ with continuous data $X_{1}, \ldots, X_{N} \sim p_{\text {true }}(x)$.

- We want to fit the data $X_{1}, \ldots, X_{N}$ (or really the underlying distribution $p_{\text {true }}$ ) well.
- We want to be able to sample from $p_{\theta}$.
- (We want to get a good latent representation.)

We first develop the mathematical discussion with 1D flows, and then generalize the discussion to high dimensions.

## Example density model: Gaussian mixture model

$$
p_{\theta}(x)=\sum_{i=1}^{k} \pi_{i} \mathcal{N}\left(x ; \mu_{i}, \sigma_{i}^{2}\right)
$$

Parameters: means and variances of components, mixture weights

$$
\theta=\left(\pi_{1}, \ldots, \pi_{k}, \mu_{1}, \ldots, \mu_{k}, \sigma_{1}, \ldots, \sigma_{k}\right)
$$

Problems with GMM:

- Highly non-convex optimization problem. Can easily get stuck in local minima.
- It is does not have the representation power to express high-dimensional data.


## Example density model: Gaussian mixture model

GMM doesn't work with high-dimensional data. The sampling process is:
1.Pick a cluster center
2.Add Gaussian noise

If this is done with natural images, a realistic image can be generated only if it is a cluster center, i.e., the clusters must already be realistic images.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-300.jpg?height=831&width=1672&top_left_y=469&top_left_x=1622)

So then how do we fit a general (complex) density model?

## Math review: 1D continuous RV

A random variable $X$ is continuous if there exists a probability density function $p_{X}(x) \geq 0$ such that

$$
\mathbb{P}(a \leq X \leq b)=\int_{a}^{b} p_{X}(x) d x
$$

$p_{X}(x)$
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-301.jpg?height=686&width=834&top_left_y=133&top_left_x=2462)

In this case, we write $X \sim p_{X}$.

The cumulative distribution function (CDF) of $X$ is defined as

$$
F_{X}(t)=\mathbb{P}(X \leq t)=\int_{-\infty}^{t} p_{X}(x) d x
$$

$F_{X}(t)$ is a nondecreasing function.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-301.jpg?height=652&width=1004&top_left_y=916&top_left_x=2275)
$F_{X}(t)$ is a continuous function if $X$ is a continuous random variable.

## Naïve approach: prameterize $p_{\theta}$ as DNN

Naïve approach for fitting a density model. Represent $p_{\theta}(x)$ with DNN.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-302.jpg?height=545&width=1685&top_left_y=629&top_left_x=663)

There are some challenges:

1. How to ensure proper distribution?
$\int_{-\infty}^{+\infty} p_{\theta}(x) d x=1, \quad p_{\theta}(x) \geq 0, \quad x \in \mathbb{R}$
2. How to sample?

## Normalization of $p_{\theta}$

For discrete random variables, one can use the soft-max function $\mu: \mathbb{R}^{k} \rightarrow \mathbb{R}^{k}$ defined as

$$
\mu_{i}(z)_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{k} e^{z_{j}}}
$$

to normalize probabilities.

For continuous random variables, we can ensure $p_{\theta} \geq 0$ with $p_{\theta}(x)=e^{f_{\theta}(x)}$, where $f_{\theta}$ is the output of the neural network. However, ensuring the normalization

$$
\int_{-\infty}^{+\infty} p_{\theta}(x) d x=1
$$

is not a simple matter. (Any Bayesian statistician can tell you how difficult this is.)

## What happens if we ignore normalization?

Do we really need this normalization thing? Yes, we do.

Without normalization, one can just assign arbitrarily large probabilities everywhere when we perform maximum likelihood estimation:

$$
\underset{\theta \in \mathbb{R}^{p}}{\operatorname{maximize}} \sum_{i=1}^{N} \log p_{\theta}\left(X_{i}\right)
$$

The solution is to set $p_{\theta}(x)=M$ with $M \rightarrow \infty$.

We want model to place large probability on data $X_{1}, \ldots, X_{N}$ while placing small probability elsewhere. Normalization forces model to place small probability where data doesn't reside.

## Key insight: Parameterize $Z=f_{\theta}(X)$ with DNN

Key insight of normalizing flow: DNN outputs random variable $Z$, rather than $p_{\theta}(X)$
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-305.jpg?height=562&width=1804&top_left_y=633&top_left_x=374)

In normalizing flow, find $\theta$ such that the flow $f_{\theta}$ normalizes the random variable $X \sim p_{X}$ into $Z \sim \mathcal{N}(0,1)^{*}$.

Important questions to resolve:

1. How to train? (How to evaluate $p_{\theta}(x)$ ? DNN outputs $f_{\theta}$, not $p_{\theta}$.)
2. How to sample $X$ ?

## 1D change of variable formula

Assume $f$ is invertible, $f$ is differentiable, and $f^{-1}$ is differentiable.
If $X \sim p_{X}$, then $Z=f(X)$ has pdf

$$
p_{Z}(z)=p_{X}\left(f^{-1}(z)\right)\left|\frac{d x}{d z}\right|
$$

If $Z \sim p_{Z}$, then $X=f^{-1}(Z)$ has pdf

$$
p_{X}(x)=p_{Z}(f(x))\left|\frac{d f(x)}{d x}\right|
$$

Since $Z=f(X)$, one might think $p_{X}(x)=p_{Z}(z)=p_{Z}(f(x)) . \leftarrow$ This is wrong.

Invertibility of $f$ is essential; it is not a minor technical issue.

## Training flow models

Train model with MLE

$$
\underset{\theta \in \mathbb{R}^{p}}{\operatorname{maximize}} \sum_{i=1}^{N} \log p_{\theta}\left(X_{i}\right)=\underset{\theta \in \mathbb{R}^{p}}{\operatorname{maximize}} \sum_{i=1}^{N} \log p_{Z}\left(f_{\theta}\left(X_{i}\right)\right)+\log \left|\frac{\partial f_{\theta}}{\partial x}\left(X_{i}\right)\right|
$$

where $f_{\theta}$ is invertible and differentiable, and $X=f_{\theta}^{-1}(Z)$ with $Z \sim p_{Z}$ so

$$
p_{X}(x)=p_{Z}\left(f_{\theta}(x)\right)\left|\frac{\partial f_{\theta}}{\partial x}(x)\right|
$$

Can optimize with SGD, if we know how to perform backprop on $\left|\frac{\partial f_{\theta}}{\partial x}\left(X_{i}\right)\right|$. More on this later.

## Sampling from flow models

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-308.jpg?height=566&width=1973&top_left_y=465&top_left_x=370)

Step 1: Sample $Z \sim p_{Z}$
Step 2: Compute $X=f_{\theta}^{-1}(Z)$

## Requirements of flow $f_{\theta}$

Theoretical requirement:

- $f_{\theta}(x)$ invertible and differentiable.

Computational requirements:

- $f_{\theta}(x)$ and $\nabla_{\theta} f_{\theta}(x)$ efficient to evaluate (for training)
- $\left|\frac{\partial f_{\theta}}{\partial x}(x)\right|$ and $\nabla_{\theta}\left|\frac{\partial f_{\theta}}{\partial x}(x)\right|$ efficient to evaluate (for training)
- $f_{\theta}^{-1}$ efficient to evaluate (for sampling)


## Example: Flow to $Z$ ~ Uniform([0,1])

## Before training

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-310.jpg?height=579&width=813&top_left_y=442&top_left_x=580)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-310.jpg?height=604&width=813&top_left_y=438&top_left_x=1405)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-310.jpg?height=596&width=838&top_left_y=438&top_left_x=2209)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-310.jpg?height=422&width=418&top_left_y=1048&top_left_x=55)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-310.jpg?height=575&width=817&top_left_y=1065&top_left_x=578)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-310.jpg?height=584&width=809&top_left_y=1052&top_left_x=1403)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-310.jpg?height=592&width=801&top_left_y=1048&top_left_x=2236)

## Example: Flow to $Z \sim \operatorname{Beta}(5,5)$

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-311.jpg?height=1281&width=3139&top_left_y=397&top_left_x=25)

## Example: Flow to $Z \sim \mathcal{N}(0,1)$

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-312.jpg?height=1491&width=2943&top_left_y=384&top_left_x=17)

## 1D flow demonstration

PyTorch demo

## Universality of flows

Are flows universal, i.e., can $f_{\theta}^{-1}(Z) \sim p_{X}$ for any $X$ provided that $f_{\theta}$ can represent any invertible function?

Yes, 1D flows are universal due to the inverse CDF sampling technique.*

Higher dimensional flows are also universal as shown by Huang et al.\# or earlier by the general theory of optimal transport. ${ }^{\%}$

## Math review: Sampling via inverse CDF

Inverse CDF sampling is a technique for sampling $X \sim p_{X}$.
If $F_{X}(t)$ is furthermore a strictly increasing function, then $F_{X}$ is invertible, i.e., $F_{X}^{-1}$ exists.

Generate a random number $U \sim \operatorname{Uniform}([0,1])$ and compute $F_{X}^{-1}(U)$. Then

$$
F_{X}^{-1}(U) \sim p_{X}
$$

since

$$
\mathbb{P}\left(F_{X}^{-1}(U) \leq t\right)=\mathbb{P}\left(U \leq F_{X}(t)\right)=F_{X}(t)
$$

Technique can be generalized to when $F_{X}$ is not invertible.

## Universality of 1D flows

Composition of flows is a flow, and inverse of a flow is a flow

Universality of 1D flows:

- Use inverse CDF as flow to transform $X \sim p_{X}$ into $U \sim \operatorname{Uniform}([0,1])$ and $Z \sim \mathcal{N}(0,1)$ into $U \sim$ Uniform ( $[0,1]$ ).
- Compose flow $X \rightarrow U$ and inverse flow $U \rightarrow Z$
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-316.jpg?height=514&width=1231&top_left_y=599&top_left_x=1973)


## Jacobian notation

Let $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}$, such that

$$
f(x)=\left[\begin{array}{c}
f_{1}(x) \\
f_{2}(x) \\
\vdots \\
f_{n}(x)
\end{array}\right]
$$

The Jacobian matrix is

$$
\frac{\partial f}{\partial x}(x)=\left[\begin{array}{cccc}
\frac{\partial f_{1}}{\partial x_{1}}(x) & \frac{\partial f_{1}}{\partial x_{2}}(x) & \cdots & \frac{\partial f_{1}}{\partial x_{n}}(x) \\
\frac{\partial f_{2}}{\partial x_{1}}(x) & \frac{\partial f_{2}}{\partial x_{2}}(x) & \cdots & \frac{\partial f_{2}}{\partial x_{n}}(x) \\
\vdots & & \ddots & \vdots \\
\frac{\partial f_{n}}{\partial x_{1}}(x) & \frac{\partial f_{n}}{\partial x_{2}}(x) & \cdots & \frac{\partial f_{n}}{\partial x_{n}}(x)
\end{array}\right]=\left[\begin{array}{c}
\left(\nabla f_{1}(x)\right)^{\top} \\
\left(\nabla f_{2}(x)\right)^{\top} \\
\vdots \\
\left(\nabla f_{n}(x)\right)^{\top}
\end{array}\right]
$$

The Jacobian determinant is $\operatorname{det}\left(\frac{\partial f}{\partial x}\right)$. We use the notation

$$
\left|\frac{\partial f}{\partial x}(x)\right|=\left|\operatorname{det}\left(\frac{\partial f}{\partial x}(x)\right)\right|
$$

where the second $|\cdot|$ is the absolute value of the determinant. (This notation is not completely standard.)

## Math review: Multivariate change of variables

Let $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}$ be an invertible function such that both $f$ and $f^{-1}$ are differentiable. Let $U \subseteq \mathbb{R}^{n}$. Then

$$
\int_{f(U)} h(v) d v=\int_{U} h(f(u))\left|\frac{\partial f}{\partial u}(u)\right| d u
$$

for any $h: \mathbb{R}^{n} \rightarrow \mathbb{R}$. (Change of variable from $v=f(u)$ to $u=f^{-1}(v)$.)

## Math review: Multivariate continuous RV

A multivariate random variable $X \in \mathbb{R}^{n}$ is continuous if there exists a probability density function $p_{X}(x)$ such that

$$
\mathbb{P}(X \in A)=\int_{A} p_{X}(x) d x
$$

where the integral is over the volume $A \subseteq \mathbb{R}^{n}$. In this case, we write $X \sim p_{X}$.

The joint cumulative distribution function (the copula) does not seem to be useful in the context of high-dimensional flow models.

## Math review: Mult. change of variables for RV

Let $f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}$ be an invertible function such that both $f$ and $f^{-1}$ are differentiable. Let $X$ be a continuous random variable with probability density function $p_{X}$ and let $Y=f(X)$ have density $p_{Y}$. Then

$$
p_{X}(x)=p_{Y}(f(x))\left|\frac{\partial f}{\partial x}(x)\right|
$$

Proof)

$$
\mathbb{P}\left(f^{-1}(Y) \in A\right)=\mathbb{P}(Y \in f(A))=\int_{f(A)} p_{Y}(y) d y=\int_{A} p_{Y}(f(x))\left|\frac{\partial f}{\partial x}(x)\right| d x=\mathbb{P}(X \in A)
$$

Invertibility of $f$ is essential; it is not a minor technical issue.

## Math review: Determinant formulae

Fact: Determinant definitions in undergraduate linear algebra textbooks require exponentially many operations to compute:

$$
\operatorname{det}(A)=\sum_{\sigma \in S_{n}}\left(\operatorname{sgn}(\sigma) \prod_{i=1}^{n} a_{i, \sigma_{i}}\right)
$$

Efficient computation of determinant for general matrices and performing backprop through the computation is difficult. Therefore, high-dimensional flow model are designed to compute determinants only on simple matrices.

Product formula: if $A$ and $B$ are square, then

$$
\operatorname{det}(A B)=\operatorname{det}(A) \operatorname{det}(B)
$$

Block lower triangular formula: if $A \in \mathbb{R}^{n \times n}$ and $C \in \mathbb{R}^{m \times m}$, then

$$
\operatorname{det}\left(\begin{array}{ll}
A & 0 \\
B & C
\end{array}\right)=\operatorname{det}(A) \operatorname{det}(C)
$$

Lower triangular formula: if $a_{1}, \ldots, a_{n} \in \mathbb{R}$ and $*$ represents arbitrary values, then

$$
\operatorname{det}\left(\begin{array}{cccc}
a_{1} & 0 & \cdots & 0 \\
* & a_{2} & & \vdots \\
* & * & \ddots & 0 \\
* & * & * & a_{n}
\end{array}\right)=\prod_{i=1}^{n} a_{i}
$$

Upper triangular formula: same as for lower triangular matrices.

## Training high-dim flow models

Train model with MLE

$$
\underset{\theta \in \mathbb{R}^{p}}{\operatorname{maximize}} \sum_{i=1}^{N} \log p_{\theta}\left(X_{i}\right)=\underset{\theta \in \mathbb{R}^{p}}{\operatorname{maximize}} \sum_{i=1}^{N} \log p_{Z}\left(f_{\theta}\left(X_{i}\right)\right)+\log \left|\frac{\partial f_{\theta}}{\partial x}\left(X_{i}\right)\right|
$$

where $f_{\theta}(z)$ is invertible and differentiable, and $X=f^{-1}(Z)$ with $Z \sim p_{Z}$ so

$$
p_{X}(x)=p_{Z}\left(f_{\theta}(x)\right)\left|\frac{\partial f_{\theta}}{\partial x}(x)\right|
$$

(Exactly the same formula as with 1D flow.)

Can optimize with SGD, if we know how to perform backprop on $\left|\frac{\partial f_{\theta}}{\partial x}\left(X_{i}\right)\right|$.

## Composing flows

Flows can be composed to increase expressiveness. (Deep NN more expressive.)
Consider composition of $k$ flows

$$
\begin{aligned}
& x \rightarrow f_{1} \rightarrow f_{2} \rightarrow \cdots \rightarrow f_{k} \rightarrow z \\
& z=f_{k} \circ \cdots \circ f_{1}(x) \\
& x=f_{1}^{-1} \circ \cdots \circ f_{k}^{-1}(z)
\end{aligned}
$$

Determinant computation splits nicely due to chain rule and product formula

$$
\begin{aligned}
& \operatorname{det}\left(\frac{\partial z}{\partial x}\right)=\operatorname{det}\left(\frac{\partial f_{k}}{\partial f_{k-1}} \cdots \frac{\partial f_{1}}{\partial f_{0}}\right)=\operatorname{det}\left(\frac{\partial f_{k}}{\partial f_{k-1}}\right) \cdots \operatorname{det}\left(\frac{\partial f_{1}}{\partial f_{0}}\right) \\
& \log p_{\theta}(x)=\log p_{\theta}(z)+\sum_{i=1}^{k} \log \left|\frac{\partial f_{i}}{\partial f_{i-1}}\right|
\end{aligned}
$$

## Basic example: Affine flows

An affine (linear) transformation

$$
f_{A, b}(x)=A^{-1}(x-b)
$$

is a flow if matrix $A$ is invertible. Then

$$
\frac{\partial f_{A, b}}{\partial x}=A^{-1}
$$

and

$$
\left|\frac{\partial f_{A, b}}{\partial x}\right|=\left|\operatorname{det}\left(A^{-1}\right)\right|=\frac{1}{|\operatorname{det}(A)|}
$$

Sampling: $X=A Z+b$, where $Z \sim \mathcal{N}(0, I)$.
Problem with affine flows:

- Computing $|\operatorname{det}(A)|$ is expensive and performing backprop over it is difficult. We want $\frac{\partial f_{A, b}}{\partial x}$ to be further structured so that determinant is easy to compute.
- One affine flow is insufficient to generate complex data. However, composing multiple affine flows yields an affine flow and therefore is pointless. We need to introduce nonlinearities.


## Coupling flows

A coupling flow is a general and practical approach for constructing non-linear flows.

Partition input into two disjoint subsets $x=\left(x^{A}, x^{B}\right)$. Then

$$
f(x)=\left(x^{A}, \hat{f}\left(x^{B} \mid \psi_{\theta}\left(x^{A}\right)\right)\right)
$$

where $\psi_{\theta}$ is a neural network and $\hat{f}\left(x^{B} \mid \psi_{\theta}\left(x^{A}\right)\right)$ is another flow whose parameters depend on $x^{A}$.

## Coupling flow: forward evaluation

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-326.jpg?height=1157&width=2995&top_left_y=510&top_left_x=233)

## Coupling flow: inverse evaluation

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-327.jpg?height=1132&width=3152&top_left_y=633&top_left_x=57)

## Jacobian of coupling flows

The Jacobian of a coupling flow has a nice block structure

$$
\frac{\partial f_{\theta}}{\partial x}(x)=\left[\begin{array}{cc}
I & 0 \\
\frac{\partial \hat{f}}{\partial x^{A}}\left(x^{B} \mid \psi_{\theta}\left(x^{A}\right)\right) & \frac{\partial \hat{f}}{\partial x^{B}}\left(x^{B} \mid \psi_{\theta}\left(x^{A}\right)\right)
\end{array}\right]
$$

which leads to the simplified determinant formula

$$
\operatorname{det}\left(\frac{\partial f_{\theta}}{\partial x}(x)\right)=\operatorname{det}\left(\frac{\partial \hat{f}}{\partial x^{B}}\left(x^{B} \mid \psi_{\theta}\left(x^{A}\right)\right)\right)
$$

Note $\frac{\partial \hat{f}}{\partial x^{A}}\left(x^{B} \mid \psi_{\theta}\left(x^{A}\right)\right)$, which will be very complicated, does not appear in the determinant.

## Coupling transformation $\hat{f}(x \mid \psi)$

Additive transformations (NICE)*

$$
\hat{f}(x \mid \psi)=x+t
$$

where $\psi=t$.

Affine transformations (Real NVP) ${ }^{\#}$

$$
\hat{f}(x \mid \psi)=e^{s} \odot x+t
$$

where $\psi=(s, t)$.

Other transformations studied throughout the literature.

## NICE (Non-linear Independent Components Estimation)

NICE uses additive coupling layers:
Split variables in half: $x_{1: n / 2}, x_{n / 2: n}$

$$
\begin{aligned}
& z_{1: n / 2}=x_{1: n / 2} \\
& z_{n / 2: n}=x_{n / 2: n}+t_{\theta}\left(x_{1: n / 2}\right)
\end{aligned}
$$

Easily invertible:

$$
\begin{aligned}
& x_{1: n / 2}=z_{1: n / 2} \\
& x_{n / 2: n}=z_{n / 2: n}-t_{\theta}\left(x_{1: n / 2}\right)
\end{aligned}
$$

Jacobian determinant is easy to compute:
$\operatorname{det} \frac{\partial f_{\theta}}{\partial x}(x)=\operatorname{det}\left[\begin{array}{cc}I & 0 \\ \frac{\partial \hat{f}}{\partial x^{A}}\left(x^{B} \mid \psi_{\theta}\left(x^{A}\right)\right) & \frac{\partial \hat{f}}{\partial x^{B}}\left(x^{B} \mid \psi_{\theta}\left(x^{A}\right)\right)\end{array}\right]=\operatorname{det}\left[\begin{array}{cc}I & 0 \\ \frac{\partial \hat{f}}{\partial x^{A}}\left(x^{B} \mid \psi_{\theta}\left(x^{A}\right)\right) & I\end{array}\right]=1$

## Real NVP (Real-valued Non-Volume Preserving)

Real NVP uses affine coupling layers:

$$
\begin{aligned}
& z_{1: n / 2}=x_{1: n / 2} \\
& z_{n / 2: n}=e^{s_{\theta}\left(x_{1: n / 2}\right)} \odot x_{n / 2: n}+t_{\theta}\left(x_{1: n / 2}\right)
\end{aligned}
$$

Easily invertible:

$$
\begin{aligned}
& x_{1: n / 2}=z_{1: n / 2} \\
& x_{n / 2: n}=\left(z_{n / 2: n}-t_{\theta}\left(x_{1: n / 2}\right)\right) \odot e^{-s_{\theta}\left(x_{1: n / 2}\right)}
\end{aligned}
$$

Jacobian determinant is easy to compute:

$$
\begin{aligned}
\operatorname{det} \frac{\partial f_{\theta}}{\partial x}(x) & =\operatorname{det}\left[\begin{array}{cc}
I & 0 \\
\frac{\partial \hat{f}}{\partial x^{A}}\left(x^{B} \mid \psi_{\theta}\left(x^{A}\right)\right) & \frac{\partial \hat{f}}{\partial x^{B}}\left(x^{B} \mid \psi_{\theta}\left(x^{A}\right)\right)
\end{array}\right] \\
& =\operatorname{det}\left[\begin{array}{cc}
I & 0 \\
\frac{\partial \hat{f}}{\partial x^{A}}\left(x^{B} \mid \psi_{\theta}\left(x^{A}\right)\right) & \operatorname{diag}\left(e^{s_{\theta}\left(x_{1: n / 2}\right)}\right)
\end{array}\right]=\exp \left(\mathbf{1}_{n / 2}^{\top} s_{\theta}\left(x_{1: n / 2}\right)\right)
\end{aligned}
$$

## Real NVP - Results

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-332.jpg?height=1366&width=2822&top_left_y=365&top_left_x=257)

## How to partition variables?

Note that the additive and affine coupling layers of NICE and Real NVP are nonlinear mappings from $x_{1: n}$ to $z_{1: n}$, since $s_{\theta}\left(x_{1: n / 2}\right)$ and $t_{\theta}\left(x_{1: n / 2}\right)$ are nonlinear.

Flow models compose multiple nonlinear flows. But if $x_{1: n / 2}$ is always unchanged, then the full composition will leave it unchanged. Therefore, we change the partitioning for every coupling layer.

## NICE architecture

PyTorch demo

## Real NVP variable partitioning

Two partition strategies:

1. Partition with
checkerboard pattern.
2. Reshape tensor and then partition channelwise.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-335.jpg?height=975&width=2036&top_left_y=720&top_left_x=1294)

## Real NVP Architecture

Input $X$ : $c \times 32 \times 32$ image with $c=3$
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-336.jpg?height=868&width=2286&top_left_y=0&top_left_x=997)

## Layer 1: Input $X: c \times 32 \times 32$

- Checkerboard $\times 3$, channel reshape into $4 c \times 16 \times 16$, channel $\times 3$
- Output: Split result to get $X_{1}: 2 c \times 16 \times 16$ and $Z_{1}: 2 c \times 16 \times 16$ (fine-grained latents) Layer 2: Input $X_{1}: 2 c \times 16 \times 16$ from layer 1
- Checkerboard $\times 3$, channel reshape into $8 c \times 8 \times 8$, channel $\times 3$
- Split result to get $X_{2}: 4 c \times 8 \times 8$ and $Z_{2}: 4 c \times 8 \times 8$ (coarser latents)

Layer 3: Input $X_{2}: 4 c \times 8 \times 8$ from layer 2

- Checkerboard $\times 3$, channel reshape into $16 c \times 4 \times 4$, channel $\times 3$
- Get $Z_{3}: 16 c \times 4 \times 4$ (latents for highest-level details)


## Batch normalization

To train deep flows, BN is helpful. However, the large model size forces the use of small batch sizes, and BN is not robust with small batch sizes. RealNVP uses a modified form of BN

$$
x \mapsto \frac{x-\tilde{\mu}}{\sqrt{\tilde{\sigma}^{2}+\varepsilon}}
$$

(No $\beta$ and $\gamma$ parameters.) This layer has the log Jacobian determinant

$$
-\frac{1}{2} \sum_{i} \log \left(\tilde{\sigma}_{i}^{2}+\varepsilon\right)
$$

The mean and variance parameters are updated with

$$
\begin{aligned}
\tilde{\mu}_{k+1} & =\rho \tilde{\mu}_{k}+(1-\rho) \hat{\mu}_{k} \\
\tilde{\sigma}_{k+1}^{2} & =\rho \tilde{\sigma}_{k}^{2}+(1-\rho) \hat{\sigma}_{k}^{2}
\end{aligned}
$$

where $\rho$ is the momentum. During gradient computation, only backprop through the current batch statistics $\hat{\mu}_{k}$ and $\hat{\sigma}_{k}^{2}$.

## $s_{\theta}$ and $t_{\theta}$ networks

The $s_{\theta}$ and $t_{\theta}$ do not need to be invertible. The original RealNVP paper does not describe its construction.

We let $\left(s_{\theta}, t_{\theta}\right)$ be a deep (20-layer) convolutional neural network using residual connections and standard batch normalization.

## Real NVP architecture

PyTorch demo

## Glow paper

The authors of the Glow paper also released a blog post.
https://openai.com/blog/glow/

## FFJORD

Instead of a discrete composition of flows, what if we have a continuous-time flow?

$$
\begin{aligned}
z_{0} & =x \\
z_{t} & =z_{0}+\int_{0}^{t} h\left(t, z_{t}\right) d t \\
f(x) & =z_{1}
\end{aligned}
$$

Inverse:

$$
\begin{aligned}
z_{1} & =z \\
z_{t} & =z_{1}-\int_{t}^{1} h\left(t, z_{t}\right) d t \\
f^{-1}(z) & =z_{0}
\end{aligned}
$$

R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud, Neural ordinary differential equations, NeurIPS, 2018.
W. GrathwohI, R. T. Q. Chen, J. Bettencourt, I. Sutskever, and D. Duvenaud, FFJORD: Free-form continuous dynamics for scalable reversible generative

## Math review: Conditional probabilities

Let $A$ and $B$ be probabilistic events. Assume $A$ has nonzero probability.

Conditional probability satisfies

$$
\mathbb{P}(B \mid A) \mathbb{P}(A)=\mathbb{P}(A \cap B)
$$

Bayes' theorem is an application of conditional probability:

$$
\mathbb{P}(B \mid A)=\frac{\mathbb{P}(A \mid B) \mathbb{P}(B)}{\mathbb{P}(A)}
$$

## Math review: Conditional densities

Let $X \in \mathbb{R}^{m}$ and $Z \in \mathbb{R}^{n}$ be continuous random variables with joint density $p(x, z)$.

The marginal densities are defined by

$$
p_{X}(x)=\int_{\mathbb{R}^{n}} p(x, z) d z, \quad p_{Z}(z)=\int_{\mathbb{R}^{m}} p(x, z) d x
$$

The conditional density function $p(z \mid x)$ has the following properties

$$
\begin{gathered}
\mathbb{P}(Z \in S \mid X=x)=\int_{S} p(z \mid x) d z \\
p(z \mid x) p_{X}(x)=p(x, z), \quad p(z \mid x)=\frac{p(x \mid z) p_{Z}(Z)}{p_{X}(X)}
\end{gathered}
$$

## Variational autoencoders (VAE)

These are synthetic (fake) images.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-344.jpg?height=1381&width=2093&top_left_y=365&top_left_x=1220)

## Variational autoencoders (VAE)

Key idea of VAE:

- Latent variable model with conditional probability distribution represented by $p_{\theta}(x \mid z)$.
- Efficiently estimate $p_{\theta}(x)=\mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}(x \mid Z)\right]$ by importance sampling with $Z \sim q_{\phi}(z \mid x)$.

We can interpret $q_{\phi}(z \mid x)$ as an encoder and $p_{\theta}(x \mid z)$ as a decoder.

VAEs differ from autoencoders as follows:

- Derivations (latent variable model vs. dimensionality reduction)
- VAE regularizes/controls latent distribution, while AE does not.


## Latent variable model

Assumption on data $X_{1}, \ldots, X_{N}$ : Assumes there is an underlying latent variable $Z$ representing the "essential structure" of the data and an observable variable $X$ which generation is conditioned on $Z$. Implicitly assumes the conditional randomness of $X \sim p_{X \mid Z}$ is significantly smaller than the overall randomness $X \sim p_{X}$.

Example: $X$ is a cat picture. $Z$ encodes information about the body position, fur color, and facial expression of a cat. Latent variable $Z$ encodes the overall content of the image, but $X$ does contain details not specified in $Z$.

Specification VAE's model: VAEs implements a latent variable model with a NN that generates $X$ given $Z$. More precisely, NN is a deterministic function that outputs the conditional distribution $p_{\theta}(x \mid Z)$, and $X$ is randomly generated according to this distribution. This structure may effectively learn the latent structure from data if the assumption on data is accurate.

## Latent variable model

Sampling process:

$$
X \sim p_{\theta}(x \mid Z), \quad Z \sim p_{Z}(z)
$$

Usually $p_{Z}$ is a Gaussian (fixed) and $p_{\theta}(x \mid z)$ is a NN parameterized by $\theta$.

Evaluating density (likelihood):
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-347.jpg?height=256&width=235&top_left_y=276&top_left_x=2111)
$p_{\theta}(x \mid z)$

$$
p_{\theta}\left(X_{i}\right)=\int_{z} p_{Z}(z) p_{\theta}\left(X_{i} \mid z\right) d z=\mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}\left(X_{i} \mid Z\right)\right]
$$

Training via MLE: $\underset{\theta \in \Theta}{\operatorname{maximize}} \sum_{i=1}^{N} \log p_{\theta}\left(X_{i}\right)=\underset{\theta \in \Theta}{\operatorname{maximize}} \sum_{i=1}^{N} \log \mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}\left(X_{i} \mid Z\right)\right]$

## Latent variable model

When $p_{Z}$ is a discrete:

$$
p_{\theta}(x)=\mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}(x \mid Z)\right]=\sum_{z} p_{Z}(z) p_{\theta}(x \mid Z)
$$

When $p_{Z}$ is a continuous:

$$
p_{\theta}(x)=\mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}(x \mid Z)\right]=\int_{z} p_{Z}(z) p_{\theta}(x \mid z) d z
$$

To clarify, specification of $p_{Z}(z)$ and $p_{\theta}(x \mid z)$ fully determines $p_{\theta}(x)$ (as above) and

$$
p_{\theta}(z \mid x)=\frac{p_{\theta}(x \mid z) p_{Z}(z)}{p_{\theta}(x)}
$$

## Latent variable model: Training

Training

$$
\underset{\theta \in \Theta}{\operatorname{maximize}} \sum_{i=1}^{N} \log p_{\theta}\left(X_{i}\right)=\underset{\theta \in \Theta}{\operatorname{maximize}} \sum_{i=1}^{N} \log \mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}\left(X_{i} \mid Z\right)\right]
$$

requires evaluation $\mathbb{E}_{Z}$.

Scenario 1: If $Z$ is discrete and takes a few of values, then compute $\sum_{z}$ exactly.

Scenario 2: If $Z$ takes many values or if it is a continuous, then $\sum_{z}$ or $\mathbb{E}_{Z}$ is impractical to compute. In this case, approximate expectation with Monte Carlo and importance sampling.

## Example latent variable model: Mixture of Gaussians

Mixture of 3 Gaussians in $\mathbb{R}^{2}$, uniform prior over components. (We can make the mixture weights a trainable parameter.)

$$
\begin{gathered}
p_{Z}(Z=A)=p_{Z}(Z=B)=p_{Z}(Z=C)=\frac{1}{3} \\
p_{\theta}(x \mid Z=k)=\frac{1}{2 \pi\left|\Sigma_{k}\right|^{\frac{1}{2}}} \exp \left(-\frac{1}{2}\left(x-\mu_{k}\right)^{\top} \Sigma_{k}^{-1}\left(x-\mu_{k}\right)\right)
\end{gathered}
$$

Training objective:

$$
\begin{aligned}
\underset{\mu, \Sigma}{\operatorname{maximize}} \sum_{i=1}^{N} \log p_{\theta}\left(X_{i}\right)=\underset{\mu, \Sigma}{\operatorname{maximize}} \sum_{i=1}^{N} \log [ & \frac{1}{3} \frac{1}{2 \pi\left|\Sigma_{A}\right|^{\frac{1}{2}}} \exp \left(-\frac{1}{2}\left(X_{i}-\mu_{A}\right)^{\top} \Sigma_{A}^{-1}\left(X_{i}-\mu_{A}\right)\right) \\
& +\frac{1}{3} \frac{1}{2 \pi\left|\Sigma_{B}\right|^{\frac{1}{2}}} \exp \left(-\frac{1}{2}\left(X_{i}-\mu_{B}\right)^{\top} \Sigma_{B}^{-1}\left(X_{i}-\mu_{B}\right)\right) \\
& \left.+\frac{1}{3} \frac{1}{2 \pi\left|\Sigma_{C}\right|^{\frac{1}{2}}} \exp \left(-\frac{1}{2}\left(X_{i}-\mu_{C}\right)^{\top} \Sigma_{C}^{-1}\left(X_{i}-\mu_{C}\right)\right)\right]
\end{aligned}
$$

## Example: 2D mixture of Gaussians

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-351.jpg?height=1056&width=2361&top_left_y=637&top_left_x=308)

## VAE outline

Train latent variable model with MLE

$$
\underset{\theta \in \Theta}{\operatorname{maximize}} \sum_{i=1}^{N} \log p_{\theta}\left(X_{i}\right)=\underset{\theta \in \Theta}{\operatorname{maximize}} \sum_{i=1}^{N} \log \mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}\left(X_{i} \mid Z\right)\right]
$$

Outline of variational autoencoder (VAE):

1. Approximate intractable objective with a single $Z$ sample

$$
\sum_{i=1}^{N} \log \mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}\left(X_{i} \mid Z\right)\right] \approx \sum_{i=1}^{N} \log p_{\theta}\left(X_{i} \mid Z_{i}\right), \quad Z_{i} \sim p_{Z}
$$

2. Improve accuracy of approximation by sampling $Z_{i}$ with importance sampling

$$
\sum_{i=1}^{N} \log \mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}\left(X_{i} \mid Z\right)\right] \approx \sum_{i=1}^{N} \log \frac{p_{\theta}\left(X_{i} \mid Z_{i}\right) p_{Z}\left(Z_{i}\right)}{q_{i}\left(Z_{i}\right)}, \quad Z_{i} \sim q_{i}
$$

3. Optimize approximate objective with SGD.

## IWAE outline

Importance weighted autoencoders (IWAE) approximates intractable with $K$ samples of $Z$ :

$$
\sum_{i=1}^{N} \log \mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}\left(X_{i} \mid Z\right)\right] \approx \sum_{i=1}^{N} \log \frac{1}{K} \sum_{k=1}^{K} \frac{p_{\theta}\left(X_{i} \mid Z_{i, k}\right) p_{Z}\left(Z_{i, k}\right)}{q_{i}\left(Z_{i, k}\right)}, \quad Z_{i, 1}, \ldots, Z_{i, K} \sim q_{i}
$$

More on this in hw 9.

## Why does VAE need IS?

Sampling $Z_{i} \sim p_{Z}$ results in a high-variance estimator:

$$
\mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}\left(X_{i} \mid Z\right)\right] \approx p_{\theta}\left(X_{i} \mid Z_{i}\right),
$$

In the Gaussian mixture example, only $1 / 3$ of the $Z$ samples meaningfully contribute to the estimate. More specifically, if $X_{i}$ is near $\mu_{A}$ but is far from $\mu_{B}$ and $\mu_{C}$, then $p_{\theta}\left(X_{i} \mid Z=A\right) \gg 0$ but $p_{\theta}\left(X_{i} \mid Z=B\right) \approx 0$ and $p_{\theta}\left(X_{i} \mid Z=C\right) \approx 0$.

The issue worsens as the observable and latent variable dimension increases.

## Naïvely using IS for each $X_{i}$

To improve estimation of $\mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}\left(X_{i} \mid Z\right)\right]$, consider importance sampling (IS) with sampling distribution $Z_{i} \sim q_{i}(z)$ :

$$
\mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}\left(X_{i} \mid Z\right)\right] \approx p_{\theta}\left(X_{i} \mid Z_{i}\right) \frac{p_{Z}\left(Z_{i}\right)}{q_{i}\left(Z_{i}\right)}
$$

Optimal IS sampling distribution

$$
q_{i}^{\star}(z)=\frac{p_{\theta}\left(X_{i} \mid z\right) p_{Z}(z)}{p_{\theta}\left(X_{i}\right)}=p_{\theta}\left(z \mid X_{i}\right)
$$

To clarify, optimal sampling distribution depends on $X_{i}$. To clarify, $p_{\theta}\left(X_{i}\right)$ is the unkown normalizing factor so $p_{\theta}\left(z \mid X_{i}\right)$ is also unkown. We call $q_{i}^{\star}(z)=p_{\theta}\left(z \mid X_{i}\right)$ the true posterior distribution and we will soon consider the approximation $q_{\phi}(z \mid x) \approx p_{\theta}(z \mid x)$, which we call the approximate posterior.

## Naïvely using IS for each $X_{i}$

For each $X_{i}$, consider

$$
\begin{gathered}
\underset{q_{i}}{\operatorname{minimize}} D_{\mathrm{KL}}\left(q_{i}(\cdot) \| p_{\theta}\left(\cdot \mid X_{i}\right)\right) \\
=\underset{q_{i}}{\operatorname{minimize}} \mathbb{E}_{Z \sim q_{i}} \log \left(\frac{q_{i}(Z)}{p_{\theta}\left(Z \mid X_{i}\right)}\right) \\
=\underset{q_{i}}{\operatorname{minimize}} \mathbb{E}_{Z \sim q_{i}} \log \left(\frac{q_{i}(Z)}{p_{\theta}\left(X_{i} \mid Z\right) p_{Z}(Z) / p_{\theta}\left(X_{i}\right)}\right) \\
=\underset{Z \sim q_{i}}{\operatorname{minimize}}\left[\log q_{i}(Z)-\log p_{Z}(Z)-\log p_{\theta}\left(X_{i} \mid Z\right)\right]+\log p_{\theta}\left(X_{i}\right)
\end{gathered}
$$

Note, $q_{i}(z), p_{Z}(z)$, and $p_{\theta}(x \mid z)$ are tractable/known while $p_{\theta}\left(X_{i}\right)$ and $p_{\theta}\left(z \mid X_{i}\right)$ are intractable/unknown. Since $\log p_{\theta}\left(X_{i}\right)$ does not depend on $q_{i}$, all quantities needed in the optimization problems are tractable. However, solving this minimization problem to obtain each $q_{i}$ for each data point $X_{i}$ is computationally too expensive.

## Non-amortized inference

Individual inference (not amortized): For each $X_{1}, \ldots, X_{N}$, find corresponding optimal $q_{1}, \ldots, q_{N}$ by solving

$$
\underset{q_{i}}{\operatorname{minimize}} \quad D_{\mathrm{KL}}\left(q_{i}(\cdot) \| p_{\theta}\left(\cdot \mid X_{i}\right)\right)
$$

This is expensive as it requires solving $N$ separate optimization problems.

We need variational approach and amortized inference.

## Variational approach and amortized inference

General principle of variational approach: We can't directly use the $q$ we want. So, instead, we propose a parameterized distribution $q_{\phi}$ that we can work with easily (in this case, sample from easily), and find a parameter setting that makes it as good as possible.

Parametrization of VAE:

$$
q_{\phi}\left(z \mid X_{i}\right) \approx q_{i}^{\star}(z)=p_{\theta}\left(z \mid X_{i}\right) \quad \text { for all } i=1, \ldots, N
$$

Amortized inference: Train a neural network $q_{\phi}(\cdot \mid x)$ such that $q_{\phi}\left(\cdot \mid X_{i}\right)$ approximates the optimal $q_{i}(\cdot)$.

$$
\underset{\phi \in \Phi}{\operatorname{minimize}} \sum_{i=1}^{N} D_{\mathrm{KL}}\left(q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{\theta}\left(\cdot \mid X_{i}\right)\right)
$$

Approximation $q_{\phi}\left(z \mid X_{i}\right) \approx p_{\theta}\left(z \mid X_{i}\right)$ is often less precise than that of individual inference $q_{i}(z) \approx$ $p_{\theta}\left(z \mid X_{i}\right)$, but amortized inference is often significantly faster.

## Encoder $q_{\phi}$ optimization

In analogy with autoencoders, we call $q_{\phi}$ the encoder.

Optimization problem for encoder

$$
\begin{aligned}
& \underset{\phi \in \Phi}{\operatorname{minimize}} \sum_{i=1}^{N} D_{\mathrm{KL}}\left(q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{\theta}\left(\cdot \mid X_{i}\right)\right) \\
& \quad=\underset{\phi \in \Phi}{\operatorname{maximize}} \sum_{i=1}^{N} \mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\log \left(\frac{p_{\theta}\left(X_{i} \mid Z\right) p_{Z}(Z)}{q_{\phi}\left(Z \mid X_{i}\right)}\right)\right]+\text { constant independent of } \phi \\
& \quad=\underset{\phi \in \Phi}{\operatorname{maximize}} \sum_{i=1}^{N} \mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\log p_{\theta}\left(X_{i} \mid Z\right)\right]-D_{\mathrm{KL}}\left(q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{Z}(\cdot)\right)
\end{aligned}
$$

## Decoder $p_{\theta}$ optimization

In analogy with autoencoders, we call $p_{\theta}$ the decoder. Perform approximate MLE with

$$
\begin{aligned}
& \underset{\theta \in \Theta}{\operatorname{maximize}} \sum_{i=1}^{N} \log p_{\theta}\left(X_{i}\right)=\underset{\theta \in \Theta}{\operatorname{maximize}} \sum_{i=1}^{N} \log \mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}\left(X_{i} \mid Z\right)\right] \\
& \stackrel{(a)}{\approx} \underset{\theta \in \Theta}{\operatorname{maximize}} \sum_{i=1}^{N} \log \left(\frac{p_{\theta}\left(X_{i} \mid Z_{i}\right) p_{Z}\left(Z_{i}\right)}{q_{\phi}\left(Z_{i} \mid X_{i}\right)}\right), \quad Z_{i} \sim q_{\phi}\left(z \mid X_{i}\right) \\
& \stackrel{(b)}{\approx} \underset{\theta \in \Theta}{\operatorname{maximize}} \\
& \sum_{i=1}^{N} \mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\log \left(\frac{p_{\theta}\left(X_{i} \mid Z\right) p_{Z}(Z)}{q_{\phi}\left(Z \mid X_{i}\right)}\right)\right] \\
&=\underset{\theta \in \Theta}{\operatorname{maximize}} \sum_{i=1}^{N} \mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\log p_{\theta}\left(X_{i} \mid Z\right)\right]-D_{\mathrm{KL}}\left(q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{Z}(\cdot)\right)
\end{aligned}
$$

The $\stackrel{(a)}{\approx}$ step replaces expectation inside the log with an estimate with $Z_{i}$. The $\stackrel{(b)}{\approx}$ step replaces the random variable with the expectation. These steps take $\mathbb{E}_{Z}$ outside of the log. More on this later.

## VAE optimization

The optimization objectives for the encoder and decoder are the same.

Simultaneously train $p_{\theta}$ and $q_{\phi}$ by solving

$$
\underset{\theta \in \Theta, \phi \in \Phi}{\operatorname{maximize}} \sum_{i=1}^{N} \underbrace{\mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\log p_{\theta}\left(X_{i} \mid Z\right)\right]-D_{\mathrm{KL}}\left(q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{Z}(\cdot)\right)}_{\stackrel{\text { def }}{=} \mathrm{VLB}_{\theta, \phi}\left(X_{i}\right)}
$$

We refer to the optimization objective as the variational lower bound (VLB) or evidence lower bound (ELBO) for reasons that will be explained soon.

## VAE standard instance

A standard VAE setup:
Remember from hw6 that

$$
p_{Z}=\mathcal{N}(0, I) \quad D_{\mathrm{KL}}\left(\mathcal{N}\left(\mu_{\phi}(X), \Sigma_{\phi}(X)\right) \| \mathcal{N}(0, I)\right)
$$

$$
\begin{aligned}
& q_{\phi}(z \mid x)=\mathcal{N}\left(\mu_{\phi}(x), \Sigma_{\phi}(x)\right) \text { with diagonal } \Sigma_{\phi}=\frac{1}{2}\left(\operatorname{tr}\left(\Sigma_{\phi}(X)\right)+\left\|\mu_{\phi}(X)\right\|^{2}-d-\log \operatorname{det}\left(\Sigma_{\phi}(X)\right)\right) \\
& p_{\theta}(x \mid z)=\mathcal{N}\left(f_{\theta}(z), \sigma^{2} I\right)
\end{aligned}
$$

$\mu_{\phi}(x), \Sigma_{\phi}^{2}(x)$, and $f_{\theta}(z)$ are deterministic NN . The training objective

$$
\underset{\theta \in \Theta, \phi \in \Phi}{\operatorname{maximize}} \sum_{i=1}^{N} \mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\log p_{\theta}\left(X_{i} \mid Z\right)\right]-D_{\mathrm{KL}}\left(q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{Z}(\cdot)\right)
$$

becomes
$\underset{\theta \in \Theta, \phi \in \Phi}{\operatorname{minimize}} \sum_{i=1}^{N} \frac{1}{\sigma^{2}} \mathbb{E}_{Z \sim \mathcal{N}\left(\mu_{\phi}\left(X_{i}\right), \Sigma_{\phi}\left(X_{i}\right)\right)}\left\|X_{i}-f_{\theta}(Z)\right\|^{2}+\operatorname{tr}\left(\Sigma_{\phi}\left(X_{i}\right)\right)+\left\|\mu_{\phi}\left(X_{i}\right)\right\|^{2}-\log \operatorname{det}\left(\Sigma_{\phi}\left(X_{i}\right)\right)$

## With reparameterization trick

The standard instance of VAE
$\underset{\theta \in \Theta, \phi \in \Phi}{\operatorname{minimize}} \sum_{i=1}^{N} \frac{1}{\sigma^{2}} \mathbb{E}_{Z \sim \mathcal{N}\left(\mu_{\phi}\left(X_{i}\right), \Sigma_{\phi}\left(X_{i}\right)\right)}\left\|X_{i}-f_{\theta}(Z)\right\|^{2}+\operatorname{tr}\left(\Sigma_{\phi}\left(X_{i}\right)\right)+\left\|\mu_{\phi}\left(X_{i}\right)\right\|^{2}-\log \operatorname{det}\left(\Sigma_{\phi}\left(X_{i}\right)\right)$
can be equivalently written with the reparameterization trick

$$
\begin{gathered}
\operatorname{minimize}_{\theta \in \Theta, \phi \in \Phi} \\
\sum_{i=1}^{N} \frac{1}{\sigma^{2}} \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, I)}\left\|X_{i}-f_{\theta}\left(\mu_{\phi}\left(X_{i}\right)+\Sigma_{\phi}^{1 / 2}\left(X_{i}\right) \varepsilon\right)\right\|^{2}+\operatorname{tr}\left(\Sigma_{\phi}\left(X_{i}\right)\right)+\left\|\mu_{\phi}\left(X_{i}\right)\right\|^{2}-\log \operatorname{det}\left(\Sigma_{\phi}\left(X_{i}\right)\right) \\
\text { where } \Sigma_{\phi}^{1 / 2} \text { is diagonal with } \sqrt{ } \text { of the diagonal elements of } \Sigma_{\phi} \text {. (Remember, } \Sigma_{\phi} \text { is diagonal.) }
\end{gathered}
$$

To clarify $Z \stackrel{\mathcal{D}}{=} \mu_{\phi}\left(X_{i}\right)+\Sigma_{\phi}^{1 / 2}\left(X_{i}\right) \varepsilon$, where $\stackrel{\mathcal{D}}{\underline{\mathcal{D}}}$ denotes equality in distribution.

We now have an objective amenable to stochastic optimization.

## VAE standard instance architecture: Training

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-364.jpg?height=1179&width=1391&top_left_y=491&top_left_x=104)

With reparameterization trick
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-364.jpg?height=1046&width=1240&top_left_y=491&top_left_x=2009)

## VAE standard instance architecture: Sampling

During sampling, only the decoder network is used.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-365.jpg?height=592&width=536&top_left_y=742&top_left_x=1263)

## Discussions

Review of terminology

- Likelihood $p_{\theta}(x)$ (exact evaluation intractable)
- Prior $p_{Z}(z)$
- Conditional distribution $p_{\theta}(x \mid z)$
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-366.jpg?height=413&width=813&top_left_y=474&top_left_x=1677)
- True posterior $p_{\theta}(z \mid x)$ (exact evaluation intractable)
- Approximate posterior $q_{\phi}(z \mid x)$

Conditional distribution $p_{\theta}(x \mid z)$ and prior $p_{Z}(z)$ determines the posterior $p_{\theta}(z \mid x)$.

There is no easy way to evaluate $p_{\theta}(x)$, but we can sample $X \sim p_{\theta}(x)$ easily: $Z \sim p_{Z}(z)$ then $X \sim p_{\theta}(x \mid Z)$.

NN in VAE do not directly generate random output. NN outputs parameters for random sampling.

## Training VAE with RT

To obtain stochastic gradients of the VAE objective
$\underset{\theta \in \Theta, \phi \in \Phi}{\operatorname{minimize}} \quad \sum_{i=1}^{N} \underbrace{\frac{1}{\sigma^{2}} \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, I)}\left\|X_{i}-f_{\theta}\left(\mu_{\phi}\left(X_{i}\right)+\Sigma_{\phi}^{1 / 2}\left(X_{i}\right) \varepsilon\right)\right\|^{2}+\operatorname{tr}\left(\Sigma_{\phi}\left(X_{i}\right)\right)+\left\|\mu_{\phi}\left(X_{i}\right)\right\|^{2}-\log \operatorname{det}\left(\Sigma_{\phi}\left(X_{i}\right)\right)}$
select a data $X_{i}$, sample $\varepsilon_{i} \sim \mathcal{N}(0, I)$, evaluate $\xlongequal{\text { def }-\mathrm{VLB}_{\theta, \phi}\left(X_{i}\right)}$
$-\operatorname{VLB}_{\theta, \phi}\left(X_{i}, \varepsilon_{i}\right) \xlongequal{\text { def }} \frac{1}{\sigma^{2}}\left\|X_{i}-f_{\theta}\left(\mu_{\phi}\left(X_{i}\right)+\Sigma_{\phi}^{1 / 2}\left(X_{i}\right) \varepsilon_{i}\right)\right\|^{2}+\operatorname{tr}\left(\Sigma_{\phi}\left(X_{i}\right)\right)+\left\|\mu_{\phi}\left(X_{i}\right)\right\|^{2}-\log \operatorname{det}\left(\Sigma_{\phi}\left(X_{i}\right)\right)$ and backprop on $\operatorname{VLB}_{\theta, \phi}\left(X_{i}, \varepsilon_{i}\right)$.

Usually, batch of $X_{i}$ is selected.
One can sample multiple $Z_{i, 1}, \ldots, Z_{i, K}$ (equivalently $\varepsilon_{i, 1}, \ldots, \varepsilon_{i, K}$ ) for each $X_{i}$.

## Training VAE with log-derivative trick

Computing stochastic gradients without the reparameterization trick.

$$
\underset{\theta \in \Theta, \phi \in \Phi}{\operatorname{maximize}} \sum_{i=1}^{N} \underbrace{\mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\log \left(\frac{p_{\theta}\left(X_{i} \mid Z\right) p_{Z}(Z)}{q_{\phi}\left(Z \mid X_{i}\right)}\right)\right]}_{\stackrel{\text { def }}{=} \operatorname{VLB}_{\theta, \phi}\left(X_{i}\right)}
$$

To obtain unbiased estimates of $\nabla_{\theta}$, compute

$$
\frac{1}{K} \sum_{k=1}^{K} \log p_{\theta}\left(X_{i} \mid Z_{i, k}\right), \quad Z_{i, 1}, \ldots, Z_{i, K} \sim q_{\phi}\left(z \mid X_{i}\right)
$$

and backprop with respect to $\theta$.

## Training VAE with log-derivative trick

We differentiate the VLB objectives (cf. hw 8 problem 8)

$$
\begin{aligned}
\nabla_{\phi} \mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\log \left(\frac{p_{\theta}\left(X_{i} \mid Z\right) p_{Z}(Z)}{q_{\phi}\left(Z \mid X_{i}\right)}\right)\right] & =\nabla_{\phi} \int \log \left(\frac{p_{\theta}\left(X_{i} \mid z\right) p_{Z}(z)}{q_{\phi}\left(z \mid X_{i}\right)}\right) q_{\phi}\left(z \mid X_{i}\right) d z \\
& =\mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\left(\nabla_{\phi} \log q_{\phi}\left(Z \mid X_{i}\right)\right) \log \left(\frac{p_{\theta}\left(X_{i} \mid Z\right) p_{Z}(Z)}{q_{\phi}\left(Z \mid X_{i}\right)}\right)\right]
\end{aligned}
$$

To obtain unbiased estimates of $\nabla_{\phi}$, compute

$$
\frac{1}{K} \sum_{k=1}^{K}\left(\nabla_{\phi} \log q_{\phi}\left(Z_{i, k} \mid X_{i}\right)\right) \log \left(\frac{p_{\theta}\left(X_{i} \mid Z_{i, k}\right) p_{Z}\left(Z_{i, k}\right)}{q_{\phi}\left(Z_{i, k} \mid X_{i}\right)}\right), \quad Z_{i, 1}, \ldots, Z_{i, K} \sim q_{\phi}\left(z \mid X_{i}\right)
$$

## Why variational "autoencoder"?

VAE loss (VLB) contains a reconstruction loss resembling that of an autoencoder.

$$
\begin{aligned}
\operatorname{VLB}_{\theta, \phi}\left(X_{i}\right) & =\mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\log p_{\theta}\left(X_{i} \mid Z\right)\right]-D_{\mathrm{KL}}\left(q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{Z}(\cdot)\right) \\
& =-\frac{1}{2 \sigma^{2}} \mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\left\|X_{i}-f_{\theta}(Z)\right\|^{2}\right]-D_{\mathrm{KL}}\left(q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{Z}(\cdot)\right) \\
& =-\underbrace{\frac{1}{2 \sigma^{2}} \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, I)}\left\|X_{i}-f_{\theta}\left(\mu_{\phi}\left(X_{i}\right)+\Sigma_{\phi}^{1 / 2}\left(X_{i}\right) \varepsilon\right)\right\|^{2}}_{\text {Reconstruction loss }}-\underbrace{D_{\mathrm{KL}}\left(q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{Z}(\cdot)\right)}_{\text {Regularization }}
\end{aligned}
$$

VLB also contains a regularization term on the output of the encoder, which is not present in standard autoencoder losses.

The choice of $\sigma$ determines the relative weight between the reconstruction loss and the regularization.

## How tight is the VLB?

How accurate is the approximation?

$$
\begin{aligned}
& \underset{\theta \in \Theta}{\operatorname{maximize}} \sum_{i=1}^{N} \log p_{\theta}\left(X_{i}\right)=\underset{\theta \in \Theta}{\operatorname{maximize}} \sum_{i=1}^{N} \log \mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\frac{p_{\theta}\left(X_{i} \mid Z\right) p_{Z}(Z)}{q_{\phi}\left(Z \mid X_{i}\right)}\right] \\
& \stackrel{?}{\approx} \underset{\theta \in \Theta, \phi \in \Phi}{\operatorname{maximize}} \sum_{i=1}^{N} \mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\log \left(\frac{p_{\theta}\left(X_{i} \mid Z\right) p_{Z}(Z)}{q_{\phi}\left(Z \mid X_{i}\right)}\right)\right] \\
&=\underset{\theta \in \Theta, \phi \in \Phi}{\operatorname{maximize}} \sum_{i=1}^{N} \operatorname{VLB}_{\theta, \phi}\left(X_{i}\right)
\end{aligned}
$$

This turns out that $\log p_{\theta}\left(X_{i}\right) \geq \operatorname{VLB}_{\theta, \phi}\left(X_{i}\right)$. So we are maximizing a lower bound of the log likelihood. How large is the gap?

## Log-likelihood $\geq$ VLB: Derivation 1

Derivation via Jensen:

$$
\begin{aligned}
\log p_{\theta}\left(X_{i}\right) & =\log \mathbb{E}_{Z \sim p_{Z}}\left[p_{\theta}\left(X_{i} \mid Z\right)\right] \\
& =\log \left(\mathbb{E}_{Z \sim q_{\phi}\left(Z \mid X_{i}\right)}\left[p_{\theta}\left(X_{i} \mid Z\right) \frac{p_{Z}(Z)}{q_{\phi}\left(Z \mid X_{i}\right)}\right]\right) \\
& \geq \mathbb{E}_{Z \sim q_{\phi}\left(Z \mid X_{i}\right)}\left[\log \left(p_{\theta}\left(X_{i} \mid Z\right) \frac{p_{Z}(Z)}{q_{\phi}\left(Z \mid X_{i}\right)}\right)\right] \\
& \stackrel{\text { def }}{=} \mathrm{VLB}_{\theta, \phi}\left(X_{i}\right)
\end{aligned}
$$

Does not explicitly characterize gap.

## Log-likelihood $\geq$ VLB: Derivation 2

Derivation via KL divergence:

$$
\begin{aligned}
D_{\mathrm{KL}}\left[q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{\theta}\left(\cdot \mid X_{i}\right)\right] & =\mathbb{E}_{Z \sim q_{\theta}\left(z \mid X_{i}\right)}\left[\log q_{\theta}\left(Z \mid X_{i}\right)-\log p_{\theta}\left(Z \mid X_{i}\right)\right] \\
& =\underbrace{\mathbb{E}_{Z \sim q_{\theta}\left(z \mid X_{i}\right)}\left[\log q_{\theta}\left(Z \mid X_{i}\right)-\log p_{Z}(Z)-\log p_{\theta}\left(X_{i} \mid Z\right)\right]}_{=-\mathrm{VLB}_{\theta, \phi}\left(X_{i}\right)}+\log p_{\theta}\left(X_{i}\right)
\end{aligned}
$$

and

$$
\begin{aligned}
\log p_{\theta}\left(X_{i}\right) & =\operatorname{VLB}_{\theta, \phi}\left(X_{i}\right)+D_{\mathrm{KL}}\left[q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{\theta}\left(\cdot \mid X_{i}\right)\right] \\
& \geq \operatorname{VLB}_{\theta, \phi}\left(X_{i}\right)
\end{aligned}
$$

This derivation explicitly characterizes the gap as $D_{\mathrm{KL}}\left[q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{\theta}\left(\cdot \mid X_{i}\right)\right]$.

## VLB is tight if encoder infinitely powerful

If the encoder $q_{\phi}$ is powerful enough such that there is a $\phi^{\star}$ achieving

$$
q_{\phi^{\star}}\left(\cdot \mid X_{i}\right)=p_{\theta}\left(\cdot \mid X_{i}\right)
$$

or equivalently

$$
D_{\mathrm{KL}}\left[q_{\phi^{\star}}\left(\cdot \mid X_{i}\right) \| p_{\theta}\left(\cdot \mid X_{i}\right)\right]=0
$$

Then

$$
\underset{\theta \in \Theta}{\operatorname{maximize}} \sum_{i=1}^{N} \log p_{\theta}\left(X_{i}\right)={\underset{\theta}{\theta \in \Theta, \phi \in \Phi}}_{\operatorname{maximize}} \sum_{i=1}^{N} \operatorname{VLB}_{\theta, \phi}\left(X_{i}\right)
$$

This follows from

$$
\log p_{\theta}\left(X_{i}\right)=\operatorname{VLB}_{\theta, \phi}\left(X_{i}\right)+\underbrace{D_{\mathrm{KL}}\left[q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{\theta}\left(\cdot \mid X_{i}\right)\right]}_{\geq 0}
$$

and hw 8 problem 2.

## VQ-VAE

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-375.jpg?height=821&width=3250&top_left_y=542&top_left_x=29)

Figure 2: Left: ImageNet $128 \times 128 \times 3$ images, right: reconstructions from a VQ-VAE with a $32 \times 32 \times 1$ latent space, with $\mathrm{K}=512$.

## VQ-VAE

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-376.jpg?height=1187&width=3197&top_left_y=376&top_left_x=63)

Figure 3: Samples (128x128) from a VQ-VAE with a PixelCNN prior trained on ImageNet images.

## VQ-VAE-2

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-377.jpg?height=1710&width=2750&top_left_y=0&top_left_x=170)

## VQ-VAE-2

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-378.jpg?height=1303&width=2610&top_left_y=399&top_left_x=338)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-379.jpg?height=738&width=3107&top_left_y=154&top_left_x=224)

$$
\ell_{\theta, \phi}\left(X_{i}\right)=\mathbb{E}_{Z \sim q_{\phi}\left(z \mid X_{i}\right)}\left[\log p_{\theta}\left(X_{i} \mid Z\right)\right]-\beta D_{\mathrm{KL}}\left(q_{\phi}\left(\cdot \mid X_{i}\right) \| p_{Z}(\cdot)\right)
$$

when $\beta=1, \ell_{\theta, \phi}\left(X_{i}\right)=\operatorname{VLB}_{\theta, \phi}\left(X_{i}\right)$, i.e., $\beta$-VAE coincides with VAE when $\beta=1$.

With $\beta>1$, authors observed better feature disentanglement.

## Minimax optimization

In a minimax optimization problem we minimize with respect to one variable and maximize with respect to another:

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \underset{\phi \in \Phi}{\operatorname{maximize}} \mathcal{L}(\theta, \phi)
$$

We say $\left(\theta^{\star}, \phi^{\star}\right)$ is a solution to the minimax problem if $\theta^{\star} \in \Theta, \phi^{\star} \in \Phi$, and

$$
\mathcal{L}\left(\theta^{\star}, \phi\right) \leq \mathcal{L}\left(\theta^{\star}, \phi^{\star}\right) \leq \mathcal{L}\left(\theta, \phi^{\star}\right), \quad \forall \theta \in \Theta, \phi \in \Phi .
$$

In other words, unilaterally deviating from $\theta^{\star} \in \Theta$ increases the value of $\mathcal{L}(\theta, \phi)$ while unilaterally deviating from $\phi^{\star} \in \Phi$ decreases the value of $\mathcal{L}(\theta, \phi)$. In yet other words, the solution is defined as a Nash equilibrium in a 2-player zero-sum game.

## Minimax optimization

So far, we trained NN by solving minimization problems.

However, GANs are trained by solving minimax problems. Since the advent of GANs, minimax training has become more widely used in all areas of deep learning.

Examples:

- Adversarial training to make NN robust against adversarial attacks.
- Domain adversarial networks to train NN to make fair decisions (e.g. not base its decision on a persons race or gender).


## Minimax vs. maximin

When a solution (as we defined it) does not exist, then min-max is not the same as max-min:

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \underset{\phi \in \Phi}{\operatorname{maximize}} \mathcal{L}(\theta, \phi) \neq \underset{\phi \in \Phi}{\operatorname{maximize}} \underset{\theta \in \Theta}{\operatorname{minimize}} \mathcal{L}(\theta, \phi)
$$

This is a technical distinction that we will not explore in this class.

## Minimax optimization algorithm

First, consider deterministic gradient setup. Let $\alpha$ and $\beta$ be the stepsizes (learning rates) for the descent and ascent steps respectively.

Simultaneous gradient ascent-descent:

$$
\begin{aligned}
\phi^{k+1} & =\phi^{k}+\beta \nabla_{\phi} \mathcal{L}\left(\theta^{k}, \phi^{k}\right) \\
\theta^{k+1} & =\theta^{k}-\alpha \nabla_{\theta} \mathcal{L}\left(\theta^{k}, \phi^{k}\right)
\end{aligned}
$$

Alternating gradient ascent-descent:

$$
\begin{aligned}
\phi^{k+1} & =\phi^{k}+\beta \nabla_{\phi} \mathcal{L}\left(\theta^{k}, \phi^{k}\right) \\
\theta^{k+1} & =\theta^{k}-\alpha \nabla_{\theta} \mathcal{L}\left(\theta^{k}, \phi^{k+1}\right)
\end{aligned}
$$

## Minimax optimization algorithm

Gradient multi-ascent-single-descent:

$$
\begin{aligned}
\phi_{0}^{k+1} & =\phi_{n_{\mathrm{dis}}}^{k} \\
\phi_{i+1}^{k+1} & =\phi_{i}^{k+1}+\beta \nabla_{\phi} \mathcal{L}\left(\theta^{k}, \phi_{i}^{k+1}\right), \quad \text { for } i=0, \ldots, n_{\mathrm{dis}}-1 \\
\theta^{k+1} & =\theta^{k}-\alpha \nabla_{\theta} \mathcal{L}\left(\theta^{k}, \phi_{n_{\mathrm{dis}}}^{k+1}\right)
\end{aligned}
$$

( $n_{\text {dis }}$ stands for number of discriminator updates.) When $n_{\text {dis }}=1$, this algorithm reduces to alternating ascent-descent.

## Stochastic minimax optimization

In deep learning, however, we have access to stochastic gradients.

Stochastic gradient simultaneous ascent-descent

$$
\begin{array}{rlrl}
\phi^{k+1} & =\phi^{k}+\beta g_{\phi}^{k}, & \mathbb{E}\left[g_{\phi}^{k}\right]=\nabla_{\phi} \mathcal{L}\left(\theta^{k}, \phi^{k}\right) \\
\theta^{k+1} & =\theta^{k}-\alpha g_{\theta}^{k}, & & \mathbb{E}\left[g_{\theta}^{k}\right]=\nabla_{\theta} \mathcal{L}\left(\theta^{k}, \phi^{k}\right)
\end{array}
$$

Stochastic gradient alternating ascent-descent

$$
\begin{array}{rlrl}
\phi^{k+1} & =\phi^{k}+\beta g_{\phi}^{k}, & \mathbb{E}\left[g_{\phi}^{k}\right]=\nabla_{\phi} \mathcal{L}\left(\theta^{k}, \phi^{k}\right) \\
\theta^{k+1}=\theta^{k}-\alpha g_{\theta}^{k}, & \mathbb{E}\left[g_{\theta}^{k}\right]=\nabla_{\theta} \mathcal{L}\left(\theta^{k}, \phi^{k+1}\right)
\end{array}
$$

Stochastic gradient multi-ascent-single-descent

$$
\begin{array}{rlr}
\phi_{0}^{k+1} & =\phi_{n_{\text {dis }}}^{k} \\
\phi_{i+1}^{k+1} & =\phi_{i}^{k+1}+\beta \nabla_{\phi} g_{\phi}^{k, i}, \quad \mathbb{E}\left[g_{\phi}^{k, i}\right]=\nabla_{\phi} \mathcal{L}\left(\theta^{k}, \phi_{i}^{k+1}\right), \quad \text { for } i=0, \ldots, n_{\text {dis }}-1 \\
\theta^{k+1} & =\theta^{k}-\alpha g_{\theta}^{k}, \quad \mathbb{E}\left[g_{\theta}^{k}\right]=\nabla_{\theta} \mathcal{L}\left(\theta^{k}, \phi_{n_{\text {dis }}^{k}}^{k+1}\right)
\end{array}
$$

## Minimax optimization in PyTorch

To perform minimax optimization in PyTorch, we maintain two separate optimizers, one for the ascent, one for the descent. The OPTIMIZER can be anything like SGD or Adam.

```
G = Generator(...).to(device)
D = Discriminator(...).to(device)
D_optimizer = optim.OPTIMIZER(D.parameters(), lr = beta)
G_optimizer = optim.OPTIMIZER(G.parameters(), lr = alpha)
```

Simultaneous ascent-descent:

```
Evaluate D_loss
D_loss.backward()
Evaluate G_loss
G_loss.backward()
D_optimizer.step()
G_optimizer.step()
```


## Minimax optimization in PyTorch

Alternating ascent-descent

```
Evaluate D_loss
D_loss.backward()
D_optimizer.step()
Evaluate G_loss
G_loss.backward()
G_optimizer.step()
```


## Minimax optimization in PyTorch

Multi-ascent-single-descent

```
for _ in range(ndis) :
    Evaluate D loss
    D_loss.backward()
    D_optimizer.step()
Evaluate G_loss
G_loss.backward()
G_optimizer.step()
```


## Generative adversarial networks (GAN)

These are synthetic (fake) images.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-389.jpg?height=1153&width=3330&top_left_y=597&top_left_x=2)

## GAN

In generative adversarial networks (GAN) a generator network and a discriminator network compete adversarially.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-390.jpg?height=792&width=1778&top_left_y=29&top_left_x=1552)

Given data $X_{1}, \ldots, X_{N} \sim p_{\text {true }}$. GAN aims to learn $p_{\theta} \approx p_{\text {true }}$.

Generator aims to generate fake data similar to training data.

Discriminator aims to distinguish the training data from fake data.

Analogy: Criminal creating fake money vs. police distinguishing fake money from real.

## Generator network

The generator $G_{\theta}: \mathbb{R}^{k} \rightarrow \mathbb{R}^{n}$ is a neural network parameterized by $\theta \in \Theta$. The generator takes a random latent vector $Z \sim p_{Z}$ as input and outputs generated (fake) data $\tilde{X}=G_{\theta}(Z)$. The latent distribution is usually $p_{Z}=\mathcal{N}(0, I)$.

Write $p_{\theta}$ for the probability distribution of $\tilde{X}=G_{\theta}(Z)$. Although we can't evaluate the density $p_{\theta}(x)$, neither exactly nor approximately, we can sample from $\tilde{X} \sim p_{\theta}$.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-391.jpg?height=422&width=1226&top_left_y=1226&top_left_x=1798)

## Discriminator network

The discriminator $D_{\phi}: \mathbb{R}^{n} \rightarrow(0,1)$ is a neural network parameterized by $\phi \in \Phi$. The discriminator takes an image $X$ as input and outputs whether $X$ is a real or fake.\#

- $D_{\phi}(X) \approx 1$ : discriminator confidently predicts $X$ is real.
- $D_{\phi}(X) \approx 0$ : discriminator confidently predicts $X$ is fake.
- $D_{\phi}(X) \approx 0.5$ : discriminator is unsure whether $X$ is real or fake.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-392.jpg?height=784&width=1532&top_left_y=1071&top_left_x=250)
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-392.jpg?height=809&width=1060&top_left_y=1058&top_left_x=1860)


## Discriminator loss

Cost of incorrectly classifying real as fake (type I error):

$$
\mathbb{E}_{X \sim p_{\text {true }}}\left[-\log D_{\phi}(X)\right]
$$

Cost of incorrectly classifying fake as real (type II error):

$$
\mathbb{E}_{\tilde{X} \sim p_{\theta}}\left[-\log \left(1-D_{\phi}(\tilde{X})\right)\right]=\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[-\log \left(1-D_{\phi}\left(G_{\theta}(Z)\right)\right)\right]
$$

Discriminator solves

$$
\underset{\phi \in \Phi}{\operatorname{maximize}} \quad \mathbb{E}_{X \sim p_{\text {true }}}\left[\log D_{\phi}(X)\right]+\mathbb{E}_{\tilde{X} \sim p_{\theta}}\left[\log \left(1-D_{\phi}(\tilde{X})\right)\right]
$$

which is equivalent to

$$
\underset{\phi \in \Phi}{\operatorname{maximize}} \quad \mathbb{E}_{X \sim p_{\text {true }}}\left[\log D_{\phi}(X)\right]+\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[\log \left(1-D_{\phi}\left(G_{\theta}(Z)\right)\right)\right]
$$

## Discriminator loss

We can view

$$
\mathbb{E}_{\tilde{X} \sim p_{\theta}}\left[\log \left(1-D_{\phi}(\tilde{X})\right)\right]=\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[\log \left(1-D_{\phi}\left(G_{\theta}(Z)\right)\right)\right]
$$

as an instance of the reparameterization technique.

The loss

$$
\mathbb{E}_{X \sim p_{\text {true }}}\left[\log D_{\phi}(X)\right]+\mathbb{E}_{\tilde{X} \sim p_{\theta}}\left[\log \left(1-D_{\phi}(\tilde{X})\right)\right]
$$

puts equal weight on type I and type II errors. Alternatively, one can use the loss

$$
\mathbb{E}_{X \sim p_{\text {true }}}\left[\log D_{\phi}(X)\right]+\lambda \mathbb{E}_{\tilde{X} \sim p_{\theta}}\left[\log \left(1-D_{\phi}(\tilde{X})\right)\right]
$$

where $\lambda>0$ represents the relative significance of a type II error over a type I error.

## Generator loss

Since the goal of the generator is to deceive the discriminator, the generator minimizes the same loss.

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \quad \mathbb{E}_{X \sim p_{\text {true }}}\left[\log D_{\phi}(X)\right]+\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[\log \left(1-D_{\phi}\left(G_{\theta}(Z)\right)\right)\right]
$$

(The generator and discriminator operate under a zero-sum game.)

Note, only the second term depend on $\theta$, while the both terms depend on $\phi$.

## Empirical risk minimization

In practice, we have finite samples $X_{1}, \ldots, X_{N}$, so we instead use the loss

$$
\frac{1}{N} \sum_{i=1}^{N} \log D_{\phi}\left(X_{i}\right)+\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[\log \left(1-D_{\phi}\left(G_{\theta}(Z)\right)\right)\right]
$$

Since $\tilde{X}=G_{\theta}(Z)$ is generated with $Z \sim p_{Z}$, we have unlimited $\tilde{X}$ samples. So we replace $\mathbb{E}_{X} \approx \frac{1}{N} \sum$ while leaving $\mathbb{E}_{Z}$ as is.

## Minimax training (zero-sum game)

Train generator and discriminator simultaneously by solving

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \underset{\phi \in \Phi}{\operatorname{maximize}} \mathcal{L}(\theta, \phi)
$$

where

$$
\mathcal{L}(\theta, \phi)=\frac{1}{N} \sum_{i=1}^{N} \log D_{\phi}\left(X_{i}\right)+\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[\log \left(1-D_{\phi}\left(G_{\theta}(Z)\right)\right)\right]
$$

It remains to specify the architectures for $G_{\theta}$ and $D_{\phi}$.

## GAN demo

PyTorch demo

## DCGAN

The original GAN was also deep and convolutional. However, Radford et al.'s Deep Convolutional Generative Adversarial Networks (DCGAN) paper proposed the following architectures, which crucially utilize batchnorm.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-399.jpg?height=1005&width=2434&top_left_y=756&top_left_x=233)

## Math review: f-divergence

The f -divergence of $p$ from $q$, where $f$ is a convex function such that $f(1)=0$, is

$$
D_{f}(p \| q)=\int f\left(\frac{p(x)}{q(x)}\right) q(x) d x,
$$

This includes the KL divergence:

- If $f(u)=u \log u$, then $D_{f}(p \| q)=D_{\mathrm{KL}}(p \| q)$.
- If $f(u)=-\log u$, then $D_{f}(p \| q)=D_{\text {KL }}(q \| p)$.


## Math review: JS-divergence

Jensen-Shannon-divergence (JS-divergence) is

$$
D_{\mathrm{JS}}(p, q)=\frac{1}{2} D_{\mathrm{KL}}\left(p \| \frac{1}{2}(p+q)\right)+\frac{1}{2} D_{\mathrm{KL}}\left(q \| \frac{1}{2}(p+q)\right)
$$

With, $f(u)=\left\{\begin{array}{ll}\frac{1}{2}\left(u \log u-(u+1) \log \frac{u+1}{2}\right) & \text { for } u \geq 0 \\ \infty & \text { otherwise }\end{array}\right.$ we have $D_{f}=D_{\mathrm{IS}}$.

With, $f(u)=\left\{\begin{array}{ll}u \log u-(u+1) \log (u+1)+\log 4 & \text { for } u \geq 0 \\ \infty & \text { otherwise }\end{array}\right.$ we have $D_{f}=2 D_{\mathrm{JS}}$.

## GAN $\approx$ JSD minimization

Let us understand the minimax problem

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \underset{\phi \in \Phi}{\operatorname{maximize}} \mathcal{L}(\theta, \phi)
$$

via the minimization problem

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \mathcal{J}(\theta)
$$

where

$$
\mathcal{J}(\theta)=\sup _{\phi \in \Phi} \mathcal{L}(\theta, \phi)
$$

For simplicity, assume the discriminator is infinitely powerful, i.e., $D_{\phi}(x)$ can represent any arbitrary function.

## GAN $\approx$ JSD minimization

Note

$$
\begin{aligned}
\mathcal{L}(\theta, \phi) & =\mathbb{E}_{X \sim p_{\text {true }}}\left[\log D_{\phi}(X)\right]+\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[\log \left(1-D_{\phi}\left(G_{\theta}(Z)\right)\right)\right] \\
& =\mathbb{E}_{X \sim p_{\text {true }}}\left[\log D_{\phi}(X)\right]+\mathbb{E}_{\tilde{X} \sim p_{\theta}}\left[\log \left(1-D_{\phi}(\tilde{X})\right)\right] \\
& =\int p_{\text {true }}(x) \log D_{\phi}(x)+p_{\theta}(x) \log \left(1-D_{\phi}(x)\right) d x
\end{aligned}
$$

Since

$$
\frac{d}{d y}(a \log y+b \log (1-y))=0 \quad \Rightarrow \quad y^{\star}=\frac{a}{a+b}
$$

The integral is maximized by

$$
D_{\phi^{\star}}(x)=\frac{p_{\text {true }}(x)}{p_{\text {true }}(x)+p_{\theta}(x)}
$$

## $G A N \approx J S D$ minimization

If we plug in the optimal discriminator,

$$
D_{\phi^{\star}}(x)=\frac{p_{\text {true }}(x)}{p_{\text {true }}(x)+p_{\theta}(x)}
$$

we get

$$
\begin{aligned}
\mathcal{L}\left(\theta, \phi^{\star}\right) & =\mathbb{E}_{X \sim p_{\text {true }}}\left[\log \frac{p_{\text {true }}(X)}{p_{\text {true }}(X)+p_{\theta}(X)}\right]+\mathbb{E}_{\tilde{X} \sim p_{\theta}}\left[\log \frac{p_{\theta}(\tilde{X})}{p_{\text {true }}(\tilde{X})+p_{\theta}(\tilde{X})}\right] \\
& =2 D_{\mathrm{JS}}\left(p_{\text {true }}, p_{\theta}\right)-\log (4)
\end{aligned}
$$

Therefore,

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \underset{\phi \in \Phi}{\operatorname{maximize}} \mathcal{L}(\theta, \phi) \approx \underset{\theta \in \Theta}{\operatorname{minimize}} D_{\mathrm{JS}}\left(p_{\text {true }}, p_{\theta}\right)
$$

## f-GAN

With GANs, we started from a minimax formulation and later reinterpreted it as minimizing the JS-divergence.

Let us instead the start from an f-divergence minimization

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \quad D_{f}\left(p_{\text {true }} \| p_{\theta}\right)
$$

and then variationally approximate $D_{f}$ to obtain a minimax formulation.

Variational approach: Evaluating $D_{f}$ directly is difficult, so we pose it as a maximization problem and parameterize the maximizing function as a "discriminator" neural network.

## f-GAN

For simplicity, however, we only consider the order

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \quad D_{f}\left(p_{\text {true }} \| p_{\theta}\right)
$$

However, one can also consider

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \quad D_{f}\left(p_{\theta} \| p_{\text {true }}\right)
$$

to obtain similar results.
(During our coverage of f-GANs, we will have notational conflict between $D_{f}$, the fdivergence, and $D_{\phi}$, the discriminator network. Hopefully there won't be any confusion.)

## Convex conjugate

Let $f: \mathbb{R} \rightarrow \mathbb{R} \cup\{\infty\}$. Define the convex conjugate of $f$ as

$$
f^{*}(t)=\sup _{u \in \mathbb{R}}\{t u-f(u)\}
$$

where $f^{*}: \mathbb{R} \rightarrow \mathbb{R} \cup\{\infty\}$. This is also referred to as the Legendre transform.

If $f$ is a nice ${ }^{\#}$ convex function, then $f^{*}$ is convex and $f^{* *}=f$, i.e., the conjugate of the conjugate is the original function. ${ }^{\%}$ So

$$
f(u)=\sup _{t \in \mathbb{R}}\left\{t u-f^{*}(t)\right\}
$$

## Convex conjugate: Examples

The following are some examples. Computation of $f^{*}$ uses basic calculus.

$$
\begin{aligned}
& f_{\mathrm{KL}}(u)= \begin{cases}u \log u & \text { for } u \geq 0 \\
\infty & \text { otherwise } \\
-\log u & \text { for } u>0 \\
\infty & \text { otherwise }\end{cases} \\
& f_{\mathrm{LK}}(u)=\left\{\begin{array}{ll}
\mathrm{KL}
\end{array}\right)=\exp (t-1) \\
& f_{\mathrm{SH}}(u)=(\sqrt{u}-1)^{2} \\
& f_{\mathrm{JS}}(u)= \begin{cases}-1-\log (-t) & \text { for } t<0 \\
\infty & f_{\mathrm{SH}}^{*}(t)= \begin{cases}\frac{1}{1 / t-1} & \text { for } t<1 \\
\infty \log u-(u+1) \log (u+1)+\log 4 & \text { for } u \geq 0 \\
\infty & \text { otherwise }\end{cases} \\
f_{\mathrm{JS}}^{*}(t)= \begin{cases}-\log (1-\exp (t))-\log 4 & \text { for } t<0 \\
\infty & \text { otherwise }\end{cases} \end{cases}
\end{aligned}
$$

(Keeping track of the $\infty$ output is necessary.)
$K L=K L$, LK=reverse-KL, SH=squared Hellinger distance, JS=JS

## Convex conjugate: Examples

We get the following f-divergences: $D_{f_{\mathrm{KL}}}(p \| q)=D_{\mathrm{KL}}(p \| q)$

$$
\begin{aligned}
& D_{f_{\mathrm{LK}}}(p \| q)=D_{\mathrm{KL}}(q \| p) \\
& D_{f_{\mathrm{SH}}}(p \| q)=D_{\mathrm{SH}}(q, p) \\
& D_{f_{\mathrm{JS}}}(p \| q)=2 D_{\mathrm{JS}}(q, p)
\end{aligned}
$$

We don't use the following property, but it's interesting so we mention it. If $f$ and $f^{*}$ are differentiable, then $\left(f^{\prime}\right)^{-1}=\left(f^{*}\right)^{\prime}$ :

$$
\begin{array}{ll}
\frac{d}{d u} f_{\mathrm{KL}}(u)=1+\log u & \frac{d}{d t} f_{\mathrm{KL}}^{*}(t)=\exp (t-1) \\
\frac{d}{d u} f_{\mathrm{LK}}(u)=-\frac{1}{u} & \frac{d}{d t} f_{\mathrm{LK}}^{*}(t)=-\frac{1}{t} \\
\frac{d}{d u} f_{\mathrm{SH}}(u)=1-\frac{1}{\sqrt{u}} & \frac{d}{d t} f_{\mathrm{SH}}^{*}(t)=\frac{1}{(1-t)^{2}} \\
\frac{d}{d u} f_{\mathrm{JS}}(u)=\log \frac{u}{1+u} & \frac{d}{d t} f_{\mathrm{JS}}^{*}(t)=\frac{1}{e^{-t}-1}
\end{array}
$$

## Variational formulation of f-divergence

Variational formulation of f-divergence:

$$
\begin{aligned}
D_{f}(p \| q) & =\int q(x) f\left(\frac{p(x)}{q(x)}\right) d x \\
& =\int q(x) \sup _{t}\left\{t \frac{p(x)}{q(x)}-f^{*}(t)\right\} d x=\int q(x) T^{\star}(x) \frac{p(x)}{q(x)}-q(x) f^{*}\left(T^{\star}(x)\right) d x \\
& =\sup _{T \in \mathcal{T}}\left(\int p(x) T(x) d x-\int q(x) f^{*}(T(x)) d x\right)=\sup _{T \in \mathcal{T}}\left(\mathbb{E}_{X \sim p}[T(X)]-\mathbb{E}_{\tilde{X} \sim q}\left[f^{*}(T(\tilde{X}))\right]\right) \\
& \geq \sup _{\phi \in \Phi}\left(\mathbb{E}_{X \sim p}\left[D_{\phi}(X)\right]-\mathbb{E}_{\tilde{X} \sim q}\left[f^{*}\left(D_{\phi}(\tilde{X})\right)\right]\right)
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-410.jpg?height=171&width=2804&top_left_y=1424&top_left_x=231) $D_{\phi}$ is a neural network parameterized by $\phi$.

## f-GAN minimax formulation

Minimax formulation of f-GANs.

$$
\begin{gathered}
\underset{\theta \in \Theta}{\operatorname{minimize}} D_{f}\left(p_{\text {true }} \| p_{\theta}\right) \\
\approx \underset{\theta \in \Theta}{\operatorname{minimize}} \underset{\phi \in \Phi}{\operatorname{maximize}} \mathbb{E}_{X \sim p_{\text {true }}}\left[D_{\phi}(X)\right]-\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[f^{*}\left(D_{\phi}\left(G_{\theta}(Z)\right)\right)\right]
\end{gathered}
$$

## f-GAN with KL-divergence

Instantiate f-GAN with KL-divergence: $f^{*}(t)=e^{t-1}$.

$$
\begin{aligned}
& \underset{\theta \in \Theta}{\operatorname{minimize}} \quad D_{\mathrm{KL}}\left(p_{\text {true }} \| p_{\theta}\right) \\
& \approx \underset{\theta \in \Theta}{\operatorname{minimize}} \underset{\phi \in \Phi}{\operatorname{maximize}} \mathbb{E}_{X \sim p_{\text {true }}}\left[D_{\phi}(X)\right]-\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[e^{D_{\phi}\left(G_{\theta}(Z)\right)-1}\right] \\
& \stackrel{(*)}{=} \underset{\theta \in \Theta}{\operatorname{minimize}} \underset{\phi \in \Phi}{\operatorname{maximize}} 1+\mathbb{E}_{X \sim p_{\text {true }}}\left[D_{\phi}(X)\right]-\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[e^{D_{\phi}\left(G_{\theta}(Z)\right)}\right] \\
& =\underset{\theta \in \Theta}{\operatorname{minimize}} \underset{\phi \in \Phi}{\operatorname{maximize}} \mathbb{E}_{X \sim p_{\text {true }}}\left[D_{\phi}(X)\right]-\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[e^{D_{\phi}\left(G_{\theta}(Z)\right)}\right]
\end{aligned}
$$

Step (*) uses the substitution $D_{\phi} \mapsto D_{\phi}+1$, which is valid if the final layer of $D_{\phi}$ has a trainable bias term. ( $D_{\phi}: \mathbb{R}^{n} \rightarrow \mathbb{R}$.)

## f-GAN with squared Hellinger

Instantiate f-GAN with squared Hellinger distance $\#: f^{*}(t)= \begin{cases}\frac{1}{1 / t-1} & \text { if } t<1 \\ \infty & \text { otherwise }\end{cases}$

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \quad D_{\mathrm{SH}}\left(p_{\text {true }}, p_{\theta}\right)
$$

$$
\approx \begin{array}{lll}
\underset{\theta \in \Theta}{\operatorname{minimize}} & \begin{array}{l}
\operatorname{maximize} \\
\\
\\
\text { subject to }
\end{array} & \mathbb{E}_{X \sim p_{\text {true }}}\left[D_{\phi}(X)\right]-\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[\frac{1}{1 /\left(D_{\phi}\left(G_{\theta}(Z)\right)\right)-1}\right] \\
\left.D_{\theta}(z)\right)<1 \text { for all } z \in \mathbb{R}^{k}
\end{array}
$$

When the constraint is violated, the $f^{*}(t)=\infty$ case makes the maximization objective $-\infty$. However, directly enforcing the neural networks to satisfy $D_{\phi}\left(G_{\theta}(z)\right)<1$ is awkward.

## Solution: Output activation $\rho$

When $D_{\phi}: \mathbb{R}^{n} \rightarrow \mathbb{R}$ and $\left\{t \mid f^{*}(t)<\infty\right\} \neq \mathbb{R}$, then $f^{*}\left(D_{\phi}(\tilde{X})\right)=\infty$ is possible. To prevent this, substitute $T(x) \mapsto \rho(\tilde{T}(x))$, where $\rho: \mathbb{R} \rightarrow\left\{t \mid f^{*}(t)<\infty\right\}$ is a one-to-one function:

$$
\begin{aligned}
D_{f}(p \| q) & =\sup _{T \in \mathcal{T}}\left\{\mathbb{E}_{X \sim p}[T(X)]-\mathbb{E}_{\tilde{X} \sim q}\left[f^{*}(T(\tilde{X}))\right]\right\} \\
& \stackrel{(*)}{=} \sup _{\substack{T \in \mathcal{T}}}\left\{\mathbb{E}_{X \sim p}[T(X)]-\mathbb{E}_{\tilde{X} \sim q}\left[f^{*}(T(\tilde{X}))\right]\right\} \\
& \stackrel{(* *)}{=} \sup _{\tilde{T} \in \mathcal{T}}\left\{\mathbb{E}_{X \sim p}[\rho(\tilde{T}(X))]-\mathbb{E}_{\tilde{X} \sim q}\left[f^{*}(\rho(\tilde{T}(\tilde{X})))\right]\right\} \\
& \geq \sup _{\phi \in \Phi}\left\{\mathbb{E}_{X \sim p}\left[\rho\left(D_{\phi}(X)\right)\right]-\mathbb{E}_{\tilde{X} \sim q}\left[f^{*}\left(\rho\left(D_{\phi}(\tilde{X})\right)\right)\right]\right\}
\end{aligned}
$$

(*) We can restrict the search over $T$ since if $f^{*}(T(x))=\infty$, then the objective becomes $-\infty$.\# (**) With $T=\rho \circ \tilde{T}$, have $\left[T \in \mathcal{T}\right.$ and $\left.f^{*}(T(x))<\infty\right] \Leftrightarrow[\tilde{T} \in \mathcal{T}]$ since $\rho$ is one-to-one.

## f-GAN with output activation

Formulate f-GAN with output activation function $\rho$ :

$$
\begin{gathered}
\operatorname{minimize}_{\theta \in \Theta}^{\min } \quad D_{f}\left(p_{\text {true }} \| p_{\theta}\right) \\
\approx \underset{\theta \in \Theta}{\operatorname{minimize}} \underset{\phi \in \Phi}{\operatorname{maximize}} \quad \mathbb{E}_{X \sim p_{\text {true }}}\left[\rho\left(D_{\phi}(X)\right)\right]-\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[f^{*}\left(\rho\left(D_{\phi}\left(G_{\theta}(Z)\right)\right)\right)\right]
\end{gathered}
$$

## f-GAN with squared Hellinger

Instantiate f-GAN with squared Hellinger distance using $\rho(r)=1-e^{-r}$ and

$$
f^{*}(t)= \begin{cases}\frac{1}{1 / t-1} & \text { if } t<1 \\ \infty & \text { otherwise }\end{cases}
$$

Note that $f^{*}(\rho(r))=-1+e^{r}$.

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \quad D_{\mathrm{SH}}\left(p_{\text {true }}, p_{\theta}\right)
$$

$$
\begin{array}{cc}
\approx \underset{\theta \in \Theta}{\operatorname{minimize}} & \underset{\phi \in \Phi}{\operatorname{maximize}} \\
=\underset{\theta \in \Theta}{\operatorname{minimize}} & 2-\mathbb{E}_{X \sim p_{\text {true }}}\left[e^{-D_{\phi}(X)}\right]-\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[e^{D_{\phi}\left(G_{\theta}(Z)\right)}\right] \\
\underset{\phi \in \Phi}{ } \operatorname{maximize}-\mathbb{E}_{X \sim p_{\text {true }}}\left[e^{-D_{\phi}(X)}\right]-\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[e^{D_{\phi}\left(G_{\theta}(Z)\right)}\right]
\end{array}
$$

## f-GAN with reverse KL

Instantiate f-GAN with reverse KL using $\rho(r)=-e^{r}$ and

$$
f^{*}(t)= \begin{cases}-1-\log (-t) & \text { if } t<0 \\ \infty & \text { otherwise }\end{cases}
$$

Note that $f^{*}(\rho(r))=-1-r$.

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \quad D_{\mathrm{KL}}\left(p_{\theta} \| p_{\text {true }}\right)
$$

$$
\begin{aligned}
& \approx \underset{\theta \in \Theta}{\operatorname{minimize}} \underset{\phi \in \Phi}{\operatorname{maximize}} 1-\mathbb{E}_{X \sim p_{\text {true }}}\left[e^{D_{\phi}(X)}\right]+\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[D_{\phi}\left(G_{\theta}(Z)\right)\right] \\
& =\underset{\theta \in \Theta}{\operatorname{minimize}} \underset{\phi \in \Phi}{\operatorname{maximize}}-\mathbb{E}_{X \sim p_{\text {true }}}\left[e^{D_{\phi}(X)}\right]+\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[D_{\phi}\left(G_{\theta}(Z)\right)\right]
\end{aligned}
$$

## Recovering standard GAN

We recover standard GAN with

$$
\rho(r)=\log (\sigma(r)), \quad \sigma(r)=\frac{1}{1+e^{-r}}, \quad f^{*}(t)= \begin{cases}-\log (1-\exp (t))-\log 4 & \text { for } t<0 \\ \infty & \text { otherwise }\end{cases}
$$

Note that $\sigma$ is the familiar sigmoid and

$$
f^{*}(\rho(r))=-\log (1-\sigma(r))-\log 4
$$

$$
\begin{gathered}
\operatorname{minimize}_{\theta \in \Theta} D_{\mathrm{JS}}\left(p_{\text {true }}, p_{\theta}\right) \\
\underset{\theta \in \Theta}{\operatorname{minimize}} \underset{\phi \in \Phi}{\operatorname{maximize}} \mathbb{E}_{X \sim p_{\text {true }}}\left[\log \sigma\left(D_{\phi}(X)\right)\right]+\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[\log \left(1-\sigma\left(D_{\phi}\left(G_{\theta}(X)\right)\right)\right)\right]
\end{gathered}
$$

where $D_{\phi}: \mathbb{R}^{n} \rightarrow \mathbb{R}$.
(Standard GAN has $D_{\phi}: \mathbb{R}^{n} \rightarrow(0,1)$. Here, $\left(\sigma \circ D_{\phi}\right): \mathbb{R}^{n} \rightarrow(0,1)$ serves the same purpose.)

## WGAN

The Wasserstein GAN (WGAN) minimizes the Wasserstein distance:

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \quad W\left(p_{\text {true }}, p_{\theta}\right)
$$

The $W(p, q)$ is a distance (metric) on probability distributions defined as

$$
W(p, q)=\inf _{f} \mathbb{E}_{(X, Y) \sim f(x, y)}\|X-Y\|
$$

where the infimum is taken over joint probability distributions $f$ with marginals $p$ and $q$, i.e.,

$$
p(x)=\int f(x, y) d y, \quad q(y)=\int f(x, y) d x
$$

(The mathematics of $W(p, q)$ exceeds the scope of this class, but I still want to give you a high-level exposure to WGANs.)

## $W(p, q)$ by optimal transport

Another equivalent formulation of the Wasserstein distance is by the theory of optimal transport. Given distributions $p$ and $q$ (initial and target)

$$
W(p, q)=\inf _{T} \int\|x-T(x)\| p(x) d x
$$

where $T$ is a transport plan that transports $p$ to $q .{ }^{\%}$ Figuratively speaking, we are transporting grains of sand from one pile to another, and we wan to minimize the aggregate transport distance.
![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-420.jpg?height=1216&width=2243&top_left_y=391&top_left_x=1087)

Transport plan $T$ from $p$ to $q$.

## Minimax via KR duality

Kantorovich-Rubinstein duality\# establishes:

$$
W\left(p_{\text {true }}, p_{\theta}\right)=\sup _{\|T\|_{L \leq 1}} \mathbb{E}_{X \sim p_{\text {true }}}[T(X)]-\mathbb{E}_{\tilde{X} \sim p_{\theta}}[T(\tilde{X})]
$$

Minimax formulation of WGAN:

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} W\left(p_{\text {true }}, p_{\theta}\right)
$$

$$
\approx \underset{\theta \in \Theta}{\operatorname{minimize}} \begin{array}{ll}
\operatorname{maximize} \\
\text { subject to } & \mathbb{E}_{X \sim p_{\text {truu }}}\left[D_{\phi}(X)\right]-\mathbb{E}_{\tilde{X} \sim p_{\theta}} \text { is 1-Lipschitz }
\end{array}
$$

## Spectral normalization

How do we enforce the constraint that $D_{\phi}$ is 1-Lipschitz? Consider an MLP:

$$
\begin{aligned}
y_{L}= & A_{L} y_{L-1}+b_{L} \\
y_{L-1}= & \sigma\left(A_{L-1} y_{L-2}+b_{L-1}\right) \\
& \vdots \\
y_{2}= & \sigma\left(A_{2} y_{1}+b_{2}\right) \\
y_{1}= & \sigma\left(A_{1} x+b_{1}\right),
\end{aligned}
$$

where $\sigma$ is a 1-Lipschitz continuous activation function, such as ReLU and tanh. If

$$
\left\|A_{i}\right\|_{\mathrm{op}}=\sigma_{\max }\left(A_{i}\right) \leq 1
$$

for $i=1, \ldots, L$, where $\sigma_{\text {max }}$ denotes the largest singular value, then each layer is 1-Lipschitz continuous and the entire mapping $x \mapsto y_{L}$ is 1 -Lipschitz. (A sufficient, but not a necessary, condition.)

## Spectral normalization

Replace Lipschitz constraint with a singular-value constraint

$$
\begin{array}{rll}
\underset{\theta \in \Theta}{\operatorname{minimize}} & \underset{\phi \in \Phi}{\operatorname{maximize}} & \frac{1}{N} \sum_{i=1}^{N} D_{\phi}\left(X_{i}\right)-\mathbb{E}_{Z \sim \mathcal{N}(0, I)}\left[D_{\phi}\left(G_{\theta}(Z)\right)\right] \\
& \text { subject to } & \sigma_{\max }\left(A_{i}\right) \leq 1, \quad i=1, \ldots, L
\end{array}
$$

Constraint is handled with a projected gradient method. (Note that $A_{1}, \ldots, A_{L}$ are part of the discriminator parameters $\phi$.)
(Specifically, one performs an (approximate) projection after the ascent step in the stochastic gradient ascent-descent methods. The approximate projection involves computing the largest singular with the power iteration.)

## Conclusion

We discussed the following unsupervised learning techniques:

- Autoencoders
- Flow models
- Variational autoencoders
- GANs

Unsupervised learning techniques, particularly generative models, tend to utilize more math in their formulations. This chapter provided a brief and gentle introduction to the mathematical foundations of these formulations.

# Appendix A: <br> Basics of Monte Carlo 

Mathematical Foundations of Deep Neural Networks
Spring 2024
Department of Mathematical Sciences
Ernest K. Ryu
Seoul National University

## Monte Carlo

We quickly cover some basic notions of Monte Carlo simulations.

These concepts will be used with VAEs.

These ideas are also extensively used in reinforcement learning (although not a topic of this course).

## Monte Carlo estimation

Consider IID data $X_{1}, \ldots, X_{N} \sim f$. Let $\phi(X) \geq 0$ be some function*. Consider the problem of estimating

$$
I=\mathbb{E}_{X \sim f}[\phi(X)]=\int \phi(x) f(x) d x
$$

One commonly uses

$$
\hat{I}_{N}=\frac{1}{N} \sum_{i=1}^{N} \phi\left(X_{i}\right)
$$

to estimate $I$. After all, $\mathbb{E}\left[\hat{I}_{N}\right]=I$ and $\hat{I}_{N} \rightarrow I$ by the law of large numbers.\#

## Monte Carlo estimation

We can quantify convergence with variance:

$$
\operatorname{Var}_{X \sim f}\left(\hat{I}_{N}\right)=\sum_{i=1}^{N} \operatorname{Var}_{X_{i} \sim f}\left(\frac{\phi\left(X_{i}\right)}{N}\right)=\frac{1}{N} \operatorname{Var}_{X \sim f}(\phi(X))
$$

In other words

$$
\mathbb{E}\left[\left(\hat{I}_{N}-I\right)^{2}\right]=\frac{1}{N} \operatorname{Var}_{X \sim f}(\phi(X))
$$

and

$$
\mathbb{E}\left[\left(\hat{I}_{N}-I\right)^{2}\right] \rightarrow 0
$$

as $N \rightarrow \infty$.\#

## Empirical risk minimization

In machine learning and statistics, we often wish to solve

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \quad \mathcal{L}(\theta)
$$

where the objective function

$$
\mathcal{L}(\theta)=\mathbb{E}_{X \sim p_{X}}\left[\ell\left(f_{\theta}(X), f_{\star}(X)\right)\right]
$$

Is the (true) risk. However, the evaluation of $\mathbb{E}_{X \sim p_{X}}$ is impossible (if $p_{X}$ is unknown) or intractable (if $p_{X}$ is known but the expectation has no closed-form solution). Therefore, we define the proxy loss function

$$
\mathcal{L}_{N}(\theta)=\frac{1}{N} \sum_{i=1}^{N} \ell\left(f_{\theta}\left(X_{i}\right), f_{\star}\left(X_{i}\right)\right)
$$

which we call the empirical risk, and solve

$$
\underset{\theta \in \Theta}{\operatorname{minimize}} \quad \mathcal{L}_{N}(\theta)
$$

## Empirical risk minimization

This is called empirical risk minimization (ERM). The idea is that

$$
\mathcal{L}_{N}(\theta) \approx \mathcal{L}(\theta)
$$

with high probability, so minimizing $\mathcal{L}_{N}(\theta)$ should be similar to minimizing $\mathcal{L}(\theta)$.

Technical note) The law of large numbers tells us that

$$
\mathbb{P}\left(\left|\mathcal{L}_{N}(\theta)-\mathcal{L}(\theta)\right|>\varepsilon\right)=\text { small }
$$

for any given $\theta$, but we need

$$
\mathbb{P}\left(\sup _{\theta \in \Theta}\left|\mathcal{L}_{N}(\theta)-\mathcal{L}(\theta)\right|>\varepsilon\right)=\text { small }
$$

for all compact $\Theta$ in order to conclude that the argmins of the two losses to be similar. These types of results are established by a uniform law of large numbers.

## Importance sampling

Importance sampling (IS) is a technique for reducing the variance of a Monte Carlo estimator.

Key insight of important sampling:

$$
I=\int \phi(x) f(x) d x=\int \frac{\phi(x) f(x)}{g(x)} g(x) d x=\mathbb{E}_{X \sim g}\left[\frac{\phi(X) f(X)}{g(X)}\right]
$$

(We do have to be mindful of division by 0.) Then

$$
\hat{I}_{N}=\frac{1}{N} \sum_{i=1}^{N} \phi\left(X_{i}\right) \frac{f\left(X_{i}\right)}{g\left(X_{i}\right)}
$$

with $X_{1}, \ldots, X_{N} \sim g$ is also an estimator of $I$. Indeed, $\mathbb{E}\left[\hat{I}_{N}\right]=I$ and $\hat{I}_{N} \rightarrow I$. The weight $\frac{f(x)}{g(x)}$ is called the likelihood ratio or the Radon-Nikodym derivative.

So we can use samples from $g$ to compute expectation with respect to $f$.

## IS example: Low probability events

Consider the setup of estimating the probability

$$
\mathbb{P}(X>3)=0.00135
$$

where $X \sim \mathcal{N}(0,1)$. If we use the regular Monte Carlo estimator

$$
\hat{I}_{N}=\frac{1}{N} \sum_{i=1}^{N} \mathbf{1}_{\left\{X_{i}>3\right\}}
$$

where $X_{i} \sim \mathcal{N}(0,1)$, if $N$ is not sufficiently large, we can have $\hat{I}_{N}=0$. Inaccurate estimate.

If we use the IS estimator

$$
\hat{I}_{N}=\frac{1}{N} \sum_{i=1}^{N} \mathbf{1}_{\left\{Y_{i}>3\right\}} \exp \left(\frac{\left(Y_{i}-3\right)^{2}-Y_{i}^{2}}{2}\right)
$$

where $Y_{i} \sim \mathcal{N}(3,1)$, having $\hat{I}_{N}=0$ is much less likely. Estimate is much more accurate.

## Importance sampling

Benefit of IS quantified by with variance:

$$
\operatorname{Var}_{X \sim g}\left(\hat{I}_{N}\right)=\sum_{i=1}^{N} \operatorname{Var}_{X \sim g}\left(\frac{\phi\left(X_{i}\right) f\left(X_{i}\right)}{n g\left(X_{i}\right)}\right)=\frac{1}{N} \operatorname{Var}_{X \sim g}\left(\frac{\phi(X) f(X)}{g(X)}\right)
$$

If $\operatorname{Var}_{X \sim g}\left(\frac{\phi(X) f(X)}{g(X)}\right)<\operatorname{Var}_{X \sim f}(\phi(X))$, then IS provides variance reduction.

We call $g$ the importance or sampling distribution. Choosing $g$ poorly can increase the variance. What is the best choice of $g$ ?

## Optimal sampling distribution

The sampling distribution

$$
g(x)=\frac{\phi(x) f(x)}{I}
$$

makes $\operatorname{Var}_{X \sim g}\left(\frac{\phi(X) f(X)}{g(X)}\right)=\operatorname{Var}_{X \sim g}(I)=0$ and therefore is optimal. (I serves as the normalizing factor that ensures the density $g$ integrates to 1.)

Problem: Since we do not know the normalizing factor $I$, the answer we wish to estimate, sampling from $g$ is usually difficult.

## Optimized/trained sampling distribution

Instead, we consider the optimization problem

$$
\underset{g \in \mathcal{G}}{\operatorname{minimize}} \quad D_{\mathrm{KL}}\left(g \| \frac{\phi f}{I}\right)
$$

and compute a suboptimal, but good, sampling distribution within a class of sampling distributions $\mathcal{G}$. (In ML, $\mathcal{G}=\left\{g_{\theta} \mid \theta \in \Theta\right\}$ is parameterized by neural networks.)

Importantly, this optimization problem does not require knowledge of $I$.

$$
\begin{aligned}
D_{\mathrm{KL}}\left(g_{\theta} \| \phi f / I\right) & =\mathbb{E}_{X \sim g_{\theta}}\left[\log \left(\frac{I g_{\theta}(X)}{\phi(X) f(X)}\right)\right] \\
& =\mathbb{E}_{X \sim g_{\theta}}\left[\log \left(\frac{g_{\theta}(X)}{\phi(X) f(X)}\right)\right]+\log I \\
& =\mathbb{E}_{X \sim g_{\theta}}\left[\log \left(\frac{g_{\theta}(X)}{\phi(X) f(X)}\right)\right]+\text { constant independent of } \theta
\end{aligned}
$$

How do we compute stochastic gradients?

## Log-derivative trick

Generally, consider the setup where we wish to solve

$$
\underset{\theta \in \mathbb{R}^{p}}{\operatorname{minimize}} \mathbb{E}_{X \sim f_{\theta}}[\phi(X)]
$$

with SGD.
(Previous slide had $\theta$-dependence both on and inside the expectation. For now, let's simplify the problem so that $\phi$ does not depend on $\theta$.)

Incorrect gradient computation:

$$
\nabla_{\theta} \mathbb{E}_{X \sim f_{\theta}}[\phi(X)] \stackrel{?}{=} \mathbb{E}_{X \sim f_{\theta}}\left[\nabla_{\theta} \phi(X)\right]=\mathbb{E}_{X \sim f_{\theta}}[0]=0
$$

## Log-derivative trick

Correct gradient computation:

$$
\begin{aligned}
\nabla_{\theta} \mathbb{E}_{X \sim f_{\theta}}[\phi(X)] & =\nabla_{\theta} \int \phi(x) f_{\theta}(x) d x=\int \phi(x) \nabla_{\theta} f_{\theta}(x) d x \\
& =\int \phi(x) \frac{\nabla_{\theta} f_{\theta}(x)}{f_{\theta}(x)} f_{\theta}(x) d x=\mathbb{E}_{X \sim f_{\theta}}\left[\phi(X) \frac{\nabla_{\theta} f_{\theta}(X)}{f_{\theta}(X)}\right] \\
& =\mathbb{E}_{X \sim f_{\theta}}\left[\phi(X) \nabla_{\theta} \log \left(f_{\theta}(X)\right)\right]
\end{aligned}
$$

Therefore, $\phi(X) \nabla_{\theta} \log \left(f_{\theta}(X)\right)$ with $X \sim f_{\theta}$ is a stochastic gradient of the loss function. This technique is called the log-derivative trick, the likelihood ratio gradient\#, or REINFORCE*.

Formula with the log-derivative $\left(\nabla_{\theta} \log (\cdot)\right)$ is convenient when dealing with Gaussians, or more generally exponential families, since the densities are of the form

$$
f_{\theta}(x)=h(x) \exp (\text { function of } \theta)
$$

## Log-derivative trick example

Learn $\mu \in \mathbb{R}^{2}$ to minimize the objective below.

$$
\underset{\mu \in \mathbb{R}^{2}}{\operatorname{minimize}} \mathbb{E}_{X \sim \mathcal{N}(\mu, I)}\left\|X-\binom{5}{5}\right\|^{2}
$$

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-438.jpg?height=732&width=881&top_left_y=0&top_left_x=2451)

Then the loss function is

$$
\mathcal{L}(\mu)=\mathbb{E}_{X \sim \mathcal{N}(\mu, I)}\left\|X-\binom{5}{5}\right\|^{2}=\int\left\|x-\binom{5}{5}\right\|^{2} \frac{1}{2 \pi} \exp \left(-\frac{1}{2}\|x-\mu\|^{2}\right) d x
$$

And, using $X_{1}, \ldots, X_{B} \sim \mathcal{N}(\mu, I)$, we have stochastic gradients

$$
\nabla_{\mu} \mathcal{L}(\mu)=\mathbb{E}_{X \sim q_{\mu}}\left[\left\|x-\binom{5}{5}\right\|^{2} \nabla_{\mu}\left(-\frac{1}{2}\|x-\mu\|^{2}\right)\right] \approx \frac{1}{B} \sum_{i=1}^{B}\left\|X_{i}-\binom{5}{5}\right\|^{2}\left(X_{i}-\mu\right)
$$

These stochastic gradients have large variance and thus SGD is slow.

## Log-derivative trick example

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-439.jpg?height=1438&width=1953&top_left_y=395&top_left_x=601)

## Reparameterization trick

The reparameterization trick (RT) or the pathwise derivative (PD) relies on the key insight.

$$
\mathbb{E}_{X \sim \mathcal{N}\left(\mu, \sigma^{2}\right)}[\phi(X)]=\mathbb{E}_{Y \sim \mathcal{N}(0,1)}[\phi(\mu+\sigma Y)]
$$

Gradient computation:

$$
\begin{aligned}
\nabla_{\mu, \sigma} \mathbb{E}_{X \sim \mathcal{N}\left(\mu, \sigma^{2}\right)}[\phi(X)] & =\mathbb{E}_{Y \sim \mathcal{N}(0,1)}\left[\nabla_{\mu, \sigma} \phi(\mu+\sigma Y)\right]=\mathbb{E}_{Y \sim \mathcal{N}(0,1)}\left[\phi^{\prime}(\mu+\sigma Y)\left[\begin{array}{c}
1 \\
Y
\end{array}\right]\right] \\
& \approx \frac{1}{B} \sum_{i=1}^{B} \phi^{\prime}\left(\mu+\sigma Y_{i}\right)\left[\begin{array}{c}
1 \\
Y_{i}
\end{array}\right], \quad Y_{1}, \ldots, Y_{B} \sim \mathcal{N}(0, I)
\end{aligned}
$$

RT is less general than log-derivative trick, but it usually produces stochastic gradients with lower variance.

## Reparameterization trick example

Consider the same example as before

$$
\mathcal{L}(\mu)=\mathbb{E}_{X \sim \mathcal{N}(\mu, I)}\left\|X-\binom{5}{5}\right\|^{2}=\mathbb{E}_{Y \sim \mathcal{N}(0, I)}\left\|Y+\mu-\binom{5}{5}\right\|^{2}
$$

Gradient computation:

$$
\begin{aligned}
\nabla_{\mu} \mathcal{L}(\mu) & =\mathbb{E}_{Y \sim \mathcal{N}(0, I)} \nabla_{\mu}\left\|Y+\mu-\binom{5}{5}\right\|^{2}=2 \mathbb{E}_{Y \sim \mathcal{N}(0, I)}\left(Y+\mu-\binom{5}{5}\right) \\
& \approx \frac{2}{B} \sum_{i=1}^{B}\left(Y_{i}+\mu-\binom{5}{5}\right), \quad Y_{1}, \ldots, Y_{B} \sim \mathcal{N}(0, I)
\end{aligned}
$$

These stochastic gradients have smaller variance and thus SGD is faster.

## Reparameterization trick example

![](https://cdn.mathpix.com/cropped/2025_01_07_dc39ed2e168ab0674c3ag-442.jpg?height=1434&width=1974&top_left_y=397&top_left_x=595)


[^0]:    Convolution+ReLU
    Dropout (0.5)
    Local response normalization (preserves spatial dimension\&channel \#s) (outdated technique)
    Max pool $f=3, s=2$ (overlapping max pool)
    Fully connected layer+ReLU

[^1]:    *R. M. Schmidt, F. Schneider, and P. Hennig, Descending through a crowded valley — benchmarking deep learning optimizers, ICML, 2021.
    ${ }^{\dagger}$ M. Tan and Q. V. Le, EfficientNet: Rethinking model scaling for convolutional neural networks, ICML, 2019.
    \#A. C. Wilson, R. Roelofs, M. Stern, N. Srebro, and B. Recht, The marginal value of adaptive gradient methods in machine learning, NeurlPS, 2017.

