---
layout: post
title: "Two Views on L2 Regularization"
date: 2026-09-12
mathjax: true
---

<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

L2 regularization is a well-known technique in machine learning, and its benefits are at least two-fold, as any practitioner will confirm:

1. Small weights tend to produce models that generalize better.
2. Small weights help control collinearity in models sensitive to it.

Many popular libraries offer it as a built-in option: in classical ML it's called ridge regression (scikit-learn implements it as `Ridge` in the `linear_model` module), and PyTorch exposes it as `weight_decay`. It is also known as Tikhonov regularization, after the Russian mathematician who first described and justified the method.

In this note I'd like to share two views on the method that I find particularly fascinating. They are largely unrelated to each other, but that's precisely the beauty of it: a striking convergence of two seemingly distant paths arriving at the same place. This is not meant to be a detailed, self-contained exposition — think of it more as a tour of two vantage points.

## Setting the Stage

The central object of machine learning is a **loss function**: something we optimize to uncover a relationship between the data and the target variable we're modeling. Popular choices include MSE (Mean Squared Error), MAE (Mean Absolute Error), Huber loss, categorical cross-entropy, hinge loss, and many more. Let's write a generic loss as $$L(\mathbf{x}, \mathbf{w})$$, where $$\mathbf{x}$$ is a vector of inputs and $$\mathbf{w}$$ is the vector of model weights we're trying to find. Here $$\mathbf{x}$$ is known, and $$\mathbf{w}$$ is what we optimize over. The task at hand is

$$\arg\min_{\mathbf{w}} L(\mathbf{x}, \mathbf{w}),$$

"find the weights $$\mathbf{w}$$ that minimize the loss."

As mentioned above, we often also want these weights to be *small*. We achieve this by folding an additional objective into the optimization problem:

$$\arg\min_{\mathbf{w}} \; L(\mathbf{x}, \mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2,$$

"find the weights that minimize the loss *and* keep the vector $$\mathbf{w}$$ as short as possible." The parameter $$\lambda$$ is a knob controlling how much emphasis to place on shrinking $$\mathbf{w}$$: large $$\lambda$$ forces smaller weights, while $$\lambda$$ near zero gives the model more freedom.

Note that $$\|\mathbf{w}\|^2 := \sum_i w_i^2$$, so the sign of each $$w_i$$ is irrelevant, and the penalty grows quadratically with the size of each component — larger components contribute disproportionately to the objective. The actual *shrinkage* each weight ends up experiencing, though, is best understood not coordinate-by-coordinate but direction-by-direction: as the PCA section below will show, ridge shrinks different directions in parameter space by different amounts, depending on the underlying geometry of the features — not simply on how large a given $$w_i$$ happens to be.

Now that the setup and notation are out of the way, let's look at the first surprising fact: the Bayesian connection.

## The Bayesian Connection

### Bayes' Theorem

One of the most important theorems in machine learning — arguably *the* most important — is Bayes' theorem:

$$p(\mathbf{w} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathbf{w})\, p(\mathbf{w})}{p(\mathcal{D})}$$

The left-hand side is the **posterior**: our updated belief about the weights after seeing the data $$\mathcal{D}$$. In the numerator, $$p(\mathcal{D} \mid \mathbf{w})$$ is the **likelihood** and $$p(\mathbf{w})$$ is the **prior**. This formula elegantly folds our prior beliefs about the weights into their computation given data.

Crucially, the posterior is a full *distribution* over possible weight vectors, not a single point. This is usually intractable to work with directly. As it turns out, there's a useful middle ground between full Bayesian inference (carrying around the entire distribution) and a pure optimization approach (picking one *best* set of parameters): find the single most probable weight vector under the posterior. This is called **MAP estimation** — Maximum A Posteriori.

### Setting Up the Model

Let's work this out concretely for linear regression, using the same weight vector $$\mathbf{w}$$ as above. Let $$\boldsymbol{\phi}(\mathbf{x}) \in \mathbb{R}^M$$ be a **feature vector** (also called basis functions) — a fixed transformation of the raw input $$\mathbf{x}$$. This is general enough to include:

- Plain linear regression: $$\boldsymbol{\phi}(\mathbf{x}) = \mathbf{x}$$
- Polynomial regression: $$\boldsymbol{\phi}(x) = (1, x, x^2, \ldots, x^{M-1})^\top$$
- Any other fixed nonlinear transformation of the inputs

The key point is that the model stays *linear in* $$\mathbf{w}$$, even when it's nonlinear in $$\mathbf{x}$$.

**Likelihood.** We model each output as linear in the parameters, plus Gaussian noise:

$$t_i = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}_i) + \epsilon_i, \qquad \epsilon_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$$

For a dataset $$\mathcal{D} = \{(\mathbf{x}_i, t_i)\}_{i=1}^n$$, stack everything into matrix form:

$$\mathbf{t} = \Phi\mathbf{w} + \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I)$$

where $$\Phi \in \mathbb{R}^{n \times M}$$ is the **design matrix**, with rows $$\boldsymbol{\phi}(\mathbf{x}_i)^\top$$. This gives a likelihood

$$p(\mathbf{t} \mid \mathbf{w}, \Phi, \sigma^2) = \mathcal{N}(\mathbf{t};\, \Phi\mathbf{w},\, \sigma^2 I) \;\propto\; \exp\!\left(-\frac{1}{2\sigma^2}\|\mathbf{t} - \Phi\mathbf{w}\|^2\right)$$

**Prior.** Now for the part that matters most: choosing a prior on $$\mathbf{w}$$. We place a zero-mean, isotropic Gaussian prior:

$$p(\mathbf{w}) = \mathcal{N}(\mathbf{w};\, \mathbf{0},\, \alpha^{-1}I) \;\propto\; \exp\!\left(-\frac{\alpha}{2}\|\mathbf{w}\|^2\right)$$

where $$\alpha > 0$$ is the **prior precision** (inverse variance). Larger $$\alpha$$ means a tighter prior — we're assuming, before seeing any data, that the weights are small.

### Deriving the Posterior

Plugging into Bayes' theorem and dropping the normalizing constant $$p(\mathcal{D})$$, which doesn't depend on $$\mathbf{w}$$:

$$p(\mathbf{w} \mid \mathcal{D}) \;\propto\; p(\mathbf{t} \mid \mathbf{w}, \Phi, \sigma^2)\, p(\mathbf{w})$$

Products of exponentials are unwieldy to optimize directly, so we take logarithms. Since $$\log$$ is monotonically increasing, it doesn't change *where* the maximum is — only how easy the function is to work with. And it's considerably easier: the product on the right becomes a sum, and each exponential term collapses to the (much friendlier) quantity in its exponent:

$$\log p(\mathbf{w} \mid \mathcal{D}) = -\frac{1}{2\sigma^2}\|\mathbf{t} - \Phi\mathbf{w}\|^2 - \frac{\alpha}{2}\|\mathbf{w}\|^2 + \text{const}$$

Expanding the squared norm and collecting terms in $$\mathbf{w}$$:

$$= -\frac{1}{2}\,\mathbf{w}^\top\!\left(\frac{\Phi^\top\Phi}{\sigma^2} + \alpha I\right)\!\mathbf{w} + \frac{1}{\sigma^2}\,\mathbf{t}^\top\Phi\mathbf{w} + \text{const}$$

This has the form $$-\frac12\mathbf{w}^\top A \mathbf{w} + \mathbf{b}^\top\mathbf{w} + \text{const}$$, with $$A = \Phi^\top\Phi/\sigma^2 + \alpha I$$ and $$\mathbf{b} = \Phi^\top\mathbf{t}/\sigma^2$$. Completing the square puts this directly into the log-density of a Gaussian — no need to invoke conjugate-prior machinery as a black box, it's just algebra — so the posterior $$p(\mathbf{w} \mid \mathcal{D})$$ is itself Gaussian. For a Gaussian, the mode and the mean coincide, so finding the *most probable* $$\mathbf{w}$$ is the same as finding the mean of this posterior.

### From Posterior to MAP — and Back to L2

The MAP estimate is, by definition, the mode of the posterior:

$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\max_{\mathbf{w}} p(\mathbf{w} \mid \mathcal{D}) = \arg\max_{\mathbf{w}} \log p(\mathbf{w} \mid \mathcal{D})$$

Maximizing the log-posterior above is the same as minimizing its negative:

$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\min_{\mathbf{w}} \left[\frac{1}{2\sigma^2}\|\mathbf{t} - \Phi\mathbf{w}\|^2 + \frac{\alpha}{2}\|\mathbf{w}\|^2\right]$$

Look at this objective next to the one we wrote down at the very start of this article. The first term is (up to the constant $$1/\sigma^2$$) a squared-error loss $$L(\mathbf{x}, \mathbf{w})$$. The second term is *exactly* our L2 penalty $$\frac{\lambda}{2}\|\mathbf{w}\|^2$$, with

$$\lambda = \alpha\sigma^2$$

In other words: **ridge regression *is* MAP estimation**, under a zero-mean isotropic Gaussian prior on the weights. They are not merely analogous — they are literally the same optimization problem, just arrived at from two different directions: one by penalizing a loss function by hand, the other by taking a prior belief about the weights seriously and asking Bayes' theorem what follows.

Both routes lead to the same closed-form solution — solve $$\frac{1}{2\sigma^2}\|\mathbf{t}-\Phi\mathbf{w}\|^2 + \frac{\alpha}{2}\|\mathbf{w}\|^2$$ directly, or equivalently substitute $$\lambda = \alpha\sigma^2$$ into the plain-form ridge objective — and either way you land on:

$$\hat{\mathbf{w}}_{\text{MAP}} = \left(\Phi^\top\Phi + \lambda\, I\right)^{-1} \Phi^\top \mathbf{t}, \qquad \lambda = \alpha\sigma^2$$

The regularization strength $$\lambda$$, which we introduced as an arbitrary knob, turns out to have a precise meaning: it's the ratio of the noise variance $$\sigma^2$$ to the prior variance $$\alpha^{-1}$$ on the weights. A noisier likelihood or a tighter prior — either one — pushes $$\lambda$$ up and the weights down.

### A Frequentist Postscript

It's worth being honest about how this plays out in practice. In the frequentist workflow, nobody sets $$\sigma^2$$ and $$\alpha$$ individually and multiplies them together — $$\lambda$$ is simply treated as a hyperparameter, chosen by cross-validation or a grid search over held-out data. That procedure never separates out $$\sigma^2$$ and $$\alpha$$ at all; it only ever sees their product, so it's silent on which of the two — noisier data or a tighter prior — is doing the work.

Getting $$\sigma^2$$ and $$\alpha$$ individually, as meaningful quantities rather than a tuned product, is a taller order. $$\sigma^2$$ is at least in principle estimable from data — from residual variance after fitting, from repeated measurements at the same input, or from known measurement precision in the data-generating process. $$\alpha$$ is harder: it's a genuine prior belief about the *scale* of the weights, fixed before seeing $$\mathcal{D}$$, and in most applications there's no such belief lying around. In practice, when people do want $$\alpha$$ (and $$\sigma^2$$) separately rather than folded into a single $$\lambda$$, they usually turn to *empirical Bayes* — estimating them from the data itself by maximizing the marginal likelihood $$p(\mathcal{D})$$ — which quietly reintroduces the data-fitting that a "true" prior is supposed to be free of.

## The PCA Connection

### What Is PCA, Really

Principal Component Analysis starts from a different question than regression: forget the target $$t$$ entirely, and ask which directions in feature space capture the most *variance* in the inputs themselves.

Take the design matrix $$\Phi \in \mathbb{R}^{n \times M}$$ from before, with rows $$\boldsymbol{\phi}(\mathbf{x}_i)^\top$$ — but now, critically, assume its columns are **mean-centered** (each feature has had its sample mean subtracted off). This centering step is not optional bookkeeping the way it might be in regression; PCA is fundamentally about *how the data varies around its mean*, so if you skip centering, the "variance directions" you find are contaminated by the mean itself.

The (unnormalized) **covariance matrix** — more precisely, the scatter matrix, since we're skipping the $$1/n$$ or $$1/(n-1)$$ normalization for now — is

$$\Phi^\top \Phi \in \mathbb{R}^{M \times M}$$

Being symmetric and positive semi-definite, it admits an eigendecomposition

$$\Phi^\top \Phi = V D V^\top$$

where $$V = [\mathbf{v}_1, \ldots, \mathbf{v}_M]$$ is an orthonormal matrix of **eigenvectors** (the *principal directions*), and $$D = \mathrm{diag}(d_1, \ldots, d_M)$$ holds the corresponding **eigenvalues**, sorted $$d_1 \geq d_2 \geq \cdots \geq d_M \geq 0$$. Up to the normalization constant, $$d_j$$ is exactly the variance of the data along direction $$\mathbf{v}_j$$: it tells you how spread out the data is once you look at it from that particular angle.

PCA's usual move is dimensionality reduction: keep only the top $$k$$ eigenvectors — the $$k$$ directions of greatest variance — and either project the data down onto them, $$\mathbf{z}_i = V_k^\top \boldsymbol{\phi}(\mathbf{x}_i)$$, or reconstruct each point using only that subspace, $$\hat{\boldsymbol{\phi}}(\mathbf{x}_i) = V_k V_k^\top \boldsymbol{\phi}(\mathbf{x}_i)$$. Notice the character of this decision: each direction $$\mathbf{v}_j$$ either survives entirely (weight $$1$$, if $$j \leq k$$) or is discarded entirely (weight $$0$$, if $$j > k$$). It's a hard, discrete cutoff — you draw a line after the $$k$$-th eigenvalue and everything on the far side of it is gone.

### From the Ridge Objective to the Covariance Matrix

Now go back to the ridge objective from the Bayesian section and take its gradient with respect to $$\mathbf{w}$$:

$$\nabla_{\mathbf{w}} \left[\frac{1}{2}\|\mathbf{t} - \Phi\mathbf{w}\|^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2\right] = -\Phi^\top(\mathbf{t} - \Phi\mathbf{w}) + \lambda\mathbf{w}$$

Setting this to zero gives the **regularized normal equations**:

$$\left(\Phi^\top \Phi + \lambda I\right)\mathbf{w} = \Phi^\top \mathbf{t}$$

The quadratic term $$\Phi^\top\Phi$$ that falls out of this gradient is the *exact same matrix* we just eigendecomposed above. This is the first hint of the connection: the curvature of the least-squares loss, and the covariance structure PCA cares about, are literally the same object.

### Ridge Regression in the Eigenbasis: Shrinkage Factors

Since $$V$$ is orthonormal, $$I = VV^\top$$, so we can write

$$\Phi^\top\Phi + \lambda I = VDV^\top + \lambda VV^\top = V(D + \lambda I)V^\top$$

and therefore

$$\hat{\mathbf{w}}_{\text{ridge}} = \left(\Phi^\top\Phi + \lambda I\right)^{-1}\Phi^\top\mathbf{t} = V(D+\lambda I)^{-1}V^\top \Phi^\top\mathbf{t}$$

Compare this to the unregularized least-squares solution, $$\hat{\mathbf{w}}_{\text{OLS}} = (\Phi^\top\Phi)^{-1}\Phi^\top\mathbf{t} = VD^{-1}V^\top\Phi^\top\mathbf{t}$$ (assuming $$\Phi^\top\Phi$$ is invertible). Writing $$\mathbf{u} := V^\top \hat{\mathbf{w}}_{\text{OLS}}$$ — the OLS solution expressed in the eigenbasis of $$\Phi^\top\Phi$$ — the normal equations give $$V^\top\Phi^\top\mathbf{t} = D\mathbf{u}$$, so

$$\hat{\mathbf{w}}_{\text{ridge}} = V(D+\lambda I)^{-1}D\,\mathbf{u} = V \,\mathrm{diag}\!\left(\frac{d_j}{d_j + \lambda}\right)\mathbf{u}$$

In words: express the OLS solution in the coordinate system given by the principal directions of $$\Phi^\top\Phi$$, and ridge regression rescales *each coordinate* by a factor $$d_j/(d_j+\lambda) \in (0,1)$$, then rotates back. The eigenvectors don't move — they're the same $$\mathbf{v}_j$$ as in PCA. Only how much weight each one is given changes.

### The Continuous View

Now look at what that shrinkage factor does as a function of $$d_j$$:

- When $$d_j \gg \lambda$$ (a high-variance direction — exactly the kind of direction PCA would keep), $$\dfrac{d_j}{d_j+\lambda} \approx 1$$: essentially no shrinkage.
- When $$d_j \ll \lambda$$ (a low-variance direction — exactly the kind of direction a truncated PCA would discard), $$\dfrac{d_j}{d_j+\lambda} \approx 0$$: the coordinate is crushed toward zero. Worth being precise here: low variance in $$\Phi$$ is not the same thing as irrelevance to $$\mathbf{t}$$ — a direction the data barely moves along can still carry real predictive signal. Ridge's advantage over a hard truncation is exactly that it never has to gamble on that distinction; it discounts a shaky direction instead of deleting it outright.
- When $$d_j = 0$$ exactly, the factor is exactly $$0$$ regardless of $$\lambda$$ — a direction with no variance at all in the data contributes nothing, the one case where ridge's soft gate and PCA's hard gate agree completely.
- As $$\lambda \to 0$$, every factor tends to $$1$$ and $$\hat{\mathbf{w}}_{\text{ridge}} \to \hat{\mathbf{w}}_{\text{OLS}}$$.
- As $$\lambda \to \infty$$, every factor tends to $$0$$ and $$\hat{\mathbf{w}}_{\text{ridge}} \to \mathbf{0}$$.

This is precisely PCA's logic — sort directions by eigenvalue, keep the important ones, suppress the unimportant ones — except the step function $$\{0, 1\}$$ of hard truncation has been replaced by the smooth function $$d_j/(d_j+\lambda)$$ of soft shrinkage. Where PCA asks a binary question ("is this direction in the top $$k$$?"), ridge asks a continuous one ("how large is $$d_j$$ relative to $$\lambda$$?"), and answers it with a graded weight rather than a verdict. Same eigenvectors, same underlying matrix, same ranking by eigenvalue — just a discrete gate softened into a continuous one.

### A Postscript: PCA on Its Own Terms

It's worth resisting the temptation to see PCA as merely "ridge regression's hard-thresholded cousin." PCA never looks at $$\mathbf{t}$$ — it's entirely unsupervised, a statement purely about the geometry of $$\Phi$$. That's exactly what makes it useful far outside any regression context: as a first pass on a new dataset to see how many directions actually carry meaningful variance, as a way to decorrelate or whiten features, for visualizing high-dimensional data in two or three dimensions, for denoising, or for compressing data before it ever meets a target variable. Ridge regression, by contrast, is inherently supervised — the "right" amount of shrinkage along each direction depends on $$\mathbf{t}$$ and $$\lambda$$, not on the variance structure of $$\Phi$$ alone.

The two techniques also diverge in a very practical way: **Principal Component Regression** (regress on the top-$$k$$ principal components) requires choosing $$k$$, a discrete and somewhat awkward hyperparameter to tune. Ridge replaces that discrete choice with a continuous one: rather than deciding which directions survive and which vanish, $$\lambda$$ controls how strongly every direction with nonzero variance is shrunk. That continuity — a dial instead of a cutoff — is often the more convenient thing to tune in practice, even without invoking any of the eigen-structure above.

## Epilogue

There's a story from nineteenth-century physics that this whole exercise kept reminding me of. In the 1850s and 60s, two constants had been measured completely independently: the electric constant $$\varepsilon_0$$ and the magnetic constant $$\mu_0$$, both pulled out of static, benchtop experiments with capacitors and coils — nothing to do with light. Separately, people had been measuring the speed of light itself, optically, since Fizeau's rotating-toothed-wheel experiment in 1849. When Maxwell derived, from his equations of electromagnetism, that a disturbance in the electromagnetic field should propagate at $$c = 1/\sqrt{\varepsilon_0\mu_0}$$, he plugged in the measured values of $$\varepsilon_0$$ and $$\mu_0$$ and got a number that matched the independently measured speed of light almost exactly. Maxwell's own reaction, more or less, was: this is too close to be an accident — light itself must *be* an electromagnetic wave. He was right, and it took physics somewhere it had no reason to expect to go, unifying optics with electromagnetism decades before anyone had a deeper theory of why it had to be so.

It's tempting to reach for the same language here — Bayesian inference and linear algebra converging on the identical formula feels like it should mean something similarly deep is going on underneath. I think it's worth being a bit more careful than that, though, and the honest version of the story is less mystical but still genuinely satisfying. Maxwell's coincidence was a statement about the physical world: two quantities measured in unrelated experiments turned out to be the same *fact* about nature, and the explanation (light is electromagnetism) was new physical content nobody had put in by hand. What's happening with L2 regularization is a statement about a single piece of mathematics wearing two hats. The MAP derivation and the eigenbasis derivation aren't two independent measurements of some deeper truth — they're two people, standing in different rooms, staring at the exact same object: the quadratic form $$\|\mathbf{t}-\Phi\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|^2$$, or equivalently the linear system $$(\Phi^\top\Phi + \lambda I)\mathbf{w} = \Phi^\top\mathbf{t}$$. One room asks "what distribution is this the log-density of, and where's its peak?" — and gets a Gaussian posterior whose mode happens to satisfy that equation. The other room asks "how does this matrix act, direction by direction?" — and gets an eigendecomposition whose shrinkage factors happen to satisfy the same equation. Of course they agree: it's the same equation. The genuine payoff isn't that two unrelated fields secretly agree by cosmic coincidence — it's that this one quadratic form is rich enough to be *legible* from two completely different directions, Bayesian belief-updating and spectral geometry, without either one having to know the other exists.

If that sounds like a smaller claim than Maxwell's, it is — and it should be, because we didn't do an experiment, we did algebra, and algebra isn't supposed to surprise you about the world the way a benchtop measurement can. But it's not nothing, either. It's a reminder that a good piece of mathematics is rarely the property of the field that invented it. The same quadratic form has been sitting at the intersection of probability, optimization, and linear algebra the whole time; it just took someone in each field independently deciding to stare at it before anyone noticed all three were describing the same picture.
