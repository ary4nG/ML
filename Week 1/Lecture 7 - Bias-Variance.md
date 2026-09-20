# Lecture 7 — Bias-Variance Tradeoff

This lecture answers a really important, practical question: **why doesn't the most flexible, most powerful model always win?** Any model's total error can be split into three separate, distinct sources of error — understanding each piece tells you exactly why "more complexity" isn't automatically "better."

## Step 0: A new idea we need first — the model itself is random

So far we've talked about randomness in the *data* — noisy measurements, spread-out outputs. Now we need a slightly different idea: **the fitted model itself is random**, because it depends on which particular training set you happened to collect.

Thought experiment: imagine you could magically collect 100 *different* training datasets, all from the same underlying real-world process. Each time, you fit your model (say, a straight line, or a k-NN model) on that dataset. Since each dataset is a little different (different random samples), **the fitted model will come out slightly different each time too.**

So $\hat f(x)$ — your fitted prediction at some point $x$ — isn't one fixed thing. It's a random quantity, because it depends on which random training set you happened to get. This is the key mental shift for this lecture.

## Step 1: Two very different ways a model can be "bad"

Given that $\hat f(x)$ varies across different training sets, there are two completely different ways your predictions can go wrong:

**1. Bias — being systematically wrong, on average.**
Imagine you always use a straight line to fit data that's actually curved. No matter how much data you collect, or which particular sample you happen to get, a straight line just *cannot* bend to match the curve. So even if you averaged your fitted line's prediction across a thousand different training sets, that average would still be off from the truth — consistently, in the same direction. This systematic "always a bit wrong in the same way" error is called **bias**. It comes from your model being too simple/rigid to represent the real pattern.

**2. Variance — being inconsistent across different training sets.**
Now imagine the opposite: an extremely flexible model (like a wiggly high-degree polynomial) that bends to pass through every single training point exactly. On one training set, it might curve up sharply near some region; give it a slightly different training set (with slightly different noise), and it curves completely differently in that same region. The model isn't wrong on average — but it swings wildly depending on exactly which random sample it saw. This inconsistency is called **variance**. It comes from your model being so flexible that it starts fitting the random noise in the data, not just the true pattern.

## Step 2: The dartboard picture

- **Bias** = how far off-center your average shot lands (are you aiming at the wrong spot?)
- **Variance** = how spread out your shots are around wherever their average lands (are you consistent, even if aiming wrong?)

You want both low — dart throws tightly clustered right on the bullseye. There's a tension between these two: models that reduce one tend to increase the other.

## Step 3: A third, unavoidable source of error

Even the perfect model, that somehow knew the *exact* true underlying pattern $f(x)$, still can't predict $y$ perfectly — because real data has **noise**:

$$
y = f(x) + \varepsilon
$$

where $f(x)$ is the true underlying pattern, and $\varepsilon$ is random noise (mean 0, spread/variance $\sigma^2$). This is exactly the "irreducible error" idea from Lecture 5/6 — no model, however good, can predict pure randomness. This is the **third source of error**, completely separate from bias and variance.

## Step 4: Deriving the exact formula

We want: at some test point $x_0$, what's the *expected* squared error of our prediction, averaging over **two** sources of randomness — the noise in the test observation, and the randomness of which training set we happened to fit on?

$$
\text{Err}(x_0) = \mathbb{E}\big[(y_0 - \hat f(x_0))^2\big], \qquad y_0 = f(x_0) + \varepsilon
$$

**Substep A — split off the noise term.**
Substitute $y_0 = f(x_0) + \varepsilon$ and expand the square:

$$
(y_0 - \hat f)^2 = \varepsilon^2 + 2\varepsilon(f(x_0) - \hat f(x_0)) + (f(x_0) - \hat f(x_0))^2
$$

Take the expectation of each term:
- $\mathbb{E}[\varepsilon^2] = \sigma^2$ (definition of variance of the noise, since its mean is 0).
- The middle "cross term" vanishes to 0: the test noise $\varepsilon$ is independent of the training set that produced $\hat f$, and $\mathbb{E}[\varepsilon] = 0$. Multiplying something independent-and-zero-mean by anything else, then averaging, gives zero.
- The last term stays: $\mathbb{E}[(f(x_0) - \hat f(x_0))^2]$.

$$
\text{Err}(x_0) = \sigma^2 + \mathbb{E}[(f(x_0) - \hat f(x_0))^2]
$$

We've peeled off the irreducible noise term. Now split the remaining piece into bias and variance.

**Substep B — split the remaining term into bias and variance.**
Define $\bar f = \mathbb{E}[\hat f(x_0)]$ — "the average prediction you'd get, if you averaged $\hat f(x_0)$ across every possible random training set." It's a fixed number, representing the "typical" behavior of your fitting procedure.

Add and subtract $\bar f$ inside the term:

$$
f(x_0) - \hat f(x_0) = \underbrace{(f(x_0) - \bar f)}_{\text{a fixed number}} + \underbrace{(\bar f - \hat f(x_0))}_{\text{random, but averages to } 0}
$$

The second piece averages to 0 because $\bar f$ is defined as the average of $\hat f(x_0)$: $\mathbb{E}[\bar f - \hat f(x_0)] = \bar f - \mathbb{E}[\hat f(x_0)] = \bar f - \bar f = 0$.

Square this sum and take the expectation. The cross term vanishes again (one part is fixed, the other averages to zero), leaving:

$$
\mathbb{E}[(f(x_0) - \hat f(x_0))^2] = \underbrace{(f(x_0) - \bar f)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat f(x_0) - \bar f)^2]}_{\text{Variance}}
$$

$(f(x_0) - \bar f)^2$ is "how far is the average prediction from the truth, squared" — **Bias, squared**. $\mathbb{E}[(\hat f(x_0) - \bar f)^2]$ is "how much does the actual prediction typically differ from its own average" — **Variance**.

## Step 5: Putting it all together

$$
\boxed{\mathbb{E}[(y_0 - \hat f(x_0))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2}
$$

Three separate, additive sources of error:
- **$\sigma^2$**: pure noise — can never be reduced by any model, ever.
- **Bias²**: error from the model's assumptions being wrong or too rigid — doesn't go away even with infinite data, if the model is fundamentally too simple.
- **Variance**: error from the model being overly sensitive to exactly which random training set it happened to see — shrinks with more data, grows as the model is made more flexible.

## Step 6: Why this creates a tradeoff, not a free win

As model complexity increases:
- **Bias goes down** (a more flexible model can bend closer to the true pattern).
- **Variance goes up** (a more flexible model also has more freedom to chase random noise, wobbling more across different training sets).

Since total error = Bias² + Variance + $\sigma^2$, and these two pieces move in *opposite* directions as complexity changes, total error traces a **U-shape** as complexity increases:

- Too simple → **underfitting**: high bias, low variance.
- Too complex → **overfitting**: low bias, high variance.
- Somewhere in the middle is the sweet spot — the complexity that minimizes *total* error, not just bias or just variance alone.

This directly explains:
- **k-NN**: small $k$ (like $k=1$) is very flexible → low bias, high variance (each prediction depends on one nearby point — noisy). Large $k$ averages over more neighbours → high bias (averaging pulls the prediction toward a smoother, less locally-accurate value), low variance.
- **Polynomial regression**: degree-1 (a straight line) is rigid → high bias. Degree-15 wiggles through every point → high variance.

## Worked numeric example

True value at some point: $f(x_0) = 10$, noise variance $\sigma^2 = 1$. Simulate training on 4 different random datasets, record what each fitted model predicts.

**Estimator A — simple and biased, but stable:** predictions $\{8, 8, 9, 9\}$.
- Average prediction: $\bar f_A = (8+8+9+9)/4 = 8.5$
- Bias $= 8.5 - 10 = -1.5$, so Bias² $= 2.25$
- Variance $= \frac{(8-8.5)^2 \times 2 + (9-8.5)^2\times 2}{4} = \frac{0.25\times4}{4} = 0.25$
- Total expected error $= 2.25 + 0.25 + 1 = \mathbf{3.5}$

**Estimator B — flexible and unbiased, but jittery:** predictions $\{6, 14, 7, 13\}$.
- Average prediction: $\bar f_B = (6+14+7+13)/4 = 10$ — exactly right on average!
- Bias $= 10 - 10 = 0$, so Bias² $= 0$
- Variance $= \frac{(6-10)^2+(14-10)^2+(7-10)^2+(13-10)^2}{4} = \frac{16+16+9+9}{4} = 12.5$
- Total expected error $= 0 + 12.5 + 1 = \mathbf{13.5}$

**The punchline:** Estimator B is *perfectly unbiased* — its average prediction hits the true value exactly. Yet it performs **much worse** overall (13.5 vs 3.5) than the biased Estimator A, purely because its predictions are wildly inconsistent from one dataset to the next. This proves the lecture's key insight: **a little bias is not automatically bad — a slightly-wrong-but-stable model can easily beat a correct-on-average-but-erratic one.**
