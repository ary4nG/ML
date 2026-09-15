# Lecture 5 — Statistical Decision Theory for Regression

## The one question this whole lecture answers

Imagine you have to predict the temperature at 3 a.m. tomorrow. You can't say "it depends" — you must commit to **one number**.

But here's the problem: temperature at 3 a.m. isn't fixed. Look at 3 a.m. on different nights and you'll see different values — say 19°C, 21°C, 23°C. There's noise (weather is random) and hidden factors (which day, humidity, etc.) that you can't fully account for.

**So the real question is:** given that the truth is really a *spread* of possible values, what single number should you guess to be as close as possible, on average?

The intuitive answer is **the mean** (the average) — and that single fact is the seed of this entire lecture. If you have several actual readings at 3 a.m. — say 19, 21, 23 — and you have to pick **one number** to represent all of them, guessing the average (21) is the safest bet. If you guess too high, you overshoot on the low nights; if you guess too low, you undershoot on the high nights. The average balances these over/undershoots as evenly as possible.

This idea — "the best single guess is the average of what actually happens" — has a formal name: the **conditional expectation**, written $\mathbb{E}[Y \mid X = x]$. In plain words it means: *"the average value of Y, restricted to only the situations where X equals x."*

The professor calls this the **regression function**, and proving that this is truly the best possible guess (not just intuitively, but mathematically) is the main content of the lecture.

---

## Step 1: Setting up the problem precisely

Ingredients, using the notation you'll see everywhere in this course:

- $X$ = your input (like "time of day", or several inputs like "time, humidity, altitude")
- $Y$ = the actual output you're trying to predict (like temperature) — this is *random*, meaning it varies
- $f(x)$ = your prediction function — given an input $x$, it spits out your guess
- **Loss function**: how do we measure "how wrong" a guess is? The standard choice in regression is **squared error**: $(Y - f(X))^2$. Why squared and not just the plain difference? A plain difference could be negative (undershoot) or positive (overshoot), and they'd cancel out when averaged, hiding how wrong you actually are. Squaring makes every error positive, so errors always add up honestly.

The number we want to minimize is the **Expected Prediction Error (EPE)**:

$$
\text{EPE}(f) = \mathbb{E}[(Y - f(X))^2]
$$

In plain English: *"If I used this prediction function $f$ on every possible input, and compared it to every possible true output, what would my average squared mistake be?"* We want the $f$ that makes this number as small as possible.

**Key exam detail:** the expectation here is over the **joint distribution** of $X$ and $Y$ together — averaging over every possible combination of input *and* output, weighted by how likely each combination is. Both "which inputs show up" and "what outputs occur at that input" are random.

---

## Step 2: Breaking the big average into two smaller averages

Classic math trick: instead of averaging over $X$ and $Y$ jointly all at once, split it using a basic probability rule:

$$
p(x, y) = p(y \mid x) \cdot p(x)
$$

In words: *"the chance of seeing (x, y) together = the chance of seeing x, times the chance of seeing y given that x already happened."*

Using this, EPE becomes:

$$
\text{EPE}(f) = \mathbb{E}_X \Big[ \ \mathbb{E}_{Y \mid X}\big[(Y - f(X))^2 \mid X\big] \Big]
$$

Two nested steps:
1. **Inner step**: Freeze $x$ at one specific value. Look at just the spread of $Y$ values that occur at that $x$, and compute the average squared error there.
2. **Outer step**: Do that inner step for every possible $x$, then average those results together, weighted by how often each $x$ occurs.

Why split it like this? Because it lets us optimize $f$ **one input at a time** instead of over the whole messy joint distribution at once.

---

## Step 3: Minimize one input at a time

Since $f$ can be *any* function (no assumed shape yet), we're free to choose $f(x)$ for each $x$ completely independently of what it does elsewhere. Since the outer average is just a sum of non-negative pieces (one per $x$), making the total sum smallest means making **every individual piece** as small as possible.

Fix one particular $x$. Here, your prediction $f(x)$ is just *some number* — call it $c$. We ask: what value of $c$ minimizes

$$
g(c) = \mathbb{E}[(Y - c)^2 \mid X = x]
$$

Now it's a simple one-variable calculus problem.

---

## Step 4: Solve it with calculus

Take the derivative w.r.t. $c$, set it to zero:

$$
\frac{d}{dc}\mathbb{E}[(Y-c)^2 \mid X=x] = \mathbb{E}[-2(Y-c) \mid X=x] = 0
$$

Solving for $c$:

$$
\mathbb{E}[Y \mid X=x] - c = 0 \quad\Rightarrow\quad c = \mathbb{E}[Y \mid X=x]
$$

(Second derivative is $2 > 0$, confirming a minimum.)

We've now **proven**:

$$
\boxed{f(x) = \mathbb{E}[Y \mid X = x]}
$$

This is the **regression function**: the best possible prediction, under squared-error loss, is the conditional mean.

---

## Step 5: What if we used a different loss?

This mean result is *specific to squared-error loss*. With **absolute difference** loss, $|Y - f(X)|$, the best predictor becomes the **median** instead of the mean.

Why it matters: the mean is very sensitive to outliers; the median barely moves. Example: readings are 19, 21, 23 (mean = median = 21). Add a broken sensor reading of 60: 19, 21, 23, 60.
- New mean = (19+21+23+60)/4 = **30.75** — dragged way up by the outlier.
- New median = middle of 21 and 23 = **22** — barely changed.

The choice of loss function directly determines what kind of "average" you're computing, and whether your model is robust to bad data.

---

## Step 6: The catch — this perfect formula is useless in practice

We proved $f(x) = \mathbb{E}[Y \mid X=x]$ is the best possible guess. Two real problems block us from computing it directly:

1. We don't actually know $P(Y \mid X)$ — the true underlying probability rule of nature. We only have a limited set of observed data points.
2. Even if we tried to literally average all $y$ values where $x_i$ equals our query $x$ exactly — for continuous inputs, almost never will any training point land *exactly* on the $x$ we're querying. Nothing to average.

This is where the two famous algorithms come from — two different ways of patching up this "unusable in practice" formula.

---

## Step 7: Fix #1 — k-Nearest Neighbours (k-NN)

Instead of "average $y$ at exactly this $x$" (impossible), relax to: "average $y$ over the $k$ closest training points to $x$":

$$
\hat f(x) = \frac{1}{k} \sum_{x_i \in N_k(x)} y_i
$$

where $N_k(x)$ = "the set of $k$ nearest training points to $x$."

Two approximations compared to the ideal:
- **True expectation → sample average** (using actual observed points instead of a theoretical average)
- **Exact point → neighbourhood region** (implicitly assumes $f$ is roughly constant within that small neighbourhood)

**When does this work well?** As data grows ($n \to \infty$) and $k$ grows too, but slower than $n$ (so $k/n \to 0$), k-NN's estimate provably converges to the true regression function. In practice:
- You never truly have infinite data.
- **Curse of dimensionality**: with many input features (high $p$), the "k nearest neighbours" spread across a huge, sparse region — so "roughly constant nearby" stops being safe, and k-NN's accuracy degrades badly.

---

## Step 8: Fix #2 — Linear Regression

Instead of relaxing "exact point" into "nearby region" (flexible but local), linear regression bets the opposite way: assume $f$ has **one fixed global shape** — a straight line (or plane):

$$
f(x) = x^\top \beta
$$

Minimize the *training* version of the error directly:

$$
\hat R(\beta) = \|y - X\beta\|^2
$$

Taking the derivative and setting it to zero gives the **normal equations**, solved by:

$$
\boxed{\hat\beta = (X^\top X)^{-1} X^\top y}
$$

**Geometric picture:** think of $X$'s columns as spanning a flat subspace inside a bigger space. The prediction $\hat y = X\hat\beta$ is literally the **shadow** (orthogonal projection) of the true $y$ onto that flat subspace — the closest point on that "plane" to the actual data. The leftover error, $y - \hat y$, points perfectly perpendicular to that plane.

---

## The big unifying idea

Both algorithms start from the exact same place — minimize EPE, true answer is the conditional mean $\mathbb{E}[Y|X=x]$ — but they patch the "can't compute it directly" problem in two opposite ways:

| Assumption | Algorithm |
|---|---|
| f is locally constant (flexible, but only trusts nearby data) | k-Nearest Neighbours |
| f is globally linear (rigid, but uses all data at once) | Linear Regression |

**Core takeaway:** one principle, different assumptions, different algorithms.
