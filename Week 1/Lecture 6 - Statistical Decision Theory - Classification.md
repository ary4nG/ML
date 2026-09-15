# Lecture 6 — Statistical Decision Theory for Classification

This lecture takes the exact same recipe from Lecture 5 and applies it to a different kind of output: instead of predicting a number (like temperature), you're now predicting a **category** (like "has disease" vs "doesn't have disease").

## Step 0: Why can't we just reuse squared error?

In Lecture 5, our output $Y$ was a real number, so "how wrong were you" could be measured with subtraction and squaring: $(Y - f(X))^2$.

Now the output is a **label**, like "cat" vs "dog," or "disease" vs "no disease." There's no numerical distance between labels — "dog" minus "cat" doesn't mean anything. So squared error is meaningless here. We need a completely different way to measure "how bad was my guess."

## Step 1: Introducing the loss matrix

Instead of a formula, we just write down, by hand, a table of costs: *"if the truth is class $k$, and I guessed class $\ell$, how much does that cost me?"* This table is called the **loss matrix** $L$, and $L_{k\ell}$ means "the cost of guessing $\ell$ when the truth was $k$."

The simplest and most common version is **0-1 loss**: every mistake costs exactly 1, and being correct costs 0. So the diagonal (guess = truth) is all zeros, and everywhere else is 1. For 3 classes:

$$
L = \begin{pmatrix} 0&1&1 \\ 1&0&1 \\ 1&1&0 \end{pmatrix}
$$

This just encodes "all wrong answers are equally bad" — no mistake is worse than another (this assumption isn't always realistic — misdiagnosing a serious disease is worse than a false alarm).

## Step 2: What's a "posterior probability"?

$P(G = k \mid X = x)$ reads as *"the probability that the true class is $k$, given that we observed input $x$."* This is called the **posterior probability** of class $k$.

Concrete example: a patient walks in with symptoms $x$. Among all patients who look like $x$, maybe 70% actually have the disease and 30% don't. So $P(\text{disease} \mid x) = 0.7$ and $P(\text{no disease} \mid x) = 0.3$. You don't get to know which specific bucket *this* patient falls in — you only know the probabilities.

## Step 3: Setting up the same "minimize expected error" question

Just like Lecture 5, we want to find the prediction rule $f(x)$ that minimizes the **expected loss**, averaged over the randomness in the true class:

$$
\text{EPE} = \mathbb{E}_X\Big[\ \mathbb{E}_{G \mid X}\big[L(G, f(X)) \mid X\big]\Big]
$$

Identical structure to Lecture 5 — outer average over which input occurs, inner average over the randomness of the true label given that input. The only difference: since $G$ (the class) is discrete, the "inner expectation" isn't an integral anymore — it's a finite **sum** over the possible classes.

For a fixed $x$, if we commit to predicting some specific class $g$, our expected loss is:

$$
\mathbb{E}[L(G, g) \mid X = x] = \sum_{k=1}^{K} L_{kg} \cdot P(G = k \mid X = x)
$$

In plain words: *"for each possible true class $k$, multiply the cost of guessing $g$ when truth is $k$, by the probability that $k$ is actually the truth — then add all of that up."*

## Step 4: Minimizing this to get the Bayes classifier

We want to pick whichever $g$ makes that sum smallest:

$$
f^\star(x) = \arg\min_{g} \sum_{k=1}^K L_{kg} \, P(G=k \mid x)
$$

($\arg\min$ means "the value of $g$ that makes this expression smallest," not the minimum value itself.)

Plug in the 0-1 loss specifically. Since $L_{kg} = 1$ for every $k \ne g$, and $0$ when $k = g$:

$$
\sum_k L_{kg} P(G=k\mid x) = \sum_{k \ne g} P(G=k \mid x) = 1 - P(G = g \mid x)
$$

Makes sense: "the expected cost of guessing $g$" is just "the total probability that the truth was something *other* than $g$" — every wrong guess costs exactly 1, matching $g$ contributes zero.

So minimizing $1 - P(G=g\mid x)$ is the same as **maximizing** $P(G=g \mid x)$:

$$
\boxed{f^\star(x) = \arg\max_k P(G=k \mid X=x)}
$$

This is the **Bayes optimal classifier**: at every input $x$, predict whichever class has the highest posterior probability. In the disease example: since 70% > 30%, always bet "disease" — wrong only 30% of the time, beating "no disease" (wrong 70% of the time).

## Step 5: The Bayes error — mistakes you can never avoid

Even the perfect, all-knowing Bayes classifier still makes mistakes! At a given $x$, it's wrong whenever the true label isn't the majority class — probability $1 - \max_k P(G=k \mid x)$ (in the disease example, the 30% who don't fit the majority pattern).

Averaging this over every possible input gives the **Bayes error rate**:

$$
\text{Err}_{\text{Bayes}} = \mathbb{E}_X\big[1 - \max_k P(G=k \mid X)\big]
$$

This is called **irreducible** — no classifier, however clever or data-rich, can ever beat it. It's not a limitation of the model; it's a limitation of the problem itself, because classes genuinely overlap (two people with identical symptoms $x$ can have different true outcomes). This is the classification version of "noise," which reappears in the bias-variance tradeoff (Lecture 7).

## Step 6: Practical estimation — we don't know the true posterior

$f^\star(x) = \arg\max_k P(G=k\mid x)$ is exact but useless directly — we don't actually know the true probabilities; we only have training data.

**Fix: k-NN classifier.** Estimate the posterior locally: among the $k$ nearest neighbours of $x$, count what fraction belong to each class:

$$
\hat P(G=k \mid x) = \frac{1}{k}\sum_{x_i \in N_k(x)} \mathbb{1}[y_i = k]
$$

($\mathbb{1}[\cdot]$ means "1 if true, 0 if false" — counting how many neighbours have label $k$, divided by $k$.)

Since the denominator ($k$) is the same regardless of class, taking the argmax of these estimated probabilities is the same as picking whichever class appears **most often** among the neighbours — i.e., **majority vote**. k-NN classification is a plug-in approximation of the Bayes classifier, using nearby data to estimate probabilities we don't actually know. Same caveats as regression apply — needs enough data, struggles in high dimensions.

## Step 7: Linear regression as a classifier

Take a two-class problem, encode labels as numbers: $y=1$ for one class, $y=0$ for the other, and fit ordinary linear regression. If a bunch of training points at the same $x$ have labels 1,1,1,0,0, regression's best guess there (the mean, from Lecture 5) is $3/5 = 0.6$ — exactly the fraction of neighbours with label 1, i.e. an estimate of $P(G=1\mid x)$.

Treat the regression output as an estimated probability, then threshold: predict class 1 if $\hat f(x) \ge 0.5$, else class 0.

**Catch:** linear regression's output can go below 0 or above 1 (a straight line has no bounds), which makes no sense as a probability. **Logistic regression** (a later lecture) fixes this properly.

---

## Worked example (from the lecture)

Three classes at some input $x$: $P(1\mid x) = 0.6$, $P(2\mid x)=0.2$, $P(3\mid x)=0.2$, using 0-1 loss.

Expected loss for each possible guess $g$ (recall: expected loss $= 1 - P(g\mid x)$):

| Guess $g$ | Expected loss |
|---|---|
| $g=1$ | $0.2+0.2 = 0.4$ |
| $g=2$ | $0.6+0.2 = 0.8$ |
| $g=3$ | $0.6+0.2 = 0.8$ |

Lowest expected loss is at $g=1$ — matching $\arg\max_k P(k\mid x) = 1$. Bayes classifier picks class 1; unavoidable error rate at this $x$ is $1 - 0.6 = 0.4$ (40%).

---

## The big picture — same recipe as Lecture 5

| | Regression (Lecture 5) | Classification (Lecture 6) |
|---|---|---|
| Output | continuous number | discrete label |
| Loss | squared error | loss matrix (usually 0-1 loss) |
| Optimal rule | $f(x) = \mathbb{E}[Y\mid x]$ (the mean) | $f(x) = \arg\max_k P(G=k\mid x)$ (most likely class) |
| Unavoidable limit | noise variance | Bayes error rate |
| Practical fix | k-NN average / linear regression | k-NN majority vote / linear regression + threshold |

Same underlying idea both times: figure out what's truly optimal given perfect knowledge of the probabilities, realize you don't actually have that knowledge, then approximate it using nearby data.
