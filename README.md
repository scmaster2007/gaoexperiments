# Hypotheses and Questions for the Gao Lab Project

The main question I would like to explore in considerable depth is "Given some realistic deep, feature-rich neural network and a weight-adaptive optimizer, which function does it choose after the model is finished training and why is the model inclined to choose this function?"

Some potential subdirections / hypotheses are:
1. Studying the innate structure of data, developing some language to talk about data and complexity of datasets / difficulty of a task and feature using classical tools in statistical learning theory (kolmogorov complexity, bit length, sharpness, etc.)
2. First observing how the function space (NOT parameter space) is altered by the adaptive update either using PDEs / saddle-to-saddle method / some other related mathematical tool
3. If there is some universal conservation law for adaptive optimizers relating to function spaces that could imply an existence of a "law" to choose simplest solution, etc.

Hypotheses:
1. The normalization that \sqrt(v_t) induces in the parameter update rule acts as a sort of negative feedback loop which makes it implicitly optimize a measure of generalization (be it sharpness, or hessian, etc) -> easily testable
2. Because the beta coefficients are traditionally very small, ADAM has a behavior of increasing its update by the magnitude of the learning rate, |\eta|. This means that it effectively maximizes the l_1 norm: does the extra sidestep of going "1" unit versus \sqrt(2) increase time complexity, potentially relating adaptive optimizers to saddle-to-saddle dynamics? -> easily testable with coding agent.
3. There exists a clean conservation law for adaptive optimizers in a realistic functional space landscape which implies the existence of an a priori method to determine final function at EOT -> need to split this up into smaller subquestions further for easy testing + garnering intuition

## Results so far

Most of these scripts I ran on small CPU pilots to gauge the interestingness of the direction, the scripts are in the experiments/README.md file.

- **Hypothesis 1:** 50% correct -> the feedback loop from normalization is real, but it minimizes a different variant of sharpness than the traditional sharpness metric. In fact, compared to GD, Adam ends up in a sharper region of the loss landscape. To also test that this is caused by the feedback loop, in one instance we freeze v_t -> this resulted in near identical result of GD and frozen adam.
- **Hypothesis 2:** partly correct -> in lin reg., adam does not find the GD's closest-to-init solution that was theory in /thoughts. instead, it landed near sign-GD, which implies that my intuition about increasing by sign was correct. But, this sidestep does not seem to change time complexity.
- **Hypothesis 3:** not tested yet still, need to break down into smaller theoretical questions that I can think about + run.

## Layout

```
aixi/          Earlier AIXI experiments: exact Solomonoff-style prior vs. trained transformers (experiments 1–4). See aixi/README.md.
experiments/   Quick experiments on the hypotheses above and the follow-up questions in thoughts/ (findings in experiments/README.md)
thoughts/      ideation on specific papers that cover same subject material, exist to serve as notes repository + add to personal intuition
```

Scripts in `aixi/` use paths relative to that folder, so run them from inside it (e.g. `cd aixi && python3 src/analyze.py`). The full setup, results tables and reproduce commands are in `aixi/notes/details.md`. Previously, this directory used to function as a storage for experiments related to AIXI (the universal learner), but I do not think that this is as tractable of a question as the new one.
